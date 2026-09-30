"""Static-data social skill evolution primitives. Execute tests/runs through Slurm."""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import math
import os
import multiprocessing
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import rung4
from social_baseline import HierarchicalSocialUCB


CONDITIONS = {
    'llm_solo': ('llm', 'none'),
    'llm_social_payoff': ('llm', 'payoff'),
    'openevolve_solo': ('openevolve', 'none'),
    'ucb_social': ('ucb', 'payoff'),
    'uniform_social': ('uniform', 'payoff'),
}
BASE_CONDITIONS = tuple(CONDITIONS)
TIMING_CONDITIONS = ('forced_early', 'forced_middle', 'forced_late', 'forced_distributed',
                     'optional_early', 'optional_middle', 'optional_late')
CONDITIONS.update({name: ('llm', 'payoff') for name in TIMING_CONDITIONS})

EVOLUTION_PROTOCOL = 'offline_parity_v2'


def timing_spec(condition, seed, agents, smoke=False):
    """Preassign schedules without touching learner RNG or consulting outcomes."""
    mode, window = condition.split('_', 1)
    if condition not in TIMING_CONDITIONS:
        raise ValueError('Unknown timing condition')
    intervals = {'early': [(1, 10)], 'middle': [(20, 29)], 'late': [(40, 49)],
                 'distributed': [(1, 12), (13, 24), (25, 36), (37, 49)]}
    ranges = [[1, 1]] if smoke else [list(pair) for pair in intervals[window]]
    count = 1 if smoke else 4
    schedules = {}
    for agent in range(agents):
        rng = random.Random(rung4.seed_for(seed, agent, condition, 'observation-timing-v1'))
        schedules[str(agent)] = (sorted(rng.sample(range(ranges[0][0], ranges[0][1] + 1), count))
            if len(ranges) == 1 else [rng.randint(lo, hi) for lo, hi in ranges]) if mode == 'forced' else []
    return dict(mode=mode, window=window, intervals=ranges, count=count if mode == 'forced' else None,
                schedules=schedules)


def observation_access(config, agent, r):
    spec = config.get('observation_timing')
    if spec is None:
        return True, False
    if spec['mode'] == 'forced':
        forced = r in spec['schedules'][str(agent)]
        return forced, forced
    return any(lo <= r <= hi for lo, hi in spec['intervals']), False


class CheckpointSet(set):
    """Replay add/discard history to preserve OpenEvolve's native set ordering.

    Sampling indexes lists made from these sets. Rebuilding from sorted members
    can select a different parent with the very same RNG seed. The launcher pins
    PYTHONHASHSEED; replaying mutations preserves the original set table layout.
    This does not change iteration or selection semantics during a live run.
    """
    def __init__(self, history=()):
        super().__init__()
        self.history = list(history)
        for op, value in history:
            getattr(super(), op)(value)

    def add(self, value):
        super().add(value)
        self.history.append(('add', value))

    def discard(self, value):
        super().discard(value)
        self.history.append(('discard', value))

    def remove(self, value):
        super().remove(value)
        self.history.append(('discard', value))


def learner_seed(config, agent_id):
    """Agent 0 reuses the original solo seed; other agents have explicit seeds."""
    seeds = config['agent_seeds']
    if len(seeds) != config['num_agents'] or len(set(seeds)) != len(seeds):
        raise ValueError('Expected one distinct, explicit seed per agent')
    return seeds[agent_id]


def restore_random_state(value):
    # JSON turns the inner RNG-state tuple into a list.
    random.setstate((value[0], tuple(value[1]), value[2]))


def acquire_observation(learner, snapshot: dict, info: str, round_index: int) -> dict:
    """Acquire in one observation, without forcing deployment or copying private feedback."""
    if info not in {'action', 'payoff'}:
        raise ValueError('Observation requires social access')
    if snapshot['agent'] == learner.id or snapshot['round'] >= round_index:
        raise ValueError('Only another agent\'s completed prior round is observable')
    vid = snapshot['version']
    if rung4.digest(snapshot['artifact']['files'])[:24] != vid:
        raise ValueError('Observed version does not match immutable file contents')
    # Preserve source lineage, not the source's development scores or feedback.
    if vid not in learner.versions:
        artifact = copy.deepcopy(snapshot['artifact'])
        artifact['development_scores'] = []
        learner.versions[vid] = artifact
    learner.acquisitions.setdefault(vid, dict(origin='social', round=round_index,
                                             source=snapshot['agent']))
    result = dict(version=vid, files=copy.deepcopy(snapshot['artifact']['files']))
    if info == 'payoff':
        result['training_score'] = snapshot['training_score']
    learner.observations.append(dict(target=snapshot['agent'], source_round=snapshot['round'],
                                    acquired_round=round_index, **copy.deepcopy(result)))
    return result


def public_library(learner, known_scores: dict) -> list[dict]:
    """Only the recipient's acquired files and legitimately known scores enter its prompt."""
    return [dict(version=vid, files=copy.deepcopy(value['files']),
                 training_score=known_scores.get(vid),
                 origin=learner.acquisitions[vid]['origin'])
            for vid, value in sorted(learner.versions.items())]


def source_controller(agent_id: int, seed: int, count: int, saved: dict | None = None):
    # Scores lie in [0,1]. Fix the common prior at .5 and the self advantage at
    # .05 (the original rule's half-noise-SD with declared adapter scale .1).
    selector = HierarchicalSocialUCB(agent_id, seed, count, 1, .5, .1)
    if saved is not None:
        selector.source_counts[:] = saved['counts']
        selector.source_reward_sums[:] = saved['sums']
    return selector


def source_state(selector) -> dict:
    return dict(counts=selector.source_counts.tolist(), sums=selector.source_reward_sums.tolist())


def evaluate_cached(model, benchmark, tasks: list[str], files: dict, directory: Path,
                    seed: int, phase: str, per_task_tokens: int, *,
                    holdout_repeat: int = 0) -> tuple[dict, rung4.Budget]:
    """Save each paid answer independently; cached results retain their original costs.

    This directory belongs to one fixed evaluation, not to an entire artifact:
    fresh evaluation seeds must never accidentally reuse old fitness samples.
    """
    if len(set(tasks)) != len(tasks) or not tasks:
        raise ValueError('Evaluation needs unique, nonempty task IDs')
    if phase == 'holdout' and not set(tasks) <= set(benchmark.holdout_ids):
        raise ValueError('Held-out evaluation contains training tasks')
    if phase != 'holdout' and not set(tasks) <= set(benchmark.train_ids):
        raise ValueError('Learning evaluation contains hidden tasks')
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / 'complete.json'
    saved = json.loads(complete.read_text()) if complete.exists() else None
    saved_by_task = {v['identity']['task']: v for v in saved} if saved is not None else None
    if saved_by_task is not None and set(saved_by_task) != set(tasks):
        raise ValueError('Refusing incompatible completed task cohort')
    def one(task):
        execution_seed = (rung4.seed_for(seed, task, holdout_repeat, 'holdout')
                          if phase == 'holdout' else rung4.seed_for(seed, task))
        identity = dict(task=task, version=rung4.digest(files), seed=execution_seed,
                        phase=phase, budget=per_task_tokens, dataset=rung4.digest(benchmark.manifest),
                        model=rung4.experiment_config({'model': model.config})['model'])
        path = directory / (rung4.digest(task)[:20] + '.json')
        if saved_by_task is not None:
            value = saved_by_task[task]
            if value['identity'] != identity:
                raise ValueError('Refusing incompatible cached evaluation')
            return value
        if path.exists():
            value = json.loads(path.read_text())
            if value['identity'] != identity:
                raise ValueError('Refusing incompatible cached evaluation')
            return value
        budget = rung4.Budget(per_task_tokens)
        result = rung4.execute(model, benchmark, None, task, files, budget,
                               identity['seed'], phase)
        value = dict(identity=identity, result=result, budget=vars(budget))
        rung4.atomic_json(path, value)
        return value
    with ThreadPoolExecutor(max_workers=min(len(tasks), model.config.get('parallel_requests', 1))) as pool:
        values = list(pool.map(one, tasks))
    if saved is None:
        # Keep per-task durability during work, then collapse the finished cohort
        # to one inode. Partial cleanup is harmless: complete.json is authoritative.
        rung4.atomic_json(complete, values)
    for task in tasks:
        (directory / (rung4.digest(task)[:20] + '.json')).unlink(missing_ok=True)
    budget = rung4.Budget(len(tasks) * per_task_tokens)
    for value in values:
        budget.absorb(rung4.Budget(**value['budget']))
    results = [v['result'] for v in values]
    return dict(reward=sum(r['constraint_accuracy'] for r in results) / len(results),
                prompt_accuracy=sum(r['prompt_level_loose'] for r in results) / len(results),
                evaluated_count=len(results), evaluations=results,
                submitted_rate=sum(r['submitted'] for r in results) / len(results)), budget


class PopulationAgent:
    """An independently checkpointable learner; peers enter only via prior snapshots."""

    def __init__(self, config, agent_id, model, benchmark, directory, saved=None):
        from rung4_instruction import InstructionLearner, MINIMAL_SKILL
        self.config, self.id, self.model = config, agent_id, model
        self.benchmark, self.directory = benchmark, directory
        self.seed = learner_seed(config, agent_id)
        if saved and saved.get('evolution_protocol') != EVOLUTION_PROTOCOL:
            raise ValueError('Old population checkpoint: parity correction requires a new run')
        # Exactly the two permutations used by rung4.run for offline training.
        development = random.Random(rung4.seed_for(self.seed, 'offline-training')).sample(
            benchmark.train_ids, len(benchmark.train_ids))
        self.tasks = random.Random(rung4.seed_for(self.seed, 'population-tasks')).sample(
            development, len(development))
        self.kind, self.info = CONDITIONS[config['condition']]
        if saved:
            state = saved['learner']
            self.learner = InstructionLearner(agent_id, rung4.seed_for(self.seed, 0), state['versions'], state['active'],
                                              state['acquisitions'], state['observations'], MINIMAL_SKILL)
        else:
            self.learner = InstructionLearner(agent_id, rung4.seed_for(self.seed, 0), initial_skill=MINIMAL_SKILL)
        self.learner.archive_config = rung4.full_archive_config(self.seed, config)
        self.learner.archive = rung4.ProgramDatabase(self.learner.archive_config)
        self.learner.sampler.config.use_template_stochasticity = True
        if saved:
            if saved['python_hash_probe'] != hash('agentmarket-openevolve-parity'):
                raise ValueError('Resume requires the same PYTHONHASHSEED (launcher uses 0)')
            archive = copy.deepcopy(saved['archive'])
            # atomic_json sorts dictionary keys; insertion order breaks score ties
            # and enters prompt sampling, so restore it explicitly.
            archive['programs'] = {key: archive['programs'][key] for key in saved['program_order']}
            rung4.restore_archive(self.learner.archive, archive)
            # OpenEvolve keys this cache by Python's process-local string hash.
            self.learner.archive.diversity_cache = {
                hash(code): entry for code, entry in saved['diversity_cache']}
            self.learner.archive.islands = [CheckpointSet(h) for h in saved['island_set_history']]
            self.learner.archive.archive = CheckpointSet(saved['archive_set_history'])
            if ([sorted(s) for s in self.learner.archive.islands] != archive['islands']
                    or sorted(self.learner.archive.archive) != archive['archive']):
                raise ValueError('Corrupt archive set history')
        else:
            self.learner.archive.islands = [CheckpointSet() for _ in self.learner.archive.islands]
            self.learner.archive.archive = CheckpointSet()
        self.scores = copy.deepcopy(saved['scores']) if saved else {}
        self.program_lineage = copy.deepcopy(saved['program_lineage']) if saved else {}
        # The original revision prompt uses json.dumps(feedback) without sorting.
        # Preserve that text ordering through atomic_json's sorted checkpoints.
        self.feedback = json.loads(saved['feedback_json']) if saved else []
        self.source = source_controller(agent_id, config['seed'], config['num_agents'],
                                        saved['source'] if saved else None)
        self.random_state = saved['random_state'] if saved else random.getstate()

    def state(self):
        archive = self.learner.archive
        codes = dict.fromkeys([v['files']['SKILL.md'] for v in self.learner.versions.values()]
                             + [p.code for p in archive.programs.values()])
        return dict(evolution_protocol=EVOLUTION_PROTOCOL, random_state=self.random_state,
                    python_hash_probe=hash('agentmarket-openevolve-parity'),
                    island_set_history=[s.history for s in archive.islands],
                    archive_set_history=archive.archive.history,
                    program_order=list(archive.programs),
                    diversity_cache=[(code, archive.diversity_cache[hash(code)]) for code in codes
                                     if hash(code) in archive.diversity_cache],
                    learner=self.learner.state(), archive=rung4.archive_state(self.learner.archive),
                    scores=self.scores, program_lineage=self.program_lineage,
                    feedback_json=json.dumps(self.feedback), source=source_state(self.source))

    def program_for_version(self, vid):
        """LLMs retain acquired skills even if MAP-Elites evicts their archive cell."""
        code = self.learner.versions[vid]['files']['SKILL.md']
        program = self.learner.archive.programs.get(vid)
        if program is None:
            program = next((p for p in self.learner.archive.programs.values() if p.code == code), None)
        if program is not None:
            return program
        return rung4.Program(id=vid, code=code, language='markdown',
            metrics={'combined_score': self.scores[vid]}, **self.program_lineage[vid])

    def register_score(self, vid, score, round_index, *, parent=None, island=0):
        """Match original insertion, not a running-average/re-registration policy."""
        learner = self.learner
        learner.versions[vid]['development_scores'].append(score)
        self.scores[vid] = score
        self.program_lineage[vid] = dict(parent_id=parent.id if parent else None,
                                        generation=parent.generation + 1 if parent else 0)
        learner.archive.add(rung4.Program(id=vid, code=learner.versions[vid]['files']['SKILL.md'],
            language='markdown', parent_id=parent.id if parent else None,
            generation=parent.generation + 1 if parent else 0,
            iteration_found=round_index, metrics={'combined_score': score},
            metadata={'island': island}), iteration=round_index, target_island=island)

    def evaluate(self, vid, r, phase, budget):
        result, used = evaluate_cached(self.model, self.benchmark, self.tasks,
            self.learner.versions[vid]['files'], self.directory / 'evaluations' / f'{r:04d}-{phase}-{vid}',
            rung4.seed_for(self.seed, r, 'matched-minibatch'), phase, self.config['training_task_tokens'])
        budget.absorb(used)
        return result

    def decision(self, r, stage, prior, budget, events):
        learner = self.learner
        catalog = [dict(version=v['version'], training_score=v['training_score'], origin=v['origin'],
                        description=v['files']['SKILL.md'][:600])
                   for v in public_library(learner, self.scores)]
        # No peer score/file is visible until observation has been selected.
        available = [i for i, p in enumerate(prior) if i != self.id and p is not None] if self.info != 'none' else []
        access, forced = observation_access(self.config, self.id, r)
        if not access:
            available = []
        allowed = ['deploy'] if stage == 'deployment' else ['revise', 'evaluate', 'keep'] + (['observe'] if available else [])
        if forced and stage != 'deployment':
            if not available:
                raise ValueError('Assigned observation has no available peer')
            allowed = ['observe']
        prompt = dict(stage=stage, round=r, agent=self.id, current_version=learner.active,
                      current_skill=learner.files, repertoire=catalog, observable_peers=available,
                      allowed_actions=allowed,
                      recent_observation=learner.observations[-1:] if learner.observations else [])
        instructions = (
            'You improve a reusable instruction-following skill on a fixed training set. '
            'Choose how to allocate your learning, then choose which acquired skill to execute. '
            'Scores are fractions of satisfied training constraints, not hidden test scores. '
            'Observation automatically acquires the peer\'s last deployed full skill and its training score. '
            'It never forces execution and requires no adoption action. '
            'You have one learning action per round, followed by a deployment choice. '
            'Learning actions: revise (choose a known parent; OpenEvolve proposes and evaluates a revision), '
            'observe (choose an available peer), evaluate (test a known skill), or keep (no learning expense). '
            'Deployment chooses any acquired version; no revision or observation occurs during deployment. '
            'Return only JSON with one action from allowed_actions. Include version for revise, '
            'evaluate, or deploy, and target for observe. Only relevant fields are required. '
            'Invalid decisions preserve your incumbent; they do not silently trigger algorithmic selection.')
        instructions += (
            ' This call is the DEPLOYMENT stage. The only allowed action is deploy. '
            'Choose an acquired version to execute, e.g. {"action":"deploy","version":"known ID"}.'
            if stage == 'deployment' else
            ' This call is the LEARNING stage. Choose revise, evaluate, keep, or observe if a peer is available. '
            'To skip learning and keep your current skill, return {"action":"keep"}. '
            'Deploy is not allowed in this call; a separate deployment choice follows.')
        if forced and stage != 'deployment':
            instructions += ' This round is assigned to observation. Choose which available peer to observe; only observe is allowed.'
        remaining = min(self.config['controller_tokens'], budget.remaining - self.config['execution_tokens'])
        if remaining <= 0:
            events.append(dict(kind='decision_error', stage=stage, error='No controller budget'))
            return dict(action='keep', version=learner.active)
        message = self.model.call([dict(role='system', content=instructions),
                                  dict(role='user', content=json.dumps(prompt))],
                                 budget, stage, rung4.seed_for(self.seed, r, stage),
                                 reserve=budget.remaining - remaining)
        text = ''
        try:
            text = (message or {}).get('content', '').strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1].rsplit('```', 1)[0]
            choice = json.loads(text)
            if choice.get('action') not in allowed:
                raise ValueError('Invalid action for decision stage')
            if choice['action'] == 'observe':
                if type(choice.get('target')) is not int or choice['target'] not in available:
                    raise ValueError('Unavailable peer')
            elif choice['action'] != 'keep' and choice.get('version') not in learner.versions:
                raise ValueError('Unknown version')
            return choice
        except (ValueError, TypeError, AttributeError) as exc:
            events.append(dict(kind='decision_error', stage=stage, error=str(exc)[:300],
                               response_preview=text[:200]))
            return dict(action='keep', version=learner.active)

    def step(self, r, prior):
        learner = self.learner
        restore_random_state(self.random_state)
        started = time.monotonic()
        budget = rung4.Budget(self.config['tokens_per_round'])
        events = []
        incumbent = learner.active
        parent_evaluation = candidate_evaluation = None
        parent_id, candidate_id, oe_parent, island = incumbent, None, None, None
        if r == 0:
            island = 0
            parent_evaluation = self.evaluate(incumbent, r, 'development_parent', budget)
            self.register_score(incumbent, parent_evaluation['reward'], r, island=0)
            action = dict(action='initialize')
        else:
            if self.kind == 'llm':
                action = self.decision(r, 'selection', prior, budget, events)
                access, forced = observation_access(self.config, self.id, r)
                if self.config.get('observation_timing'):
                    events.append(dict(kind='observation_schedule', available=access, forced=forced))
                if forced and action['action'] != 'observe':
                    # Preserve assigned dose after malformed output, without another paid call.
                    peers = [i for i, p in enumerate(prior) if p is not None and i != self.id]
                    if not peers:
                        raise ValueError('Assigned observation has no available peer')
                    target = random.Random(rung4.seed_for(self.seed, r, 'forced-source-fallback')).choice(peers)
                    events.append(dict(kind='forced_source_fallback', target=target))
                    action = dict(action='observe', target=target)
                if not access and action['action'] == 'observe':
                    raise ValueError('Observation outside assigned access window')
            else:
                available = [self.id] + ([i for i, p in enumerate(prior) if p is not None and i != self.id]
                                          if self.info != 'none' else [])
                if self.kind == 'ucb':
                    source = self.source.choose_source(r, available)
                elif self.kind == 'uniform':
                    source = int(self.source.rng(r, 0).choice(sorted(available)))
                else:
                    source = self.id
                action = dict(action='revise' if source == self.id else 'observe', target=source)
            if action['action'] == 'observe':
                observed = acquire_observation(learner, prior[action['target']], self.info, r)
                if observed['version'] not in self.scores:
                    # Social extension: import the advertised fitness once, without
                    # pretending this was a private evaluation or generating feedback.
                    vid = observed['version']
                    self.scores[vid] = observed['training_score']
                    self.program_lineage[vid] = dict(parent_id=learner.versions[vid]['parent'],
                        generation=prior[action['target']].get('skill_generation', 0))
                    island = r % learner.archive_config.num_islands
                    learner.archive.add(rung4.Program(id=vid, code=observed['files']['SKILL.md'],
                        language='markdown', parent_id=learner.versions[vid]['parent'],
                        generation=prior[action['target']].get('skill_generation', 0),
                        iteration_found=r, metrics={'combined_score': observed['training_score']},
                        metadata={'island': island, 'social_source': action['target']}),
                        iteration=r, target_island=island)
                if self.kind == 'ucb':
                    self.source.update_source(action['target'], observed['training_score'])
                events.append(dict(kind='observe', target=action['target'], **observed))
            elif action['action'] in {'revise', 'evaluate'}:
                inspirations = []
                island = r % learner.archive_config.num_islands
                if self.kind == 'llm':
                    learner.active = action['version']
                    # An archive migration can assign another ID to identical code.
                    oe_parent = self.program_for_version(learner.active)
                else:
                    state = random.getstate()
                    random.seed(rung4.seed_for(self.seed, r, 'openevolve-selection'))
                    try:
                        oe_parent, inspirations = learner.archive.sample_from_island(
                            island, num_inspirations=5)
                    finally:
                        random.setstate(state)
                    learner.active = learner.install({'SKILL.md': oe_parent.code}, incumbent, 'openevolve_archive', r)
                    # Sampling a migrated copy is not new evidence or insertion.
                    self.scores.setdefault(learner.active, oe_parent.metrics['combined_score'])
                    self.program_lineage.setdefault(learner.active,
                        dict(parent_id=oe_parent.parent_id, generation=oe_parent.generation))
                parent_id = learner.active
                if action['action'] == 'revise':
                    future = (self.config['training_task_tokens'] * len(self.tasks)
                              + self.config['controller_tokens'] + self.config['execution_tokens'])
                    # Original loop leaves exactly one 32K allowance for revision.
                    revision_allowance = min(self.config.get('revision_tokens', 32768),
                                             max(0, budget.remaining - future))
                    reserve = budget.remaining - revision_allowance
                    changed = learner.revise(self.model, budget, self.feedback, r, reserve,
                        rung4.seed_for(self.seed, r, 'revision'),
                        experience_chars=self.config.get('revision_experience_max_chars', 350000),
                        inspirations=inspirations, top_programs=learner.archive.get_top_programs(5),
                        feature_dimensions=learner.archive_config.feature_dimensions)
                    events.append(dict(kind='revision', parent=parent_id, candidate=learner.active,
                                       changed=changed, error=learner.last_revision_error))
                    if changed:
                        candidate_id = learner.active
                        candidate_evaluation = self.evaluate(candidate_id, r, 'development_candidate', budget)
                        self.register_score(candidate_id, candidate_evaluation['reward'], r,
                                            parent=oe_parent, island=island)
                        learner.archive.increment_island_generation(island)
                        if learner.archive.should_migrate():
                            learner.archive.migrate_programs()
                else:
                    parent_evaluation = self.evaluate(parent_id, r, 'development_parent', budget)
                    # Explicit reevaluation is an LLM-only extension. Preserve ancestry.
                    learner.versions[parent_id]['development_scores'].append(parent_evaluation['reward'])
                    self.scores[parent_id] = parent_evaluation['reward']
                    for program in learner.archive.programs.values():
                        if program.code == learner.files['SKILL.md']:
                            program.metrics['combined_score'] = parent_evaluation['reward']
                    if not any(p.code == learner.files['SKILL.md'] for p in learner.archive.programs.values()):
                        oe_parent.metrics['combined_score'] = parent_evaluation['reward']
                        learner.archive.add(oe_parent, iteration=r, target_island=island)
                    learner.archive.best_program_id = None
            if self.kind == 'llm':
                choice = self.decision(r, 'deployment', prior, budget, events)
                learner.active = choice.get('version', learner.active)
            else:
                best = learner.archive.get_best_program()
                learner.active = learner.install({'SKILL.md': best.code}, parent_id, 'openevolve_archive', r)
                self.scores[learner.active] = best.metrics['combined_score']
            if self.kind == 'ucb' and action['action'] == 'revise':
                self.source.update_source(self.id, self.scores[learner.active])
        if action['action'] in {'initialize', 'revise', 'evaluate'}:
            # Even an unsuccessful revision leaves the same empty-evaluation
            # record as the original loop; observation adds no private feedback.
            self.feedback.append(rung4.compact_experience(self.benchmark, self.tasks,
                parent_evaluation, candidate_evaluation, r, 'fixed_offline_training_set'))
        # A protected fresh execution measures what the selected skill actually
        # does, separately from its possibly cached search-fitness estimate.
        task = self.benchmark.train_ids[rung4.seed_for(self.seed, r, 'deployment-task') % len(self.benchmark.train_ids)]
        execution, used = evaluate_cached(self.model, self.benchmark, [task], learner.files,
            self.directory / 'executions' / f'{r:04d}', rung4.seed_for(self.seed, r, 'deployment-execution'),
            'execution', self.config['execution_tokens'])
        budget.absorb(used)
        for vid, artifact in learner.versions.items():
            path = self.directory / 'artifacts' / f'{vid}.json'
            if not path.exists():
                rung4.atomic_json(path, artifact)
        self.random_state = random.getstate()
        deployed_program = self.program_for_version(learner.active)
        return dict(agent=self.id, round=r, action=action, version=learner.active,
                    learner_seed=self.seed, skill_generation=deployed_program.generation,
                    parent_version=parent_id, candidate_version=candidate_id,
                    openevolve_parent_id=oe_parent.id if oe_parent else None,
                    openevolve_island=island,
                    parent_reward=(parent_evaluation or {}).get('reward'),
                    candidate_reward=(candidate_evaluation or {}).get('reward'),
                    training_score=self.scores.get(learner.active), archive_best=learner.archive.get_best_program().metrics['combined_score'],
                    execution_constraint=execution['reward'], execution_prompt=execution['prompt_accuracy'],
                    execution_submitted_rate=execution['submitted_rate'],
                    training_ranked_alternative=learner.archive.get_best_program().code,
                    artifact=copy.deepcopy(learner.versions[learner.active]), events=events,
                    budget=vars(budget), wall_seconds=time.monotonic() - started)


def prior_spending(root: Path):
    """Freeze completed predecessor ledger exposure; unknown charges stay reserved."""
    allocation = json.loads((root / 'allocation.json').read_text())
    if not json.loads((root / 'completion_audit.json').read_text()).get('passed'):
        raise ValueError('Predecessor needs a passing completion audit')
    reported = exposure = 0.
    weights, hashes = {}, {}
    for slot in allocation['slots']:
        config = slot['config']
        output = root / slot['label']
        if not config['smoke'] and not (output / 'completed.json').exists():
            raise ValueError('Cannot carry forward a still-incomplete predecessor')
        accounting = Path(config.get('accounting_directory', output))
        for agent in range(config['num_agents']):
            path = accounting / 'agents' / str(agent) / 'usage.json'
            data = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
            rows = json.loads(data).values()
            actual = sum(v.get('cost_usd') or 0. for v in rows)
            maximum = sum(v['cost_usd'] if v.get('cost_usd') is not None else v.get('reserved_usd', 0.) for v in rows)
            reported += actual
            exposure += maximum
            if not config['smoke']:
                key = (config['model']['name'], config['condition'])
                weights[key] = weights.get(key, 0.) + actual
    return dict(root=str(root.resolve()), reported_cost=reported, exposure=exposure,
                ledger_sha256=hashes), weights


def prepare(root: Path, prior_root: Path | None = None):
    """Immutable allocation: production $975 plus smoke $25, including all retries."""
    base = json.loads((Path(__file__).parents[1] / 'configs/rung4_gpt_oss_120b_ifbench_offline.json').read_text())
    slots = []
    for smoke in (True, False):
        for model_index, (name, allowance, price) in enumerate([
                ('openai/gpt-oss-120b', 2.2, dict(prompt=.06, completion=.25)),
                ('z-ai/glm-5.3-flash', 4.3, dict(prompt=.15, completion=.5))]):
            for condition in BASE_CONDITIONS:
                for seed in range(1 if smoke else 6):
                    config = copy.deepcopy(base)
                    config.update(condition=condition, seed=seed, num_agents=5, rounds=3 if smoke else 50,
                        evolution_protocol=EVOLUTION_PROTOCOL,
                        agent_seeds=[seed + 6 * i for i in range(5)],
                        initial_skill='minimal', evolution_design='static_population',
                        dataset=str((Path(__file__).parents[1] / 'data/ifbench_train100_test200.json').resolve()),
                        training_task_tokens=8704, controller_tokens=32768, execution_tokens=32768,
                        revision_tokens=32768, offline_training_size=100,
                        development_cohort_size=100, development_tokens=870400,
                        tokens_per_round=100*8704 + 4*32768,
                        snapshots=[0, 1, 2] if smoke else [0, 10, 25, 49], smoke=smoke)
                    if smoke:
                        config.update(smoke_train_tasks=10, smoke_holdout_tasks=20)
                    config['model'].update(name=name, run_cost_limit_usd=.5 if smoke else allowance,
                                           total_cost_limit_usd=.5 if smoke else allowance)
                    config['model']['request_options']['provider']['max_price'] = price
                    label = f'{"smoke" if smoke else "full"}/model_{model_index}/{condition}/seed_{seed}'
                    slots.append(dict(label=label, config=config))
    manifest = dict(cap_usd=1000, production_seeds=6, slots=slots)
    if prior_root is not None:
        carried, weights = prior_spending(prior_root)
        available = 1000 - carried['exposure']
        smoke_allowance = 8.
        if available <= smoke_allowance or set(weights) != {(s['config']['model']['name'], s['config']['condition']) for s in slots}:
            raise ValueError('Insufficient carry-forward budget or missing predecessor conditions')
        for slot in slots:
            config = slot['config']
            allowance = (smoke_allowance / 50 if config['smoke'] else
                         (available - smoke_allowance) * weights[(config['model']['name'], config['condition'])]
                         / sum(weights.values()) / 30)
            config['model'].update(run_cost_limit_usd=allowance, total_cost_limit_usd=allowance)
        manifest['prior_spending'] = carried
    total = sum(s['config']['num_agents'] * s['config']['model']['total_cost_limit_usd'] for s in slots)
    assert abs(total + manifest.get('prior_spending', {}).get('exposure', 0.) - 1000) < 1e-8
    path = root / 'allocation.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Refusing to change existing spending allocation')
    else:
        rung4.atomic_json(path, manifest)
    print(json.dumps(dict(slots=len(slots), new_allowance_usd=total,
                          prior_exposure_usd=manifest.get('prior_spending', {}).get('exposure', 0.),
                          combined_cap_usd=1000)), flush=True)


def agent_model(config, directory):
    model_config = copy.deepcopy(config['model'])
    if config.get('accounting_directory'):
        directory = Path(config['accounting_directory']) / 'agents' / directory.name
    model_config['usage_ledger'] = str(directory / 'usage.json')
    model = rung4.Model(model_config, directory / 'api_cache')
    # Accounting-only recovery must not change prompts or paid-evaluation identities.
    if '_operational_cost_limit' in config:
        model.client.config.update(run_cost_limit_usd=config['_operational_cost_limit'],
                                   total_cost_limit_usd=config['_operational_cost_limit'])
    return model


def prepare_timing(root, prior_root, full=False, cap=400.):
    """New intervention allocation; completed controls are read-only references."""
    prior_root = prior_root.resolve()
    old = json.loads((prior_root / 'allocation.json').read_text())
    controls = []
    templates = {}
    for slot in old['slots']:
        cfg = slot['config']
        if cfg['smoke'] or cfg['condition'] not in {'llm_solo', 'llm_social_payoff'}:
            continue
        directory = prior_root / slot['label']
        complete = json.loads((directory / 'completed.json').read_text())
        if complete != dict(rounds=50, snapshots=[0, 10, 25, 49]):
            raise ValueError('Expected complete, matched 50-round controls')
        for r in cfg['snapshots']:
            for agent in range(cfg['num_agents']):
                test = json.loads((directory / 'test_summary' / f'{r:04d}-{agent}.json').read_text())
                if test['tasks'] != 200:
                    raise ValueError('Control has wrong test split')
        provenance = json.loads((directory / 'provenance.json').read_text())
        if provenance['config'] != cfg:
            raise ValueError('Control allocation/provenance mismatch')
        # The timing intervention changes only population orchestration.
        for name, digest in provenance['source_sha256'].items():
            if name != 'rung4_population.py' and hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Control implementation mismatch: {name}')
        controls.append(dict(label=slot['label'], directory=str(directory), config=cfg,
            provenance_sha256=hashlib.sha256((directory / 'provenance.json').read_bytes()).hexdigest(), reused=True))
        if cfg['condition'] == 'llm_social_payoff':
            templates[(cfg['model']['name'], cfg['seed'])] = cfg
    models = ['openai/gpt-oss-120b', 'z-ai/glm-5.3-flash']
    if len(controls) != 24 or set(templates) != {(m, s) for m in models for s in range(6)}:
        raise ValueError('Expected two models and six completed matched seeds')
    for control in controls:
        cfg = control['config']
        template = templates[(cfg['model']['name'], cfg['seed'])]
        def scientific_config(value):
            result = copy.deepcopy(value)
            result.pop('condition')
            for key in ('run_cost_limit_usd', 'total_cost_limit_usd'):
                result['model'].pop(key, None)
            return result
        if scientific_config(cfg) != scientific_config(template):
            raise ValueError('Solo/social control scientific settings differ')
    selected = TIMING_CONDITIONS if full else ('forced_early', 'forced_distributed', 'optional_early')
    if not math.isfinite(cap) or cap <= 4:
        raise ValueError('Invalid timing budget')
    slots = []
    for smoke in (True, False):
        for mi, model in enumerate(models):
            for condition in selected:
                for seed in range(1 if smoke else 6):
                    cfg = copy.deepcopy(templates[(model, seed)])
                    cfg.update(condition=condition, smoke=smoke, rounds=3 if smoke else 50,
                        snapshots=[0, 1, 2] if smoke else [0, 5, 10, 20, 25, 30, 40, 49],
                        observation_timing=timing_spec(condition, seed, cfg['num_agents'], smoke))
                    for key in ('accounting_directory', '_operational_cost_limit'):
                        cfg.pop(key, None)
                    if smoke:
                        cfg.update(smoke_train_tasks=10, smoke_holdout_tasks=20)
                    # Fixed per-agent ceilings sum to the cap even under concurrency/retries.
                    allowance = (4 / (2 * len(selected) * 5) if smoke else
                                 (cap - 4) * (4 if mi == 0 else 7) / (11 * len(selected) * 6 * 5))
                    cfg['model'].update(run_cost_limit_usd=allowance, total_cost_limit_usd=allowance)
                    slots.append(dict(label=f'{"smoke" if smoke else "full"}/model_{mi}/{condition}/seed_{seed}', config=cfg))
    manifest = dict(cap_usd=cap, study='observation_timing_v1', stage='full' if full else 'first',
                    production_seeds=6, expected_smoke_conditions=2 * len(selected), slots=slots,
                    reused_controls=controls, comparison_snapshots=[0, 10, 25, 49],
                    planned_conditions=list(TIMING_CONDITIONS))
    assert abs(sum(s['config']['num_agents'] * s['config']['model']['total_cost_limit_usd'] for s in slots) - cap) < 1e-8
    path = root / 'allocation.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing to change existing timing allocation')
    rung4.atomic_json(path, manifest)
    print(json.dumps(dict(production_slots=sum(not s['config']['smoke'] for s in slots),
                          smoke_slots=2 * len(selected), reused_controls=len(controls), cap_usd=cap)), flush=True)


def effective_cost_limit(allocation, slot_index):
    overrides = allocation.get('budget_overrides', {})
    if any(str(int(k)) != k or not 0 <= int(k) < len(allocation['slots']) for k in overrides):
        raise ValueError('Invalid recovery budget slot')
    limits = [float(overrides.get(str(i), s['config']['model']['total_cost_limit_usd']))
              for i, s in enumerate(allocation['slots'])]
    if any(not math.isfinite(v) or v <= 0 for v in limits):
        raise ValueError('Invalid recovery budget ceiling')
    prior = allocation.get('prior_spending', {}).get('exposure', 0.)
    if not math.isfinite(prior) or prior < 0:
        raise ValueError('Invalid prior spending exposure')
    if prior + sum(v * s['config']['num_agents'] for v, s in zip(limits, allocation['slots'])) > allocation['cap_usd'] + 1e-8:
        raise ValueError('Recovery exceeds hard spending allocation')
    return limits[slot_index]


def configure_worker():
    from rung4_instruction import configure_runner
    configure_runner()


def population_pool(count):
    # Fresh interpreters avoid inheriting locks from imported numerical libraries.
    return ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn'),
                               initializer=configure_worker)


def population_benchmark(config):
    from rung4_instruction import InstructionBenchmark
    benchmark = InstructionBenchmark(config['dataset'])
    if config.get('smoke_train_tasks'):
        benchmark.train_ids = random.Random(rung4.seed_for(config['seed'], 'smoke-train')).sample(
            benchmark.train_ids, config['smoke_train_tasks'])
        benchmark.holdout_ids = random.Random(rung4.seed_for(config['seed'], 'smoke-holdout')).sample(
            benchmark.holdout_ids, config['smoke_holdout_tasks'])
    return benchmark


def prepare_small_smoke(root):
    """Replace only smoke work; its existing $25 accounting allocation persists."""
    path = root / 'allocation.json'
    allocation = json.loads(path.read_text())
    if 'prior_spending' in allocation:
        raise ValueError('Carry-forward allocations already use small smoke; do not rebalance as a fresh $1000 sweep')
    if allocation.get('smoke_budget_rebalanced'):
        return
    original = root / 'allocation.before_small_smoke.json'
    if not original.exists():
        rung4.atomic_json(original, allocation)
    smoke = []
    for slot in allocation['slots']:
        if slot['config']['smoke']:
            if (root / slot['label'] / 'provenance.json').exists() and slot['label'].startswith('parallel_smoke/'):
                raise ValueError('Cannot rebalance a started replacement smoke')
            config = slot['config']
            config.setdefault('accounting_directory', str(root / slot['label']))
            config.update(smoke_train_tasks=10, smoke_holdout_tasks=20)
            if not slot['label'].startswith('parallel_smoke/'):
                slot['label'] = slot['label'].replace('smoke/', 'parallel_smoke/', 1)
            exposure = []
            for i in range(config['num_agents']):
                ledger = Path(config['accounting_directory']) / 'agents' / str(i) / 'usage.json'
                records = json.loads(ledger.read_text()) if ledger.exists() else {}
                exposure.append(sum(v['cost_usd'] if v.get('cost_usd') is not None
                                    else v.get('reserved_usd', 0) for v in records.values()))
            smoke.append((config, max(exposure)))
    # Give every agent equal fresh admission headroom while retaining unknown
    # charges from canceled requests. The total smoke allowance remains $25.
    headroom = (25 - sum(c['num_agents'] * used for c, used in smoke)) / sum(c['num_agents'] for c, _ in smoke)
    if headroom <= 0:
        raise ValueError('No smoke headroom remains within its $25 allocation')
    for config, used in smoke:
        config['model'].update(run_cost_limit_usd=used + headroom,
                               total_cost_limit_usd=used + headroom)
    assert abs(sum(s['config']['num_agents'] * s['config']['model']['total_cost_limit_usd']
                   for s in allocation['slots']) - 1000) < 1e-8
    allocation['smoke_budget_rebalanced'] = True
    rung4.atomic_json(path, allocation)


def run_agent_round(payload):
    """One process owns an agent's state; completed work survives other failures."""
    config, output, agent_id, r, prior = payload
    directory = Path(output) / 'agents' / str(agent_id)
    checkpoint = directory / 'checkpoint.json'
    row_path = directory / 'rounds' / f'{r:04d}.json'
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    if saved and saved['round'] >= r:
        return json.loads(row_path.read_text())
    model = agent_model(config, directory)
    agent = PopulationAgent(config, agent_id, model, population_benchmark(config),
                            directory, saved['state'] if saved else None)
    # step restores the per-agent RNG carried forward from its previous round.
    row = agent.step(r, prior)
    rung4.atomic_json(row_path, row)
    rung4.atomic_json(checkpoint, dict(round=r, state=agent.state()))
    model.client.clear_cache()
    return row


def holdout_work(config, output):
    """Deduplicate snapshots before concurrent submission, separately per agent."""
    groups = {}
    for r in config['snapshots']:
        rows = json.loads((output / 'rounds' / f'{r:04d}.json').read_text())
        for row in rows:
            key = (row['agent'], row['version'])
            group = groups.setdefault(key, dict(files=row['artifact']['files'], summaries=[]))
            group['summaries'].append((f'{r:04d}-{row["agent"]}.json', dict(round=r)))
    for row in rows:
        files = {'SKILL.md': row['training_ranked_alternative']}
        key = (row['agent'], rung4.digest(files)[:24])
        group = groups.setdefault(key, dict(files=files, summaries=[]))
        group['summaries'].append((f'alternative-{row["agent"]}.json',
                                  dict(selection_rule='final archive training rank')))
    return [(config, str(output), agent_id, vid, group)
            for (agent_id, vid), group in groups.items()]


def run_holdout_group(payload):
    config, output, agent_id, vid, group = payload
    from rung4_instruction import InstructionBenchmark
    output = Path(output)
    directory = output / 'agents' / str(agent_id)
    benchmark = population_benchmark(config)
    model = agent_model(config, directory)
    scores, used = evaluate_cached(model, benchmark, benchmark.holdout_ids, group['files'],
        directory / 'holdout' / vid, learner_seed(config, agent_id),
        'holdout', config['holdout_tokens'])
    for filename, metadata in group['summaries']:
        rung4.atomic_json(output / 'test_summary' / filename,
            dict(**metadata, agent=agent_id, version=vid, constraint_accuracy=scores['reward'],
                 prompt_accuracy=scores['prompt_accuracy'], tasks=scores['evaluated_count'],
                 budget=vars(used)))
    # Other versions of this agent can still have paid requests in flight.
    # Cache cleanup occurs only after the entire holdout pool has joined.
    return agent_id, vid


def run_slot(root: Path, slot_index: int, replay_only=False):
    from rung4_instruction import configure_runner, InstructionBenchmark
    configure_runner()
    allocation = json.loads((root / 'allocation.json').read_text())
    if not math.isfinite(allocation['cap_usd']) or allocation['cap_usd'] <= 0 or allocation.get('prior_spending', {}).get('exposure', 0.) + sum(s['config']['num_agents'] *
            s['config']['model']['total_cost_limit_usd'] for s in allocation['slots']) > allocation['cap_usd'] + 1e-8:
        raise ValueError('Invalid hard spending allocation')
    slot = allocation['slots'][slot_index]
    config = copy.deepcopy(slot['config'])
    if config['condition'] in TIMING_CONDITIONS and config.get('observation_timing') != timing_spec(
            config['condition'], config['seed'], config['num_agents'], config['smoke']):
        raise ValueError('Timing schedule changed or is missing')
    if config.get('evolution_protocol') != EVOLUTION_PROTOCOL:
        raise ValueError('Parity correction requires a new allocation and output directory')
    output = root / slot['label']
    output.mkdir(parents=True, exist_ok=True)
    benchmark = InstructionBenchmark(config['dataset'])
    if len(benchmark.train_ids) != 100 or len(benchmark.holdout_ids) != 200:
        raise ValueError('Expected unchanged 100/200 split')
    files = ['rung4_population.py', 'rung4_instruction.py', 'rung4.py', 'openrouter.py',
             'rung4_benchmark.py', 'rung4_server.py', 'social_baseline.py', 'main.py', 'agent.py', 'env.py']
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in files}
    identity = dict(config=config, dataset=rung4.digest(benchmark.manifest), source_sha256=sources)
    provenance = output / 'provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text()) != identity:
            raise ValueError('Refusing changed code/config/data on resume')
    else:
        for name in files:
            target = output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((Path(__file__).parent / name).read_bytes())
        rung4.atomic_json(provenance, identity)
        rung4.atomic_json(output / 'config.json', config)
    # Ceilings come only from the immutable slot allocation, never an inherited override.
    for key in ('AGENT_MARKET_RUN_COST_LIMIT_USD', 'AGENT_MARKET_TOTAL_COST_LIMIT_USD'):
        os.environ.pop(key, None)
    config['_operational_cost_limit'] = effective_cost_limit(allocation, slot_index)
    # Each worker has isolated OpenEvolve RNG/model state and its own task pool.
    with population_pool(config['num_agents']) as pool:
        if not replay_only:
            for r in range(config['rounds']):
                round_path = output / 'rounds' / f'{r:04d}.json'
                if round_path.exists():
                    continue
                prior = json.loads((output / 'rounds' / f'{r-1:04d}.json').read_text()) if r else [None] * config['num_agents']
                results = list(pool.map(run_agent_round, [(config, str(output), i, r, prior)
                                                         for i in range(config['num_agents'])]))
                rung4.atomic_json(round_path, results)
                if r in config['snapshots']:
                    rung4.atomic_json(output / 'snapshots' / f'{r:04d}.json', [dict(agent=x['agent'],
                        version=x['version'], training_score=x['training_score']) for x in results])
                print(json.dumps(dict(round=r, training_scores=[x['training_score'] for x in results])), flush=True)
            rung4.atomic_json(output / 'training_completed.json', dict(rounds=config['rounds']))
        # All learning is committed before any hidden score is computed.
        list(pool.map(run_holdout_group, holdout_work(config, output)))
    for i in range(config['num_agents']):
        agent_model(config, output / 'agents' / str(i)).client.clear_cache()
    rung4.atomic_json(output / 'completed.json', dict(rounds=config['rounds'], snapshots=config['snapshots']))


def audit_smoke(root: Path):
    """Evidence gate, not a check that the policies happen to improve accuracy."""
    allocation = json.loads((root / 'allocation.json').read_text())
    reports = []
    for slot in allocation['slots']:
        config = slot['config']
        if not config['smoke']:
            continue
        output = root / slot['label']
        if not (output / 'completed.json').exists():
            raise ValueError(f'Incomplete smoke: {slot["label"]}')
        rows = [x for r in range(config['rounds'])
                for x in json.loads((output / 'rounds' / f'{r:04d}.json').read_text())]
        assert len(rows) == config['num_agents'] * config['rounds']
        decisions = [e for row in rows for e in row['events'] if e['kind'] == 'decision_error']
        if len(decisions) > 1:
            raise ValueError(f'Excess invalid controller decisions: {slot["label"]}')
        if statistics.mean(row['execution_submitted_rate'] for row in rows) < .9:
            raise ValueError(f'Execution completion below 90%: {slot["label"]}')
        if any(row['budget']['phase_cap_violations'] or row['budget']['provider_generation_errors'] for row in rows):
            raise ValueError(f'Provider/budget failure in {slot["label"]}')
        observations = [e for row in rows for e in row['events'] if e['kind'] == 'observe']
        if config.get('observation_timing'):
            for row in rows:
                access, forced = observation_access(config, row['agent'], row['round'])
                copied = sum(e['kind'] == 'observe' for e in row['events'])
                if (not access and copied) or (forced and copied != 1):
                    raise ValueError('Smoke violated assigned observation schedule')
            if any(e['kind'] == 'forced_source_fallback' for row in rows for e in row['events']):
                raise ValueError('Smoke required source-selection fallback')
        revisions = [e for row in rows for e in row['events'] if e['kind'] == 'revision']
        if config['condition'] in {'openevolve_solo', 'llm_solo'} and not any(e['changed'] for e in revisions):
            raise ValueError(f'No successful private revision in {slot["label"]}')
        if config['condition'] in {'llm_solo', 'openevolve_solo'} and observations:
            raise ValueError('Solo population accessed peer')
        tests = [json.loads(p.read_text()) for p in (output / 'test_summary').glob('*.json')]
        assert len(tests) == config['num_agents'] * (len(config['snapshots']) + 1)
        assert all(x['tasks'] == config.get('smoke_holdout_tasks', 200) for x in tests)
        cost, exposure = 0., 0.
        for i in range(config['num_agents']):
            accounting = Path(config.get('accounting_directory', output))
            ledger = json.loads((accounting / 'agents' / str(i) / 'usage.json').read_text())
            actual = sum(x.get('cost_usd') or 0 for x in ledger.values())
            reserved = sum(x['cost_usd'] if x.get('cost_usd') is not None else x.get('reserved_usd', 0)
                           for x in ledger.values())
            assert reserved <= config['model']['total_cost_limit_usd'] + 1e-6
            cost += actual
            exposure += reserved
        reports.append(dict(slot=slot['label'], reported_cost=cost, conservative_exposure=exposure,
                            decision_errors=len(decisions), observations=len(observations),
                            valid_revisions=sum(e['changed'] for e in revisions)))
    expected = allocation.get('expected_smoke_conditions', 10)
    assert len(reports) == expected
    rung4.atomic_json(root / 'smoke_audit.json', dict(passed=True, conditions=reports))
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--prior-root', type=Path, help='Completed sweep whose spending remains inside the $1000 cap')
    parser.add_argument('--slot', type=int)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--audit-smoke', action='store_true')
    parser.add_argument('--prepare-small-smoke', action='store_true')
    parser.add_argument('--prepare-timing', action='store_true')
    parser.add_argument('--full-timing', action='store_true', help='Prepare all seven conditions; does not submit jobs')
    parser.add_argument('--timing-cap', type=float, default=400.)
    args = parser.parse_args()
    if args.prepare_timing:
        if args.prior_root is None:
            parser.error('--prepare-timing requires --prior-root')
        prepare_timing(args.root, args.prior_root, args.full_timing, args.timing_cap)
    elif args.prepare_small_smoke:
        prepare_small_smoke(args.root)
    elif args.prepare:
        prepare(args.root, args.prior_root)
    elif args.audit_smoke:
        audit_smoke(args.root)
    else:
        if args.slot is None:
            parser.error('--slot required')
        run_slot(args.root, args.slot, args.replay)
