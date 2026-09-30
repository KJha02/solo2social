"""Scientific invariants for the approved static social protocol."""
import copy
from types import SimpleNamespace

import pytest

import rung4
from rung4_population import (acquire_observation, public_library, source_controller,
                              source_state, evaluate_cached, CONDITIONS, PopulationAgent, prepare)
from social_baseline import HierarchicalSocialUCB


@pytest.fixture(autouse=True)
def restore_instruction_adapter_globals():
    """configure_runner mutates shared modules; do not contaminate coding tests."""
    import rung4_benchmark
    originals = (rung4.INITIAL_SKILLS, rung4.Learner, rung4.execute,
                 rung4_benchmark.LiveCodeBench, rung4_benchmark.Sandbox)
    yield
    (rung4.INITIAL_SKILLS, rung4.Learner, rung4.execute,
     rung4_benchmark.LiveCodeBench, rung4_benchmark.Sandbox) = originals


def test_recovery_budget_preserves_global_cap_and_scientific_config(monkeypatch, tmp_path):
    import rung4_population as population
    allocation = dict(cap_usd=10, slots=[dict(config=dict(num_agents=1,
        model=dict(total_cost_limit_usd=5))) for _ in range(2)],
        budget_overrides={'0': 6, '1': 4})
    assert population.effective_cost_limit(allocation, 0) == 6
    allocation['budget_overrides']['0'] = 7
    with pytest.raises(ValueError, match='hard spending'):
        population.effective_cost_limit(allocation, 0)
    allocation['budget_overrides']['0'] = float('nan')
    with pytest.raises(ValueError, match='ceiling'):
        population.effective_cost_limit(allocation, 0)
    monkeypatch.setattr(rung4, 'Model', lambda config, directory:
                        SimpleNamespace(config=config, client=SimpleNamespace(config=dict(config))))
    config = dict(model=dict(run_cost_limit_usd=2.2, total_cost_limit_usd=2.2),
                  _operational_cost_limit=3.)
    model = population.agent_model(config, tmp_path)
    assert model.config['total_cost_limit_usd'] == 2.2
    assert model.client.config['total_cost_limit_usd'] == 3.


def snapshot():
    files = {'SKILL.md': 'Peer procedure'}
    return dict(agent=1, round=0, version=rung4.digest(files)[:24], training_score=.9,
                artifact=dict(files=files, parent=None, creator=1, created_round=0,
                              development_scores=[.9]))


def test_observation_acquires_without_deploying_or_leaking_score():
    learner = rung4.Learner(0, 0)
    original = learner.active
    peer = snapshot()
    observed = acquire_observation(learner, peer, 'action', 1)
    assert learner.active == original
    assert peer['version'] in learner.versions
    assert learner.versions[peer['version']]['development_scores'] == []
    assert 'training_score' not in observed
    assert peer['artifact']['development_scores'] == [.9]
    assert next(x for x in public_library(learner, {}) if x['version'] == peer['version'])['training_score'] is None
    acquire_observation(learner, peer, 'action', 2)
    assert len(learner.versions) == 2
    assert learner.acquisitions[peer['version']]['round'] == 1


def test_payoff_observation_and_timing():
    learner = rung4.Learner(0, 0)
    peer = snapshot()
    assert acquire_observation(learner, peer, 'payoff', 1)['training_score'] == .9
    for info, change in [('none', {}), ('payoff', {'round': 1}), ('payoff', {'agent': 0})]:
        with pytest.raises(ValueError):
            acquire_observation(learner, dict(peer, **change), info, 1)


def test_source_ucb_exact_trace_and_resume():
    adapted = source_controller(0, 17, 5)
    original = HierarchicalSocialUCB(0, 17, 5, 1, .5, .1)
    for r in range(20):
        available = [0] if r == 0 else list(range(5))
        selected = adapted.choose_source(r, available)
        assert selected == original.choose_source(r, available)
        payoff = ((r * 3 + selected) % 10) / 10
        adapted.update_source(selected, payoff)
        original.update_source(selected, payoff)
        adapted = source_controller(0, 17, 5, source_state(adapted))


def test_partial_evaluation_resume_and_split_isolation(tmp_path, monkeypatch):
    calls = []
    def execute(model, benchmark, sandbox, task, files, budget, seed, phase):
        calls.append(task)
        if task == 'b' and len(calls) == 2:
            raise RuntimeError('preempted')
        return dict(constraint_accuracy=.5, prompt_level_loose=0., submitted=True)
    monkeypatch.setattr(rung4, 'execute', execute)
    model = SimpleNamespace(config={'name': 'fixture', 'parallel_requests': 1})
    benchmark = SimpleNamespace(train_ids=['a', 'b'], holdout_ids=['h'], manifest={'fixture': True})
    args = (model, benchmark, ['a', 'b'], {'SKILL.md': 'x'}, tmp_path, 0, 'development', 100)
    with pytest.raises(RuntimeError):
        evaluate_cached(*args)
    score, _ = evaluate_cached(*args)
    assert calls == ['a', 'b', 'b']
    assert score['reward'] == .5 and score['prompt_accuracy'] == 0.
    assert [p.name for p in tmp_path.glob('*.json')] == ['complete.json']
    evaluate_cached(*args)
    assert calls == ['a', 'b', 'b']
    with pytest.raises(ValueError, match='hidden'):
        evaluate_cached(model, benchmark, ['h'], {}, tmp_path, 0, 'development', 100)
    with pytest.raises(ValueError, match='incompatible'):
        evaluate_cached(model, benchmark, ['a'], {'SKILL.md': 'changed'}, tmp_path, 0, 'development', 100)


def test_allocation_hard_ceiling_and_no_skill_only_sweep(tmp_path):
    import json
    prepare(tmp_path)
    allocation = json.loads((tmp_path / 'allocation.json').read_text())
    slots = allocation['slots']
    assert len(slots) == 70
    assert sum(not x['config']['smoke'] for x in slots) == 60
    assert len({x['label'] for x in slots}) == 70
    assert sum(x['config']['model']['total_cost_limit_usd'] * 5 for x in slots) == pytest.approx(1000)
    assert 'llm_social_skill' not in CONDITIONS
    prepare(tmp_path)
    from rung4_population import prepare_small_smoke
    before = copy.deepcopy(slots)
    prepare_small_smoke(tmp_path)
    revised = json.loads((tmp_path / 'allocation.json').read_text())
    for old, new in zip(before, revised['slots']):
        if old['config']['smoke']:
            assert new['config']['accounting_directory'] == str(tmp_path / old['label'])
            assert new['config']['model'] == old['config']['model']
            assert new['config']['smoke_train_tasks'] == 10
            assert new['config']['smoke_holdout_tasks'] == 20
        else:
            assert new == old
    prepare_small_smoke(tmp_path)


@pytest.mark.parametrize('condition', list(CONDITIONS))
def test_two_round_population_protocol_without_api(tmp_path, monkeypatch, condition):
    import json
    from rung4_instruction import configure_runner
    configure_runner()
    def execute(model, benchmark, sandbox, task, files, budget, seed, phase):
        return dict(constraint_accuracy=.8, prompt_level_loose=1., submitted=True,
                    task_id=task, reward=.8, passed_test_fraction_lower_bound=.8,
                    reflection={'constraint_accuracy': .8})
    monkeypatch.setattr(rung4, 'execute', execute)
    class FixtureModel:
        config = {'name': 'fixture', 'parallel_requests': 2}
        def call(self, messages, budget, phase, seed, **kwargs):
            if phase == 'revision':
                return dict(content=f'```markdown\n# Revised procedure {seed}\nCheck constraints.\n```')
            prompt = json.loads(messages[-1]['content'])
            if phase == 'deployment':
                assert prompt['allowed_actions'] == ['deploy']
                assert 'DEPLOYMENT stage' in messages[0]['content']
                choice = dict(action='deploy', version=prompt['repertoire'][-1]['version'])
            elif prompt['observable_peers']:
                assert 'deploy' not in prompt['allowed_actions']
                assert 'observe' in prompt['allowed_actions']
                choice = dict(action='observe', target=prompt['observable_peers'][0])
            else:
                assert 'deploy' not in prompt['allowed_actions']
                assert 'observe' not in prompt['allowed_actions']
                assert 'LEARNING stage' in messages[0]['content']
                choice = dict(action='revise', version=prompt['current_version'])
            return dict(content=json.dumps(choice))
    benchmark = SimpleNamespace(train_ids=['a', 'b'], holdout_ids=['h'], manifest={'fixture': True},
                                public_task=lambda task: {'id': task, 'prompt': 'Fixture task'})
    config = dict(condition=condition, seed=0, agent_seeds=[0, 6], num_agents=2, training_task_tokens=100,
                  tokens_per_round=2000, controller_tokens=100, execution_tokens=100)
    from rung4_population import TIMING_CONDITIONS, timing_spec
    if condition in TIMING_CONDITIONS:
        config['observation_timing'] = timing_spec(condition, 0, 2, smoke=True)
    agents = [PopulationAgent(config, i, FixtureModel(), benchmark, tmp_path / str(i)) for i in range(2)]
    prior = [None, None]
    for r in range(3):
        rows = [agent.step(r, copy.deepcopy(prior)) for agent in agents]
        assert all(row['execution_constraint'] == .8 for row in rows)
        assert all(row['training_score'] == .8 for row in rows)
        for agent in agents:
            saved = agent.state()
            restored = PopulationAgent(config, agent.id, agent.model, benchmark, agent.directory, saved)
            assert restored.learner.active == agent.learner.active
            assert source_state(restored.source) == source_state(agent.source)
        prior = rows
    if condition == 'llm_social_payoff':
        assert all(any(o['acquired_round'] == 1 for o in a.learner.observations) for a in agents)
    if condition in {'llm_solo', 'openevolve_solo'}:
        assert all(not a.learner.observations for a in agents)
    if condition in TIMING_CONDITIONS:
        assert all([o['acquired_round'] for o in a.learner.observations] == [1] for a in agents)


def test_analysis_uses_population_seeds_and_excludes_partial_tests(tmp_path):
    import csv
    from analyze_rung4_population import analyze
    slots = []
    for condition in ('llm_solo', 'llm_social_payoff'):
        for seed in range(2):
            config = dict(condition=condition, seed=seed, num_agents=5, rounds=2,
                          snapshots=[0, 1, 2], smoke=True, model={'name': 'fixture'})
            label = f'{condition}/{seed}'
            slots.append(dict(config=config, label=label))
            for r in range(3):
                rows = []
                for i in range(5):
                    value = .3 + seed * .02 + i * .01 + r * .1 + (condition != 'llm_solo') * .1
                    rows.append(dict(round=r, budget=dict(cost_usd=.01, completion=10), events=[],
                        execution_constraint=value, execution_prompt=value,
                        training_score=value, execution_submitted_rate=1.))
                    # One incomplete population/checkpoint must not enter means.
                    if not (condition == 'llm_solo' and seed == 1 and r == 2 and i == 4):
                        rung4.atomic_json(tmp_path / label / 'test_summary' / f'{r:04d}-{i}.json',
                            dict(round=r, tasks=200, constraint_accuracy=value, prompt_accuracy=value))
                rung4.atomic_json(tmp_path / label / 'rounds' / f'{r:04d}.json', rows)
    rung4.atomic_json(tmp_path / 'allocation.json', {'slots': slots})
    analyze(tmp_path, tmp_path / 'analysis', smoke=True)
    with (tmp_path / 'analysis/paired_by_seed.csv').open() as stream:
        paired = list(csv.DictReader(stream))
    assert len(paired) == 2  # two metrics, only one complete paired population seed
    assert all(float(x['final_gain']) == pytest.approx(.1) for x in paired)
    with (tmp_path / 'analysis/test_summary.csv').open() as stream:
        summaries = list(csv.DictReader(stream))
    assert all(int(x['seeds']) == 2 for x in summaries if x['condition'] == 'llm_social_payoff')


def test_parallel_population_barrier_resume_and_holdout_dedup(tmp_path, monkeypatch):
    """A serial implementation deadlocks; completed peers must survive a failure."""
    import json
    import multiprocessing
    import rung4_instruction
    from rung4_population import run_slot, holdout_work
    rung4_instruction.configure_runner()
    monkeypatch.setattr(rung4_instruction, 'configure_runner', lambda: None)
    ctx = multiprocessing.get_context('fork')
    # Fork only this fixture so its fake model/counters remain available. The
    # separate spawn test exercises production worker startup without API calls.
    import rung4_population
    from concurrent.futures import ProcessPoolExecutor
    monkeypatch.setattr(rung4_population, 'population_pool',
                        lambda count: ProcessPoolExecutor(max_workers=count, mp_context=ctx))
    barrier = ctx.Barrier(5, timeout=20)
    calls = ctx.Value('i', 0)
    failed = ctx.Value('i', 0)
    first_task = [None]
    benchmark = SimpleNamespace(train_ids=[f'a{i}' for i in range(100)],
        holdout_ids=[f'h{i}' for i in range(200)], manifest={'fixture': True},
        public_task=lambda task: {'id': task, 'prompt': 'Fixture task'})
    monkeypatch.setattr(rung4_instruction, 'InstructionBenchmark', lambda path: copy.copy(benchmark))

    class FixtureModel:
        def __init__(self, config, directory):
            self.config = config
            self.agent_id = int(directory.parent.name)
            self.client = SimpleNamespace(clear_cache=lambda: None, config=dict(config))
        def call(self, messages, budget, phase, seed, **kwargs):
            prompt = json.loads(messages[-1]['content'])
            choice = (dict(action='deploy', version=prompt['current_version']) if phase == 'deployment'
                      else dict(action='observe', target=prompt['observable_peers'][0]))
            return dict(content=json.dumps(choice))
    monkeypatch.setattr(rung4, 'Model', FixtureModel)

    def execute(model, benchmark, sandbox, task, files, budget, seed, phase):
        with calls.get_lock():
            calls.value += 1
        if phase == 'development_parent' and task == first_task[model.agent_id]:
            if not failed.value:
                barrier.wait()
            if model.agent_id == 1:
                with failed.get_lock():
                    if not failed.value:
                        failed.value = 1
                        raise RuntimeError('fixture preemption')
        return dict(constraint_accuracy=.8, prompt_level_loose=1., submitted=True,
                    task_id=task, reward=.8, passed_test_fraction_lower_bound=.8,
                    reflection={'constraint_accuracy': .8})
    monkeypatch.setattr(rung4, 'execute', execute)
    prepare(tmp_path)
    allocation = json.loads((tmp_path / 'allocation.json').read_text())
    index = next(i for i, s in enumerate(allocation['slots'])
                 if s['config']['smoke'] and s['config']['condition'] == 'llm_social_payoff')
    output = tmp_path / allocation['slots'][index]['label']
    from rung4_population import population_benchmark
    from rung4_population import learner_seed
    import random
    cfg = allocation['slots'][index]['config']
    tasks = population_benchmark(cfg).train_ids
    first_task[:] = []
    for i in range(5):
        seed = learner_seed(cfg, i)
        development = random.Random(rung4.seed_for(seed, 'offline-training')).sample(tasks, len(tasks))
        first_task.append(random.Random(rung4.seed_for(seed, 'population-tasks')).sample(development, len(tasks))[0])
    with pytest.raises(RuntimeError, match='fixture preemption'):
        run_slot(tmp_path, index)
    assert sum((output / 'agents' / str(i) / 'checkpoint.json').exists() for i in range(5)) == 4
    assert not (output / 'rounds' / '0000.json').exists()
    run_slot(tmp_path, index)
    assert (output / 'completed.json').exists()
    for r in (1, 2):
        rows = json.loads((output / 'rounds' / f'{r:04d}.json').read_text())
        assert [row['agent'] for row in rows] == list(range(5))
    for i in range(5):
        state = json.loads((output / 'agents' / str(i) / 'checkpoint.json').read_text())['state']
        assert all(e['source_round'] == e['acquired_round'] - 1
                   for e in state['learner']['observations'])
    groups = holdout_work(allocation['slots'][index]['config'], output)
    assert len(groups) == 5  # same skill across snapshots/alternative, one group per agent
    assert all(len(group[-1]['summaries']) == 4 for group in groups)
    assert len(list((output / 'test_summary').glob('*.json'))) == 20
    before = calls.value
    # A dollar-only reallocation must reuse all paid evaluations on resume.
    donor = next(i for i in range(len(allocation['slots'])) if i != index)
    allocation['budget_overrides'] = {str(index): .6, str(donor): .4}
    rung4.atomic_json(tmp_path / 'allocation.json', allocation)
    run_slot(tmp_path, index)
    assert calls.value == before


def test_spawn_workers_reuse_completed_agent_rounds_without_api(tmp_path):
    from rung4_population import population_pool, run_agent_round
    for i in range(2):
        directory = tmp_path / 'agents' / str(i)
        rung4.atomic_json(directory / 'checkpoint.json', dict(round=0, state={}))
        rung4.atomic_json(directory / 'rounds' / '0000.json', dict(agent=i, round=0))
    with population_pool(2) as pool:
        rows = list(pool.map(run_agent_round, [({}, str(tmp_path), i, 0, []) for i in range(2)]))
    assert rows == [dict(agent=i, round=0) for i in range(2)]


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('restart', [False, True])
def test_original_offline_loop_parity_across_revisions_and_migration(tmp_path, monkeypatch, seed, restart):
    """Use the unchanged original runner as oracle, not a reimplementation in tests.

    Compare actual revision prompts/seeds, execution inputs, private feedback,
    selected skills, parent IDs, generation/island metadata, and archive state.
    Timestamp/UUID sources are frozen solely to compare otherwise random IDs.
    No API or inference call is made.
    """
    import json
    import random
    import uuid
    import openevolve.database as database
    from rung4_instruction import configure_runner, InstructionLearner, MINIMAL_SKILL
    from rung4_population import learner_seed
    configure_runner()
    original_init = rung4.Program.__init__
    def fixed_init(self, *args, **kwargs):
        kwargs.setdefault('timestamp', 0.)
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(rung4.Program, '__init__', fixed_init)
    serial = [0]
    def fixed_uuid():
        serial[0] += 1
        return uuid.UUID(int=serial[0])
    monkeypatch.setattr(database.uuid, 'uuid4', fixed_uuid)
    benchmark = SimpleNamespace(train_ids=['a', 'b', 'c', 'd'], holdout_ids=['hidden'],
        manifest={'fixture': True}, public_task=lambda task: {'id': task, 'prompt': f'Instruction {task}'})
    evaluations = []
    def execute(model, benchmark, sandbox, task, files, budget, execution_seed, phase):
        if phase.startswith('development'):
            evaluations.append((task, copy.deepcopy(files), execution_seed, phase, budget.limit))
        reward = (int(rung4.digest((files, task, execution_seed))[:8], 16) % 90 + 10) / 100
        budget.charge({'completion_tokens': 1, 'prompt_tokens': 1}, 0., phase)
        return dict(task_id=task, reward=reward, constraint_accuracy=reward,
            prompt_level_loose=float(reward > .5), submitted=True,
            passed_test_fraction_lower_bound=reward, reflection={'score': reward})
    monkeypatch.setattr(rung4, 'execute', execute)
    class Model:
        config = {'name': 'fixture', 'parallel_requests': 2, 'max_call_tokens': 200}
        def __init__(self): self.calls = []
        def call(self, messages, budget, phase, call_seed, reserve=0):
            assert phase == 'revision'
            self.calls.append((copy.deepcopy(messages), call_seed, budget.remaining - reserve))
            budget.charge({'completion_tokens': 1, 'prompt_tokens': 1}, 0., phase)
            if call_seed == rung4.seed_for(seed, 3, 'revision'):
                return None  # Failed revisions must preserve identical feedback cadence.
            if call_seed == rung4.seed_for(seed, 5, 'revision'):
                return {'content': self.last_reply}
            self.last_reply = '```markdown\n# Procedure\nCheck ' + rung4.digest(messages)[:12] + '.\n```'
            return {'content': self.last_reply}
    config = dict(seed=seed, agent_seeds=[seed], num_agents=1, condition='openevolve_solo',
        rounds=50, policy='openevolve_full', evolution_design='offline_openevolve',
        development_cohort_size=4, development_tokens=200, tokens_per_round=600,
        training_task_tokens=50, revision_tokens=200, controller_tokens=50, execution_tokens=50,
        revision_experience_max_chars=5000,
        openevolve_database=dict(num_islands=2, migration_interval=2, migration_rate=.5))
    assert learner_seed(config, 0) == seed
    original_model = Model()
    learner = InstructionLearner(0, rung4.seed_for(seed, 0), initial_skill=MINIMAL_SKILL)
    development = random.Random(rung4.seed_for(seed, 'offline-training')).sample(benchmark.train_ids, 4)
    tasks = random.Random(rung4.seed_for(seed, 'population-tasks')).sample(development, 4)
    original = tmp_path / 'original'
    rung4.run_matched_growing(config, original, benchmark, None, original_model,
                             [learner], [[]], [0], 0, 'fixture', tasks)
    expected_evaluations = copy.deepcopy(evaluations)
    expected_archive = rung4.archive_state(learner.archive)
    expected_feedback = json.loads((original / 'checkpoint.json').read_text())['feedback'][0]
    serial[0] = 0
    evaluations.clear()
    model = Model()
    agent = PopulationAgent(config, 0, model, benchmark, tmp_path / 'population')
    for r in range(config['rounds']):
        if restart:
            # JSON round-trip and unrelated RNG use reproduce a different worker.
            saved = json.loads(json.dumps(agent.state(), sort_keys=True))
            random.seed(123456 + r)
            agent = PopulationAgent(config, 0, model, benchmark, agent.directory, saved)
        actual = agent.step(r, [None])
        expected = json.loads((original / 'rounds' / f'{r:04d}.json').read_text())['results'][0]
        for key in ('version', 'parent_version', 'candidate_version', 'openevolve_parent_id', 'openevolve_island'):
            assert actual[key] == expected[key], (r, key)
    assert model.calls == original_model.calls
    # Thread completion order is not an experimental difference.
    assert sorted(evaluations, key=str) == sorted(expected_evaluations, key=str)
    assert agent.feedback == expected_feedback
    assert agent.learner.state() == learner.state()
    assert rung4.archive_state(agent.learner.archive) == expected_archive


def test_holdout_seed_matches_original_replay(tmp_path, monkeypatch):
    from rung4_population import learner_seed
    seeds = []
    def execute(model, benchmark, sandbox, task, files, budget, seed, phase):
        seeds.append(seed)
        return dict(constraint_accuracy=1., prompt_level_loose=1., submitted=True)
    monkeypatch.setattr(rung4, 'execute', execute)
    benchmark = SimpleNamespace(train_ids=['a'], holdout_ids=['h'], manifest={})
    model = SimpleNamespace(config={'name': 'fixture'})
    for original_seed in range(8):
        evaluate_cached(model, benchmark, ['h'], {}, tmp_path / str(original_seed),
                        original_seed, 'holdout', 32768)
    assert seeds == [rung4.seed_for(s, 'h', 0, 'holdout') for s in range(8)]


def test_reject_old_population_state_and_duplicate_seeds(tmp_path):
    from rung4_population import learner_seed
    config = dict(agent_seeds=[0], num_agents=1, condition='openevolve_solo', seed=0)
    with pytest.raises(ValueError, match='new run'):
        PopulationAgent(config, 0, None, None, tmp_path, {'learner': {}})
    with pytest.raises(ValueError, match='distinct'):
        learner_seed(dict(agent_seeds=[0, 0], num_agents=2), 0)


def test_social_import_then_revision_preserves_candidate_semantics(tmp_path, monkeypatch):
    """Peer fitness is public evidence, not private training or a fresh sample."""
    from rung4_instruction import configure_runner
    configure_runner()
    def execute(model, benchmark, sandbox, task, files, budget, seed, phase):
        return dict(task_id=task, reward=.8, constraint_accuracy=.8,
                    prompt_level_loose=1., submitted=True, passed_test_fraction_lower_bound=.8,
                    reflection={'score': .8})
    monkeypatch.setattr(rung4, 'execute', execute)
    class Model:
        config = {'name': 'fixture', 'parallel_requests': 2}
        def call(self, *args, **kwargs):
            return dict(content='```markdown\n# New revision\nCheck all constraints.\n```')
    benchmark = SimpleNamespace(train_ids=['a', 'b'], holdout_ids=['h'], manifest={},
                                public_task=lambda task: {'id': task})
    config = dict(condition='llm_social_payoff', seed=0, agent_seeds=[0, 6], num_agents=2,
                  training_task_tokens=100, tokens_per_round=2000, controller_tokens=100,
                  execution_tokens=100, revision_tokens=100)
    agent = PopulationAgent(config, 0, Model(), benchmark, tmp_path)
    agent.step(0, [None, None])
    peer = snapshot()
    peer['skill_generation'] = 7
    def decision(r, stage, prior, budget, events):
        if stage == 'deployment': return dict(action='deploy', version=agent.learner.active)
        return dict(action='observe', target=1) if r < 3 else dict(action='revise', version=peer['version'])
    monkeypatch.setattr(agent, 'decision', decision)
    for r in (1, 2):
        peer['round'] = r - 1
        agent.step(r, [None, peer])
        assert len(agent.feedback) == 1
        assert agent.scores[peer['version']] == .9
        assert agent.learner.versions[peer['version']]['development_scores'] == []
        assert agent.program_for_version(peer['version']).generation == 7
    # The LLM may choose a retained skill whose MAP-Elites cell was evicted.
    agent.learner.archive.programs.pop(peer['version'])
    peer['round'] = 2
    row = agent.step(3, [None, peer])
    assert row['parent_version'] == peer['version']
    child = agent.learner.archive.programs[row['candidate_version']]
    assert child.parent_id == peer['version'] and child.generation == 8
    assert child.metadata['island'] == 3 % agent.learner.archive_config.num_islands
    assert all(task['parent'] is None and task['candidate'] is not None
               for task in agent.feedback[-1]['tasks'])
    assert peer['artifact']['development_scores'] == [.9]


def test_seed_allocation_keeps_original_seeds_and_pairs_conditions(tmp_path):
    import json
    from rung4_population import BASE_CONDITIONS
    prepare(tmp_path)
    slots = json.loads((tmp_path / 'allocation.json').read_text())['slots']
    for model in {s['config']['model']['name'] for s in slots}:
        for condition in BASE_CONDITIONS:
            configs = [s['config'] for s in slots if not s['config']['smoke']
                       and s['config']['model']['name'] == model and s['config']['condition'] == condition]
            assert {seed for c in configs for seed in c['agent_seeds']} == set(range(30))
            assert all(c['agent_seeds'][0] == c['seed'] for c in configs)
            assert all(c['rounds'] == 50 and c['snapshots'][-1] == 49 for c in configs)


def test_carry_forward_cap_includes_unknown_charges(tmp_path, monkeypatch):
    import json
    import rung4_population as population
    previous = tmp_path / 'previous'
    prepare(previous)
    old = json.loads((previous / 'allocation.json').read_text())
    for slot in old['slots']:
        output = previous / slot['label']
        rung4.atomic_json(output / 'completed.json', {})
        for i in range(5):
            rung4.atomic_json(output / 'agents' / str(i) / 'usage.json', {
                'known': dict(cost_usd=.5), 'unknown': dict(cost_usd=None, reserved_usd=.5)})
    rung4.atomic_json(previous / 'completion_audit.json', {'passed': True})
    current = tmp_path / 'current'
    prepare(current, previous)
    allocation = json.loads((current / 'allocation.json').read_text())
    assert allocation['prior_spending']['reported_cost'] == 175
    assert allocation['prior_spending']['exposure'] == 350
    assert sum(s['config']['num_agents'] * s['config']['model']['total_cost_limit_usd']
               for s in allocation['slots']) == pytest.approx(650)
    population.effective_cost_limit(allocation, 0)
    allocation['budget_overrides'] = {'0': 20}
    with pytest.raises(ValueError, match='hard spending'):
        population.effective_cost_limit(allocation, 0)
    with pytest.raises(ValueError, match='Carry-forward'):
        population.prepare_small_smoke(current)
