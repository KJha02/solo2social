"""Tool-using skill evolution on LiveCodeBench; shared costs for solo and social.

OpenEvolve supplies revision prompts, rewrite parsing, and the private archive.
The population loop owns action timing and social access, not island migration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import random
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openevolve.config import DatabaseConfig, PromptConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import parse_full_rewrite

INITIAL_SKILL = """# Coding procedure
Read the problem and constraints. Choose an algorithm, implement it in Python,
and check the provided examples before submitting. Write the solution to solution.py.
"""
MINIMAL_SKILL = """# Coding procedure
Solve the problem and write the answer to solution.py.
"""
INITIAL_SKILLS = {"minimal": MINIMAL_SKILL, "strong": INITIAL_SKILL}
PARSER_ERROR_PREFIX = '[rung4_harmony_parse_error] '


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_for(*values: Any) -> int:
    return int(digest(values)[:8], 16) % (2**31)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def experiment_config(config: dict) -> dict:
    """The local server port may change after preemption; scientific settings may not."""
    value = json.loads(json.dumps(config))
    value['model'].pop('base_url', None)
    return value


@dataclass
class Budget:
    limit: int
    completion: int = 0
    prompt: int = 0
    inference_seconds: float = 0.0
    tool_seconds: float = 0.0
    calls: int = 0
    context_exhaustions: int = 0
    parser_errors: int = 0
    unreported_completion_allowance: int = 0
    provider_generation_errors: int = 0
    provider_errors_by_phase: dict = field(default_factory=dict)
    provider_cap_overruns: int = 0
    phase_cap_violations: int = 0
    empty_completions: int = 0
    calls_without_cost: int = 0
    cost_usd: float = 0.0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    providers: dict = field(default_factory=dict)
    served_models: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)

    def absorb(self, other: 'Budget') -> None:
        """Merge an independently capped parallel model call into this budget."""
        fields = ('completion', 'prompt', 'inference_seconds', 'tool_seconds', 'calls',
                  'context_exhaustions', 'parser_errors', 'unreported_completion_allowance',
                  'provider_generation_errors', 'provider_cap_overruns',
                  'phase_cap_violations', 'empty_completions', 'calls_without_cost',
                  'cost_usd', 'reasoning_tokens', 'cached_prompt_tokens')
        for name in fields:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in ('provider_errors_by_phase', 'providers', 'served_models'):
            target = getattr(self, name)
            for key, value in getattr(other, name).items():
                target[key] = target.get(key, 0) + value
        for phase, values in other.phases.items():
            target = self.phases.setdefault(phase, {key: 0 for key in values})
            for key, value in values.items():
                target[key] = target.get(key, 0) + value

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.completion - self.unreported_completion_allowance)

    def charge(self, usage: dict, seconds: float, phase: str,
               response: dict | None = None, allow_overrun: bool = False) -> None:
        if 'completion_tokens' not in usage or 'prompt_tokens' not in usage:
            raise RuntimeError('Endpoint must report actual token usage; cannot run cost-controlled study')
        completion, prompt = int(usage['completion_tokens']), int(usage['prompt_tokens'])
        if completion < 0 or prompt < 0 or (completion > self.remaining and not allow_overrun):
            raise RuntimeError('Endpoint exceeded requested completion budget')
        self.completion += completion
        self.prompt += prompt
        self.inference_seconds += seconds
        self.calls += 1
        details = usage.get('completion_tokens_details') or {}
        prompt_details = usage.get('prompt_tokens_details') or {}
        reasoning = int(details.get('reasoning_tokens', 0) or 0)
        cached = int(prompt_details.get('cached_tokens', 0) or 0)
        cost = float(usage.get('cost', 0.) or 0.)
        if response is not None and usage.get('cost') is None:
            self.calls_without_cost += 1
        self.reasoning_tokens += reasoning
        self.cached_prompt_tokens += cached
        self.cost_usd += cost
        if response:
            for field, target in (('provider', self.providers), ('model', self.served_models)):
                value = response.get(field)
                if value:
                    target[value] = target.get(value, 0) + 1
        part = self.phases.setdefault(phase, dict(completion_tokens=0, prompt_tokens=0,
            reasoning_tokens=0, cached_prompt_tokens=0, cost_usd=0., seconds=0., calls=0))
        part['completion_tokens'] += completion
        part['prompt_tokens'] += prompt
        part['reasoning_tokens'] += reasoning
        part['cached_prompt_tokens'] += cached
        part['cost_usd'] += cost
        part['seconds'] += seconds
        part['calls'] += 1


class Model:
    """Native chat/tool API against local vLLM or OpenRouter."""
    def __init__(self, config: dict, cache_dir: Path | None = None):
        self.config = config
        self.client = None
        if config.get('provider', 'vllm') == 'openrouter':
            from openrouter import OpenRouterClient
            if cache_dir is None and os.environ.get('AGENT_MARKET_API_CACHE_DIR'):
                cache_dir = Path(os.environ['AGENT_MARKET_API_CACHE_DIR'])
            self.client = OpenRouterClient(config, cache_dir)

    def call(self, messages: list, budget: Budget, phase: str, seed: int,
             tools: list | None = None, reserve: int = 0) -> dict | None:
        available = budget.remaining - reserve
        cap = min(available, self.config.get('max_call_tokens', 4096))
        if self.client is not None:
            cap -= self.config.get('token_cap_headroom', 0)
            if cap < self.config.get('min_call_tokens', 1):
                return None
        if cap <= 0:
            return None
        body = dict(model=self.config['name'], messages=messages, max_tokens=cap,
                    temperature=self.config.get('temperature', .6), seed=seed)
        body.update(self.config.get('request_options', {}))
        # Mistral's native tokenizer rejects even an empty chat-template override.
        if body.get('chat_template_kwargs') == {}:
            body.pop('chat_template_kwargs')
        # The experimental cap cannot be overwritten by endpoint options.
        body['max_tokens'] = cap
        if tools:
            body.update(tools=tools, tool_choice='auto')
            if self.client is None:
                body['parallel_tool_calls'] = False
        started = time.monotonic()
        try:
            if self.client is not None:
                self.client.phase = phase
                result = self.client.chat_completions([body])[0]
            else:
                request = urllib.request.Request(
                    self.config['base_url'].rstrip('/') + '/chat/completions',
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=self.config.get('timeout', 600)) as response:
                    result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors='replace')
            if exc.code == 400 and ('maximum context length' in detail.lower() or 'max_model_len' in detail.lower()):
                budget.context_exhaustions += 1
                return None
            raise RuntimeError(f'Inference endpoint failed ({exc.code}): {detail[:1000]}') from exc
        except RuntimeError as exc:
            if 'maximum context length' in str(exc).lower():
                budget.context_exhaustions += 1
                return None
            raise
        if result.get('generation_error') == 'MALFORMED_FUNCTION_CALL':
            # Provider omits usage for these rejected generations. Bound compute
            # conservatively without pretending the allowance is measured usage.
            budget.unreported_completion_allowance += cap
            budget.provider_generation_errors += 1
            budget.provider_errors_by_phase[phase] = budget.provider_errors_by_phase.get(phase, 0) + 1
            budget.parser_errors += 1
            budget.calls += 1
            budget.inference_seconds += time.monotonic() - started
            if result.get('usage', {}).get('cost') is not None:
                budget.cost_usd += result['usage']['cost']
            else:
                budget.calls_without_cost += 1
            return dict(role='assistant', content=PARSER_ERROR_PREFIX +
                        'Provider rejected MALFORMED_FUNCTION_CALL; token usage unavailable. '
                        'Use a valid tool call with JSON arguments.')
        reported = result.get('usage', {}).get('completion_tokens', 0)
        if reported > cap:
            if self.client is None:
                raise RuntimeError('Endpoint exceeded requested call cap, violating the reserve')
            budget.provider_cap_overruns += 1
        budget.charge(result.get('usage', {}), time.monotonic() - started, phase, result,
                      allow_overrun=self.client is not None)
        if reported > available:
            budget.phase_cap_violations += 1
            return None  # Preserve the actual usage; do not execute an over-budget answer.
        message = result['choices'][0]['message']
        if self.client is not None and not message.get('content') and not message.get('tool_calls'):
            budget.empty_completions += 1
            return None  # Never spend another batch request on a zero-progress loop.
        if (message.get('content') or '').startswith(PARSER_ERROR_PREFIX):
            budget.parser_errors += 1
        return message


def tool(name: str, description: str, properties: dict) -> dict:
    return dict(type='function', function=dict(name=name, description=description,
        parameters=dict(type='object', properties=properties, required=list(properties), additionalProperties=False)))


TEXT = {'type': 'string'}
INT = {'type': 'integer'}
SKILL_PATH = {'type': 'string', 'description': 'Relative Markdown path inside the skill directory, e.g. SKILL.md. Do not include /work/ or the skills/ directory prefix.'}
FILE_TOOLS = [
    tool('read_file', 'Read a file in your isolated task workspace.', {'path': TEXT}),
    tool('write_file', 'Write a task file, including solution.py or your own tests.', {'path': TEXT, 'content': TEXT}),
    tool('run', 'Execute a shell command in your isolated workspace; only public examples and your own tests are available.', {'command': TEXT}),
    tool('submit', 'Submit solution.py and finish this task.', {}),
]
META_TOOLS = [
    tool('read_skill', 'Read one of your persistent skill files.', {'path': SKILL_PATH}),
    tool('write_skill', 'Replace one persistent Markdown skill file with the complete content supplied, or create a new file. Use read_skill first to preserve existing text. Maximum 8 files, 16000 characters per file, 32000 total. This creates an immutable child version.', {'path': SKILL_PATH, 'content': {'type': 'string', 'maxLength': 16000, 'description': 'Complete replacement file text, not a diff or patch.'}}),
    tool('revise', 'Use OpenEvolve to propose a reusable skill revision from your private feedback. All revision tokens cost your budget.', {}),
    tool('evaluate', 'Test your current skill on your fixed private development cohort, charging all solution tokens to this round.', {}),
    tool('select', 'Load a previously acquired skill version from your own library.', {'version': TEXT}),
    tool('solve', 'Spend the remaining budget solving this round\'s private scored problem.', {}),
]
OBSERVE = tool('observe', 'Inspect one peer\'s previously used skill; no current-round or hidden evaluation results are visible.', {'target': INT})
IMPORT = tool('import_skill', 'Adopt a skill version previously acquired by observation.', {'version': TEXT})


class Learner:
    def __init__(self, agent_id: int, seed: int, versions: dict | None = None,
                 active: str | None = None, acquisitions: dict | None = None,
                 observations: list | None = None, initial_skill: str = INITIAL_SKILL):
        self.id = agent_id
        self.versions = versions or {}
        self.acquisitions = acquisitions or {}
        self.observations = observations or []
        self.active = active or self.install({'SKILL.md': initial_skill}, None, 'initial', 0)
        self.archive_config = DatabaseConfig(
            num_islands=1, migration_rate=0., population_size=10000,
            archive_size=1000, log_prompts=False, random_seed=seed)
        self.rebuild_archive()
        self.sampler = PromptSampler(PromptConfig(
            system_message='Improve reusable Markdown coding skills for solving NEW problems. '
            'Do not store a solution to a particular problem. Return the complete SKILL.md in a markdown code fence.',
            use_template_stochasticity=False))

    def install(self, files: dict, parent: str | None, origin: str, round_index: int,
                source: int | None = None) -> str:
        for name, content in files.items():
            self.skill_path(name)
            if not isinstance(content, str) or len(content) > 16000:
                raise ValueError('Each skill must be text of at most 16000 characters')
        if len(files) > 8 or sum(map(len, files.values())) > 32000:
            raise ValueError('Skill library exceeds the declared size limit')
        vid = digest(files)[:24]
        self.versions.setdefault(vid, dict(files=files.copy(), parent=parent,
            creator=self.id, created_round=round_index, development_scores=[]))
        self.acquisitions.setdefault(vid, dict(origin=origin, round=round_index, source=source))
        return vid

    @staticmethod
    def skill_path(name: str) -> None:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or p.suffix != '.md' or len(p.parts) > 3:
            raise ValueError('Skills must be relative Markdown paths, without parent traversal')

    @property
    def files(self) -> dict:
        return self.versions[self.active]['files']

    def sync_skills(self, directory: Path) -> None:
        """Materialize the agent's persistent harness files for inspection and use."""
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.rglob('*.md'):
            if str(path.relative_to(directory)) not in self.files:
                path.unlink()
        for name, content in self.files.items():
            path = workspace_path(directory, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def add_evidence(self, vid: str, score: float | None) -> None:
        version = self.versions[vid]
        if score is not None:
            version['development_scores'].append(score)
        self.rebuild_archive()

    def rebuild_archive(self) -> None:
        # Stable insertion and timestamps preserve OE sampling/ties after JSON resume.
        # Rebuilding also invalidates upstream's cached best after a mean score changes.
        state = random.getstate()
        try:
            self.archive = ProgramDatabase(self.archive_config)
            for vid, version in sorted(self.versions.items(), key=lambda pair: (pair[1]['created_round'], pair[0])):
                values = version['development_scores']
                if values:
                    self.archive.add(Program(id=vid, code=version['files'].get('SKILL.md', ''),
                        language='markdown', parent_id=version['parent'], timestamp=float(version['created_round']),
                        metrics={'combined_score': sum(values) / len(values)}, metadata={'agent_id': self.id}))
        finally:
            random.setstate(state)

    def revise(self, model: Model, budget: Budget, feedback: list, round_index: int,
               reserve: int, seed: int, experience_chars: int | None = None,
               inspirations: list | None = None, top_programs: list | None = None,
               feature_dimensions: list | None = None) -> bool:
        self.last_revision_error = None
        parent = self.active
        scores = self.versions[parent]['development_scores']
        evidence = feedback[-4:]
        if experience_chars is not None:
            # The complete compact buffer remains in the checkpoint. Supply as much
            # accumulated experience as fits the declared context allowance, keeping
            # the newest evidence when the bound is reached.
            evidence = []
            used = 2
            for record in reversed(feedback):
                encoded = json.dumps(record, sort_keys=True)
                if evidence and used + len(encoded) + 1 > experience_chars:
                    break
                if len(encoded) + 2 > experience_chars:
                    evidence.append({'truncated_record': encoded[:max(0, experience_chars - 2)]})
                    break
                evidence.append(record)
                used += len(encoded) + 1
            evidence.reverse()
        self.last_revision_experience_records = len(evidence)
        self.last_revision_experience_chars = len(json.dumps(evidence))
        prompt = self.sampler.build_prompt(
            current_program=self.files.get('SKILL.md', ''), language='markdown',
            program_metrics={'combined_score': sum(scores) / len(scores)} if scores else {},
            inspirations=[x.to_dict() for x in (inspirations or [])],
            top_programs=[x.to_dict() for x in (top_programs or [])],
            feature_dimensions=feature_dimensions or [],
            program_artifacts={'private_feedback': json.dumps(evidence)},
            diff_based_evolution=False, evolution_round=round_index)
        result = model.call([{'role': 'system', 'content': prompt['system']},
            {'role': 'user', 'content': prompt['user']}], budget, 'revision', seed, reserve=reserve)
        if result is None or not result.get('content'):
            self.last_revision_error = 'No complete revision returned'
            return False
        if result['content'].startswith(PARSER_ERROR_PREFIX):
            self.last_revision_error = result['content'][:400]
            return False
        content = parse_full_rewrite(result['content'], 'markdown')
        if not content:
            return False
        files = self.files.copy()
        files['SKILL.md'] = content
        known = set(self.versions)
        try:
            self.active = self.install(files, parent, 'independent', round_index)
        except ValueError as exc:
            self.last_revision_error = str(exc)
            return False
        return self.active not in known

    def state(self) -> dict:
        return dict(agent_id=self.id, versions=self.versions, active=self.active,
                    acquisitions=self.acquisitions, observations=self.observations)


def workspace_path(workspace: Path, name: str) -> Path:
    target = (workspace / name).resolve()
    if not target.is_relative_to(workspace.resolve()):
        raise ValueError('Path is outside the task workspace')
    if target.is_symlink():
        raise ValueError('Symlink paths are not allowed')
    return target


def execute(model: Model, benchmark, sandbox, task_id: str, files: dict,
            budget: Budget, seed: int, phase: str, reserve: int = 0) -> dict:
    """Tool-using executor; identical for endogenous, frozen and replay conditions."""
    import tempfile
    from rung4_benchmark import score
    public = benchmark.public_task(task_id)
    started = time.monotonic()
    tool_events = []
    reflection = []
    submitted = False
    with tempfile.TemporaryDirectory(prefix='rung4-task-') as directory:
        workspace = Path(directory)
        for name, content in files.items():
            path = workspace / 'skills' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        messages = [dict(role='system', content=(
            'Solve the coding problem using Python 3.11 and the available file and shell tools. '
            'Your reusable skill files are in skills/. Read and apply them as useful. '
            'Write solution.py, test with public examples or your own inputs, then call submit. '
            'Hidden tests are unavailable. Skills are fixed during this task; revisions happen between tasks.\n'
            + json.dumps(files))), dict(role='user', content=json.dumps(public))]
        # Calls have a declared limit in addition to the token budget, identical across conditions.
        for step in range(32):
            message = model.call(messages, budget, phase, seed_for(seed, step), FILE_TOOLS, reserve)
            if message is None:
                break
            messages.append(message)
            calls = message.get('tool_calls') or []
            if not calls:
                tool_events.append(dict(tool='no_tool_call', error=True,
                    error_message=(message.get('content') or '')[:400] if
                    (message.get('content') or '').startswith(PARSER_ERROR_PREFIX) else None))
                messages.append(dict(role='user', content='Use the provided tools. Submit solution.py with the submit tool when ready.'))
                continue
            for call in calls:
                function = call['function']
                name = function['name']
                began = time.monotonic()
                try:
                    args = json.loads(function['arguments'])
                    if name == 'read_file':
                        result = workspace_path(workspace, args['path']).read_text()[:16000]
                    elif name == 'write_file':
                        path = workspace_path(workspace, args['path'])
                        if path.is_relative_to(workspace / 'skills'):
                            raise ValueError('Skills are fixed during execution')
                        if len(args['content']) > 64000:
                            raise ValueError('Task file exceeds 64000 characters')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(args['content'])
                        result = 'written'
                    elif name == 'run':
                        result = sandbox.run(workspace, args['command'], timeout=15,
                                             _mounts=((workspace / 'skills', '/work/skills'),))
                    elif name == 'submit':
                        submitted = workspace_path(workspace, 'solution.py').is_file()
                        result = 'submitted' if submitted else 'Write solution.py first'
                    else:
                        raise ValueError('Unknown tool')
                    error = False
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    result, error = str(exc)[:400], True
                elapsed = time.monotonic() - began
                budget.tool_seconds += elapsed
                tool_events.append(dict(tool=name, seconds=elapsed, error=error,
                    error_message=str(result)[:400] if error else None,
                    argument_preview=function['arguments'][:256] if error else None,
                    returncode=result.get('returncode') if isinstance(result, dict) else None,
                    timed_out=result.get('timed_out') if isinstance(result, dict) else None))
                reflection.append(dict(tool=name, result=json.dumps(result)[:800]))
                messages.append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(result)))
            if submitted:
                break
        try:
            solution = workspace_path(workspace, 'solution.py')
        except ValueError:
            solution, submitted = None, False
        submitted = submitted and solution is not None and solution.is_file()
        # Failed completion earns zero, even when a partially written file happens to pass.
        scored = score(benchmark.task(task_id), solution, sandbox) if submitted else dict(
            reward=0., pass_at_1=False, passed_test_fraction_lower_bound=0., total_tests=0, passed_tests=0, timed_out=False, seconds=0.)
        return dict(task_id=task_id, submitted=submitted, **scored,
                    wall_seconds=time.monotonic() - started, tools=tool_events,
                    reflection=dict(public_tool_results=reflection[-4:],
                        solution_excerpt=solution.read_text()[:6000] if solution is not None and solution.is_file() else ''),
                    solution_sha256=hashlib.sha256(solution.read_bytes()).hexdigest() if solution is not None and solution.is_file() else None)


def evaluate_development(model, benchmark, sandbox, tasks, files, budget, seed, tokens,
                         cascade=False, phase='development'):
    """Matched private cohort, equal per-problem caps, all nested costs charged."""
    if budget.remaining < tokens or tokens < len(tasks):
        raise ValueError('Insufficient budget for the complete development cohort')
    cap = tokens // len(tasks)
    parallelism = min(len(tasks), int(getattr(model, 'config', {}).get('parallel_requests', 1)))
    if parallelism > 1 and not cascade:
        def evaluate_one(task):
            child = Budget(cap)
            result = execute(model, benchmark, sandbox, task, files, child,
                             seed_for(seed, task), phase)
            return result, child
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            evaluated = list(pool.map(evaluate_one, tasks))
        results = [result for result, _ in evaluated]
        for _, child in evaluated:
            budget.absorb(child)
    else:
        results = []
        for task in tasks:
            result = execute(model, benchmark, sandbox, task, files, budget,
                             seed_for(seed, task), phase, budget.remaining - cap)
            results.append(result)
            if cascade and result['reward'] < 1:
                break
    denominator = len(tasks)
    return dict(reward=sum(x['reward'] for x in results) / denominator,
        submitted=len(results) == denominator and all(x['submitted'] for x in results),
        submitted_rate=sum(x['submitted'] for x in results) / denominator,
        passed_test_fraction_lower_bound=sum(x['passed_test_fraction_lower_bound'] for x in results) / denominator,
        evaluated_count=len(results), filtered_early=len(results) < denominator,
        reflection=[dict(task=benchmark.public_task(x['task_id']), **x['reflection']) for x in results],
        evaluations=results)


def compact_experience(benchmark, tasks: list, parent: dict | None,
                       candidate: dict | None, round_index: int,
                       kind: str = 'matched_fresh_minibatch') -> dict:
    """Keep cumulative learning evidence useful without duplicating raw traces."""
    def excerpt(value: Any, limit: int) -> str:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
        return encoded if len(encoded) <= limit else encoded[:limit] + '...[truncated]'

    def outcome(evaluation: dict | None, index: int) -> dict | None:
        if evaluation is None:
            return None
        item = evaluation['evaluations'][index]
        return dict(reward=item.get('reward', 0.), submitted=item.get('submitted', False),
                    passed_fraction=item.get('passed_test_fraction_lower_bound', 0.),
                    failure=item.get('failure'),
                    answer_and_feedback_excerpt=excerpt(item.get('reflection') or {}, 900))
    return dict(kind=kind, round=round_index,
                tasks=[dict(task_id=task,
                            problem_excerpt=excerpt(benchmark.public_task(task), 1200),
                            parent=outcome(parent, index),
                            candidate=outcome(candidate, index) if candidate is not None else None)
                       for index, task in enumerate(tasks)])


def full_archive_config(seed: int, config: dict) -> DatabaseConfig:
    settings = config.get('openevolve_database', {})
    return DatabaseConfig(num_islands=settings.get('num_islands', 5),
        population_size=settings.get('population_size', 1000),
        archive_size=settings.get('archive_size', 100),
        elite_selection_ratio=settings.get('elite_selection_ratio', .1),
        exploration_ratio=settings.get('exploration_ratio', .2),
        exploitation_ratio=settings.get('exploitation_ratio', .7),
        feature_dimensions=settings.get('feature_dimensions', ['complexity', 'diversity']),
        feature_bins=settings.get('feature_bins', 10),
        migration_interval=settings.get('migration_interval', 5),
        migration_rate=settings.get('migration_rate', .1),
        log_prompts=False, random_seed=seed)


def archive_state(database: ProgramDatabase) -> dict:
    return dict(programs={key: value.to_dict() for key, value in database.programs.items()},
        island_feature_maps=database.island_feature_maps,
        islands=[sorted(x) for x in database.islands], archive=sorted(database.archive),
        best_program_id=database.best_program_id,
        island_best_programs=database.island_best_programs,
        current_island=database.current_island,
        island_generations=database.island_generations,
        last_migration_generation=database.last_migration_generation,
        last_iteration=database.last_iteration,
        feature_stats=database.feature_stats,
        diversity_reference_set=database.diversity_reference_set)


def restore_archive(database: ProgramDatabase, state: dict) -> None:
    database.programs = {key: Program(**value) for key, value in state['programs'].items()}
    database.island_feature_maps = state['island_feature_maps']
    database.islands = [set(x) for x in state['islands']]
    database.archive = set(state['archive'])
    database.best_program_id = state['best_program_id']
    database.island_best_programs = state['island_best_programs']
    database.current_island = state['current_island']
    database.island_generations = state['island_generations']
    database.last_migration_generation = state['last_migration_generation']
    database.last_iteration = state['last_iteration']
    database.feature_stats = state['feature_stats']
    database.diversity_reference_set = state['diversity_reference_set']


def run_matched_growing(config: dict, output: Path, benchmark, sandbox, model,
                        learners: list, feedback: list, dev_counts: list,
                        start: int, fingerprint: str, task_order: list,
                        saved_archive: dict | None = None) -> None:
    """Matched RSI on fresh online minibatches or one fixed offline training set."""
    count, rounds, seed = config['num_agents'], config['rounds'], config['seed']
    policy = config['policy']
    cohort_size = config['development_cohort_size']
    evaluation_tokens = config['development_tokens']
    experience_chars = config.get('revision_experience_max_chars', 350000)
    if count != 1 or policy not in {'frozen', 'openevolve', 'openevolve_full'} or config.get('social_info', 'none') != 'none':
        raise ValueError('matched_growing is a one-agent frozen/OpenEvolve solo comparison')
    if type(experience_chars) is not int or experience_chars < 1000:
        raise ValueError('revision_experience_max_chars must be at least 1000')
    if policy in {'openevolve', 'openevolve_full'} and 2 * evaluation_tokens >= config['tokens_per_round']:
        raise ValueError('OpenEvolve needs room for revision plus two matched evaluations')
    full = policy == 'openevolve_full'
    offline = config.get('evolution_design') == 'offline_openevolve'
    if full:
        learner = learners[0]
        learner.archive_config = full_archive_config(seed, config)
        learner.archive = ProgramDatabase(learner.archive_config)
        if saved_archive is not None:
            restore_archive(learner.archive, saved_archive)
        learner.sampler.config.use_template_stochasticity = True
    checkpoint = output / 'checkpoint.json'
    if offline and policy == 'frozen':
        # The offline frozen control does not train or inspect the 64-problem
        # evolution set. Materialize the untouched initial artifact so the
        # launcher can proceed directly to the matched hidden replay.
        learner = learners[0]
        if start == 0:
            artifact = learner.versions[learner.active]
            atomic_json(output / 'artifacts' / f'{learner.active}.json', artifact)
            budget = Budget(config['tokens_per_round'])
            row = dict(agent=0, round=0, task_ids=[], version=learner.active,
                       deployed_version=learner.active, origin='initial', winner='frozen',
                       reward=0., submitted=False, execution_attempted=False,
                       offline_training_evaluation_skipped=True,
                       budget=vars(budget), wall_seconds_round=0.)
            atomic_json(output / 'rounds/0000.json', dict(results=[row], events=[]))
            atomic_json(checkpoint, dict(fingerprint=fingerprint, next_round=config['rounds'],
                learners=[learner.state()], feedback=feedback, prior=[None], dev_counts=dev_counts))
        atomic_json(output / 'completed.json', dict(rounds=config['rounds'], fingerprint=fingerprint))
        return
    for r in range(start, rounds):
        learner = learners[0]
        began = time.monotonic()
        budget = Budget(config['tokens_per_round'])
        tasks = task_order[:cohort_size] if offline else task_order[
            r * cohort_size:(r + 1) * cohort_size]
        parent = learner.active
        oe_parent = None
        oe_island = None
        inspirations = []
        candidate = None
        changed = False
        revision_error = None
        revision_experience_batches = 0
        revision_experience_chars = 0
        if full and learner.archive.programs:
            oe_island = r % learner.archive_config.num_islands
            state = random.getstate()
            random.seed(seed_for(seed, r, 'openevolve-selection'))
            try:
                oe_parent, inspirations = learner.archive.sample_from_island(
                    oe_island, num_inspirations=5)
            finally:
                random.setstate(state)
            parent = learner.install({'SKILL.md': oe_parent.code}, learner.active,
                                     'openevolve_archive', r)
            learner.active = parent
        if policy == 'openevolve' or full and oe_parent is not None:
            top_programs = learner.archive.get_top_programs(5) if full else []
            revision_options = {'experience_chars': experience_chars}
            if full:
                revision_options.update(inspirations=inspirations, top_programs=top_programs,
                    feature_dimensions=learner.archive_config.feature_dimensions)
            changed = learner.revise(model, budget, feedback[0], r,
                                     2 * evaluation_tokens, seed_for(seed, r, 'revision'),
                                     **revision_options)
            candidate = learner.active if changed else None
            revision_error = learner.last_revision_error
            revision_experience_batches = getattr(
                learner, 'last_revision_experience_records', len(feedback[0]))
            revision_experience_chars = getattr(
                learner, 'last_revision_experience_chars', len(json.dumps(feedback[0])))

        learner.active = parent
        cached_offline_parent = offline and full and bool(learner.archive.programs)
        parent_evaluation = None if cached_offline_parent else evaluate_development(
            model, benchmark, sandbox, tasks, learner.files, budget,
            seed_for(seed, r, 'matched-minibatch'), evaluation_tokens,
            cascade=False, phase='development_parent')
        if full and parent_evaluation is not None:
            learner.versions[parent]['development_scores'].append(parent_evaluation['reward'])
            if not learner.archive.programs:
                oe_island = 0
                learner.archive.add(Program(id=parent, code=learner.files['SKILL.md'],
                    language='markdown', iteration_found=r,
                    metrics={'combined_score': parent_evaluation['reward']},
                    metadata={'island': oe_island}), iteration=r, target_island=oe_island)
        elif not full:
            learner.add_evidence(parent, parent_evaluation['reward'])

        candidate_evaluation = None
        winner = 'parent'
        if candidate is not None:
            learner.active = candidate
            candidate_evaluation = evaluate_development(
                model, benchmark, sandbox, tasks, learner.files, budget,
                seed_for(seed, r, 'matched-minibatch'), evaluation_tokens,
                cascade=False, phase='development_candidate')
            if full:
                learner.versions[candidate]['development_scores'].append(candidate_evaluation['reward'])
                learner.archive.add(Program(id=candidate, code=learner.files['SKILL.md'],
                    language='markdown', parent_id=oe_parent.id,
                    generation=oe_parent.generation + 1, iteration_found=r,
                    metrics={'combined_score': candidate_evaluation['reward']},
                    metadata={'island': oe_island}), iteration=r, target_island=oe_island)
                learner.archive.increment_island_generation(oe_island)
                if learner.archive.should_migrate():
                    learner.archive.migrate_programs()
            else:
                learner.add_evidence(candidate, candidate_evaluation['reward'])
                if candidate_evaluation['reward'] > parent_evaluation['reward']:
                    winner = 'candidate'
                else:
                    learner.active = parent
        else:
            learner.active = parent

        if full:
            best = learner.archive.get_best_program()
            learner.active = learner.install({'SKILL.md': best.code}, parent,
                                             'openevolve_archive', r)
            winner = 'openevolve_archive'

        deployed = learner.active
        feedback[0].append(compact_experience(
            benchmark, tasks, parent_evaluation, candidate_evaluation, r,
            'fixed_offline_training_set' if offline else 'matched_fresh_minibatch'))
        dev_counts[0] += len(tasks) * (int(parent_evaluation is not None) +
                                      int(candidate_evaluation is not None))
        parent_reward = (parent_evaluation['reward'] if parent_evaluation is not None
                         else oe_parent.metrics.get('combined_score'))
        row = dict(agent=0, round=r, task_ids=tasks, version=deployed,
                   parent_version=parent, candidate_version=candidate,
                   deployed_version=deployed, winner=winner,
                   revision_changed=changed, revision_error=revision_error,
                   revision_experience_batches=revision_experience_batches,
                   revision_experience_chars=revision_experience_chars,
                   openevolve_parent_id=oe_parent.id if oe_parent is not None else None,
                   openevolve_island=oe_island,
                   openevolve_archive_size=len(learner.archive.archive) if full else None,
                   openevolve_population_size=len(learner.archive.programs) if full else None,
                   openevolve_map_cells=(sum(len(x) for x in learner.archive.island_feature_maps)
                                         if full else None),
                   reward=parent_reward,
                   prequential_parent_reward=parent_reward,
                   candidate_reward=(candidate_evaluation or {}).get('reward'),
                   deployed_batch_reward=(None if full else
                       candidate_evaluation['reward'] if winner == 'candidate'
                       else parent_evaluation['reward']),
                   paired_revision_uplift=(candidate_evaluation['reward'] - parent_evaluation['reward']
                                           if candidate_evaluation is not None and
                                           parent_evaluation is not None else None),
                   parent_evaluation=parent_evaluation,
                   candidate_evaluation=candidate_evaluation,
                   accumulated_experience_batches=len(feedback[0]),
                   accumulated_experience_tasks=sum(len(x['tasks']) for x in feedback[0]),
                   budget=vars(budget), wall_seconds_round=time.monotonic() - began)
        atomic_json(output / 'rounds' / f'{r:04d}.json', dict(results=[row], events=[]))
        for version, artifact in learner.versions.items():
            path = output / 'artifacts' / f'{version}.json'
            if not path.exists():
                atomic_json(path, artifact)
        saved = dict(fingerprint=fingerprint, next_round=r + 1,
            learners=[learner.state()], feedback=feedback, prior=[None], dev_counts=dev_counts)
        if full:
            saved['openevolve_archive'] = archive_state(learner.archive)
        atomic_json(checkpoint, saved)
        if getattr(model, 'client', None) is not None:
            model.client.clear_cache()
        print(json.dumps(dict(round=r, reward=row['reward'],
            candidate_reward=row['candidate_reward'], winner=winner,
            experience_tasks=row['accumulated_experience_tasks'],
            completion_tokens=budget.completion, cost_usd=budget.cost_usd)), flush=True)
    atomic_json(output / 'completed.json', dict(rounds=rounds, fingerprint=fingerprint))


def run(config: dict, output: Path) -> None:
    from rung4_benchmark import LiveCodeBench, Sandbox
    benchmark = LiveCodeBench(config['dataset'])
    sandbox = Sandbox(config['sandbox_image'])
    os.environ['AGENT_MARKET_API_CACHE_DIR'] = str(output / 'api_batches')
    model = Model(config['model'])
    count, rounds, seed = config['num_agents'], config['rounds'], config['seed']
    info = config.get('social_info', 'none')
    if info not in {'none', 'action', 'payoff'}:
        raise ValueError('social_info must be none/action/payoff')
    if config.get('policy', 'endogenous') not in {'endogenous', 'frozen', 'openevolve', 'openevolve_full'}:
        raise ValueError('Unknown policy')
    if config.get('policy') in {'frozen', 'openevolve', 'openevolve_full'} and info != 'none':
        raise ValueError('Frozen and OpenEvolve reference policies are solo controls')
    grant = config['tokens_per_round']
    initial_name = config.get('initial_skill', 'strong')
    if initial_name not in INITIAL_SKILLS:
        raise ValueError('initial_skill must be minimal/strong')
    initial_skill = INITIAL_SKILLS[initial_name]
    reserve = config.get('execution_reserve_tokens', 0)
    execution_tokens = config.get('execution_tokens', grant)
    if type(execution_tokens) is not int or not 0 < execution_tokens <= grant:
        raise ValueError('execution_tokens must be a positive integer within the round grant')
    development_tokens = config.get('development_tokens', grant // 4)
    independent = config.get('independent_search_probability', 0.)
    if not 0 <= reserve < grant or not 0 <= independent <= 1 or not 0 < development_tokens <= grant - reserve:
        raise ValueError('Invalid fixed-budget intervention')
    design = config.get('evolution_design')
    growing = design == 'matched_growing'
    offline = design == 'offline_openevolve'
    train = benchmark.train_ids
    if len(train) < 4 or not benchmark.holdout_ids:
        raise ValueError('Need separate development, online and hidden holdout problems')
    dev_pool_size = 0 if growing or offline else config.get('development_pool_size', min(16, len(train) // 2))
    if growing:
        development, online = [], train
    elif offline:
        offline_size = config.get('offline_training_size', 16)
        if type(offline_size) is not int or not 1 <= offline_size <= len(train):
            raise ValueError('Invalid offline training-set size')
        development = random.Random(seed_for(seed, 'offline-training')).sample(train, offline_size)
        online = []
    else:
        if not 1 <= dev_pool_size < len(train):
            raise ValueError('Invalid development pool size')
        development, online = train[:dev_pool_size], train[dev_pool_size:]
    cohort_source = online if growing else development
    cohort_size = config.get('development_cohort_size', min(4, len(cohort_source)))
    if not 1 <= cohort_size <= len(cohort_source) or development_tokens < cohort_size:
        raise ValueError('Invalid private development cohort size/budget')
    tasks_per_round = cohort_size if growing else 1
    if not offline and count * rounds * tasks_per_round > len(online):
        raise ValueError('Scored problems must be disjoint across the population: prepare a larger split or reduce agent-rounds')
    task_source = development if offline else online
    task_order = random.Random(seed_for(seed, 'population-tasks')).sample(task_source, len(task_source))
    fingerprint = digest(dict(config=experiment_config(config), dataset=benchmark.manifest))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'checkpoint.json'
    saved_archive = None
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved['fingerprint'] != fingerprint:
            raise ValueError('Refusing resume with changed config or dataset')
        provenance = json.loads((output / 'provenance.json').read_text())
        for name, expected in provenance.get('source_sha256', {}).items():
            source = Path(__file__).resolve().parents[1] / name if name == 'pyproject.toml' else Path(__file__).resolve().parent / name
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Refusing resume after source change: {name}; use a new run directory')
        learners = [Learner(x['agent_id'], seed_for(seed, x['agent_id']),
                    x['versions'], x['active'], x['acquisitions'], x.get('observations'),
                    initial_skill) for x in saved['learners']]
        feedback, prior, dev_counts = saved['feedback'], saved['prior'], saved['dev_counts']
        saved_archive = saved.get('openevolve_archive')
        start = saved['next_round']
    else:
        learners = [Learner(i, seed_for(seed, i), initial_skill=initial_skill) for i in range(count)]
        feedback, prior, dev_counts = [[] for _ in learners], [None for _ in learners], [0] * count
        start = 0
        atomic_json(output / 'config.json', config)
        source_hashes = {}
        for name in ('rung4.py', 'rung4_benchmark.py', 'rung4_server.py', 'openrouter.py', 'pyproject.toml'):
            source = Path(__file__).resolve().parents[1] / name if name == 'pyproject.toml' else Path(__file__).resolve().parent / name
            data = source.read_bytes()
            target = output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            source_hashes[name] = hashlib.sha256(data).hexdigest()
        atomic_json(output / 'provenance.json', dict(fingerprint=fingerprint,
            dataset={k: v for k, v in benchmark.manifest.items() if k != 'tasks'},
            development_ids=development, online_ids=online, holdout_ids=benchmark.holdout_ids,
            initial_skill=initial_name, initial_version=digest({'SKILL.md': initial_skill})[:24],
            source_sha256=source_hashes,
            openevolve_commit='411fb59c886c18704caaffb611e17cf9e7d824d2'))
    if growing or offline:
        run_matched_growing(config, output, benchmark, sandbox, model, learners,
                            feedback, dev_counts, start, fingerprint, task_order, saved_archive)
        return
    for r in range(start, rounds):
        results, next_prior, events = [], [], []
        for learner in learners:
            i = learner.id
            dev_tasks = random.Random(seed_for(seed, i, 'development-cohort')).sample(development, cohort_size)
            skill_directory = output / 'agents' / str(i) / 'skills'
            learner.sync_skills(skill_directory)
            budget = Budget(grant)
            began = time.monotonic()
            assigned = random.Random(seed_for(seed, r, i, 'independent')).random() < independent
            independent_done = False
            policy = config.get('policy', 'endogenous')
            system = (
                'You solve a sequence of private coding problems using a persistent skill library. '
                'Before each scored problem, choose how much to invest in improving or selecting your skills. '
                'Use write_skill to edit files, revise for an OpenEvolve proposal, evaluate for development feedback, '
                'and solve to execute the scored problem with the remaining budget. '
                'Development solves and revisions consume the SAME completion-token budget as scored execution. '
                'Rewards are the fraction of scored problems fully solved. Public examples can be tested with tools. '
                'Development test feedback is aggregate only. Hidden held-out evaluations are never shown. '
                f'Your total round budget is {grant} completion tokens; {reserve} are reserved for scored execution. '
                f'A development evaluation needs {development_tokens} available tokens and tests {cohort_size} fixed private problems with equal caps. '
                f'Your agent ID is {i}. '
                + (f'You may observe agents {[j for j in range(count) if j != i]}; each observation reveals '
                   'their last scored skill, and only payoff information when enabled. You must explicitly import to adopt it. '
                   if info != 'none' else '')
                + ('This round requires an independent revision before social observation or solving. ' if assigned else '')
            )
            messages = [dict(role='system', content=system), dict(role='user', content=json.dumps(dict(
                round=r, active=learner.active, files=list(learner.files),
                library={v: dict(acquisition=learner.acquisitions[v], development_scores=x['development_scores'])
                         for v, x in learner.versions.items()},
                observations=learner.observations[-32:], feedback=feedback[i][-4:])))]
            if policy == 'openevolve':
                # Algorithmic reference: OpenEvolve chooses a private archived parent, then revises.
                # One island per principal, with no implicit cross-agent migration.
                if not learner.archive.programs:
                    initial = evaluate_development(model, benchmark, sandbox, dev_tasks, learner.files,
                                                   budget, seed_for(seed, i, 'development'), development_tokens,
                                                   **({'cascade': True} if config.get('development_cascade') else {}))
                    learner.add_evidence(learner.active, initial['reward'])
                    feedback[i].append(dict(kind='development', version=learner.active,
                        tasks=[benchmark.public_task(t) for t in dev_tasks], reward=initial['reward'], submitted=initial['submitted'],
                        reflection=initial.get('reflection')))
                    events.append(dict(agent=i, round=r, tool='openevolve_initial_evaluate',
                                       version=learner.active, evaluation=initial))
                if learner.archive.programs:
                    state = random.getstate()
                    random.seed(seed_for(seed, r, i, 'parent'))
                    try:
                        parent, _ = learner.archive.sample()
                        learner.active = parent.id
                    finally:
                        random.setstate(state)
                independent_done = learner.revise(model, budget, feedback[i], r, reserve, seed_for(seed, r, i, 'revision'))
                events.append(dict(agent=i, round=r, tool='openevolve_revision',
                    version=learner.active, changed=independent_done, error=learner.last_revision_error))
                if budget.remaining - reserve >= development_tokens:
                    evaluation = evaluate_development(model, benchmark, sandbox, dev_tasks, learner.files,
                                                      budget, seed_for(seed, i, 'development'), development_tokens,
                                                      **({'cascade': True} if config.get('development_cascade') else {}))
                    learner.add_evidence(learner.active, evaluation['reward'])
                    feedback[i].append(dict(kind='development', version=learner.active,
                        tasks=[benchmark.public_task(t) for t in dev_tasks], reward=evaluation['reward'],
                        submitted=evaluation['submitted'], reflection=evaluation.get('reflection')))
                    events.append(dict(agent=i, round=r, tool='openevolve_evaluate',
                                       version=learner.active, evaluation=evaluation))
                learner.active = learner.archive.get_best_program().id
            elif policy != 'frozen':
                for step in range(32):
                    available = META_TOOLS + ([OBSERVE, IMPORT] if info != 'none' and not (assigned and not independent_done) else [])
                    call_completion, call_prompt, call_seconds = budget.completion, budget.prompt, budget.inference_seconds
                    message = model.call(messages, budget, 'selection', seed_for(seed, r, i, 'meta', step), available, reserve)
                    decision_usage = dict(call_id=f'{r}:{i}:{step}',
                        decision_completion_tokens=budget.completion - call_completion,
                        decision_prompt_tokens=budget.prompt - call_prompt,
                        decision_inference_seconds=budget.inference_seconds - call_seconds)
                    if message is None:
                        break
                    messages.append(message)
                    calls = message.get('tool_calls') or []
                    if not calls:
                        events.append(dict(agent=i, round=r, tool='no_tool_call', error=True,
                            error_message=(message.get('content') or '')[:400] if
                            (message.get('content') or '').startswith(PARSER_ERROR_PREFIX) else None,
                            **decision_usage))
                        messages.append(dict(role='user', content='Choose a tool action; call solve when ready.'))
                        continue
                    end = False
                    for call in calls:
                        name = call['function']['name']
                        before = learner.active
                        event = dict(agent=i, round=r, tool=name, before=before,
                                     completion_before=budget.completion, **decision_usage)
                        try:
                            args = json.loads(call['function']['arguments'])
                            if name == 'read_skill':
                                result = learner.files[args['path']]
                            elif name == 'write_skill':
                                files = learner.files.copy()
                                files[args['path']] = args['content']
                                known = set(learner.versions)
                                learner.active = learner.install(files, before, 'independent', r)
                                independent_done |= learner.active not in known
                                result = learner.active
                            elif name == 'revise':
                                changed = learner.revise(model, budget, feedback[i], r, reserve,
                                                         seed_for(seed, r, i, 'revision', step))
                                independent_done |= changed
                                result = dict(changed=changed, version=learner.active, error=learner.last_revision_error)
                            elif name == 'evaluate':
                                if budget.remaining - reserve < development_tokens:
                                    raise ValueError(f'Development evaluation requires {development_tokens} available tokens outside the execution reserve')
                                dev_counts[i] += 1
                                result = evaluate_development(model, benchmark, sandbox, dev_tasks, learner.files,
                                                              budget, seed_for(seed, i, 'development'), development_tokens,
                                                              **({'cascade': True} if config.get('development_cascade') else {}))
                                learner.add_evidence(learner.active, result['reward'])
                                feedback[i].append(dict(kind='development', version=learner.active,
                                    tasks=[benchmark.public_task(t) for t in dev_tasks], reward=result['reward'],
                                    submitted=result['submitted'], passed_test_fraction_lower_bound=result["passed_test_fraction_lower_bound"],
                                    reflection=result.get('reflection')))
                                event['evaluation'] = result
                            elif name == 'observe':
                                target = int(args['target'])
                                if info == 'none' or assigned and not independent_done or target == i or not 0 <= target < count:
                                    raise ValueError('Observation unavailable')
                                snapshot = prior[target]
                                event['target'] = target
                                if snapshot is None:
                                    result = 'No previous scored action'
                                else:
                                    vid = snapshot['version']
                                    # Copy immutable source metadata, never private development evidence.
                                    if vid not in learner.versions:
                                        learner.versions[vid] = dict(snapshot['artifact'], development_scores=[])
                                    learner.acquisitions.setdefault(vid, dict(origin='social', round=r, source=target))
                                    result = dict(version=vid, files=snapshot['artifact']['files'])
                                    if info == 'payoff':
                                        result['reward'] = snapshot['reward']
                                    learner.observations.append(dict(target=target, source_round=r - 1,
                                        acquired_round=r, version=vid,
                                        reward=snapshot['reward'] if info == 'payoff' else None))
                                    event['version'] = vid
                            elif name in {'select', 'import_skill'}:
                                vid = args['version']
                                if vid not in learner.versions:
                                    raise ValueError('Unknown version')
                                if name == 'import_skill' and learner.acquisitions[vid]['origin'] != 'social':
                                    raise ValueError('Not a socially acquired version')
                                learner.active = vid
                                result = vid
                            elif name == 'solve':
                                if assigned and not independent_done:
                                    raise ValueError('Complete an independent revision first')
                                result, end = 'Starting scored execution', True
                            else:
                                raise ValueError('Unknown tool')
                            event['error'] = False
                        except (ValueError, KeyError, OSError, TypeError) as exc:
                            result, event['error'] = str(exc)[:400], True
                            event.update(error_message=result, argument_preview=call['function']['arguments'][:256])
                        event.update(after=learner.active, completion_after=budget.completion)
                        if learner.active != before:
                            learner.sync_skills(skill_directory)
                        events.append(event)
                        messages.append(dict(role='tool', tool_call_id=call['id'], content=json.dumps(result)))
                    if end:
                        break
            # Explicit reserve intervention starts execution when selection's spendable portion ends.
            # In the unprotected condition, an exhausted budget yields no completed solution.
            vid = learner.active
            learner.sync_skills(skill_directory)
            task_id = task_order[r * count + i]
            if assigned and not independent_done:
                result = dict(task_id=task_id, submitted=False, reward=0., pass_at_1=False,
                    passed_test_fraction_lower_bound=0., total_tests=0, passed_tests=0,
                    timed_out=False, seconds=0., wall_seconds=0., tools=[], solution_sha256=None,
                    failure='required_independent_revision_not_completed')
            else:
                result = execute(model, benchmark, sandbox, task_id, learner.files, budget,
                                 seed_for(seed, r, i, 'execution'), 'execution',
                                 max(0, budget.remaining - execution_tokens))
            feedback[i].append(dict(kind='scored', version=vid, task=benchmark.public_task(task_id),
                                   reward=result['reward'], submitted=result['submitted'],
                                   reflection=result.get('reflection')))
            feedback[i] = feedback[i][-8:]
            artifact = learner.versions[vid]
            attempted = (budget.phases.get('execution', {}).get('calls', 0) > 0
                         or budget.provider_errors_by_phase.get('execution', 0) > 0)
            next_prior.append(dict(version=vid, artifact={k: v for k, v in artifact.items() if k != 'development_scores'}, reward=result['reward']) if attempted else None)
            row = dict(agent=i, round=r, version=vid, origin=learner.acquisitions[vid]['origin'],
                       execution_attempted=attempted,
                       independent_assigned=assigned, independent_completed=independent_done,
                       **result, budget=vars(budget), wall_seconds_round=time.monotonic() - began)
            results.append(row)
            for version, artifact in learner.versions.items():
                path = output / 'artifacts' / f'{version}.json'
                if not path.exists():
                    atomic_json(path, artifact)
        atomic_json(output / 'rounds' / f'{r:04d}.json', dict(results=results, events=events))
        prior = next_prior
        atomic_json(checkpoint, dict(fingerprint=fingerprint, next_round=r + 1,
            learners=[x.state() for x in learners], feedback=feedback, prior=prior, dev_counts=dev_counts))
        if getattr(model, 'client', None) is not None:
            model.client.clear_cache()
        print(json.dumps(dict(round=r, reward=sum(x['reward'] for x in results) / count,
            completion_tokens=sum(x['budget']['completion'] for x in results),
            provider_cap_overruns=sum(x['budget']['provider_cap_overruns'] for x in results),
            phase_cap_violations=sum(x['budget']['phase_cap_violations'] for x in results),
            empty_completions=sum(x['budget']['empty_completions'] for x in results),
            calls_without_cost=sum(x['budget']['calls_without_cost'] for x in results),
            cost_usd=sum(x['budget']['cost_usd'] for x in results))), flush=True)
    atomic_json(output / 'completed.json', dict(rounds=rounds, fingerprint=fingerprint))


def selected_holdout_ids(available: list, output: Path) -> list:
    """A prespecified evaluation subset can shrink testing without retraining."""
    selection = output / 'evaluation_selection.json'
    if not selection.exists():
        return available
    ids = json.loads(selection.read_text())['task_ids']
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(available):
        raise ValueError('Invalid prespecified held-out task selection')
    return ids


def replay(config: dict, output: Path) -> None:
    """Hidden common-task evaluation; never modify learning state or feedback."""
    from rung4_benchmark import LiveCodeBench, Sandbox
    benchmark = LiveCodeBench(config['dataset'])
    holdout_ids = selected_holdout_ids(benchmark.holdout_ids, output)
    os.environ['AGENT_MARKET_API_CACHE_DIR'] = str(output / 'api_batches_replay')
    model, sandbox = Model(config['model']), Sandbox(config['sandbox_image'])
    source_hashes = {name: hashlib.sha256((Path(__file__).resolve().parent / name).read_bytes()).hexdigest()
                     for name in ('rung4.py', 'rung4_benchmark.py', 'rung4_server.py', 'openrouter.py')}
    # Predetermine checkpoints in the config; no outcome-based cherry-picking.
    checkpoints = config.get('holdout_rounds', [0, config['rounds'] - 1])
    initial_name = config.get('initial_skill', 'strong')
    if initial_name not in INITIAL_SKILLS:
        raise ValueError('initial_skill must be minimal/strong')
    versions = {digest({'SKILL.md': INITIAL_SKILLS[initial_name]})[:24]}
    for r in checkpoints:
        path = output / 'rounds' / f'{r:04d}.json'
        if not path.exists():
            raise ValueError(f'Missing declared checkpoint {r}')
        versions.update(row['version'] for row in json.loads(path.read_text())['results'])
    scope = config.get('holdout_scope', 'selected')
    if scope not in {'selected', 'acquired'}:
        raise ValueError('holdout_scope must be selected/acquired')
    if scope == 'acquired':
        saved = json.loads((output / 'checkpoint.json').read_text())
        for learner in saved['learners']:
            versions.update(vid for vid, acquisition in learner['acquisitions'].items()
                            if acquisition['round'] <= max(checkpoints))
    parallelism = max(1, int(config['model'].get('parallel_requests', 1)))
    for vid in sorted(versions):
        files = json.loads((output / 'artifacts' / f'{vid}.json').read_text())['files']
        pending = []
        for task in holdout_ids:
            for repeat in range(config.get('holdout_repeats', 2)):
                path = output / 'holdout' / f'{vid}-{digest(task)[:12]}-{repeat}.json'
                identity = dict(version=vid, task=task, repeat=repeat,
                                model=experiment_config(config)['model'], budget=config['holdout_tokens'],
                                dataset=digest(benchmark.manifest), source_sha256=source_hashes,
                                execution_seed=seed_for(config['seed'], task, repeat, 'holdout'))
                if path.exists():
                    if json.loads(path.read_text())['identity'] != identity:
                        raise ValueError('Hidden replay configuration changed')
                    continue
                pending.append((path, task, identity))
        def evaluate_one(item):
            path, task, identity = item
            budget = Budget(config['holdout_tokens'])
            result = execute(model, benchmark, sandbox, task, files, budget,
                             identity['execution_seed'], 'holdout')
            # Persist each finished task immediately, even if a later task fails.
            atomic_json(path, dict(identity=identity, result=result, budget=vars(budget)))
            return path
        with ThreadPoolExecutor(max_workers=min(parallelism, len(pending) or 1)) as pool:
            list(pool.map(evaluate_one, pending))
        if getattr(model, 'client', None) is not None:
            model.client.clear_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    (replay if args.replay else run)(json.loads(args.config.read_text()), args.output)
