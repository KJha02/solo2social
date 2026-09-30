"""Scientific invariants of nested budgets, immutable skills and hidden replay."""
import json
from pathlib import Path

import pytest

from rung4 import Budget, Learner, Model, PARSER_ERROR_PREFIX, digest, workspace_path


def test_prespecified_test_subset_keeps_training_manifest_unchanged(tmp_path):
    from rung4 import selected_holdout_ids
    available = [f'test-{i}' for i in range(200)]
    assert selected_holdout_ids(available, tmp_path) == available
    (tmp_path / 'evaluation_selection.json').write_text(json.dumps({'task_ids': available[:50]}))
    assert selected_holdout_ids(available, tmp_path) == available[:50]
    assert len(available) == 200
    (tmp_path / 'evaluation_selection.json').write_text(json.dumps({'task_ids': ['train-0']}))
    with pytest.raises(ValueError, match='held-out'):
        selected_holdout_ids(available, tmp_path)


def test_nested_calls_share_actual_usage_and_reserve(monkeypatch):
    bodies = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({'usage': {'completion_tokens': 7, 'prompt_tokens': 20},
                               'choices': [{'message': {'role': 'assistant', 'content': 'x'}}]}).encode()
    def request(req, timeout):
        bodies.append(json.loads(req.data))
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', request)
    model = Model({'name': 'test', 'base_url': 'http://localhost/v1'})
    budget = Budget(20)
    model.call([], budget, 'selection', 0, reserve=10)
    model.call([], budget, 'execution', 1)
    assert [x['max_tokens'] for x in bodies] == [10, 13]
    assert (budget.completion, budget.prompt) == (14, 40)
    assert model.call([], budget, 'revision', 0, reserve=10) is None
    assert budget.phases['selection']['completion_tokens'] == 7
    with pytest.raises(RuntimeError):
        budget.charge({'completion_tokens': 7, 'prompt_tokens': 1}, 0, 'revision')


def test_skill_versions_provenance_and_resume():
    learner = Learner(0, 123)
    initial = learner.active
    files = {'SKILL.md': 'Check constraints; test boundary cases.'}
    new = learner.install(files, initial, 'independent', 2)
    learner.active = new
    learner.add_evidence(new, 1.)
    assert learner.install(files, initial, 'social', 3, source=1) == new
    assert learner.acquisitions[new]['origin'] == 'independent'
    assert learner.versions[initial]['files'] != files
    state = json.loads(json.dumps(learner.state()))
    resumed = Learner(0, 123, state['versions'], state['active'], state['acquisitions'])
    assert resumed.state() == learner.state()
    assert resumed.archive.get_best_program().id == new
    with pytest.raises(ValueError):
        learner.install({'../bad.md': 'bad'}, new, 'independent', 4)


def test_provider_tool_failure_preserves_reserve_without_inventing_usage(monkeypatch):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    model = Model({'provider': 'openrouter', 'name': 'test/model:batch'})
    monkeypatch.setattr(model.client, 'chat_completions', lambda bodies: [
        {'generation_error': 'MALFORMED_FUNCTION_CALL', 'usage_unavailable': True}])
    budget = Budget(20)
    message = model.call([], budget, 'development', 0, reserve=12)
    assert message['content'].startswith(PARSER_ERROR_PREFIX)
    assert budget.remaining == 12
    assert budget.completion == budget.prompt == 0
    assert budget.unreported_completion_allowance == 8
    assert budget.provider_generation_errors == budget.parser_errors == 1
    assert budget.provider_errors_by_phase == {'development': 1}
    assert model.call([], budget, 'development', 0, reserve=12) is None
    budget.charge({'completion_tokens': 12, 'prompt_tokens': 1}, 0, 'execution')
    assert budget.remaining == 0


def test_openrouter_headroom_overrun_and_empty_response(monkeypatch):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    model = Model({'provider': 'openrouter', 'name': 'test/model',
                   'max_call_tokens': 20, 'min_call_tokens': 5, 'token_cap_headroom': 4})
    bodies = []
    response = {'usage': {'completion_tokens': 18, 'prompt_tokens': 2, 'cost': .001},
                'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}]}
    def generate(requests):
        bodies.extend(requests)
        return [response]
    monkeypatch.setattr(model.client, 'chat_completions', generate)
    budget = Budget(30)
    assert model.call([], budget, 'development', 0, reserve=10)['content'] == 'ok'
    assert bodies[0]['max_tokens'] == 16
    assert budget.remaining == 12 and budget.provider_cap_overruns == 1
    assert model.call([], budget, 'development', 1, reserve=10) is None
    assert len(bodies) == 1
    response['usage']['completion_tokens'] = 0
    response['choices'][0]['message']['content'] = None
    assert model.call([], budget, 'execution', 2) is None
    assert budget.empty_completions == 1


def test_true_provider_phase_overrun_is_charged_but_discarded(monkeypatch):
    monkeypatch.setenv('OPENROUTER_KEY', 'test-key')
    model = Model({'provider': 'openrouter', 'name': 'test/model'})
    monkeypatch.setattr(model.client, 'chat_completions', lambda bodies: [{
        'usage': {'completion_tokens': 25, 'prompt_tokens': 2, 'cost': .01},
        'choices': [{'message': {'role': 'assistant', 'content': 'over budget'}}]}])
    budget = Budget(20)
    assert model.call([], budget, 'execution', 0) is None
    assert budget.completion == 25 and budget.remaining == 0
    assert budget.phase_cap_violations == 1 and budget.cost_usd == .01


def test_serving_parse_error_is_charged_and_never_installed(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({'usage': {'completion_tokens': 7, 'prompt_tokens': 20},
                'choices': [{'message': {'role': 'assistant',
                    'content': PARSER_ERROR_PREFIX + 'invalid header', 'tool_calls': []}}]}).encode()
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: Response())
    model = Model({'name': 'test', 'base_url': 'http://localhost/v1'})
    budget, learner = Budget(20), Learner(0, 0)
    initial = learner.active
    assert not learner.revise(model, budget, [], 0, 0, 0)
    assert learner.active == initial and len(learner.versions) == 1
    assert (budget.completion, budget.prompt, budget.parser_errors) == (7, 20, 1)
    assert learner.last_revision_error.startswith(PARSER_ERROR_PREFIX)


def test_workspace_does_not_follow_host_symlinks(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'outside').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        workspace_path(work, 'outside/hidden.json')
    with pytest.raises(ValueError):
        workspace_path(work, '../hidden.json')


def test_population_tools_snapshot_resume_and_hidden_replay(monkeypatch, tmp_path):
    """Exercise the real population loop and executor with scripted native tool messages."""
    import rung4
    import rung4_benchmark

    class Benchmark:
        def __init__(self, _):
            self.train_ids = ['d0', 'o0', 'd1', 'o1', 'd2', 'o2', 'd3', 'o3']
            self.holdout_ids = ['h0']
            self.manifest = {'tasks': self.train_ids + self.holdout_ids}
        def public_task(self, task):
            return {'id': task, 'question_content': 'Print zero.'}
        def task(self, task):
            return {'id': task}

    class Sandbox:
        def __init__(self, _): pass
        def run(self, work, command, **kwargs):
            assert command == 'python3 solution.py'
            assert (work / 'solution.py').read_text() == 'print(0)'
            return {'returncode': 0, 'output': '0', 'seconds': 0., 'timed_out': False}

    def score(task, solution, sandbox):
        assert solution.read_text() == 'print(0)'
        return dict(reward=1., pass_at_1=True, passed_test_fraction_lower_bound=1.,
                    total_tests=1, passed_tests=1, timed_out=False, seconds=0.)

    class NativeTools:
        total_calls = 0
        def __init__(self, _): pass
        def call(self, messages, budget, phase, seed, tools=None, reserve=0):
            if budget.remaining <= reserve:
                return None
            NativeTools.total_calls += 1
            budget.charge({'completion_tokens': 1, 'prompt_tokens': 10}, .01, phase)
            step = sum(m['role'] == 'assistant' for m in messages)
            if phase == 'selection':
                state = json.loads(messages[1]['content'])
                agent = int(messages[0]['content'].split('Your agent ID is ')[1].split('.')[0])
                if state['round'] == 0:
                    name, args = ('write_skill', {'path': 'SKILL.md', 'content': f'# Skill from {agent}'}) if step == 0 else ('solve', {})
                elif step == 0:
                    name, args = 'observe', {'target': 1 - agent}
                elif step == 1:
                    observation = json.loads(messages[-1]['content'])
                    name, args = 'import_skill', {'version': observation['version']}
                else:
                    name, args = 'solve', {}
            else:
                name, args = [
                    ('read_file', {'path': 'skills/SKILL.md'}),
                    ('write_file', {'path': 'solution.py', 'content': 'print(0)'}),
                    ('run', {'command': 'python3 solution.py'}),
                    ('submit', {}),
                ][step]
            return {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': f'call{step}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}

    monkeypatch.setattr(rung4, 'Model', NativeTools)
    monkeypatch.setattr(rung4_benchmark, 'LiveCodeBench', Benchmark)
    monkeypatch.setattr(rung4_benchmark, 'Sandbox', Sandbox)
    monkeypatch.setattr(rung4_benchmark, 'score', score)
    config = dict(dataset='fixture', sandbox_image='fixture', model={'name': 'fixture'},
                  num_agents=2, rounds=2, seed=7, social_info='payoff', tokens_per_round=16,
                  execution_reserve_tokens=4, holdout_tokens=8, holdout_repeats=1, holdout_rounds=[0, 1])
    full = tmp_path / 'full'
    rung4.run(config, full)
    first = json.loads((full / 'rounds/0000.json').read_text())['results']
    second = json.loads((full / 'rounds/0001.json').read_text())['results']
    assert len({x['task_id'] for x in first + second}) == 4
    assert first[0]['version'] != first[1]['version']
    assert [x['version'] for x in second] == [first[1]['version'], first[0]['version']]
    assert all(x['origin'] == 'social' and x['submitted'] for x in second)
    assert all(x['budget']['completion'] == 7 and x['budget']['prompt'] == 70 for x in second)

    interrupted = tmp_path / 'resumed'
    original = rung4.atomic_json
    def stop_after_checkpoint(path, value):
        original(path, value)
        if path.name == 'checkpoint.json' and value['next_round'] == 1:
            raise InterruptedError('scripted preemption at checkpoint')
    monkeypatch.setattr(rung4, 'atomic_json', stop_after_checkpoint)
    with pytest.raises(InterruptedError):
        rung4.run(config, interrupted)
    monkeypatch.setattr(rung4, 'atomic_json', original)
    rung4.run(config, interrupted)
    assert json.loads((full / 'checkpoint.json').read_text()) == json.loads((interrupted / 'checkpoint.json').read_text())

    checkpoint_before = (full / 'checkpoint.json').read_bytes()
    rung4.replay(config, full)
    assert (full / 'checkpoint.json').read_bytes() == checkpoint_before
    assert len(list((full / 'holdout').glob('*.json'))) == 3  # initial + two created skills
    calls = NativeTools.total_calls
    rung4.replay(config, full)
    assert NativeTools.total_calls == calls  # exact-evaluation cache reused
    with pytest.raises(ValueError, match='Hidden replay configuration changed'):
        rung4.replay(config | {'seed': config['seed'] + 1}, full)
    cached = next((full / 'holdout').glob('*.json'))
    original_cache = cached.read_bytes()
    value = json.loads(original_cache)
    value['identity']['source_sha256']['rung4.py'] = 'changed source'
    cached.write_text(json.dumps(value))
    try:
        with pytest.raises(ValueError, match='Hidden replay configuration changed'):
            rung4.replay(config, full)
    finally:
        cached.write_bytes(original_cache)


def test_archive_mean_update_and_json_resume_preserve_parent_sampling():
    import random
    learner = Learner(0, 123)
    initial = learner.active
    other = learner.install({'SKILL.md': 'Alternative procedure'}, initial, 'independent', 1)
    learner.add_evidence(initial, 1.)
    learner.add_evidence(other, .75)
    assert learner.archive.get_best_program().id == initial
    learner.add_evidence(initial, 0.)  # the old cached best now has mean .5
    assert learner.archive.get_best_program().id == other
    state = json.loads(json.dumps(learner.state(), sort_keys=True))
    resumed = Learner(0, 123, state['versions'], state['active'], state['acquisitions'], state['observations'])
    rng = random.getstate()
    try:
        for seed in range(20):
            random.seed(seed)
            first = learner.archive.sample()[0].id
            random.seed(seed)
            assert resumed.archive.sample()[0].id == first
    finally:
        random.setstate(rng)


def test_development_cohort_uses_equal_caps_without_reallocating_unused_tokens(monkeypatch):
    import rung4
    caps = []
    class Benchmark:
        def public_task(self, task): return {'id': task}
    def execute(model, benchmark, sandbox, task, files, budget, seed, phase, reserve):
        cap = budget.remaining - reserve
        caps.append(cap)
        budget.charge({'completion_tokens': 1 if task == 'a' else cap, 'prompt_tokens': 2}, .01, phase)
        return dict(task_id=task, reward=float(task == 'b'), submitted=True,
                    passed_test_fraction_lower_bound=float(task == 'b'), reflection={})
    monkeypatch.setattr(rung4, 'execute', execute)
    budget = Budget(20)
    result = rung4.evaluate_development(None, Benchmark(), None, ['a', 'b'], {}, budget, 7, 11)
    assert caps == [5, 5]  # first task's unused allowance does not enlarge second's
    assert budget.completion == 6 and budget.remaining == 14
    assert result['reward'] == .5 and len(result['evaluations']) == 2
    with pytest.raises(ValueError, match='Insufficient budget'):
        rung4.evaluate_development(None, Benchmark(), None, ['a', 'b'], {}, budget, 7, 15)


def test_development_cohort_runs_parallel_calls_and_merges_usage(monkeypatch):
    import threading
    import rung4
    barrier = threading.Barrier(4)

    class Model:
        config = {'parallel_requests': 4}
    class Benchmark:
        def public_task(self, task): return {'id': task}
    def execute(model, benchmark, sandbox, task, files, budget, seed, phase, reserve=0):
        barrier.wait(timeout=2)
        budget.charge({'completion_tokens': 2, 'prompt_tokens': 3}, .01, phase)
        return dict(task_id=task, reward=1., submitted=True,
                    passed_test_fraction_lower_bound=1., reflection={})

    monkeypatch.setattr(rung4, 'execute', execute)
    budget = Budget(40)
    result = rung4.evaluate_development(
        Model(), Benchmark(), None, ['a', 'b', 'c', 'd'], {}, budget, 7, 32)
    assert [row['task_id'] for row in result['evaluations']] == ['a', 'b', 'c', 'd']
    assert budget.completion == 8 and budget.prompt == 12 and budget.calls == 4
    assert budget.phases['development']['calls'] == 4


def test_hidden_replay_runs_holdout_tasks_in_parallel(monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace
    import rung4
    import rung4_benchmark
    barrier = threading.Barrier(4)

    class Benchmark:
        def __init__(self, _):
            self.holdout_ids = ['h0', 'h1', 'h2', 'h3']
            self.manifest = {'tasks': self.holdout_ids}
    monkeypatch.setattr(rung4_benchmark, 'LiveCodeBench', Benchmark)
    monkeypatch.setattr(rung4_benchmark, 'Sandbox', lambda _: None)
    monkeypatch.setattr(rung4, 'Model', lambda config:
                        SimpleNamespace(config=config, client=None))

    def execute(model, benchmark, sandbox, task, files, budget, seed, phase, reserve=0):
        barrier.wait(timeout=2)
        budget.charge({'completion_tokens': 2, 'prompt_tokens': 3}, .01, phase)
        return dict(task_id=task, reward=1., submitted=True)
    monkeypatch.setattr(rung4, 'execute', execute)

    config = dict(dataset='fixture', sandbox_image='fixture',
                  model={'name': 'fixture', 'parallel_requests': 4},
                  seed=3, rounds=1, initial_skill='strong', holdout_tokens=8,
                  holdout_repeats=1, holdout_rounds=[0], holdout_scope='selected')
    output = tmp_path / 'parallel-replay'
    version = rung4.digest({'SKILL.md': rung4.INITIAL_SKILL})[:24]
    rung4.atomic_json(output / 'artifacts' / f'{version}.json',
                      {'files': {'SKILL.md': rung4.INITIAL_SKILL}})
    rung4.atomic_json(output / 'rounds/0000.json',
                      {'results': [{'version': version}], 'events': []})
    rung4.atomic_json(output / 'provenance.json', {'holdout_ids': Benchmark(None).holdout_ids})
    rung4.replay(config, output)

    saved = [json.loads(path.read_text()) for path in (output / 'holdout').glob('*.json')]
    assert len(saved) == 4
    assert {row['result']['task_id'] for row in saved} == set(Benchmark(None).holdout_ids)
    assert all(row['budget']['completion'] == 2 for row in saved)


@pytest.mark.parametrize('execution_tokens', [None, 1])
def test_reference_skips_unperformed_candidate_evaluation(monkeypatch, tmp_path, execution_tokens):
    import rung4
    import rung4_benchmark
    class Benchmark:
        def __init__(self, _):
            self.train_ids = ['d0', 'o0', 'd1', 'o1']
            self.holdout_ids = ['h0']
            self.manifest = {'tasks': self.train_ids + self.holdout_ids}
        def public_task(self, task): return {'id': task}
    monkeypatch.setattr(rung4_benchmark, 'LiveCodeBench', Benchmark)
    monkeypatch.setattr(rung4_benchmark, 'Sandbox', lambda _: None)
    monkeypatch.setattr(rung4, 'Model', lambda _: None)
    evaluations = []
    def evaluate(model, benchmark, sandbox, tasks, files, budget, seed, tokens):
        evaluations.append(list(tasks))
        budget.charge({'completion_tokens': tokens, 'prompt_tokens': 1}, .01, 'development')
        return dict(reward=1., submitted=True, reflection=[], evaluations=[])
    def revise(self, model, budget, feedback, r, reserve, seed):
        budget.charge({'completion_tokens': budget.remaining - reserve, 'prompt_tokens': 1}, .01, 'revision')
        self.last_revision_error = 'scripted unsuccessful revision'
        return False
    def execute(model, benchmark, sandbox, task, files, budget, seed, phase, reserve=0):
        assert budget.remaining - reserve == (2 if execution_tokens is None else execution_tokens)
        budget.charge({'completion_tokens': budget.remaining - reserve, 'prompt_tokens': 1}, .01, phase)
        return dict(task_id=task, reward=1., submitted=True, reflection={}, tools=[])
    monkeypatch.setattr(rung4, 'evaluate_development', evaluate)
    monkeypatch.setattr(rung4.Learner, 'revise', revise)
    monkeypatch.setattr(rung4, 'execute', execute)
    config = dict(dataset='fixture', sandbox_image='fixture', model={'name': 'fixture'},
                  num_agents=1, rounds=1, seed=7, social_info='none', policy='openevolve',
                  tokens_per_round=12, execution_reserve_tokens=2, development_tokens=4,
                  development_cohort_size=2)
    if execution_tokens is not None:
        config['execution_tokens'] = execution_tokens
    rung4.run(config, tmp_path / 'reference')
    checkpoint = json.loads((tmp_path / 'reference/checkpoint.json').read_text())
    scores = [v['development_scores'] for v in checkpoint['learners'][0]['versions'].values()]
    assert len(evaluations) == 1 and scores == [[1.]]
    row = json.loads((tmp_path / 'reference/rounds/0000.json').read_text())['results'][0]
    assert row['budget']['completion'] == (12 if execution_tokens is None else 11) and row['submitted']


def test_matched_growing_uses_fresh_batches_and_deploys_strict_winner(monkeypatch, tmp_path):
    import rung4
    import rung4_benchmark

    class Benchmark:
        def __init__(self, _):
            self.train_ids = ['t0', 't1', 't2', 't3']
            self.holdout_ids = ['h0']
            self.manifest = {'tasks': self.train_ids + self.holdout_ids}
        def public_task(self, task): return {'id': task, 'prompt': f'problem {task}'}

    monkeypatch.setattr(rung4_benchmark, 'LiveCodeBench', Benchmark)
    monkeypatch.setattr(rung4_benchmark, 'Sandbox', lambda _: None)
    monkeypatch.setattr(rung4, 'Model', lambda _: None)
    seen_feedback = []
    revision_number = [0]

    def revise(self, model, budget, feedback, round_index, reserve, seed,
               experience_chars=None):
        seen_feedback.append((len(feedback), experience_chars))
        parent = self.active
        revision_number[0] += 1
        self.active = self.install({'SKILL.md': f'child {revision_number[0]}'},
                                   parent, 'independent', round_index)
        self.last_revision_error = None
        budget.charge({'completion_tokens': 1, 'prompt_tokens': 1}, 0., 'revision')
        return True

    calls = []
    def evaluate(model, benchmark, sandbox, tasks, files, budget, seed, tokens,
                 cascade=False, phase='development'):
        calls.append((list(tasks), files['SKILL.md'], phase))
        # Child 1 wins round 0; child 2 loses round 1, so ties/losses do not drift.
        rewards = {'# Coding procedure\nRead the problem and constraints. Choose an algorithm, implement it in Python,\nand check the provided examples before submitting. Write the solution to solution.py.\n': 0.,
                   'child 1': 1., 'child 2': 0.}
        reward = rewards[files['SKILL.md']]
        budget.charge({'completion_tokens': tokens, 'prompt_tokens': 1}, 0., phase)
        items = [dict(task_id=task, reward=reward, submitted=True,
                      passed_test_fraction_lower_bound=reward, reflection={}) for task in tasks]
        return dict(reward=reward, submitted=True, submitted_rate=1.,
                    passed_test_fraction_lower_bound=reward, evaluated_count=len(tasks),
                    filtered_early=False, reflection=[], evaluations=items)

    monkeypatch.setattr(rung4.Learner, 'revise', revise)
    monkeypatch.setattr(rung4, 'evaluate_development', evaluate)
    config = dict(dataset='fixture', sandbox_image='fixture', model={'name': 'fixture'},
                  num_agents=1, rounds=2, seed=7, social_info='none', policy='openevolve',
                  initial_skill='strong', evolution_design='matched_growing',
                  tokens_per_round=20, development_tokens=4,
                  development_cohort_size=2, execution_reserve_tokens=0,
                  execution_tokens=4, revision_experience_max_chars=5000,
                  holdout_tokens=4, holdout_rounds=[1])
    output = tmp_path / 'growing'
    rung4.run(config, output)

    first = json.loads((output / 'rounds/0000.json').read_text())['results'][0]
    second = json.loads((output / 'rounds/0001.json').read_text())['results'][0]
    assert seen_feedback == [(0, 5000), (1, 5000)]
    assert first['revision_experience_batches'] == 0
    assert second['revision_experience_batches'] == 1
    assert set(first['task_ids']).isdisjoint(second['task_ids'])
    assert calls[0][0] == calls[1][0] and calls[2][0] == calls[3][0]
    assert first['winner'] == 'candidate' and first['paired_revision_uplift'] == 1.
    assert second['winner'] == 'parent' and second['deployed_version'] == first['deployed_version']
    saved = json.loads((output / 'checkpoint.json').read_text())
    assert len(saved['feedback'][0]) == 2
    assert saved['feedback'][0][0]['kind'] == 'matched_fresh_minibatch'


def test_full_openevolve_streaming_selects_from_islands_and_checkpoints_archive(monkeypatch, tmp_path):
    import rung4
    import rung4_benchmark

    class Benchmark:
        def __init__(self, _):
            self.train_ids = ['t0', 't1', 't2', 't3']
            self.holdout_ids = ['h0']
            self.manifest = {'tasks': self.train_ids + self.holdout_ids}
        def public_task(self, task): return {'id': task}

    monkeypatch.setattr(rung4_benchmark, 'LiveCodeBench', Benchmark)
    monkeypatch.setattr(rung4_benchmark, 'Sandbox', lambda _: None)
    monkeypatch.setattr(rung4, 'Model', lambda _: None)
    revision_calls = []

    def revise(self, model, budget, feedback, round_index, reserve, seed, **kwargs):
        revision_calls.append(dict(round=round_index, feedback=len(feedback),
                                   inspirations=len(kwargs['inspirations'])))
        parent = self.active
        self.active = self.install({'SKILL.md': 'full child'}, parent,
                                   'independent', round_index)
        self.last_revision_error = None
        self.last_revision_experience_records = len(feedback)
        self.last_revision_experience_chars = len(json.dumps(feedback))
        budget.charge({'completion_tokens': 1, 'prompt_tokens': 1}, 0., 'revision')
        return True

    def evaluate(model, benchmark, sandbox, tasks, files, budget, seed, tokens,
                 cascade=False, phase='development'):
        reward = 1. if files['SKILL.md'] == 'full child' else .5
        budget.charge({'completion_tokens': tokens, 'prompt_tokens': 1}, 0., phase)
        items = [dict(task_id=task, reward=reward, submitted=True,
                      passed_test_fraction_lower_bound=reward, reflection={}) for task in tasks]
        return dict(reward=reward, submitted=True, submitted_rate=1.,
                    passed_test_fraction_lower_bound=reward, evaluated_count=len(tasks),
                    filtered_early=False, reflection=[], evaluations=items)

    monkeypatch.setattr(rung4.Learner, 'revise', revise)
    monkeypatch.setattr(rung4, 'evaluate_development', evaluate)
    config = dict(dataset='fixture', sandbox_image='fixture', model={'name': 'fixture'},
                  num_agents=1, rounds=2, seed=9, social_info='none',
                  policy='openevolve_full', initial_skill='strong',
                  evolution_design='matched_growing', tokens_per_round=20,
                  development_tokens=4, development_cohort_size=2,
                  execution_reserve_tokens=0, execution_tokens=4,
                  revision_experience_max_chars=5000, holdout_tokens=4,
                  holdout_rounds=[1], openevolve_database=dict(
                      num_islands=2, migration_interval=1, migration_rate=.5))
    output = tmp_path / 'full-oe'
    rung4.run(config, output)

    first = json.loads((output / 'rounds/0000.json').read_text())['results'][0]
    second = json.loads((output / 'rounds/0001.json').read_text())['results'][0]
    saved = json.loads((output / 'checkpoint.json').read_text())
    assert first['candidate_version'] is None  # time-zero initial evaluation
    assert revision_calls == [dict(round=1, feedback=1, inspirations=0)]
    assert second['winner'] == 'openevolve_archive'
    assert second['openevolve_island'] == 1
    assert second['openevolve_population_size'] >= 2
    assert saved['openevolve_archive']['island_generations'] == [0, 1]
    before = saved['openevolve_archive']
    rung4.run(config, output)
    assert json.loads((output / 'checkpoint.json').read_text())['openevolve_archive'] == before


def test_offline_openevolve_reuses_only_its_fixed_training_set(monkeypatch, tmp_path):
    import rung4
    import rung4_benchmark

    class Benchmark:
        def __init__(self, _):
            self.train_ids = [f't{i}' for i in range(6)]
            self.holdout_ids = ['hidden']
            self.manifest = {'tasks': self.train_ids + self.holdout_ids}
        def public_task(self, task): return {'id': task}

    monkeypatch.setattr(rung4_benchmark, 'LiveCodeBench', Benchmark)
    monkeypatch.setattr(rung4_benchmark, 'Sandbox', lambda _: None)
    monkeypatch.setattr(rung4, 'Model', lambda _: None)
    calls = []

    def revise(self, model, budget, feedback, round_index, reserve, seed, **kwargs):
        parent = self.active
        self.active = self.install({'SKILL.md': f'child {round_index}'}, parent,
                                   'independent', round_index)
        self.last_revision_error = None
        self.last_revision_experience_records = len(feedback)
        self.last_revision_experience_chars = len(json.dumps(feedback))
        budget.charge({'completion_tokens': 1, 'prompt_tokens': 1}, 0., 'revision')
        return True

    def evaluate(model, benchmark, sandbox, tasks, files, budget, seed, tokens,
                 cascade=False, phase='development'):
        calls.append(list(tasks))
        budget.charge({'completion_tokens': tokens, 'prompt_tokens': 1}, 0., phase)
        items = [dict(task_id=task, reward=1., submitted=True,
                      passed_test_fraction_lower_bound=1., reflection={}) for task in tasks]
        return dict(reward=1., submitted=True, submitted_rate=1.,
                    passed_test_fraction_lower_bound=1., evaluated_count=len(tasks),
                    filtered_early=False, reflection=[], evaluations=items)

    monkeypatch.setattr(rung4.Learner, 'revise', revise)
    monkeypatch.setattr(rung4, 'evaluate_development', evaluate)
    config = dict(dataset='fixture', sandbox_image='fixture', model={'name': 'fixture'},
                  num_agents=1, rounds=2, seed=4, social_info='none',
                  policy='openevolve_full', initial_skill='minimal',
                  evolution_design='offline_openevolve', offline_training_size=3,
                  tokens_per_round=20, development_tokens=4,
                  development_cohort_size=3, execution_reserve_tokens=0,
                  execution_tokens=4, revision_experience_max_chars=5000,
                  holdout_tokens=4, holdout_rounds=[1],
                  openevolve_database={'num_islands': 2})
    output = tmp_path / 'offline'
    rung4.run(config, output)

    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['online_ids'] == [] and len(provenance['development_ids']) == 3
    # Initial fitness is computed once; subsequent generations evaluate only
    # the new candidate because parent fitness remains in the archive.
    assert len(calls) == 2 and all(tasks == calls[0] for tasks in calls)
    assert 'hidden' not in calls[0]
    for r in range(2):
        row = json.loads((output / 'rounds' / f'{r:04d}.json').read_text())
        assert row['results'][0]['task_ids'] == calls[0]
        assert json.loads((output / 'checkpoint.json').read_text())['feedback'][0][r][
            'kind'] == 'fixed_offline_training_set'

    def forbidden_evaluation(*args, **kwargs):
        pytest.fail('Frozen offline control must go directly to hidden replay')
    monkeypatch.setattr(rung4, 'evaluate_development', forbidden_evaluation)
    frozen = tmp_path / 'offline-frozen'
    rung4.run(config | {'policy': 'frozen', 'rounds': 1, 'holdout_rounds': [0]}, frozen)
    row = json.loads((frozen / 'rounds/0000.json').read_text())['results'][0]
    assert row['offline_training_evaluation_skipped']
    assert row['version'] == rung4.digest({'SKILL.md': rung4.MINIMAL_SKILL})[:24]
