import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from agent import Generation, parse_action
import rung3

ROOT = Path(__file__).resolve().parents[1]


def config(enabled=True, social=False, rounds=2):
    value = json.loads((ROOT / 'configs/rung3_solo.json').read_text())
    value.update(independent_skill_acquisition=enabled,
                 prompt='prompts/v2_rung3_policy.txt', social_info='full' if social else 'none')
    value['environment'].update(num_agents=10, rounds=rounds)
    value['budget'] = {'tokens_per_round': 10, 'carry_over': False}
    return value


class ScriptedEngine:
    def __init__(self, _):
        pass

    def generate(self, requests):
        result = []
        for request in requests:
            text = ('FINAL: SCHEDULE 0' if request['state'].startswith('Apply the selected skill')
                    else 'FINAL: PULL 7')
            result.append(Generation(text=text, final_text=text, token_count=2,
                                    finish_reason='stop', action=parse_action(text),
                                    requested_max_tokens=request['max_tokens'],
                                    effective_max_tokens=request['max_tokens'],
                                    prompt_tokens=10, context_limited=False))
        return result


class Writer:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)

    def flush(self):
        pass


def run(c, checkpoint=None):
    writer = Writer()
    checkpoints = []
    summary = rung3.run_condition(config=c, seed=0, env=rung3.make_environment(c, 0),
                                 writer=writer, checkpoint=checkpoint,
                                 checkpoint_writer=lambda state: checkpoints.append(copy.deepcopy(state)))
    return summary, writer.events, checkpoints


@pytest.mark.parametrize('social', [False, True])
def test_unseen_pull_acquires_and_executes_with_same_budget(monkeypatch, social):
    monkeypatch.setattr(rung3, 'VLLMDecisionEngine', ScriptedEngine)
    summary, events, checkpoints = run(config(social=social))
    assert summary['total_independent_acquisitions'] == 9
    assert summary['total_copies'] == 0
    assert summary['missed_pulls'] == 0
    ends = [e for e in events if e['event'] == 'round_end']
    assert len(ends) == 20
    assert sum(e['independently_acquired_this_pull'] for e in ends) == 9
    assert all(e['skill_id'] == 7 and e['valid_solution'] for e in ends)
    assert all(e['closing_tokens'] == 0 for e in ends)
    assert all(e['tokens_explore'] + e['tokens_exploit'] == 4 for e in ends)
    assert len(checkpoints[-1]['agents'][0]['owned_skill_ids']) in (1, 2)


def test_old_mode_still_rejects_unowned_pulls(monkeypatch):
    monkeypatch.setattr(rung3, 'VLLMDecisionEngine', ScriptedEngine)
    summary, _, _ = run(config(enabled=False, rounds=1))
    assert summary['total_independent_acquisitions'] == 0
    assert summary['missed_pulls'] == 9


def test_acquisition_resume_matches_uninterrupted(monkeypatch):
    monkeypatch.setattr(rung3, 'VLLMDecisionEngine', ScriptedEngine)
    full, _, full_checkpoints = run(config())
    _, _, first = run(config(rounds=1))
    resumed, _, final = run(config(), first[-1])
    resumed.pop('inference_seconds_by_stage')
    full.pop('inference_seconds_by_stage')
    assert resumed == full
    assert final[-1]['agents'] == full_checkpoints[-1]['agents']
    with pytest.raises(ValueError, match='acquisition mode'):
        run(config(enabled=False), first[-1])


@pytest.mark.parametrize('social', [False, True])
def test_deoe_execution_budget_provenance_and_resume(monkeypatch, social):
    class ExecutorOnly(ScriptedEngine):
        def generate(self, requests):
            assert all(r['state'].startswith('Apply the selected skill') for r in requests)
            assert all(r['max_tokens'] == 10 for r in requests)
            return super().generate(requests)

    monkeypatch.setattr(rung3, 'VLLMDecisionEngine', ExecutorOnly)
    c = config(social=social, rounds=30)
    c['policy'] = 'social_deoe' if social else 'solo_deoe'
    summary, events, checkpoints = run(c)
    assert summary['total_independent_acquisitions'] > 0
    decisions = [e for e in events if e['event'] == 'decision']
    assert sum(e['parsed_action'] == 'pull' for e in decisions) == 300
    assert all(e['completion_tokens'] == 0 for e in decisions)
    observations = [e for e in decisions if e['observation'] is not None]
    assert bool(observations) == social
    assert all(e['observation']['source_round'] < e['round'] for e in observations)
    assert all(e['tokens_spent'] == 2 for e in events if e['event'] == 'round_end')
    # Resume from serialized state, including policy estimates and deduplication.
    resumed, _, final = run(c, json.loads(json.dumps(checkpoints[12])))
    resumed.pop('inference_seconds_by_stage')
    summary.pop('inference_seconds_by_stage')
    assert resumed == summary
    assert json.dumps(final[-1]['agents'], sort_keys=True) == json.dumps(checkpoints[-1]['agents'], sort_keys=True)
    assert final[-1]['deoe'] == checkpoints[-1]['deoe']


def test_protected_execution_fallback_preserves_fresh_search(monkeypatch):
    class ExhaustSelector(ScriptedEngine):
        def generate(self, requests):
            if requests[0]['state'].startswith('Apply the selected skill'):
                assert all(r['max_tokens'] == 4 for r in requests)
                return super().generate(requests)
            assert all(r['max_tokens'] == 6 for r in requests)
            assert all('requires independent search' in r['state'] for r in requests)
            return [Generation(text='unfinished', final_text='', token_count=6,
                               action=None, finish_reason='length', requested_max_tokens=6,
                               effective_max_tokens=6, prompt_tokens=10,
                               context_limited=False) for _ in requests]

    monkeypatch.setattr(rung3, 'VLLMDecisionEngine', ExhaustSelector)
    c = config(rounds=1)
    c['interventions'] = dict(independent_search_probability=1, execution_reserve_tokens=4)
    summary, events, _ = run(c)
    ends = [e for e in events if e['event'] == 'round_end']
    assert summary['total_independent_acquisitions'] == 10
    assert all(e['selection_fallback'] and e['independent_search_completed'] for e in ends)
    assert all(e['valid_solution'] and e['tokens_spent'] == 8 for e in ends)


@pytest.mark.parametrize('zero_execution', [False, True])
def test_fixed_executor_replays_real_selections_only(monkeypatch, tmp_path, zero_execution):
    class ExhaustOnSelection(ScriptedEngine):
        def generate(self, requests):
            return [replace(generation, token_count=10)
                    for generation in super().generate(requests)]

    monkeypatch.setattr(rung3, 'VLLMDecisionEngine',
                        ExhaustOnSelection if zero_execution else ScriptedEngine)
    original = config(enabled=zero_execution, rounds=1)
    summary, events, _ = run(original)
    (tmp_path / 'config.json').write_text(json.dumps(original))
    (tmp_path / 'summary.json').write_text(json.dumps(dict(completed=True, **summary)))
    (tmp_path / 'environment.json').write_text(json.dumps(rung3.make_environment(original, 0).description()))
    source = tmp_path / 'events.jsonl'
    source.write_text('\n'.join(json.dumps(e) for e in events))
    c = copy.deepcopy(original)
    c.update(policy='skill_replay', source_run=str(tmp_path))
    execution_seeds = []

    class ExecutorOnly(ScriptedEngine):
        def generate(self, requests):
            execution_seeds.extend(r['seed'] for r in requests)
            assert all(r['state'].startswith('Apply the selected skill') and r['max_tokens'] == 10
                       for r in requests)
            return super().generate(requests)

    monkeypatch.setattr(rung3, 'VLLMDecisionEngine', ExecutorOnly)
    replay, replay_events, checkpoints = run(c)
    assert replay['recorded_selections'] == (10 if zero_execution else 1)
    assert replay['missed_pulls'] == (0 if zero_execution else 9)
    assert replay['invalid_solutions'] == 0
    assert not any(e['event'] == 'decision' for e in replay_events)
    originals = {(e['round'], e['agent_id']): e for e in events if e['event'] == 'solution'}
    if not zero_execution:
        for e in replay_events:
            if e['event'] == 'solution':
                prior = originals[e['round'], e['agent_id']]
                assert e['executor_seed'] == prior['executor_seed']
                assert e['executor_input_sha256'] == prior['executor_input_sha256']
    if zero_execution:
        assert summary['invalid_solutions'] == 10
        assert all(e['source_execution_tokens'] == 0 for e in replay_events if e['event'] == 'solution')
    first_seeds = execution_seeds[:]
    execution_seeds.clear()
    replicate = dict(c, execution_replicate=1)
    run(replicate)
    assert len(first_seeds) == len(execution_seeds)
    assert all(a != b for a, b in zip(first_seeds, execution_seeds))
    with pytest.raises(ValueError, match='execution replicate'):
        run(replicate, checkpoints[-1])
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='replay source'):
        run(c, checkpoints[-1])


def test_provenance_and_hidden_skill_information():
    agent = rung3.Rung3Agent(0, 0)
    agent.acquire_independently(7, 2, 8)
    restored = rung3.Rung3Agent.from_state_dict(agent.state_dict())
    assert restored.acquisition_source(7) == 'independent'
    assert restored.acquisition_source(0) == 'initial'
    assert not restored.adopt(7, 3)[0]
    for invalid in (-1, 8):
        with pytest.raises(ValueError):
            restored.acquire_independently(invalid, 3, 8)
    observer = rung3.Rung3Agent(0, 0)
    observer.observations.append(rung3.SkillObservation('x', 0, 1, 0, 7))
    with pytest.raises(ValueError, match='socially observed'):
        observer.acquire_independently(7, 1, 8)
    assert observer.adopt(7, 1)[0]
    assert observer.acquisition_source(7) == 'social'
    state = rung3.Rung3Agent(0, 0).render_state(round_index=0, num_agents=10,
                skills=rung3.load_skills(), social_enabled=False, independent_acquisition=True)
    assert '1, 2, 3, 4, 5, 6, 7' in state
    assert rung3.load_skills()[7].body not in state


def test_prompt_parity_and_legacy_text():
    solo = rung3.policy_prompt(config())
    social = rung3.policy_prompt(config(social=True))
    shared = solo.split('The fixed skill menu')[1].split('Choose exactly')[0]
    assert shared in social
    assert 'You may only use your owned skill' not in solo
    assert 'You may only use your owned skill' in rung3.policy_prompt(config(enabled=False))
