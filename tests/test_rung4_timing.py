"""Timing interventions: schedule, reuse and hard-budget invariants (Slurm only)."""
import copy
import json
import random

import pytest

import rung4
from rung4_population import (TIMING_CONDITIONS, timing_spec, observation_access,
                              prepare, prepare_timing, effective_cost_limit)


@pytest.mark.parametrize('condition', TIMING_CONDITIONS)
def test_full_schedule_dose_windows_and_rng(condition):
    before = random.getstate()
    for seed in range(6):
        spec = timing_spec(condition, seed, 5)
        assert spec == timing_spec(condition, seed, 5)
        assert spec == json.loads(json.dumps(spec))
        for agent in range(5):
            access = [r for r in range(50) if observation_access({'observation_timing': spec}, agent, r)[0]]
            forced = [r for r in range(50) if observation_access({'observation_timing': spec}, agent, r)[1]]
            assert 0 not in access
            if condition.startswith('forced'):
                assert access == forced == spec['schedules'][str(agent)]
                assert len(access) == len(set(access)) == 4
            else:
                assert len(access) == 10 and not forced
            assert all(any(lo <= r <= hi for lo, hi in spec['intervals']) for r in access)
            if condition == 'forced_distributed':
                assert all(sum(lo <= r <= hi for r in access) == 1 for lo, hi in spec['intervals'])
    assert random.getstate() == before


@pytest.mark.parametrize('full,expected', [(False, 36), (True, 84)])
def test_prepare_reuses_controls_and_allocates_only_new_work(tmp_path, full, expected):
    old = tmp_path / 'old'
    prepare(old)
    allocation = json.loads((old / 'allocation.json').read_text())
    for slot in allocation['slots']:
        cfg = slot['config']
        if cfg['smoke'] or cfg['condition'] not in {'llm_solo', 'llm_social_payoff'}:
            continue
        directory = old / slot['label']
        rung4.atomic_json(directory / 'completed.json', dict(rounds=50, snapshots=[0, 10, 25, 49]))
        rung4.atomic_json(directory / 'provenance.json', dict(config=cfg, source_sha256={}))
        for r in cfg['snapshots']:
            for a in range(5):
                rung4.atomic_json(directory / 'test_summary' / f'{r:04d}-{a}.json', dict(tasks=200))
    new = tmp_path / 'new'
    cap = 1100 if full else 400
    prepare_timing(new, old, full=full, cap=cap)
    result = json.loads((new / 'allocation.json').read_text())
    assert len(result['reused_controls']) == 24
    assert sum(not s['config']['smoke'] for s in result['slots']) == expected
    assert result['expected_smoke_conditions'] == (14 if full else 6)
    assert sum(s['config']['model']['total_cost_limit_usd'] * 5 for s in result['slots']) == pytest.approx(cap)
    assert {s['config']['condition'] for s in result['slots']} == (set(TIMING_CONDITIONS) if full else
        {'forced_early', 'forced_distributed', 'optional_early'})
    for i, slot in enumerate(result['slots']):
        cfg = slot['config']
        assert effective_cost_limit(result, i) == cfg['model']['total_cost_limit_usd']
        template = next(s['config'] for s in allocation['slots'] if not s['config']['smoke']
            and s['config']['condition'] == 'llm_social_payoff' and s['config']['model']['name'] == cfg['model']['name']
            and s['config']['seed'] == cfg['seed'])
        for key in template:
            if key not in {'condition', 'smoke', 'rounds', 'snapshots', 'model'}:
                assert cfg[key] == template[key]
        old_model = {k: v for k, v in template['model'].items() if k not in {'run_cost_limit_usd', 'total_cost_limit_usd'}}
        assert {k: v for k, v in cfg['model'].items() if k in old_model} == old_model
    prepare_timing(new, old, full=full, cap=cap)
    with pytest.raises(ValueError, match='Refusing'):
        prepare_timing(new, old, full=full, cap=cap + 1)
    changed = copy.deepcopy(result)
    changed['budget_overrides'] = {'0': 10000}
    with pytest.raises(ValueError, match='hard spending'):
        effective_cost_limit(changed, 0)
    assert json.loads((old / 'allocation.json').read_text()) == allocation
