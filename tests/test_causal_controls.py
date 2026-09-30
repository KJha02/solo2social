"""Mechanism checks: charged search, protected pull phase, and matched assignment."""
import json
import re
from pathlib import Path

import pytest

import main
import rung2
from agent import Generation, independent_search_assigned, parse_action


class Sink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)

    def flush(self):
        pass


@pytest.mark.parametrize("rung", [1, 2])
@pytest.mark.parametrize("search", [False, True])
def test_search_and_reserve_conserve_budget(monkeypatch, tmp_path, rung, search):
    root = Path(__file__).resolve().parents[1]
    filename = "rung1_solo_llm.json" if rung == 1 else "rung2_unstructured_solo_llm.json"
    config = json.loads((root / "configs" / filename).read_text())
    config["environment"].update(num_agents=2, rounds=1)
    config["budget"].update(tokens_per_round=10, max_tokens_per_decision=10, carry_over=False)
    config["interventions"] = {"independent_search_probability": float(search), "execution_reserve_tokens": 4}

    class Engine:
        def __init__(self, _):
            pass

        def generate(self, requests):
            result = []
            for request in requests:
                state = request["state"]
                if "Independent-search intervention" in state:
                    text = "FINAL: PULL 0" if rung == 1 else "FINAL: INNOVATE"
                elif rung == 2 and "none; you cannot PULL" in state:
                    text = "FINAL: INNOVATE"
                elif "Protected execution phase" in state:
                    arm = re.search(r"arm (\d+): probes=", state)
                    text = f"FINAL: PULL {arm.group(1) if arm else 0}"
                else:
                    # Invalid actions still consume the pre-pull allocation.
                    text = "FINAL: OBSERVE 999"
                count = request["max_tokens"]
                result.append(Generation(text, text, count, "stop", parse_action(text), count, count, 10, False))
            return result

    module = main if rung == 1 else rung2
    monkeypatch.setattr(module, "VLLMDecisionEngine", Engine)
    sink = Sink()
    kwargs = dict(config=config, seed=2, env=main.make_environment(config, 2), writer=sink, checkpoint=None)
    if rung == 1:
        main.run_llm_condition(**kwargs, run_dir=tmp_path)
    else:
        rung2.run_condition(**kwargs, checkpoint_writer=lambda _: None)
    ends = [e for e in sink.events if e["event"] == "round_end"]
    assert all(e["pulled"] for e in ends)
    assert all(e["tokens_spent"] + e["unused_tokens_end_of_round"] == 10 for e in ends)
    assert all(e["independent_search_completed"] == search for e in ends)
    if not (rung == 1 and search):
        assert all(e["tokens_spent"] == 10 for e in ends)
        assert sum(e.get("protected_pull_phase", False) for e in sink.events) == 2


def test_assignment_is_matched_across_social_conditions():
    config = {"budget": {"tokens_per_round": 10}, "interventions": {"independent_search_probability": .25}}
    solo = [independent_search_assigned(config | {"condition": "solo"}, 7, t, i) for t in range(20) for i in range(10)]
    social = [independent_search_assigned(config | {"condition": "social"}, 7, t, i) for t in range(20) for i in range(10)]
    assert solo == social
    assert 0 < sum(solo) < len(solo)


def test_efficiency_batch_dedup_and_target_censoring(tmp_path):
    import pandas as pd
    from analyze_efficiency import read_population, summarize
    directory = tmp_path / 'seed_0'
    directory.mkdir()
    config = {'seed': 0, 'rung': 1, 'policy': 'solo_llm', 'model_label': 'test',
              'environment': {'num_agents': 2}, 'budget': {'tokens_per_round': 10}}
    (directory / 'config.json').write_text(json.dumps(config))
    events = []
    for r in range(2):
        for i in range(2):
            events.extend([
                dict(event='decision', round=r, agent_id=i, wave=0, completion_tokens=3,
                     prompt_tokens=5, batch_generation_seconds=2.),
                dict(event='round_end', round=r, agent_id=i, tokens_spent=3, reward=.5, pulled=True)])
    (directory / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    metadata, rows = read_population(directory)
    assert [r['seconds'] for r in rows] == [2., 2.]  # not 4 seconds per batch
    frame = pd.DataFrame([dict(**metadata, **r) for r in rows])
    curves, targets = summarize(frame, target=.8, window=1)
    assert curves.iloc[-1].cumulative_completion_tokens == 12
    assert curves.iloc[-1].cumulative_total_tokens == 32
    assert bool(targets.iloc[0].censored)
    assert targets.iloc[0].observed_seconds == 4
    assert pd.isna(targets.iloc[0].to_target_seconds)
    _, reached = summarize(frame, target=.4, window=2)
    assert bool(reached.iloc[0].reached) and reached.iloc[0].stop_round == 2

    # Actual rung4 schema has sequential agent-round timing, including tools.
    (directory / 'events.jsonl').unlink()
    (directory / 'rounds').mkdir()
    config.update(rung=4, num_agents=2, policy='endogenous')
    (directory / 'config.json').write_text(json.dumps(config))
    values = [dict(agent=i, round=0, reward=1, submitted=True,
                   budget={'completion': 4, 'prompt': 7}, wall_seconds_round=3.) for i in range(2)]
    (directory / 'rounds' / '0000.json').write_text(json.dumps({'results': values, 'events': []}))
    _, rows = read_population(directory)
    assert rows[0]['seconds'] == 6 and rows[0]['completion_tokens'] == 8
