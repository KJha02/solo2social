"""Scientific checks for copy quality against independent acquisition history."""
import math

from analysis import independent_copy_comparisons


def decision(action, arm=None, *, rung=1, copy=False, round=0, valid=True):
    event = dict(event="decision", agent_id=0, parsed_action=action,
                 valid=valid, rung=rung, round=round, copy_any=copy)
    if action == "observe":
        event["observation"] = {"arm_id": arm} if arm is not None else None
    else:
        event["arm_id"] = arm
    return event


def test_social_trials_never_become_independent_and_future_discovery_is_excluded():
    events = [decision("observe", 1), decision("pull", 1, copy=True),
              decision("pull", 2, valid=False), decision("pull", 1, copy=True),
              decision("pull", 2), decision("pull", 1, copy=True)]
    rows = list(independent_copy_comparisons(events, lambda t, arm: {1: 3, 2: 2}[arm]))
    assert [r["has_comparator"] for r in rows] == [False, False, True]
    assert math.isnan(rows[0]["better"])
    assert rows[2]["better"] and rows[2]["advantage"] == 1


def test_true_frontier_survives_observation_and_revalues_after_shocks():
    events = [decision("pull", 1), decision("pull", 2),
              decision("observe", 1), decision("observe", 3),
              decision("pull", 3, copy=True),
              decision("pull", 3, copy=True, round=1),
              decision("pull", 1, copy=True, round=1)]
    values = [{1: 2, 2: 5, 3: 4}, {1: 6, 2: 1, 3: 4}]
    rows = list(independent_copy_comparisons(events, lambda t, arm: values[t][arm]))
    assert [r["advantage"] for r in rows] == [-1, -2, 0]
    assert not any(r["better"] for r in rows)
    assert [r["tie"] for r in rows] == [False, False, True]
    assert [r["worse"] for r in rows] == [True, True, False]
    assert [r["social_first"] for r in rows] == [True, True, False]
    assert [r["independent_first"] for r in rows] == [False, False, True]


def test_rung2_unpulled_probe_counts_but_reprobing_social_arm_does_not():
    events = [decision("observe", 1, rung=2), decision("innovate", 1, rung=2),
              decision("pull", 1, rung=2, copy=True),
              decision("innovate", 2, rung=2), decision("pull", 1, rung=2, copy=True)]
    rows = list(independent_copy_comparisons(events, lambda t, arm: {1: 3, 2: 4}[arm]))
    assert [r["has_comparator"] for r in rows] == [False, True]
    assert rows[1]["advantage"] == -1
    assert all(r["social_first"] for r in rows)


def test_disjoint_first_acquisition_counts_all_pulls_including_later_observation():
    events = [decision("pull", 1), decision("observe", 1),
              decision("pull", 1, copy=True), decision("observe", 2),
              decision("pull", 2, copy=True), decision("pull", 2, copy=True)]
    rows = list(independent_copy_comparisons(events, lambda t, arm: arm, include_all_pulls=True))
    assert [r["social_first"] for r in rows] == [False, False, True, True]
    assert [r["independent_first"] for r in rows] == [True, True, False, False]
    assert sum(r["legacy_copy"] for r in rows) == 3
    assert sum(r["social_first"] for r in rows) == 2
    assert all(r["better"] and not r["tie"] for r in rows if r["social_first"])
