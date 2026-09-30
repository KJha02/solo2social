import numpy as np
import pytest

from social_baseline import (
    DiscountedPolicy,
    HierarchicalSocialUCB,
    simulate,
    simulate_hierarchical_ucb,
)


def policy(mode="social_payoff"):
    return DiscountedPolicy(0, 3, 10, 25, mode, 1.)


def test_observation_provenance_and_duplicate_samples():
    p = policy()
    p.acquire(2, "independent")
    p.acquire(2, "social")
    p.acquire(4, "social")
    p.acquire(4, "independent")
    assert p.origins == {2: "independent", 4: "social"}
    p.update(4, 5., 2, 1)
    p.update(4, 5., 2, 1)
    assert p.estimates[4] == (5., 1., 2)
    p.update(4, 5.1, 8, 0)
    p.update(4, 4.9, 3, 2)
    assert p.estimates[4][2] == 8  # old social data cannot refresh evidence


def test_decay_and_shock_response_without_shock_schedule():
    p = policy()
    p.acquire(1, "independent")
    p.acquire(2, "independent")
    p.update(1, 10., 0, 0)
    p.update(2, 8., 30, 0)
    assert p.best(30) == 1  # discount toward own mean, not zero
    p.update(1, -3., 31, 0)
    assert p.needs_check
    assert p.best(31) == 2
    assert p.learning_action(99) == "observe"


def test_hierarchical_ucb_self_prior_and_two_levels():
    p = HierarchicalSocialUCB(0, 3, 10, 25, 1., 1.)
    assert p.choose_source(0, range(10)) == 0
    p.update_source(0, -10.)
    assert p.choose_source(1, range(10)) != 0
    first = p.choose_arm(0)
    p.update_arm(first, 3.)
    assert p.arm_counts[first] == 1
    assert p.source_counts.sum() == 1


@pytest.mark.parametrize("rung", [1, 2])
@pytest.mark.parametrize("mode", ["solo", "social_action", "social_payoff"])
def test_population_mechanics_and_reproducibility(rung, mode):
    environment = dict(num_agents=10, rounds=30, arm_mean_scale=1., reward_noise_std=1., regime_period=10)
    environment.update(dict(num_arms=25) if rung == 1 else
                       dict(landscape="structured", coordinate_innovation=True, reference_grid_size=1001))
    config = dict(rung=rung, environment=environment)
    result, events = simulate(config, 2, mode)
    repeated, other = simulate(config, 2, mode)
    assert events.equals(other)
    assert len(events) == 300
    assert not events.duplicated(["round", "agent_id"]).any()
    assert np.isclose(result["reward"], events.reward.mean())
    assert result["completion"] == 1.
    assert (events.loc[events.observed == 1, "source_round"] < events.loc[events.observed == 1, "round"]).all()
    assert events.groupby(["agent_id", "arm_id"]).origin.nunique().max() == 1
    if mode == "solo":
        assert result["observation_rate"] == result["social_first_rate"] == 0
    if rung == 1:
        assert events.arm_id.between(0, 24).all()
    else:
        assert result["innovations"] > 0
    assert events.loc[events["round"] == 0, "arm_id"].nunique() > 1


def test_hierarchical_population_mechanics_and_reproducibility():
    mode = "hierarchical_payoff"
    config = dict(rung=1, environment=dict(
        num_agents=10, num_arms=25, rounds=30, arm_mean_scale=1.,
        reward_noise_std=1., regime_period=10,
    ))
    result, events = simulate_hierarchical_ucb(config, 2, mode)
    repeated, other = simulate_hierarchical_ucb(config, 2, mode)
    assert events.equals(other)
    assert np.isclose(result["reward"], repeated["reward"])
    assert len(events) == 300
    assert not events.duplicated(["round", "agent_id"]).any()
    assert np.isclose(result["reward"], events.reward.mean())
    assert result["completion"] == 1.
    observed = events[events.observed == 1]
    assert len(observed) > 0
    assert (observed.source_round < observed["round"]).all()
    assert (events.loc[events.action == "self", "source_credit"] ==
            events.loc[events.action == "self", "reward"]).all()
    assert (events.loc[events.action == "self", "source_id"] ==
            events.loc[events.action == "self", "agent_id"]).all()
    assert (observed.source_id != observed.agent_id).all()
    assert observed.observed_payoff.notna().all()
    assert (observed.source_credit == observed.observed_payoff).all()
    # Observation supplies evidence; it does not force copying that arm.
    assert (observed.arm_id != observed.recommended_arm).any()
    # At every disclosed regime start, only self has a current-regime action.
    assert (events.loc[events["round"].isin([0, 10, 20]), "action"] == "self").all()
