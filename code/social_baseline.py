"""Fast homogeneous explore/observe/exploit populations; see SOCIAL_BASELINE.md.

Run with Slurm: python social_baseline.py --output-root runs/v3/social_baseline
Policies receive only their own noisy samples and explicitly requested observations.
The existing environments supply the same seeded landscapes and potential rewards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from main import make_environment

VERSION = "discounted_eoe_hierarchical_ucb_v2"
DECAY = 0.95
CHECK_INTERVAL = 20
SOURCE_PRIOR_STRENGTH = 1.0
SELF_PRIOR_ADVANTAGE_NOISE_SD = 0.5
SOCIAL_SAMPLE_WEIGHT = 1.0
ROOT = Path(__file__).resolve().parents[1]


class DiscountedPolicy:
    def __init__(self, agent_id, seed, num_agents, num_arms, mode, noise_std):
        self.agent_id, self.seed = agent_id, seed
        self.num_agents, self.num_arms, self.mode = num_agents, num_arms, mode
        self.noise_std = noise_std
        self.estimates = {}  # arm -> (weighted sum, weighted count, last sample round)
        self.origins = {}
        self.seen_samples = set()
        self.last_observe = -CHECK_INTERVAL
        self.needs_check = False
        self.innovation_index = 0

    def rng(self, round_index, stream):
        return np.random.default_rng(np.random.SeedSequence(
            [self.seed, 971, self.agent_id, round_index, stream]))

    def acquire(self, arm, origin):
        self.origins.setdefault(int(arm), origin)

    def update(self, arm, reward, sample_round, source_id):
        key = (source_id, sample_round, int(arm))
        if key in self.seen_samples:
            return
        self.seen_samples.add(key)
        old = self.estimates.get(arm)
        if old is None:
            self.estimates[arm] = (float(reward), 1., sample_round)
            return
        total, count, last = old
        if sample_round < last:
            # Add an old observation with its actual age, never refresh its date.
            weight = DECAY ** (last - sample_round)
            self.estimates[arm] = (total + weight * reward, count + weight, last)
            return
        surprise = abs(reward - total / count) > 3 * self.noise_std * math.sqrt(1 + 1 / count)
        if surprise:
            self.estimates[arm] = (float(reward), 1., sample_round)
            self.needs_check = True
        else:
            weight = DECAY ** (sample_round - last)
            self.estimates[arm] = (total * weight + reward, count * weight + 1, sample_round)

    def best(self, round_index):
        unknown = sorted(set(self.origins) - set(self.estimates))
        if unknown:
            return int(self.rng(round_index, 4).choice(unknown))
        prior = np.mean([v[0] / v[1] for v in self.estimates.values()])
        scores = {arm: prior + DECAY ** (round_index - last) * (total / count - prior)
                  for arm, (total, count, last) in self.estimates.items()}
        maximum = max(scores.values())
        return int(self.rng(round_index, 4).choice(sorted(a for a, v in scores.items() if v == maximum)))

    def learning_action(self, round_index):
        probability = 1 / math.sqrt(round_index + 1)
        if not self.origins or self.rng(round_index, 0).random() < probability:
            if self.num_arms is None or len(self.origins) < self.num_arms:
                return "innovate"
        if self.mode != "solo" and (
            self.needs_check or round_index - self.last_observe >= CHECK_INTERVAL
            or self.rng(round_index, 1).random() < probability
        ):
            return "observe"
        return "exploit"

    def new_option(self, round_index):
        if self.num_arms is not None:
            return int(self.rng(round_index, 2).choice(sorted(set(range(self.num_arms)) - set(self.origins))))
        return float(self.rng(round_index, 2).random())

    def target(self, round_index):
        peers = [i for i in range(self.num_agents) if i != self.agent_id]
        return int(self.rng(round_index, 3).choice(peers))


class HierarchicalSocialUCB:
    """UCB over information sources with a lower UCB over one's own arms.

    A peer supplies its most recent arm and payoff, which update the peer's
    outer index and the lower arm UCB. The lower UCB still chooses execution;
    peer selection acquires evidence rather than forcing imitation. Selecting
    self obtains its source sample from the lower UCB's realized pull.
    """

    def __init__(self, agent_id, seed, num_agents, num_arms, mean_scale, noise_std):
        self.agent_id, self.seed = agent_id, seed
        self.num_agents, self.num_arms = num_agents, num_arms
        self.mean_scale, self.noise_std = mean_scale, noise_std
        self.origins = {}
        self.reset_values()

    def reset_values(self):
        self.arm_counts = np.zeros(self.num_arms, dtype=np.float64)
        self.arm_reward_sums = np.zeros(self.num_arms, dtype=np.float64)
        self.source_counts = np.zeros(self.num_agents, dtype=np.float64)
        self.source_reward_sums = np.zeros(self.num_agents, dtype=np.float64)

    def rng(self, round_index, stream):
        return np.random.default_rng(np.random.SeedSequence(
            [self.seed, 1971, self.agent_id, round_index, stream]))

    def acquire(self, arm, origin):
        self.origins.setdefault(int(arm), origin)

    def choose_source(self, round_index, available):
        available = np.asarray(sorted(available), dtype=np.int64)
        prior_means = np.full(len(available), self.mean_scale, dtype=np.float64)
        prior_means[available == self.agent_id] += (
            SELF_PRIOR_ADVANTAGE_NOISE_SD * self.noise_std
        )
        counts = self.source_counts[available]
        denominators = counts + SOURCE_PRIOR_STRENGTH
        means = (
            self.source_reward_sums[available]
            + SOURCE_PRIOR_STRENGTH * prior_means
        ) / denominators
        total = float(self.source_counts.sum() + SOURCE_PRIOR_STRENGTH * self.num_agents)
        indices = means + np.sqrt(2.0 * np.log(max(total, 2.0)) / denominators)
        maxima = available[np.isclose(indices, np.max(indices))]
        return int(self.rng(round_index, 0).choice(maxima))

    def choose_arm(self, round_index):
        unseen = np.flatnonzero(self.arm_counts == 0)
        if len(unseen):
            return int(self.rng(round_index, 1).choice(unseen))
        means = self.arm_reward_sums / self.arm_counts
        total = max(float(self.arm_counts.sum()), 2.0)
        indices = means + np.sqrt(2.0 * np.log(total) / self.arm_counts)
        maxima = np.flatnonzero(np.isclose(indices, np.max(indices)))
        return int(self.rng(round_index, 1).choice(maxima))

    def update_arm(self, arm, reward, weight=1.0):
        self.arm_counts[int(arm)] += weight
        self.arm_reward_sums[int(arm)] += weight * float(reward)

    def update_source(self, source_id, reward):
        self.source_counts[int(source_id)] += 1.0
        self.source_reward_sums[int(source_id)] += float(reward)


def simulate(config, seed, mode):
    started = time.perf_counter()
    env = make_environment(config, seed)
    rung = int(config.get("rung", 1))
    rounds = config["environment"]["rounds"]
    agents = [DiscountedPolicy(i, seed, env.num_agents, env.num_arms if rung == 1 else None,
                              mode, env.reward_noise_std) for i in range(env.num_agents)]
    events = []
    independent = set()
    for t in range(rounds):
        env.begin_round(t)
        snapshot = env.snapshot(t)
        for agent in agents:
            action = agent.learning_action(t)
            observed = innovations = 0
            target = source_round = None
            forced_arm = None
            if action == "innovate":
                option = agent.new_option(t)
                if rung == 1:
                    forced_arm = option
                    agent.acquire(option, "independent")
                    independent.add(option)
                else:
                    probe = env.innovate(agent_id=agent.agent_id, innovation_index=agent.innovation_index,
                                         round_index=t, coordinate=option if env.coordinate_innovation else None)
                    agent.innovation_index += 1
                    agent.acquire(probe.arm_id, "independent")
                    # Probe and pull are different noisy samples even in the same round.
                    agent.update(probe.arm_id, probe.reward, t, -1-agent.agent_id)
                    independent.add(probe.arm_id)
                    innovations = 1
            elif action == "observe":
                target = agent.target(t)
                observation = env.observe(snapshot=snapshot, observer_id=agent.agent_id, target_id=target,
                                          round_index=t, reveal_payoff=mode == "social_payoff")
                agent.last_observe, agent.needs_check = t, False
                if observation is not None:
                    assert observation.source_round < t
                    observed = 1
                    source_round = observation.source_round
                    agent.acquire(observation.arm_id, "social")
                    if observation.reward is not None:
                        agent.update(observation.arm_id, observation.reward, source_round, target)
            arm = forced_arm if forced_arm is not None else agent.best(t)
            assert arm in agent.origins
            private = [a for a, origin in agent.origins.items() if origin == "independent"]
            mean = float(env.arm_means[arm]) if rung == 1 else env.arm_mean(arm)
            private_best = max((float(env.arm_means[a]) if rung == 1 else env.arm_mean(a)
                                for a in private), default=None)
            pull = env.pull(agent_id=agent.agent_id, arm_id=arm, round_index=t)
            agent.update(arm, pull.reward, t, agent.agent_id)
            events.append(dict(round=t, agent_id=agent.agent_id, reward=pull.reward, latent_reward=mean,
                               action=action, arm_id=arm, origin=agent.origins[arm], target=target,
                               source_round=source_round, observed=observed, innovations=innovations,
                               social_better=(mean > private_best if private_best is not None and agent.origins[arm] == "social" else None),
                               unique_discovered=len(independent)))
    frame = pd.DataFrame(events)
    counts = Counter(frame.arm_id)
    probs = np.array(list(counts.values())) / len(frame)
    result = dict(reward=frame.reward.mean(), conditional_reward=frame.reward.mean(), completion=1.,
                  latent_reward=frame.latent_reward.mean(), unique_discovered=len(independent),
                  innovations=int(frame.innovations.sum()), observation_rate=float((frame.action == "observe").mean()),
                  social_first_rate=float((frame.origin == "social").mean()),
                  social_first_better=frame.social_better.dropna().mean(),
                  diversity=float(np.exp(-np.sum(probs * np.log(probs)))),
                  early_reward=frame.loc[frame["round"] < 20, "reward"].mean(),
                  late_reward=frame.loc[frame["round"] >= rounds-20, "reward"].mean(),
                  seconds=time.perf_counter()-started, completion_tokens=0)
    return result, frame


def simulate_hierarchical_ucb(config, seed, mode):
    if int(config.get("rung", 1)) != 1:
        raise ValueError("hierarchical social UCB is currently a rung-1 baseline")
    if mode != "hierarchical_payoff":
        raise ValueError(f"unknown hierarchical UCB mode: {mode}")
    started = time.perf_counter()
    env = make_environment(config, seed)
    rounds = config["environment"]["rounds"]
    agents = [HierarchicalSocialUCB(
        i, seed, env.num_agents, env.num_arms, env.arm_mean_scale,
        env.reward_noise_std,
    ) for i in range(env.num_agents)]
    events = []
    independent = set()
    for t in range(rounds):
        regime_start = bool(t and env.regime_period and t % env.regime_period == 0)
        if regime_start:
            for agent in agents:
                agent.reset_values()
        env.begin_round(t)
        snapshot = env.snapshot(t)
        for agent in agents:
            available = [agent.agent_id]
            if not regime_start:
                available.extend(i for i, pull in enumerate(snapshot)
                                 if i != agent.agent_id and pull is not None)
            source_id = agent.choose_source(t, available)
            target = source_round = None
            observed = 0
            observed_payoff = None
            recommended_arm = None
            if source_id == agent.agent_id:
                action = "self"
            else:
                action = "observe"
                target = source_id
                observation = env.observe(
                    snapshot=snapshot,
                    observer_id=agent.agent_id,
                    target_id=target,
                    round_index=t,
                    reveal_payoff=True,
                )
                if observation is None:
                    raise RuntimeError("available hierarchical-UCB source had no action")
                assert observation.source_round < t
                observed = 1
                source_round = observation.source_round
                observed_payoff = observation.reward
                recommended_arm = observation.arm_id
                agent.acquire(recommended_arm, "social")
                assert observation.reward is not None
                agent.update_source(source_id, observation.reward)
                agent.update_arm(
                    recommended_arm, observation.reward,
                    weight=SOCIAL_SAMPLE_WEIGHT,
                )

            arm = agent.choose_arm(t)
            if arm not in agent.origins:
                agent.acquire(arm, "independent")
                independent.add(arm)

            private = [a for a, origin in agent.origins.items()
                       if origin == "independent"]
            mean = float(env.arm_means[arm])
            private_best = max((float(env.arm_means[a]) for a in private), default=None)
            pull = env.pull(agent_id=agent.agent_id, arm_id=arm, round_index=t)
            agent.update_arm(arm, pull.reward)
            if source_id == agent.agent_id:
                agent.update_source(source_id, pull.reward)
            events.append(dict(
                round=t, agent_id=agent.agent_id, reward=pull.reward,
                latent_reward=mean, action=action, arm_id=arm,
                origin=agent.origins[arm], target=target, source_id=source_id,
                source_round=source_round, observed=observed,
                recommended_arm=recommended_arm,
                observed_payoff=observed_payoff,
                source_credit=(pull.reward if source_id == agent.agent_id
                               else observed_payoff),
                innovations=0,
                social_better=(mean > private_best if private_best is not None
                               and agent.origins[arm] == "social" else None),
                unique_discovered=len(independent),
            ))
    frame = pd.DataFrame(events)
    counts = Counter(frame.arm_id)
    probs = np.array(list(counts.values())) / len(frame)
    result = dict(
        reward=frame.reward.mean(), conditional_reward=frame.reward.mean(),
        completion=1., latent_reward=frame.latent_reward.mean(),
        unique_discovered=len(independent), innovations=0,
        observation_rate=float((frame.action == "observe").mean()),
        social_first_rate=float((frame.origin == "social").mean()),
        social_first_better=frame.social_better.dropna().mean(),
        diversity=float(np.exp(-np.sum(probs * np.log(probs)))),
        early_reward=frame.loc[frame["round"] < 20, "reward"].mean(),
        late_reward=frame.loc[frame["round"] >= rounds - 20, "reward"].mean(),
        seconds=time.perf_counter() - started, completion_tokens=0,
    )
    return result, frame


def collect_environments():
    environments, matches = {}, []
    for rung in (1, 2):
        metrics = pd.read_csv(ROOT / f"runs/v3/rung{rung}/analysis/report_metrics_by_seed.csv")
        for condition in sorted(metrics.loc[metrics.model != "algorithmic", "condition"].unique()):
            config = json.loads((ROOT / f"runs/v3/rung{rung}" / condition / "seed_0/config.json").read_text())
            payload = dict(rung=rung, environment=config["environment"])
            identifier = f"r{rung}_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
            environments[identifier] = payload
            matches.append(dict(rung=rung, condition=condition, environment_id=identifier))
    return environments, pd.DataFrame(matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/v3/social_baseline")
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    environments, matches = collect_environments()
    matches.to_csv(out / "condition_mapping.csv", index=False)
    deoe_modes = ["solo", "social_action", "social_payoff"]
    hierarchical_modes = ["hierarchical_payoff"]
    (out / "specification.json").write_text(json.dumps(dict(
        version=VERSION, decay=DECAY, check_interval=CHECK_INTERVAL,
        exploration="1/sqrt(round+1)", environments=environments,
        deoe_modes=deoe_modes, hierarchical_rung1_modes=hierarchical_modes,
        seeds=list(range(8)), homogeneous_population=True, inference_tokens=0,
        max_learning_actions_per_round=1,
        hierarchical_ucb=dict(
            outer_reward="selected source's realized payoff: own lower-UCB pull or observed peer payoff",
            peer_recommendation="peer's latest arm strictly before the round",
            inner_policy="Gaussian UCB over arms after either self or peer evidence",
            source_prior_strength=SOURCE_PRIOR_STRENGTH,
            self_prior_advantage_noise_sd=SELF_PRIOR_ADVANTAGE_NOISE_SD,
            social_payoff_sample_weight=SOCIAL_SAMPLE_WEIGHT,
            reset_at_disclosed_regime_boundaries=True,
        ),
    ), indent=2))
    summaries, curves = [], []
    for key, config in sorted(environments.items()):
        for seed in range(8):
            modes = deoe_modes + (hierarchical_modes if config["rung"] == 1 else [])
            for mode in modes:
                if mode.startswith("hierarchical_"):
                    summary, frame = simulate_hierarchical_ucb(config, seed, mode)
                else:
                    summary, frame = simulate(config, seed, mode)
                meta = dict(environment_id=key, rung=config["rung"], seed=seed, mode=mode)
                summaries.append(dict(**meta, **summary))
                curve = frame.groupby("round").agg(reward=("reward", "mean"),
                    latent_reward=("latent_reward", "mean"), unique_discovered=("unique_discovered", "max"),
                    observation_rate=("action", lambda a: (a == "observe").mean()),
                    social_first_rate=("origin", lambda a: (a == "social").mean())).reset_index()
                curves.append(curve.assign(**meta))
        print(key, config["environment"], flush=True)
    pd.DataFrame(summaries).to_csv(out / "metrics_by_seed.csv", index=False)
    pd.concat(curves).to_csv(out / "performance_by_seed_round.csv", index=False)
    print(f"Completed {len(summaries)} homogeneous populations", flush=True)


if __name__ == "__main__":
    main()
