"""Static noisy bandit environment for rung 1.

The environment owns only experimental mechanics. Policies and prompts live in
agent.py, and orchestration lives in main.py.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Pull:
    round: int
    agent_id: int
    arm_id: int
    reward: float


@dataclass(frozen=True)
class Observation:
    acquired_round: int
    target_id: int
    source_round: int
    arm_id: int
    reward: float | None


@dataclass(frozen=True)
class InnovationResult:
    round: int
    agent_id: int
    arm_id: int
    coordinate: float | None
    reward: float
    arm_mean: float
    novel_to_population: bool


class BanditEnvironment:
    """A fixed set of arm means with independent Gaussian realized rewards."""

    def __init__(
        self,
        *,
        seed: int,
        num_agents: int,
        num_arms: int,
        arm_mean_scale: float,
        reward_noise_std: float,
        reward_shape: float = 1.0,
        regime_period: int | None = None,
    ) -> None:
        self.seed = seed
        self.num_agents = num_agents
        self.num_arms = num_arms
        self.arm_mean_scale = arm_mean_scale
        self.reward_noise_std = reward_noise_std
        if reward_shape <= 0:
            raise ValueError("reward_shape must be positive")
        if regime_period is not None and regime_period <= 0:
            raise ValueError("regime_period must be positive or null")
        self.reward_shape = reward_shape
        self.regime_period = regime_period
        self.regime_index = -1
        self.arm_means = np.array([], dtype=np.float64)
        self.best_arm = -1
        self.best_mean = float("nan")
        self.begin_round(0)

        self.histories: list[list[Pull]] = [[] for _ in range(num_agents)]
        self._pulled_this_round: set[tuple[int, int]] = set()

    def _means_for_regime(self, regime_index: int) -> np.ndarray:
        seed_parts = (
            [self.seed, 0]
            if regime_index == 0
            else [self.seed, 0, regime_index]
        )
        rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
        if math.isclose(self.reward_shape, 1.0):
            return rng.exponential(self.arm_mean_scale, size=self.num_arms)
        unit_mean_scale = 1.0 / math.gamma(1.0 + 1.0 / self.reward_shape)
        return (
            rng.weibull(self.reward_shape, size=self.num_arms)
            * self.arm_mean_scale
            * unit_mean_scale
        )

    def begin_round(self, round_index: int) -> None:
        regime_index = (
            round_index // self.regime_period if self.regime_period is not None else 0
        )
        if regime_index == self.regime_index:
            return
        self.regime_index = regime_index
        self.arm_means = self._means_for_regime(regime_index)
        self.best_arm = int(np.argmax(self.arm_means))
        self.best_mean = float(self.arm_means[self.best_arm])

    def snapshot(self, round_index: int) -> tuple[Pull | None, ...]:
        """Freeze the latest pull strictly before this round for every agent."""
        snapshot: list[Pull | None] = []
        for history in self.histories:
            prior = history[-1] if history and history[-1].round < round_index else None
            snapshot.append(prior)
        return tuple(snapshot)

    def observe(
        self,
        *,
        snapshot: tuple[Pull | None, ...],
        observer_id: int,
        target_id: int,
        round_index: int,
        reveal_payoff: bool,
    ) -> Observation | None:
        if not 0 <= observer_id < self.num_agents:
            raise ValueError(f"invalid observer_id: {observer_id}")
        if not 0 <= target_id < self.num_agents or target_id == observer_id:
            raise ValueError(f"invalid target_id: {target_id}")

        demonstrated = snapshot[target_id]
        if demonstrated is None:
            return None
        return Observation(
            acquired_round=round_index,
            target_id=target_id,
            source_round=demonstrated.round,
            arm_id=demonstrated.arm_id,
            reward=demonstrated.reward if reveal_payoff else None,
        )

    def pull(self, *, agent_id: int, arm_id: int, round_index: int) -> Pull:
        if not 0 <= agent_id < self.num_agents:
            raise ValueError(f"invalid agent_id: {agent_id}")
        if not 0 <= arm_id < self.num_arms:
            raise ValueError(f"invalid arm_id: {arm_id}")
        key = (round_index, agent_id)
        if key in self._pulled_this_round:
            raise RuntimeError(f"agent {agent_id} pulled twice in round {round_index}")

        reward = self.potential_reward(
            round_index=round_index, agent_id=agent_id, arm_id=arm_id
        )
        pull = Pull(round_index, agent_id, arm_id, reward)
        self.histories[agent_id].append(pull)
        self._pulled_this_round.add(key)
        return pull

    def potential_reward(self, *, round_index: int, agent_id: int, arm_id: int) -> float:
        """Return a reward keyed independently of policy call order or condition."""
        reward_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 1, round_index, agent_id, arm_id])
        )
        return float(
            reward_rng.normal(float(self.arm_means[arm_id]), self.reward_noise_std)
        )

    def restore_histories(self, histories: list[list[Pull]]) -> None:
        """Restore completed pulls when resuming at a round boundary."""
        if len(histories) != self.num_agents:
            raise ValueError("checkpoint agent count does not match environment")
        restored: list[list[Pull]] = []
        pulled: set[tuple[int, int]] = set()
        for agent_id, history in enumerate(histories):
            for pull in history:
                if pull.agent_id != agent_id:
                    raise ValueError("checkpoint pull belongs to the wrong agent")
                key = (pull.round, pull.agent_id)
                if key in pulled:
                    raise ValueError("checkpoint contains duplicate pulls")
                pulled.add(key)
            restored.append(list(history))
        self.histories = restored
        self._pulled_this_round = pulled

    def description(self) -> dict:
        return {
            "seed": self.seed,
            "num_agents": self.num_agents,
            "num_arms": self.num_arms,
            "arm_distribution": "mean_normalized_weibull",
            "arm_mean_scale": self.arm_mean_scale,
            "reward_shape": self.reward_shape,
            "regime_period": self.regime_period,
            "reward_distribution": "normal",
            "reward_noise_std": self.reward_noise_std,
            "arm_means": self.arm_means.tolist(),
            "best_arm": self.best_arm,
            "best_mean": self.best_mean,
        }


class InfiniteBanditEnvironment:
    """Lazy infinite-arm reservoir with random or spatially correlated means."""

    def __init__(
        self,
        *,
        seed: int,
        num_agents: int,
        landscape: str,
        arm_mean_scale: float,
        reward_noise_std: float,
        coordinate_decimals: int = 9,
        structured_features: int = 32,
        structured_length_scale: float = 0.15,
        reference_grid_size: int = 100_001,
        reward_shape: float = 1.0,
        regime_period: int | None = None,
        coordinate_innovation: bool = False,
    ) -> None:
        if landscape not in {"unstructured", "structured"}:
            raise ValueError(f"unknown infinite-arm landscape: {landscape}")
        self.seed = seed
        self.num_agents = num_agents
        self.landscape = landscape
        self.arm_mean_scale = arm_mean_scale
        self.reward_noise_std = reward_noise_std
        self.coordinate_decimals = coordinate_decimals
        self.coordinate_scale = 10**coordinate_decimals
        self.structured_features = structured_features
        self.structured_length_scale = structured_length_scale
        self.reference_grid_size = reference_grid_size
        if reward_shape <= 0:
            raise ValueError("reward_shape must be positive")
        if regime_period is not None and regime_period <= 0:
            raise ValueError("regime_period must be positive or null")
        self.reward_shape = reward_shape
        self.regime_period = regime_period
        self.coordinate_innovation = coordinate_innovation or landscape == "structured"
        self.regime_index = -1

        self.histories: list[list[Pull]] = [[] for _ in range(num_agents)]
        self._pulled_this_round: set[tuple[int, int]] = set()
        self._population_arms: set[int] = set()
        self._population_frontier: float | None = None

        self.frequencies = np.array([], dtype=np.float64)
        self.cosine_weights = np.array([], dtype=np.float64)
        self.sine_weights = np.array([], dtype=np.float64)
        self.reference_best_arm: int | None = None
        self.reference_best_coordinate: float | None = None
        self.reference_best_mean: float | None = None
        self.begin_round(0)

    def _quantile_to_mean(self, quantile: float) -> float:
        quantile = min(max(quantile, 1e-12), 1.0 - 1e-12)
        if math.isclose(self.reward_shape, 1.0):
            return -self.arm_mean_scale * math.log1p(-quantile)
        unit_mean_scale = 1.0 / math.gamma(1.0 + 1.0 / self.reward_shape)
        return (
            self.arm_mean_scale
            * unit_mean_scale
            * (-math.log1p(-quantile)) ** (1.0 / self.reward_shape)
        )

    def _set_structured_regime(self, regime_index: int) -> None:
        seed_parts = (
            [self.seed, 20]
            if regime_index == 0
            else [self.seed, 20, regime_index]
        )
        rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
        self.frequencies = rng.normal(
            0.0,
            1.0 / self.structured_length_scale,
            size=self.structured_features,
        )
        self.cosine_weights = rng.normal(size=self.structured_features)
        self.sine_weights = rng.normal(size=self.structured_features)
        grid = np.linspace(0.0, 1.0, self.reference_grid_size)
        values = np.zeros(self.reference_grid_size, dtype=np.float64)
        for frequency, cosine, sine in zip(
            self.frequencies,
            self.cosine_weights,
            self.sine_weights,
            strict=True,
        ):
            values += cosine * np.cos(frequency * grid)
            values += sine * np.sin(frequency * grid)
        coordinate = float(grid[int(np.argmax(values))])
        self.reference_best_arm = self.coordinate_to_arm(coordinate)
        self.reference_best_coordinate = self.arm_to_coordinate(
            self.reference_best_arm
        )
        self.reference_best_mean = self.arm_mean(self.reference_best_arm)

    def begin_round(self, round_index: int) -> None:
        regime_index = (
            round_index // self.regime_period if self.regime_period is not None else 0
        )
        if regime_index == self.regime_index:
            return
        self.regime_index = regime_index
        if self.landscape == "structured":
            self._set_structured_regime(regime_index)
        if self._population_arms:
            self._population_frontier = max(
                self.arm_mean(arm_id) for arm_id in self._population_arms
            )

    @staticmethod
    def _split(value: int) -> tuple[int, int]:
        return value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF

    @staticmethod
    def _opaque_arm_id(agent_id: int, innovation_index: int) -> int:
        payload = f"{agent_id}:{innovation_index}".encode("utf-8")
        value = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        )
        return value & ((1 << 63) - 1)

    def coordinate_to_arm(self, coordinate: float) -> int:
        if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
            raise ValueError("structured innovation coordinate must be in [0, 1]")
        return int(round(coordinate * self.coordinate_scale))

    def arm_to_coordinate(self, arm_id: int) -> float:
        if not 0 <= arm_id <= self.coordinate_scale:
            raise ValueError(f"invalid structured arm ID: {arm_id}")
        return arm_id / self.coordinate_scale

    def arm_mean(self, arm_id: int) -> float:
        low, high = self._split(arm_id)
        if self.landscape == "unstructured":
            seed_parts = (
                [self.seed, 21, low, high]
                if self.regime_index == 0
                else [self.seed, 21, self.regime_index, low, high]
            )
            rng = np.random.default_rng(
                np.random.SeedSequence(seed_parts)
            )
            if math.isclose(self.reward_shape, 1.0) and not self.coordinate_innovation:
                return float(rng.exponential(self.arm_mean_scale))
            return self._quantile_to_mean(float(rng.random()))

        coordinate = self.arm_to_coordinate(arm_id)
        angles = self.frequencies * coordinate
        latent = float(
            np.sum(
                self.cosine_weights * np.cos(angles)
                + self.sine_weights * np.sin(angles)
            )
            / math.sqrt(self.structured_features)
        )
        quantile = 0.5 * (1.0 + math.erf(latent / math.sqrt(2.0)))
        return self._quantile_to_mean(quantile)

    def population_arm_ids(self) -> set[int]:
        return set(self._population_arms)

    def population_best_mean(self) -> float | None:
        return self._population_frontier

    def snapshot(self, round_index: int) -> tuple[Pull | None, ...]:
        return tuple(
            history[-1]
            if history and history[-1].round < round_index
            else None
            for history in self.histories
        )

    def observe(
        self,
        *,
        snapshot: tuple[Pull | None, ...],
        observer_id: int,
        target_id: int,
        round_index: int,
        reveal_payoff: bool,
    ) -> Observation | None:
        if not 0 <= observer_id < self.num_agents:
            raise ValueError(f"invalid observer_id: {observer_id}")
        if not 0 <= target_id < self.num_agents or target_id == observer_id:
            raise ValueError(f"invalid target_id: {target_id}")
        demonstrated = snapshot[target_id]
        if demonstrated is None:
            return None
        return Observation(
            acquired_round=round_index,
            target_id=target_id,
            source_round=demonstrated.round,
            arm_id=demonstrated.arm_id,
            reward=demonstrated.reward if reveal_payoff else None,
        )

    def innovate(
        self,
        *,
        agent_id: int,
        innovation_index: int,
        round_index: int,
        coordinate: float | None,
    ) -> InnovationResult:
        if self.coordinate_innovation:
            if coordinate is None:
                raise ValueError("coordinate innovation requires a coordinate")
            arm_id = self.coordinate_to_arm(coordinate)
        else:
            if coordinate is not None:
                raise ValueError("unstructured innovation does not take a coordinate")
            arm_id = self._opaque_arm_id(agent_id, innovation_index)

        novel_to_population = arm_id not in self._population_arms
        self._population_arms.add(arm_id)
        low, high = self._split(arm_id)
        reward_rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, 22, round_index, agent_id, innovation_index, low, high]
            )
        )
        arm_mean = self.arm_mean(arm_id)
        if self._population_frontier is None or arm_mean > self._population_frontier:
            self._population_frontier = arm_mean
        return InnovationResult(
            round=round_index,
            agent_id=agent_id,
            arm_id=arm_id,
            coordinate=(
                self.arm_to_coordinate(arm_id)
                if self.coordinate_innovation
                else None
            ),
            reward=float(reward_rng.normal(arm_mean, self.reward_noise_std)),
            arm_mean=arm_mean,
            novel_to_population=novel_to_population,
        )

    def pull(self, *, agent_id: int, arm_id: int, round_index: int) -> Pull:
        key = (round_index, agent_id)
        if key in self._pulled_this_round:
            raise RuntimeError(f"agent {agent_id} pulled twice in round {round_index}")
        low, high = self._split(arm_id)
        reward_rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, 23, round_index, agent_id, low, high]
            )
        )
        pull = Pull(
            round=round_index,
            agent_id=agent_id,
            arm_id=arm_id,
            reward=float(
                reward_rng.normal(self.arm_mean(arm_id), self.reward_noise_std)
            ),
        )
        self.histories[agent_id].append(pull)
        self._pulled_this_round.add(key)
        return pull

    def restore_state(
        self,
        *,
        histories: list[list[Pull]],
        innovation_arm_ids: list[int],
    ) -> None:
        if len(histories) != self.num_agents:
            raise ValueError("checkpoint agent count does not match environment")
        self.histories = [list(history) for history in histories]
        self._pulled_this_round = {
            (pull.round, pull.agent_id)
            for history in histories
            for pull in history
        }
        self._population_arms = set(innovation_arm_ids)
        self._population_frontier = max(
            (self.arm_mean(arm_id) for arm_id in self._population_arms),
            default=None,
        )

    def description(self) -> dict:
        description = {
            "seed": self.seed,
            "rung": 2,
            "num_agents": self.num_agents,
            "landscape": self.landscape,
            "arm_distribution": "mean_normalized_weibull",
            "arm_mean_scale": self.arm_mean_scale,
            "reward_shape": self.reward_shape,
            "regime_period": self.regime_period,
            "coordinate_innovation": self.coordinate_innovation,
            "reward_distribution": "normal",
            "reward_noise_std": self.reward_noise_std,
        }
        if self.coordinate_innovation:
            description.update(
                {
                    "coordinate_domain": [0.0, 1.0],
                    "coordinate_decimals": self.coordinate_decimals,
                }
            )
        if self.landscape == "structured":
            description.update(
                {
                    "structured_features": self.structured_features,
                    "structured_length_scale": self.structured_length_scale,
                    "frequencies": self.frequencies.tolist(),
                    "cosine_weights": self.cosine_weights.tolist(),
                    "sine_weights": self.sine_weights.tolist(),
                    "reference_grid_size": self.reference_grid_size,
                    "reference_best_arm": self.reference_best_arm,
                    "reference_best_coordinate": self.reference_best_coordinate,
                    "reference_best_mean": self.reference_best_mean,
                }
            )
        return description


def observation_dict(observation: Observation | None) -> dict | None:
    return asdict(observation) if observation is not None else None
