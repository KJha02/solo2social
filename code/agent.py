"""Agent state, LLM decision generation, and the UCB baseline."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from env import Observation, Pull


FINAL_ACTION = re.compile(
    r"(?:^|\n)\s*FINAL:\s*(OBSERVE|PULL)\s+(\d+)\s*\Z", re.IGNORECASE
)
BUDGET_RULE_MARKER = "{{BUDGET_RULE}}"
RUNG2_FINAL_ACTION = re.compile(
    r"(?:^|\n)\s*FINAL:\s*"
    r"(?:(OBSERVE)\s+(\d+)|(PULL)\s+(\d+)|"
    r"(INNOVATE)(?:\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?)\s*\Z",
    re.IGNORECASE,
)
TERMINAL_FINAL_LINE = re.compile(
    r"(?:^|\n)\s*(FINAL:\s*[^\n]+)\s*\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    value: int


@dataclass(frozen=True)
class Generation:
    text: str
    final_text: str
    token_count: int
    finish_reason: str | None
    action: ParsedAction | None
    requested_max_tokens: int
    effective_max_tokens: int
    prompt_tokens: int
    context_limited: bool
    cost_usd: float = 0.0
    reasoning_tokens: int = 0


def is_degenerate_completion(text: str, *, minimum_length: int = 128) -> bool:
    """Detect the repeated-token corruption seen in failed inference runs."""
    compact = "".join(text.split())
    return len(compact) >= minimum_length and len(set(compact)) <= 2


def is_degenerate_token_ids(
    token_ids: list[int], *, minimum_tokens: int = 128
) -> bool:
    """Detect loops even when one repeated token decodes to several characters."""
    return len(token_ids) >= minimum_tokens and len(set(token_ids)) <= 2


def completion_preview(text: str, *, limit: int = 256) -> str:
    """Return a bounded diagnostic excerpt without creating huge event logs."""
    normalized = text.replace("\x00", "�")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "…"


def terminal_final_line(text: str) -> str:
    match = TERMINAL_FINAL_LINE.search(text)
    return match.group(1).strip() if match else ""


@dataclass(frozen=True)
class Rung2Action:
    kind: str
    value: int | float | None


@dataclass(frozen=True)
class Probe:
    round: int
    arm_id: int
    coordinate: float | None
    reward: float


def parse_action(text: str) -> ParsedAction | None:
    match = FINAL_ACTION.search(text)
    if match is None:
        return None
    return ParsedAction(match.group(1).lower(), int(match.group(2)))


def parse_rung2_action(text: str) -> Rung2Action | None:
    match = RUNG2_FINAL_ACTION.search(text)
    if match is None:
        return None
    if match.group(1):
        return Rung2Action("observe", int(match.group(2)))
    if match.group(3):
        return Rung2Action("pull", int(match.group(4)))
    coordinate = float(match.group(6)) if match.group(6) is not None else None
    return Rung2Action("innovate", coordinate)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def intervention_settings(config: dict[str, Any]) -> tuple[float, int]:
    """Optional factorial controls; grants never increase under intervention."""
    values = config.get("interventions", {})
    unknown = set(values) - {"independent_search_probability", "execution_reserve_tokens"}
    if unknown:
        raise ValueError(f"unknown interventions: {sorted(unknown)}")
    probability = float(values.get("independent_search_probability", 0))
    reserve = values.get("execution_reserve_tokens", 0)
    if not 0 <= probability <= 1:
        raise ValueError("independent_search_probability must be in [0, 1]")
    if not isinstance(reserve, int) or not 0 <= reserve < config["budget"]["tokens_per_round"]:
        raise ValueError("execution reserve must be an integer below the round grant")
    return probability, reserve


def independent_search_assigned(config: dict[str, Any], seed: int, round_index: int, agent_id: int) -> bool:
    probability, _ = intervention_settings(config)
    # Exclude condition/model: matched populations receive identical assignments.
    return stable_seed("independent-search", seed, round_index, agent_id) / 2**32 < probability


def decision_token_limit(balance: int, maximum: int, reserve: int) -> int:
    """Cap pre-pull deliberation; once at the reserve, spend it on pulling."""
    available = balance - reserve if reserve and balance > reserve else balance
    return min(available, maximum)


def budget_mode(config: dict[str, Any]) -> str | None:
    budget = config.get("budget")
    if budget is None:
        return None
    return "carry" if budget.get("carry_over", True) else "use_it_or_lose_it"


def render_budget_prompt(template: str, budget: dict[str, Any]) -> str:
    """Fill the V2 budget rule without changing legacy prompt text."""
    if BUDGET_RULE_MARKER not in template:
        return template
    grant = int(budget["tokens_per_round"])
    if budget.get("carry_over", True):
        rule = (
            f"At the start of each round, {grant} completion tokens are added to "
            "your balance. Tokens not used remain in the balance and are available "
            "in later rounds."
        )
    else:
        rule = (
            f"At the start of each round, your completion-token balance is set to "
            f"{grant}. Tokens not used by the end of that round expire and cannot "
            "be used later."
        )
    return template.replace(BUDGET_RULE_MARKER, rule)


def render_experiment_prompt(template: str, config: dict[str, Any]) -> str:
    """Resolve the small set of V3 task disclosures and steering conditions."""
    prompt = render_budget_prompt(template, config["budget"])
    environment = config["environment"]
    regime_period = environment.get("regime_period")
    if regime_period:
        volatility = (
            f"Expected arm payoffs change together every {regime_period} rounds. "
            "Past evidence may therefore become stale after a change."
        )
        prompt = prompt.replace("static 25-armed", "changing 25-armed")
    else:
        volatility = "Expected arm payoffs remain fixed throughout the game."

    shape = float(environment.get("reward_shape", 1.0))
    if shape < 0.8:
        scarcity = "Exceptionally good arms are rare and disproportionately valuable."
    elif shape > 1.5:
        scarcity = "Arm quality is relatively concentrated; good arms are not extremely rare."
    else:
        scarcity = "Arm quality has a moderately long tail."

    if int(config.get("rung", 1)) == 2:
        if environment["landscape"] == "unstructured":
            structure = "Coordinates are labels only; nearby coordinates are no more similar."
        else:
            length_scale = float(environment.get("structured_length_scale", 0.15))
            strength = "weakly" if length_scale < 0.08 else "moderately" if length_scale < 0.25 else "strongly"
            structure = f"Nearby coordinates are {strength} informative about one another."
    else:
        structure = ""

    guidance = config.get("guidance", "neutral")
    if guidance == "neutral":
        guidance_rule = ""
    elif guidance == "deliberate":
        guidance_rule = (
            "Before acting, explicitly weigh obtaining social information, independent "
            "exploration or innovation, and exploiting the best evidence currently available."
        )
    elif guidance == "strategic":
        guidance_rule = (
            "Use social information selectively rather than automatically. Preserve some "
            "independent exploration or innovation so the population continues producing "
            "information; exploit strong evidence, and discount evidence that may be stale."
        )
    else:
        raise ValueError(f"unknown guidance condition: {guidance}")

    resolved = {"{{GUIDANCE_RULE}}": guidance_rule}
    if int(config.get("rung", 1)) in {1, 2}:
        resolved.update(
            {
                "{{VOLATILITY_RULE}}": volatility,
                "{{SCARCITY_RULE}}": scarcity,
                "{{STRUCTURE_RULE}}": structure,
            }
        )
    for marker, value in resolved.items():
        prompt = prompt.replace(marker, value)
    if "{{OBSERVE_RULE}}" in prompt:
        observe_rule = (
            "An observation reveals the arm and its realized scored payoff."
            if config.get("observe_payoff", False)
            else "An observation reveals the arm but never its payoff."
        )
        prompt = prompt.replace("{{OBSERVE_RULE}}", observe_rule)
    if "{{SOCIAL_RULE}}" in prompt:
        if config.get("social_info") == "none":
            social_rule = "OBSERVE is unavailable. You may only use your owned skill."
            action_lines = "FINAL: PULL <skill_id>"
        else:
            social_rule = (
                "OBSERVE buys the configured information about a chosen agent's most "
                "recent skill use strictly before this round. An observed candidate "
                "can later be adopted by pulling it."
            )
            action_lines = "FINAL: OBSERVE <agent_id>\nFINAL: PULL <skill_id>"
        prompt = prompt.replace("{{SOCIAL_RULE}}", social_rule).replace(
            "{{ACTION_LINES}}", action_lines
        )
    additions = [value for marker, value in resolved.items() if marker not in template and value]
    if additions:
        insertion = "\n\n".join(additions) + "\n\n"
        prompt = prompt.replace("Choose exactly one action now.", insertion + "Choose exactly one action now.")
    return prompt


class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        self.pulls: list[Pull] = []
        self.observations: list[Observation] = []
        self.observation_attempts: list[tuple[int, int, bool]] = []
        self.token_balance = 0
        self.personal_best_reward = -math.inf
        self.last_improvement_round: int | None = None

    def begin_round(self, tokens_per_round: int, *, carry_over: bool = True) -> int:
        if carry_over:
            self.token_balance += tokens_per_round
        else:
            self.token_balance = tokens_per_round
        return self.token_balance

    def spend(self, token_count: int) -> None:
        if token_count < 0 or token_count > self.token_balance:
            raise RuntimeError(
                f"agent {self.agent_id} cannot spend {token_count} from "
                f"balance {self.token_balance}"
            )
        self.token_balance -= token_count

    def add_observation_attempt(
        self, *, round_index: int, target_id: int, observation: Observation | None
    ) -> None:
        self.observation_attempts.append(
            (round_index, target_id, observation is not None)
        )
        if observation is not None:
            self.observations.append(observation)

    def add_pull(self, pull: Pull) -> None:
        self.pulls.append(pull)
        if pull.reward > self.personal_best_reward:
            self.personal_best_reward = pull.reward
            self.last_improvement_round = pull.round

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "pulls": [asdict(pull) for pull in self.pulls],
            "observations": [asdict(observation) for observation in self.observations],
            "observation_attempts": [list(attempt) for attempt in self.observation_attempts],
            "token_balance": self.token_balance,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> LLMAgent:
        agent = cls(int(state["agent_id"]))
        for pull in state["pulls"]:
            agent.add_pull(Pull(**pull))
        agent.observations = [
            Observation(**observation) for observation in state["observations"]
        ]
        agent.observation_attempts = [
            (int(round_index), int(target_id), bool(succeeded))
            for round_index, target_id, succeeded in state["observation_attempts"]
        ]
        agent.token_balance = int(state["token_balance"])
        return agent

    def payoff_evidence(self) -> dict[int, list[float]]:
        evidence: dict[int, list[float]] = defaultdict(list)
        for pull in self.pulls:
            evidence[pull.arm_id].append(pull.reward)
        for observation in self.observations:
            if observation.reward is not None:
                evidence[observation.arm_id].append(observation.reward)
        return dict(evidence)

    def empirical_means(self) -> dict[int, float]:
        return {
            arm_id: float(np.mean(rewards))
            for arm_id, rewards in self.payoff_evidence().items()
        }

    def pull_label(self, arm_id: int) -> str:
        means = self.empirical_means()
        if not means:
            return "explore"
        best = max(means.values())
        return (
            "exploit"
            if arm_id in means and math.isclose(means[arm_id], best, rel_tol=1e-12)
            else "explore"
        )

    def decision_features(self, round_index: int) -> dict[str, Any]:
        means = self.empirical_means()
        best_known_mean = max(means.values()) if means else None
        last_reward = self.pulls[-1].reward if self.pulls else None
        last_reward_gap = (
            last_reward - best_known_mean
            if last_reward is not None and best_known_mean is not None
            else None
        )
        known_arms = {pull.arm_id for pull in self.pulls}
        known_arms.update(obs.arm_id for obs in self.observations)
        return {
            "best_known_mean": best_known_mean,
            "last_reward": last_reward,
            "last_reward_gap": last_reward_gap,
            "rounds_since_improvement": (
                round_index - self.last_improvement_round
                if self.last_improvement_round is not None
                else None
            ),
            "known_arm_count": len(known_arms),
            "personal_pull_count": len(self.pulls),
            "social_observation_count": len(self.observations),
            "social_observation_attempt_count": len(self.observation_attempts),
            "distinct_observed_arm_count": len(
                {obs.arm_id for obs in self.observations}
            ),
        }

    def copy_features(self, arm_id: int, round_index: int) -> dict[str, Any]:
        supporting = [obs for obs in self.observations if obs.arm_id == arm_id]
        latest = self.observations[-1] if self.observations else None
        observed_rewards = [
            obs.reward for obs in supporting if obs.reward is not None
        ]
        personal_rewards: dict[int, list[float]] = defaultdict(list)
        for pull in self.pulls:
            personal_rewards[pull.arm_id].append(pull.reward)
        personal_means = {
            candidate: float(np.mean(values))
            for candidate, values in personal_rewards.items()
        }
        best_personal_arm = (
            max(personal_means, key=lambda candidate: (personal_means[candidate], -candidate))
            if personal_means
            else None
        )
        best_personal = (
            personal_means[best_personal_arm]
            if best_personal_arm is not None
            else None
        )
        personal_alternatives = {
            candidate: mean
            for candidate, mean in personal_means.items()
            if candidate != arm_id
        }
        best_personal_alternative_arm = (
            max(
                personal_alternatives,
                key=lambda candidate: (personal_alternatives[candidate], -candidate),
            )
            if personal_alternatives
            else None
        )
        best_personal_alternative = (
            personal_alternatives[best_personal_alternative_arm]
            if best_personal_alternative_arm is not None
            else None
        )
        latest_observed_payoff = observed_rewards[-1] if observed_rewards else None
        selected_observed_mean = (
            float(np.mean(observed_rewards)) if observed_rewards else None
        )
        selected_evidence = list(personal_rewards.get(arm_id, [])) + observed_rewards
        selected_evidence_mean = (
            float(np.mean(selected_evidence)) if selected_evidence else None
        )
        source = max(
            supporting,
            key=lambda observation: (
                observation.acquired_round,
                observation.source_round,
                observation.target_id,
            ),
            default=None,
        )
        return {
            "copy_any": bool(supporting),
            "personally_novel_pull": arm_id not in personal_rewards,
            "independent_exploration_pull": (
                arm_id not in personal_rewards and not supporting
            ),
            "copy_latest": latest is not None and latest.arm_id == arm_id,
            "copy_support_count": len(supporting),
            "copy_source_count": len({obs.target_id for obs in supporting}),
            "copy_recency": (
                round_index - max(obs.acquired_round for obs in supporting)
                if supporting
                else None
            ),
            "copy_source_agent_id": source.target_id if source else None,
            "copy_source_round": source.source_round if source else None,
            "latest_observed_payoff": latest_observed_payoff,
            "best_personal_arm_id_before_pull": best_personal_arm,
            "best_personal_mean_before_pull": best_personal,
            "best_personal_alternative_arm_id_before_pull": (
                best_personal_alternative_arm
            ),
            "best_personal_alternative_mean_before_pull": (
                best_personal_alternative
            ),
            "selected_personal_mean_before_pull": personal_means.get(arm_id),
            "selected_observed_mean_before_pull": selected_observed_mean,
            "selected_evidence_mean_before_pull": selected_evidence_mean,
            "selected_estimated_advantage_vs_best_personal": (
                selected_evidence_mean - best_personal
                if selected_evidence_mean is not None and best_personal is not None
                else None
            ),
            "selected_estimated_advantage_vs_best_personal_alternative": (
                selected_evidence_mean - best_personal_alternative
                if selected_evidence_mean is not None
                and best_personal_alternative is not None
                else None
            ),
            "observed_payoff_advantage": (
                latest_observed_payoff - best_personal
                if latest_observed_payoff is not None and best_personal is not None
                else None
            ),
        }

    def render_state(
        self,
        *,
        round_index: int,
        num_agents: int,
        num_arms: int,
        social_enabled: bool,
    ) -> str:
        personal: dict[int, list[float]] = defaultdict(list)
        for pull in self.pulls:
            personal[pull.arm_id].append(pull.reward)

        if personal:
            personal_lines = [
                f"arm {arm}: rewards={[round(x, 4) for x in rewards]}, "
                f"mean={np.mean(rewards):.4f}"
                for arm, rewards in sorted(personal.items())
            ]
        else:
            personal_lines = ["none"]

        if self.observations:
            social_lines = []
            for obs in self.observations:
                payoff = "hidden" if obs.reward is None else f"{obs.reward:.4f}"
                social_lines.append(
                    f"bought in round {obs.acquired_round}: agent {obs.target_id} "
                    f"pulled arm {obs.arm_id} in round {obs.source_round}; "
                    f"payoff={payoff}"
                )
        else:
            social_lines = ["none"]
        failed = [
            f"round {attempt_round}: agent {target_id} had no prior pull"
            for attempt_round, target_id, succeeded in self.observation_attempts
            if not succeeded
        ]

        social_state = ""
        if social_enabled:
            targets = ", ".join(str(i) for i in range(num_agents) if i != self.agent_id)
            social_state = (
                f"Valid observation targets: {targets}\n\n"
                "Your purchased social observations:\n"
                + "\n".join(social_lines)
                + "\n\nPurchased observations that returned nothing:\n"
                + ("\n".join(failed) if failed else "none")
            )
        return (
            f"Round: {round_index}\n"
            f"Your agent ID: {self.agent_id}\n"
            f"Completion tokens remaining this round: {self.token_balance}\n"
            f"Valid arm IDs: 0 through {num_arms - 1}\n\n"
            "Your personal payoff history by arm:\n"
            + "\n".join(personal_lines)
            + ("\n\n" + social_state if social_state else "")
        )


class Rung2Agent(LLMAgent):
    """LLM state for a repertoire that grows through innovation and copying."""

    def __init__(self, agent_id: int) -> None:
        super().__init__(agent_id)
        self.probes: list[Probe] = []

    def add_probe(self, probe: Probe) -> None:
        self.probes.append(probe)
        if probe.reward > self.personal_best_reward:
            self.personal_best_reward = probe.reward
            self.last_improvement_round = probe.round

    def known_arm_ids(self) -> set[int]:
        known = {pull.arm_id for pull in self.pulls}
        known.update(probe.arm_id for probe in self.probes)
        known.update(observation.arm_id for observation in self.observations)
        return known

    def personal_payoff_evidence(self) -> dict[int, list[float]]:
        evidence: dict[int, list[float]] = defaultdict(list)
        for pull in self.pulls:
            evidence[pull.arm_id].append(pull.reward)
        for probe in self.probes:
            evidence[probe.arm_id].append(probe.reward)
        return dict(evidence)

    def payoff_evidence(self) -> dict[int, list[float]]:
        evidence = defaultdict(list, self.personal_payoff_evidence())
        for observation in self.observations:
            if observation.reward is not None:
                evidence[observation.arm_id].append(observation.reward)
        return dict(evidence)

    def decision_features(self, round_index: int) -> dict[str, Any]:
        means = self.empirical_means()
        best_known_mean = max(means.values()) if means else None
        last_reward = self.pulls[-1].reward if self.pulls else None
        return {
            "best_known_mean": best_known_mean,
            "last_reward": last_reward,
            "last_reward_gap": (
                last_reward - best_known_mean
                if last_reward is not None and best_known_mean is not None
                else None
            ),
            "rounds_since_improvement": (
                round_index - self.last_improvement_round
                if self.last_improvement_round is not None
                else None
            ),
            "known_arm_count": len(self.known_arm_ids()),
            "personal_pull_count": len(self.pulls),
            "innovation_count": len(self.probes),
            "social_observation_count": len(self.observations),
            "social_observation_attempt_count": len(self.observation_attempts),
            "distinct_observed_arm_count": len(
                {observation.arm_id for observation in self.observations}
            ),
        }

    def copy_features(self, arm_id: int, round_index: int) -> dict[str, Any]:
        supporting = [
            observation
            for observation in self.observations
            if observation.arm_id == arm_id
        ]
        latest = self.observations[-1] if self.observations else None
        observed_rewards = [
            observation.reward
            for observation in supporting
            if observation.reward is not None
        ]
        personal_means = {
            candidate: float(np.mean(values))
            for candidate, values in self.personal_payoff_evidence().items()
        }
        best_personal_arm = (
            max(personal_means, key=lambda candidate: (personal_means[candidate], -candidate))
            if personal_means
            else None
        )
        best_personal = (
            personal_means[best_personal_arm]
            if best_personal_arm is not None
            else None
        )
        personal_alternatives = {
            candidate: mean
            for candidate, mean in personal_means.items()
            if candidate != arm_id
        }
        best_personal_alternative_arm = (
            max(
                personal_alternatives,
                key=lambda candidate: (personal_alternatives[candidate], -candidate),
            )
            if personal_alternatives
            else None
        )
        best_personal_alternative = (
            personal_alternatives[best_personal_alternative_arm]
            if best_personal_alternative_arm is not None
            else None
        )
        latest_observed_payoff = observed_rewards[-1] if observed_rewards else None
        selected_observed_mean = (
            float(np.mean(observed_rewards)) if observed_rewards else None
        )
        selected_evidence = (
            list(self.personal_payoff_evidence().get(arm_id, [])) + observed_rewards
        )
        selected_evidence_mean = (
            float(np.mean(selected_evidence)) if selected_evidence else None
        )
        source = max(
            supporting,
            key=lambda observation: (
                observation.acquired_round,
                observation.source_round,
                observation.target_id,
            ),
            default=None,
        )
        return {
            "copy_any": bool(supporting),
            "personally_novel_pull": arm_id not in personal_means,
            "independent_exploration_pull": (
                arm_id not in personal_means and not supporting
            ),
            "copy_latest": latest is not None and latest.arm_id == arm_id,
            "copy_support_count": len(supporting),
            "copy_source_count": len(
                {observation.target_id for observation in supporting}
            ),
            "copy_recency": (
                round_index
                - max(observation.acquired_round for observation in supporting)
                if supporting
                else None
            ),
            "copy_source_agent_id": source.target_id if source else None,
            "copy_source_round": source.source_round if source else None,
            "latest_observed_payoff": latest_observed_payoff,
            "best_personal_arm_id_before_pull": best_personal_arm,
            "best_personal_mean_before_pull": best_personal,
            "best_personal_alternative_arm_id_before_pull": (
                best_personal_alternative_arm
            ),
            "best_personal_alternative_mean_before_pull": (
                best_personal_alternative
            ),
            "selected_personal_mean_before_pull": personal_means.get(arm_id),
            "selected_observed_mean_before_pull": selected_observed_mean,
            "selected_evidence_mean_before_pull": selected_evidence_mean,
            "selected_estimated_advantage_vs_best_personal": (
                selected_evidence_mean - best_personal
                if selected_evidence_mean is not None and best_personal is not None
                else None
            ),
            "selected_estimated_advantage_vs_best_personal_alternative": (
                selected_evidence_mean - best_personal_alternative
                if selected_evidence_mean is not None
                and best_personal_alternative is not None
                else None
            ),
            "observed_payoff_advantage": (
                latest_observed_payoff - best_personal
                if latest_observed_payoff is not None
                and best_personal is not None
                else None
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["probes"] = [asdict(probe) for probe in self.probes]
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Rung2Agent:
        agent = cls(int(state["agent_id"]))
        records: list[tuple[int, int, Pull | Probe]] = []
        records.extend(
            (int(pull["round"]), 1, Pull(**pull)) for pull in state["pulls"]
        )
        records.extend(
            (int(probe["round"]), 0, Probe(**probe))
            for probe in state["probes"]
        )
        for _, kind, record in sorted(records, key=lambda item: (item[0], item[1])):
            if kind == 0:
                assert isinstance(record, Probe)
                agent.add_probe(record)
            else:
                assert isinstance(record, Pull)
                agent.add_pull(record)
        agent.observations = [
            Observation(**observation) for observation in state["observations"]
        ]
        agent.observation_attempts = [
            (int(round_index), int(target_id), bool(succeeded))
            for round_index, target_id, succeeded in state["observation_attempts"]
        ]
        agent.token_balance = int(state["token_balance"])
        return agent

    def render_rung2_state(
        self,
        *,
        round_index: int,
        num_agents: int,
        landscape: str,
        coordinate_scale: int,
        coordinate_innovation: bool,
        social_enabled: bool,
    ) -> str:
        personal = self.personal_payoff_evidence()
        probe_rewards: dict[int, list[float]] = defaultdict(list)
        pull_rewards: dict[int, list[float]] = defaultdict(list)
        for probe in self.probes:
            probe_rewards[probe.arm_id].append(probe.reward)
        for pull in self.pulls:
            pull_rewards[pull.arm_id].append(pull.reward)
        social_rewards: dict[int, list[float]] = defaultdict(list)
        sources: dict[int, list[str]] = defaultdict(list)
        for observation in self.observations:
            sources[observation.arm_id].append(
                f"agent {observation.target_id} in source round "
                f"{observation.source_round} (bought {observation.acquired_round})"
            )
            if observation.reward is not None:
                social_rewards[observation.arm_id].append(observation.reward)

        arm_lines = []
        for arm_id in sorted(self.known_arm_ids()):
            evidence = personal.get(arm_id, []) + social_rewards.get(arm_id, [])
            probe_values = [
                round(value, 4) for value in probe_rewards.get(arm_id, [])
            ]
            pull_values = [
                round(value, 4) for value in pull_rewards.get(arm_id, [])
            ]
            observed_values = [
                round(value, 4) for value in social_rewards.get(arm_id, [])
            ]
            location = (
                f", x={arm_id / coordinate_scale:.9f}"
                if coordinate_innovation
                else ""
            )
            empirical = (
                f"{float(np.mean(evidence)):.4f}" if evidence else "unknown"
            )
            arm_lines.append(
                f"arm {arm_id}{location}: probes={probe_values}, "
                f"scored_pulls={pull_values}, "
                f"observed={observed_values}, empirical_mean={empirical}, "
                f"observed_from={sources.get(arm_id, [])}"
            )
        if not arm_lines:
            arm_lines = ["none; you cannot PULL until you learn an arm"]

        failed = [
            f"round {attempt_round}: agent {target_id} had no prior pull"
            for attempt_round, target_id, succeeded in self.observation_attempts
            if not succeeded
        ]
        social_state = ""
        if social_enabled:
            targets = ", ".join(
                str(agent_id)
                for agent_id in range(num_agents)
                if agent_id != self.agent_id
            )
            social_state = (
                f"Valid observation targets: {targets}\n\n"
                "Observations that returned nothing:\n"
                + ("\n".join(failed) if failed else "none")
            )
        return (
            f"Round: {round_index}\n"
            f"Your agent ID: {self.agent_id}\n"
            f"Completion tokens remaining this round: {self.token_balance}\n\n"
            "Your known arms and evidence:\n"
            + "\n".join(arm_lines)
            + ("\n\n" + social_state if social_state else "")
        )


class OpenRouterDecisionEngine:
    """OpenRouter-backed decision engine with the same budget semantics as vLLM."""

    def __init__(self, model_config: dict[str, Any]) -> None:
        import os
        from pathlib import Path
        from openrouter import OpenRouterClient

        cache = os.environ.get("AGENT_MARKET_API_CACHE_DIR")
        self.model_config = model_config
        self.client = OpenRouterClient(model_config, Path(cache) if cache else None)

    def generate(self, requests: list[dict[str, Any]]) -> list[Generation]:
        bodies, limits = [], []
        for request in requests:
            requested = int(request["max_tokens"])
            effective = min(requested, int(self.model_config.get("max_call_tokens", requested)))
            body = {
                "messages": [
                    {"role": "system", "content": request["system_prompt"]},
                    {"role": "user", "content": request["state"]},
                ],
                "max_tokens": effective,
                "temperature": self.model_config.get("temperature", 0.6),
                "seed": request["seed"],
            }
            for key in ("top_p", "top_k", "min_p", "reasoning"):
                if key in self.model_config:
                    body[key] = self.model_config[key]
            body.update(self.model_config.get("request_options", {}))
            body.pop("chat_template_kwargs", None)
            body["max_tokens"] = effective
            bodies.append(body)
            limits.append((requested, effective))

        responses = self.client.chat_completions(bodies)
        generations = []
        for response, (requested, effective) in zip(responses, limits, strict=True):
            usage = response.get("usage") or {}
            if "completion_tokens" not in usage or "prompt_tokens" not in usage:
                raise RuntimeError("OpenRouter must report actual prompt and completion usage")
            completion_tokens = usage["completion_tokens"]
            prompt_tokens = usage["prompt_tokens"]
            if (type(completion_tokens) is not int or type(prompt_tokens) is not int
                    or not 0 < completion_tokens <= effective or prompt_tokens < 0):
                raise RuntimeError("OpenRouter reported invalid usage or exceeded the call budget")
            message = response["choices"][0]["message"]
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            final_text = terminal_final_line(content)
            if is_degenerate_completion(content):
                raise RuntimeError("inference health failure: degenerate repeated-token completion")
            details = usage.get("completion_tokens_details") or {}
            generations.append(Generation(
                text=content,
                final_text=final_text,
                token_count=int(usage["completion_tokens"]),
                finish_reason=response["choices"][0].get("finish_reason"),
                action=parse_action(final_text),
                requested_max_tokens=requested,
                effective_max_tokens=effective,
                prompt_tokens=int(usage["prompt_tokens"]),
                context_limited=False,
                cost_usd=float(usage.get("cost", 0.0) or 0.0),
                reasoning_tokens=int(details.get("reasoning_tokens", 0) or 0),
            ))
        return generations


class VLLMDecisionEngine:
    """Thin offline vLLM wrapper; imports vLLM only for LLM conditions."""

    def __new__(cls, model_config: dict[str, Any]):
        if model_config.get("provider", "vllm") == "openrouter":
            return OpenRouterDecisionEngine(model_config)
        return super().__new__(cls)

    def __init__(self, model_config: dict[str, Any]) -> None:
        from vllm import LLM

        self.model_config = model_config
        engine_kwargs = {
            "model": model_config["name"],
            "tensor_parallel_size": model_config["tensor_parallel_size"],
            "dtype": model_config.get("dtype", "bfloat16"),
            "max_model_len": model_config.get("max_model_len", 32768),
            "gpu_memory_utilization": model_config.get("gpu_memory_utilization", 0.90),
            "trust_remote_code": model_config.get("trust_remote_code", False),
        }
        if "limit_mm_per_prompt" in model_config:
            engine_kwargs["limit_mm_per_prompt"] = model_config["limit_mm_per_prompt"]
        for key in ("tokenizer_mode", "config_format", "load_format", "language_model_only"):
            if key in model_config:
                engine_kwargs[key] = model_config[key]
        self.llm = LLM(
            **engine_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.is_gpt_oss = self.model_config["name"].startswith("openai/gpt-oss")
        self.harmony_encoding = None
        if self.is_gpt_oss:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding

            self.harmony_encoding = load_harmony_encoding(
                HarmonyEncodingName.HARMONY_GPT_OSS
            )

    def _final_text(self, token_ids: list[int], decoded_text: str) -> str:
        if not self.is_gpt_oss:
            return terminal_final_line(decoded_text)

        from openai_harmony import Role, TextContent

        assert self.harmony_encoding is not None
        try:
            messages = self.harmony_encoding.parse_messages_from_completion_tokens(
                token_ids, role=Role.ASSISTANT, strict=False
            )
        except Exception:
            return ""
        final_messages = [message for message in messages if message.channel == "final"]
        if not final_messages:
            return ""
        final_text = "\n".join(
            "".join(
                content.text
                for content in message.content
                if isinstance(content, TextContent)
            )
            for message in final_messages
        ).strip()
        return terminal_final_line(final_text)

    def generate(
        self,
        requests: list[dict[str, Any]],
    ) -> list[Generation]:
        from vllm import SamplingParams

        prompts: list[Any] = []
        params: list[SamplingParams] = []
        request_limits: list[tuple[int, int, int, bool]] = []
        for request in requests:
            chat_template_kwargs = self.model_config.get(
                "chat_template_kwargs", {"enable_thinking": True}
            )
            messages = [
                {"role": "system", "content": request["system_prompt"]},
                {"role": "user", "content": request["state"]},
            ]
            if self.model_config.get("tokenizer_mode") == "mistral":
                prompt_token_ids = list(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )
                )
                prompts.append({"prompt_token_ids": prompt_token_ids})
                prompt_tokens = len(prompt_token_ids)
            else:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
                prompts.append(prompt)
                prompt_tokens = len(
                    self.tokenizer.encode(prompt, add_special_tokens=False)
                )
            requested_max_tokens = int(request["max_tokens"])
            context_capacity = int(
                self.model_config.get("max_model_len", 32768)
            ) - prompt_tokens
            if context_capacity < 1:
                raise RuntimeError(
                    "inference prompt leaves no completion-token capacity"
                )
            effective_max_tokens = min(requested_max_tokens, context_capacity)
            context_limited = effective_max_tokens < requested_max_tokens
            request_limits.append(
                (
                    requested_max_tokens,
                    effective_max_tokens,
                    prompt_tokens,
                    context_limited,
                )
            )
            params.append(
                SamplingParams(
                    temperature=self.model_config.get("temperature", 0.6),
                    top_p=self.model_config.get("top_p", 0.95),
                    top_k=self.model_config.get("top_k", 20),
                    min_p=self.model_config.get("min_p", 0.0),
                    max_tokens=effective_max_tokens,
                    min_tokens=1,
                    seed=request["seed"],
                )
            )

        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        generations: list[Generation] = []
        for output, limits in zip(outputs, request_limits, strict=True):
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            final_text = self._final_text(token_ids, completion.text)
            if is_degenerate_token_ids(token_ids) or is_degenerate_completion(
                completion.text
            ):
                raise RuntimeError(
                    "inference health failure: degenerate repeated-token completion: "
                    + completion_preview(completion.text)
                )
            generations.append(
                Generation(
                    text=completion.text,
                    final_text=final_text,
                    token_count=len(completion.token_ids),
                    finish_reason=completion.finish_reason,
                    action=parse_action(final_text),
                    requested_max_tokens=limits[0],
                    effective_max_tokens=limits[1],
                    prompt_tokens=limits[2],
                    context_limited=limits[3],
                )
            )
        return generations


class UCBAgent:
    """Gaussian UCB with independent, seeded tie-breaking per agent."""

    def __init__(self, *, agent_id: int, num_arms: int, seed: int) -> None:
        self.agent_id = agent_id
        self.counts = np.zeros(num_arms, dtype=np.int64)
        self.reward_sums = np.zeros(num_arms, dtype=np.float64)
        self.total_pulls = 0
        self.rng = np.random.default_rng(
            np.random.SeedSequence([seed, 2, agent_id])
        )

    def choose(self) -> int:
        unpulled = np.flatnonzero(self.counts == 0)
        if len(unpulled):
            return int(self.rng.choice(unpulled))
        means = self.reward_sums / self.counts
        bonus = np.sqrt(2.0 * np.log(self.total_pulls) / self.counts)
        indices = means + bonus
        maxima = np.flatnonzero(np.isclose(indices, np.max(indices)))
        return int(self.rng.choice(maxima))

    def update(self, arm_id: int, reward: float) -> None:
        self.counts[arm_id] += 1
        self.reward_sums[arm_id] += reward
        self.total_pulls += 1

    def pull_label(self, arm_id: int) -> str:
        observed = self.counts > 0
        if not np.any(observed):
            return "explore"
        means = np.divide(
            self.reward_sums,
            self.counts,
            out=np.full_like(self.reward_sums, -np.inf),
            where=observed,
        )
        return (
            "exploit"
            if observed[arm_id] and math.isclose(means[arm_id], np.max(means))
            else "explore"
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "counts": self.counts.tolist(),
            "reward_sums": self.reward_sums.tolist(),
            "total_pulls": self.total_pulls,
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, Any], *, num_arms: int, seed: int
    ) -> UCBAgent:
        agent = cls(
            agent_id=int(state["agent_id"]), num_arms=num_arms, seed=seed
        )
        agent.counts = np.asarray(state["counts"], dtype=np.int64)
        agent.reward_sums = np.asarray(state["reward_sums"], dtype=np.float64)
        if len(agent.counts) != num_arms or len(agent.reward_sums) != num_arms:
            raise ValueError("UCB checkpoint arm count does not match config")
        agent.total_pulls = int(state["total_pulls"])
        agent.rng.bit_generator.state = state["rng_state"]
        return agent
