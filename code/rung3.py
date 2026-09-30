"""Rung 3: social learning over a fixed menu of job-sequencing skills."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from agent import (
    VLLMDecisionEngine,
    budget_mode,
    completion_preview,
    independent_search_assigned,
    intervention_settings,
    render_experiment_prompt,
    stable_seed,
)
from env import Pull


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_ACTION = re.compile(
    r"(?:^|\n)\s*FINAL:\s*SCHEDULE\s+([0-9]+(?:\s*,\s*[0-9]+)*)\s*\Z",
    re.IGNORECASE,
)
SOCIAL_LEVELS = {"none", "id", "payoff", "reputation", "full"}


class EventSink(Protocol):
    def write(self, event: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...


@dataclass(frozen=True)
class SkillSpec:
    skill_id: int
    path: str
    description: str
    intended_rank: int
    body: str
    sha256: str


@dataclass(frozen=True)
class SkillObservation:
    observation_id: str
    acquired_round: int
    target_id: int
    source_round: int
    skill_id: int
    latest_reward: float | None = None
    description: str | None = None
    global_use_count: int | None = None
    adoption_count: int | None = None
    holder_count: int | None = None
    global_mean_reward: float | None = None
    global_reward_count: int | None = None
    target_reward_history: tuple[float, ...] | None = None
    skill_body: str | None = None

    def visible_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


SKILL_MANIFEST = [
    (0, "skill_0.md", "A direct-output scheduling procedure.", 0),
    (1, "skill_1.md", "A profit-oriented selection procedure.", 1),
    (2, "skill_2.md", "An input-order scheduling procedure.", 2),
    (3, "skill_3.md", "A deadline-prioritized scheduling procedure.", 3),
    (4, "skill_4.md", "A profit-prioritized slotting procedure.", 4),
    (5, "skill_5.md", "A profit-and-deadline heuristic with checks.", 5),
    (6, "skill_6.md", "A latest-slot profit scheduling procedure.", 6),
    (7, "skill_7.md", "A verified latest-slot profit scheduling procedure.", 7),
]


def load_skills() -> dict[int, SkillSpec]:
    skills: dict[int, SkillSpec] = {}
    for skill_id, filename, description, intended_rank in SKILL_MANIFEST:
        relative = f"skills/job_sequencing/{filename}"
        body = (ROOT / relative).read_text(encoding="utf-8").strip()
        skills[skill_id] = SkillSpec(
            skill_id=skill_id,
            path=relative,
            description=description,
            intended_rank=intended_rank,
            body=body,
            sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )
    return skills


def parse_schedule(text: str) -> list[int] | None:
    match = SCHEDULE_ACTION.search(text)
    if match is None:
        return None
    return [int(value.strip()) for value in match.group(1).split(",")]


def balanced_initial_skills(
    *, seed: int, num_agents: int, num_skills: int
) -> list[int]:
    if num_agents < num_skills:
        raise ValueError("num_agents must be at least num_skills")
    rng = np.random.default_rng(stable_seed(seed, "rung3_initial_skills"))
    shuffled_agents = list(map(int, rng.permutation(num_agents)))
    assignments = [-1] * num_agents
    for position, agent_id in enumerate(shuffled_agents):
        assignments[agent_id] = position % num_skills
    return assignments


class Rung3Environment:
    def __init__(
        self,
        *,
        seed: int,
        num_agents: int,
        num_jobs: int,
        max_deadline: int,
        profit_min: int,
        profit_max: int,
    ) -> None:
        if num_jobs < max_deadline:
            raise ValueError("num_jobs must be at least max_deadline")
        if profit_min <= 0 or profit_max < profit_min:
            raise ValueError("profits must be positive and ordered")
        self.seed = seed
        self.num_agents = num_agents
        self.num_jobs = num_jobs
        self.max_deadline = max_deadline
        self.profit_min = profit_min
        self.profit_max = profit_max
        self.skills = load_skills()
        self.num_arms = len(self.skills)

    def task(self, *, round_index: int, agent_id: int) -> list[dict[str, int]]:
        rng = np.random.default_rng(
            stable_seed(self.seed, "rung3_task", round_index, agent_id)
        )
        jobs = [
            {
                "id": job_id,
                "deadline": int(rng.integers(1, self.max_deadline + 1)),
                "profit": int(rng.integers(self.profit_min, self.profit_max + 1)),
            }
            for job_id in range(self.num_jobs)
        ]
        order = rng.permutation(self.num_jobs)
        return [jobs[int(index)] for index in order]

    @staticmethod
    def optimal_schedule(jobs: list[dict[str, int]]) -> list[int]:
        max_deadline = max((job["deadline"] for job in jobs), default=0)
        slots: list[int | None] = [None] * max_deadline
        for job in sorted(jobs, key=lambda value: (-value["profit"], value["id"])):
            for slot in range(min(job["deadline"], max_deadline) - 1, -1, -1):
                if slots[slot] is None:
                    slots[slot] = job["id"]
                    break
        return [job_id for job_id in slots if job_id is not None]

    @classmethod
    def score_schedule(
        cls, jobs: list[dict[str, int]], schedule: list[int] | None
    ) -> dict[str, Any]:
        by_id = {job["id"]: job for job in jobs}
        optimum = cls.optimal_schedule(jobs)
        optimal_profit = sum(by_id[job_id]["profit"] for job_id in optimum)
        error: str | None = None
        if schedule is None:
            error = "missing terminal schedule"
        elif len(schedule) != len(set(schedule)):
            error = "duplicate job ID"
        elif any(job_id not in by_id for job_id in schedule):
            error = "unknown job ID"
        elif any(slot > by_id[job_id]["deadline"] for slot, job_id in enumerate(schedule, 1)):
            error = "deadline violation"

        achieved_profit = (
            0 if error else sum(by_id[job_id]["profit"] for job_id in schedule or [])
        )
        reward = achieved_profit / optimal_profit if optimal_profit else 0.0
        return {
            "valid": error is None,
            "error": error,
            "schedule": schedule,
            "achieved_profit": achieved_profit,
            "optimal_profit": optimal_profit,
            "optimal_schedule": optimum,
            "reward": reward,
            "expected_regret": 1.0 - reward,
        }

    def description(self) -> dict[str, Any]:
        return {
            "task": "unit_time_job_sequencing_with_deadlines",
            "seed": self.seed,
            "num_agents": self.num_agents,
            "num_jobs": self.num_jobs,
            "max_deadline": self.max_deadline,
            "profit_min": self.profit_min,
            "profit_max": self.profit_max,
            "skill_manifest": [
                {
                    "skill_id": skill.skill_id,
                    "path": skill.path,
                    "description": skill.description,
                    "intended_rank": skill.intended_rank,
                    "sha256": skill.sha256,
                }
                for skill in self.skills.values()
            ],
        }


class Rung3Agent:
    def __init__(self, agent_id: int, initial_skill_id: int) -> None:
        self.agent_id = agent_id
        self.initial_skill_id = initial_skill_id
        self.owned_skill_ids = {initial_skill_id}
        self.adoptions: dict[int, dict[str, Any]] = {}
        self.independent_acquisitions: dict[int, int] = {}
        self.pulls: list[Pull] = []
        self.observations: list[SkillObservation] = []
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
            raise RuntimeError("rung 3 token budget conservation failed")
        self.token_balance -= token_count

    def add_pull(self, pull: Pull) -> None:
        self.pulls.append(pull)
        if pull.reward > self.personal_best_reward:
            self.personal_best_reward = pull.reward
            self.last_improvement_round = pull.round

    def add_observation(
        self,
        *,
        round_index: int,
        target_id: int,
        observation: SkillObservation | None,
    ) -> None:
        self.observation_attempts.append(
            (round_index, target_id, observation is not None)
        )
        if observation is not None:
            self.observations.append(observation)

    def latest_candidates(self) -> dict[int, SkillObservation]:
        candidates: dict[int, SkillObservation] = {}
        for observation in self.observations:
            if observation.skill_id not in self.owned_skill_ids:
                candidates[observation.skill_id] = observation
        return candidates

    def available_skill_ids(self) -> set[int]:
        return self.owned_skill_ids | set(self.latest_candidates())

    def acquire_independently(self, skill_id: int, round_index: int, num_skills: int) -> None:
        if not 0 <= skill_id < num_skills:
            raise ValueError("invalid skill ID")
        if skill_id in self.available_skill_ids():
            raise ValueError("skill is already owned or socially observed")
        self.owned_skill_ids.add(skill_id)
        self.independent_acquisitions[skill_id] = round_index

    def acquisition_source(self, skill_id: int) -> str:
        if skill_id in self.adoptions:
            return "social"
        if skill_id in self.independent_acquisitions:
            return "independent"
        return "initial"

    def personal_rewards(self) -> dict[int, list[float]]:
        rewards: dict[int, list[float]] = defaultdict(list)
        for pull in self.pulls:
            rewards[pull.arm_id].append(pull.reward)
        return dict(rewards)

    def evidence_mean(self, skill_id: int) -> float | None:
        evidence = list(self.personal_rewards().get(skill_id, []))
        for observation in self.observations:
            if observation.skill_id != skill_id:
                continue
            if observation.global_mean_reward is not None:
                evidence.append(observation.global_mean_reward)
            elif observation.latest_reward is not None:
                evidence.append(observation.latest_reward)
        return float(np.mean(evidence)) if evidence else None

    def pull_label(self, skill_id: int) -> str:
        means = {
            candidate: mean
            for candidate in self.available_skill_ids()
            if (mean := self.evidence_mean(candidate)) is not None
        }
        if not means:
            return "explore"
        best = max(means.values())
        return (
            "exploit"
            if skill_id in means
            and math.isclose(means[skill_id], best, rel_tol=1e-12)
            else "explore"
        )

    def decision_features(self, round_index: int) -> dict[str, Any]:
        means = [
            mean
            for skill_id in self.available_skill_ids()
            if (mean := self.evidence_mean(skill_id)) is not None
        ]
        best_known = max(means) if means else None
        last_reward = self.pulls[-1].reward if self.pulls else None
        return {
            "best_known_mean": best_known,
            "last_reward": last_reward,
            "last_reward_gap": (
                last_reward - best_known
                if last_reward is not None and best_known is not None
                else None
            ),
            "rounds_since_improvement": (
                round_index - self.last_improvement_round
                if self.last_improvement_round is not None
                else None
            ),
            "owned_skill_count": len(self.owned_skill_ids),
            "candidate_skill_count": len(self.latest_candidates()),
            "personal_pull_count": len(self.pulls),
            "social_observation_count": len(self.observations),
            "social_observation_attempt_count": len(self.observation_attempts),
        }

    def adopt(self, skill_id: int, round_index: int) -> tuple[bool, dict[str, Any] | None]:
        if skill_id in self.owned_skill_ids:
            return False, None
        supporting = [obs for obs in self.observations if obs.skill_id == skill_id]
        if not supporting:
            raise ValueError("cannot adopt an unobserved skill")
        latest = supporting[-1]
        cue = {
            **latest.visible_dict(),
            "source_observation_id": latest.observation_id,
            "copy_support_count": len(supporting),
            "copy_source_count": len({obs.target_id for obs in supporting}),
            "copy_recency": round_index - latest.acquired_round,
        }
        self.owned_skill_ids.add(skill_id)
        self.adoptions[skill_id] = {
            "round": round_index,
            "source_observation_id": latest.observation_id,
            "cue": cue,
        }
        return True, cue

    def render_state(
        self,
        *,
        round_index: int,
        num_agents: int,
        skills: dict[int, SkillSpec],
        social_enabled: bool,
        independent_acquisition: bool = False,
    ) -> str:
        personal = self.personal_rewards()
        owned_lines = []
        for skill_id in sorted(self.owned_skill_ids):
            values = personal.get(skill_id, [])
            mean = f"{float(np.mean(values)):.4f}" if values else "unknown"
            owned_lines.append(
                f"skill {skill_id}; personal_rewards={[round(x, 4) for x in values]}; "
                f"personal_mean={mean}\n{skills[skill_id].body}"
            )
        candidate_lines = [
            json.dumps(observation.visible_dict(), sort_keys=True)
            for observation in self.latest_candidates().values()
        ]
        social_state = ""
        if social_enabled:
            targets = ", ".join(
                str(agent_id)
                for agent_id in range(num_agents)
                if agent_id != self.agent_id
            )
            social_state = (
                f"Valid observation targets: {targets}\n\nObserved candidate cards:\n"
                + ("\n".join(candidate_lines) if candidate_lines else "none")
            )
        acquisition_state = (
            "\n\nUnowned and unobserved skill IDs available for independent PULL: "
            + ", ".join(str(k) for k in sorted(set(skills) - self.available_skill_ids()))
            if independent_acquisition else ""
        )
        return (
            f"Round: {round_index}\n"
            f"Your agent ID: {self.agent_id}\n"
            f"Completion tokens remaining this round: {self.token_balance}\n\n"
            "Owned skills and personal evidence:\n"
            + "\n---\n".join(owned_lines)
            + ("\n\n" + social_state if social_state else "")
            + acquisition_state
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "initial_skill_id": self.initial_skill_id,
            "owned_skill_ids": sorted(self.owned_skill_ids),
            "adoptions": {str(key): value for key, value in self.adoptions.items()},
            "independent_acquisitions": {str(key): value for key, value in self.independent_acquisitions.items()},
            "pulls": [asdict(pull) for pull in self.pulls],
            "observations": [asdict(observation) for observation in self.observations],
            "observation_attempts": [list(value) for value in self.observation_attempts],
            "token_balance": self.token_balance,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Rung3Agent:
        agent = cls(int(state["agent_id"]), int(state["initial_skill_id"]))
        agent.owned_skill_ids = {int(value) for value in state["owned_skill_ids"]}
        agent.adoptions = {int(key): value for key, value in state["adoptions"].items()}
        agent.independent_acquisitions = {int(key): int(value) for key, value in state.get("independent_acquisitions", {}).items()}
        for payload in state["pulls"]:
            agent.add_pull(Pull(**payload))
        agent.observations = [
            SkillObservation(
                **{
                    **payload,
                    "target_reward_history": (
                        tuple(payload["target_reward_history"])
                        if payload.get("target_reward_history") is not None
                        else None
                    ),
                }
            )
            for payload in state["observations"]
        ]
        agent.observation_attempts = [
            (int(round_index), int(target_id), bool(succeeded))
            for round_index, target_id, succeeded in state["observation_attempts"]
        ]
        agent.token_balance = int(state["token_balance"])
        return agent


def make_environment(config: dict[str, Any], seed: int) -> Rung3Environment:
    environment = config["environment"]
    return Rung3Environment(
        seed=seed,
        num_agents=environment["num_agents"],
        num_jobs=environment["num_jobs"],
        max_deadline=environment["max_deadline"],
        profit_min=environment["profit_min"],
        profit_max=environment["profit_max"],
    )


def common_event(
    *, config: dict[str, Any], seed: int, round_index: int, agent_id: int
) -> dict[str, Any]:
    return {
        "rung": 3,
        "condition": config["condition"],
        "strategy": config["strategy"],
        "social_info": config["social_info"],
        "model_label": config["model_label"],
        "model_name": config["model"]["name"],
        "policy": config["policy"],
        "seed": seed,
        "round": round_index,
        "agent_id": agent_id,
        "budget_mode": budget_mode(config),
        "guidance": config.get("guidance", "neutral"),
    }


def population_snapshot(
    agents: list[Rung3Agent], skills: dict[int, SkillSpec], social_info: str
) -> tuple[list[Pull | None], dict[int, dict[str, Any]]]:
    latest = [agent.pulls[-1] if agent.pulls else None for agent in agents]
    stats: dict[int, dict[str, Any]] = {}
    for skill_id, skill in skills.items():
        rewards = [
            pull.reward for agent in agents for pull in agent.pulls if pull.arm_id == skill_id
        ]
        stats[skill_id] = {
            "description": skill.description,
            "global_use_count": len(rewards),
            "adoption_count": sum(skill_id in agent.adoptions for agent in agents),
            "holder_count": sum(skill_id in agent.owned_skill_ids for agent in agents),
            "global_mean_reward": float(np.mean(rewards)) if rewards else None,
            "global_reward_count": len(rewards),
            "skill_body": skill.body if social_info == "full" else None,
        }
    return latest, stats


def make_observation(
    *,
    observer: Rung3Agent,
    target: Rung3Agent,
    latest_pull: Pull | None,
    stats: dict[int, dict[str, Any]],
    round_index: int,
    social_info: str,
) -> SkillObservation | None:
    if latest_pull is None:
        return None
    skill_id = latest_pull.arm_id
    population = stats[skill_id]
    target_history = tuple(
        pull.reward for pull in target.pulls if pull.arm_id == skill_id
    )
    observation_id = (
        f"r{round_index}-a{observer.agent_id}-o{len(observer.observations)}"
    )
    return SkillObservation(
        observation_id=observation_id,
        acquired_round=round_index,
        target_id=target.agent_id,
        source_round=latest_pull.round,
        skill_id=skill_id,
        latest_reward=(
            latest_pull.reward
            if social_info in {"payoff", "reputation", "full"}
            else None
        ),
        description=(
            population["description"] if social_info in {"reputation", "full"} else None
        ),
        global_use_count=(
            population["global_use_count"]
            if social_info in {"reputation", "full"}
            else None
        ),
        adoption_count=(
            population["adoption_count"]
            if social_info in {"reputation", "full"}
            else None
        ),
        holder_count=(
            population["holder_count"]
            if social_info in {"reputation", "full"}
            else None
        ),
        global_mean_reward=(
            population["global_mean_reward"]
            if social_info in {"reputation", "full"}
            else None
        ),
        global_reward_count=(
            population["global_reward_count"]
            if social_info in {"reputation", "full"}
            else None
        ),
        target_reward_history=target_history if social_info == "full" else None,
        skill_body=population["skill_body"] if social_info == "full" else None,
    )


def policy_prompt(config: dict[str, Any]) -> str:
    template = render_experiment_prompt(
        (ROOT / config["prompt"]).read_text(encoding="utf-8").strip(),
        config,
    )
    if config["social_info"] == "none":
        social_rule = "OBSERVE is unavailable. You may only use your owned skill."
        action_lines = "FINAL: PULL <skill_id>"
    else:
        social_rule = (
            "OBSERVE buys the configured information about a chosen agent's most "
            "recent skill use strictly before this round. An observed candidate can "
            "later be adopted by pulling it."
        )
        action_lines = "FINAL: OBSERVE <agent_id>\nFINAL: PULL <skill_id>"
    template = template.replace("{{SOCIAL_RULE}}", social_rule).replace(
        "{{ACTION_LINES}}", action_lines
    )
    if config.get("independent_skill_acquisition", False):
        template = template.replace(
            "OBSERVE is unavailable. You may only use your owned skill.",
            "OBSERVE is unavailable. You may independently acquire and use any skill.",
        )
        rule = (
            "The fixed skill menu has IDs 0 through 7. PULL <skill_id> may select any "
            "of these IDs, including an unowned and unobserved skill. Selecting such a "
            "skill independently acquires it and applies it to this round's private task. "
            "You see its contents when it is applied, and retain it for later rounds. "
            "No quality information about unseen skills is supplied before selection. "
            "Acquisition has no separate fee: the selection and solution completion tokens "
            "are charged to the same round balance as any other PULL. Only task execution "
            "earns reward. Observation is optional wherever available.\n\n"
        )
        template = template.replace("Choose exactly one action now.", rule + "Choose exactly one action now.")
    return template


def solution_state(skill: SkillSpec, jobs: list[dict[str, int]]) -> str:
    return (
        "Apply the selected skill to this unit-time job-sequencing instance. "
        "A job placed at position t runs in slot t and must have deadline >= t. "
        "Each job may appear at most once. Maximize total profit.\n\n"
        f"Selected skill {skill.skill_id}:\n{skill.body}\n\n"
        f"Jobs:\n{json.dumps(jobs, sort_keys=True)}\n\n"
        "End with exactly one terminal line and nothing after it:\n"
        "FINAL: SCHEDULE <comma-separated job IDs>"
    )


def generation_fields(generation: Any) -> dict[str, Any]:
    return {
        "completion_tokens": generation.token_count,
        "prompt_tokens": generation.prompt_tokens,
        "cost_usd": generation.cost_usd,
        "reasoning_tokens": generation.reasoning_tokens,
        "requested_max_tokens": generation.requested_max_tokens,
        "effective_max_tokens": generation.effective_max_tokens,
        "context_limited": generation.context_limited,
        "finish_reason": generation.finish_reason,
    }


def replay_choices(config: dict[str, Any], env: Rung3Environment) -> tuple[dict, str]:
    """Replay recorded selections, never invent a choice for a missed selection."""
    source = Path(config["source_run"])
    original = json.loads((source / "config.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    if not summary.get("completed") or int(original.get("rung", 1)) != 3:
        raise ValueError("skill replay requires a completed rung 3 source")
    if original["model"] != config["model"]:
        raise ValueError("skill replay requires the same executor model configuration")
    if json.loads((source / "environment.json").read_text()) != env.description():
        raise ValueError("skill replay task seed, parameters, or skill hashes do not match")
    if config["environment"]["rounds"] > original["environment"]["rounds"]:
        raise ValueError("skill replay horizon exceeds source")
    raw = (source / "events.jsonl").read_bytes()
    choices = {}
    for line in raw.splitlines():
        event = json.loads(line)
        if event["event"] != "solution":
            continue
        if event["seed"] != env.seed or not 0 <= event["agent_id"] < env.num_agents:
            raise ValueError("source selection does not match replay population")
        key = (event["round"], event["agent_id"])
        if key in choices or event["skill_id"] not in env.skills:
            raise ValueError("duplicate or invalid source skill selection")
        choices[key] = event
    return choices, hashlib.sha256(raw).hexdigest()


def run_condition(
    *,
    config: dict[str, Any],
    seed: int,
    env: Rung3Environment,
    writer: EventSink,
    checkpoint: dict[str, Any] | None,
    checkpoint_writer: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    rounds = config["environment"]["rounds"]
    tokens_per_round = config["budget"]["tokens_per_round"]
    carry_over = bool(config["budget"].get("carry_over", True))
    social_info = config["social_info"]
    is_oracle = config["policy"] == "skill_oracle"
    is_deoe = config["policy"] in {"solo_deoe", "social_deoe"}
    is_replay = config["policy"] == "skill_replay"
    execution_replicate = config.get("execution_replicate", 0)
    if (not isinstance(execution_replicate, int) or execution_replicate < 0 or
            (execution_replicate and not is_replay)):
        raise ValueError("execution_replicate must be a nonnegative integer for skill replay")
    independent_acquisition = bool(config.get("independent_skill_acquisition", False))
    search_probability, execution_reserve = intervention_settings(config)
    if search_probability and not independent_acquisition:
        raise ValueError("independent-search intervention requires skill acquisition")
    if (search_probability or execution_reserve) and (is_deoe or is_oracle or is_replay):
        raise ValueError("rung 3 causal interventions apply to LLM selectors")
    replay, source_hash = {}, None
    if is_replay:
        if carry_over or social_info != "none":
            raise ValueError("skill replay requires fixed expiring execution grants and social_info=none")
        replay, source_hash = replay_choices(config, env)
    if social_info not in SOCIAL_LEVELS:
        raise ValueError(f"unknown social information level: {social_info}")
    if is_deoe and (not independent_acquisition or
                    (config["policy"] == "solo_deoe") != (social_info == "none")):
        raise ValueError("DEOE requires independent skill acquisition and matching social access")

    if checkpoint is None:
        start_round = 0
        initial = balanced_initial_skills(
            seed=seed, num_agents=env.num_agents, num_skills=env.num_arms
        )
        if is_oracle:
            initial = [7] * env.num_agents
        agents = [
            Rung3Agent(agent_id, initial[agent_id])
            for agent_id in range(env.num_agents)
        ]
        total_reward = 0.0
        total_regret = 0.0
        missed_pulls = 0
        invalid_solutions = 0
        successful_observations = 0
        total_copies = 0
        context_limited_calls = 0
        allocation_totals: Counter[str] = Counter()
        inference_seconds: Counter[str] = Counter()
    else:
        if checkpoint.get("rung") != 3 or checkpoint.get("policy") != config["policy"]:
            raise ValueError("rung 3 checkpoint does not match config")
        if bool(checkpoint.get("independent_skill_acquisition", False)) != independent_acquisition:
            raise ValueError("rung 3 checkpoint acquisition mode does not match config")
        if checkpoint.get("interventions", {}) != config.get("interventions", {}):
            raise ValueError("rung 3 checkpoint interventions do not match config")
        if checkpoint.get("source_events_sha256") != source_hash:
            raise ValueError("rung 3 checkpoint replay source does not match")
        if checkpoint.get("execution_replicate", 0) != execution_replicate:
            raise ValueError("rung 3 checkpoint execution replicate does not match")
        start_round = int(checkpoint["next_round"])
        agents = [Rung3Agent.from_state_dict(value) for value in checkpoint["agents"]]
        if [agent.agent_id for agent in agents] != list(range(env.num_agents)):
            raise ValueError("rung 3 checkpoint agent IDs do not match config")
        metrics = checkpoint["metrics"]
        total_reward = float(metrics["total_reward"])
        total_regret = float(metrics["total_regret"])
        missed_pulls = int(metrics["missed_pulls"])
        invalid_solutions = int(metrics["invalid_solutions"])
        successful_observations = int(metrics["successful_observations"])
        total_copies = int(metrics["total_copies"])
        context_limited_calls = int(metrics["context_limited_calls"])
        allocation_totals = Counter(metrics["allocation_totals"])
        inference_seconds = Counter(metrics.get("inference_seconds_by_stage", {}))

    if not 0 <= start_round <= rounds:
        raise ValueError("rung 3 checkpoint round is outside the configured horizon")
    if tokens_per_round <= 0:
        raise ValueError("rung 3 tokens_per_round must be positive")

    # Reuse the same policy as rungs 1/2, with bounded scheduling rewards.
    # 0.5 is a conservative SD bound for [0,1], not an estimated hidden variance.
    # Reputation/full expose extra information that this reference does not use.
    policies = []
    if is_deoe:
        from social_baseline import DiscountedPolicy
        for agent in agents:
            policy = DiscountedPolicy(agent.agent_id, seed, env.num_agents,
                                      env.num_arms,
                                      "solo" if social_info == "none" else "social",
                                      noise_std=0.5)
            if checkpoint is None:
                for skill_id in agent.owned_skill_ids:
                    policy.acquire(skill_id, "initial")
            else:
                saved = checkpoint["deoe"][agent.agent_id]
                policy.origins = {int(k): v for k, v in saved["origins"].items()}
                policy.estimates = {int(k): tuple(v) for k, v in saved["estimates"].items()}
                policy.seen_samples = {tuple(v) for v in saved["seen_samples"]}
                policy.last_observe = saved["last_observe"]
                policy.needs_check = saved["needs_check"]
            policies.append(policy)
    system_prompt = policy_prompt(config) if not (is_deoe or is_replay) else ""
    engine = VLLMDecisionEngine(config["model"]) if start_round < rounds else None

    for round_index in range(start_round, rounds):
        assert engine is not None
        latest_snapshot, stats_snapshot = population_snapshot(
            agents, env.skills, social_info
        )
        opening = {
            agent.agent_id: agent.begin_round(
                tokens_per_round, carry_over=carry_over
            )
            for agent in agents
        }
        allocations = {agent.agent_id: Counter() for agent in agents}
        pulls_this_round: dict[int, dict[str, Any]] = {}
        search_assigned = {
            agent.agent_id: independent_search_assigned(config, seed, round_index, agent.agent_id)
            for agent in agents
        }
        fresh_skills = {
            agent.agent_id: sorted(set(env.skills) - agent.available_skill_ids())
            for agent in agents
        }

        if is_replay:
            selected = {}
            for agent in agents:
                source_choice = replay.get((round_index, agent.agent_id))
                if source_choice is None:
                    continue
                skill_id = source_choice["skill_id"]
                agent.owned_skill_ids.add(skill_id)
                selected[agent.agent_id] = dict(
                    skill_id=skill_id, pull_type="exploit", copied_this_pull=False,
                    copy_cue=None, policy_tokens=0, acquisition_source="replay")
        elif is_deoe:
            selected = {}
            for agent, policy in zip(agents, policies, strict=True):
                started = time.perf_counter()
                # Execute the balanced assigned skill first, then explore the
                # same fixed menu available to the acquisition-enabled LLM.
                action = policy.learning_action(round_index) if agent.pulls else "exploit"
                forced_skill = None
                observation = None
                target = None
                if action == "innovate":
                    forced_skill = policy.new_option(round_index)
                    policy.acquire(forced_skill, "independent")
                    agent.acquire_independently(forced_skill, round_index, env.num_arms)
                elif action == "observe" and env.num_agents > 1:
                    target = policy.target(round_index)
                    policy.last_observe = round_index
                    policy.needs_check = False
                    observation = make_observation(
                        observer=agent, target=agents[target],
                        latest_pull=latest_snapshot[target], stats=stats_snapshot,
                        round_index=round_index, social_info=social_info)
                    agent.add_observation(round_index=round_index, target_id=target,
                                          observation=observation)
                    successful_observations += observation is not None
                    if observation is not None:
                        policy.acquire(observation.skill_id, "social")
                        if observation.latest_reward is not None:
                            policy.update(observation.skill_id, observation.latest_reward,
                                          observation.source_round, target)
                    writer.write({
                        **common_event(config=config, seed=seed, round_index=round_index,
                                       agent_id=agent.agent_id),
                        "event": "decision", "stage": "policy", "wave": 0,
                        "parsed_action": "observe", "parsed_value": target,
                        "valid": True, "error": None, "allocation": "observe",
                        "observation": observation.visible_dict() if observation else None,
                        "completion_tokens": 0, "prompt_tokens": 0,
                        "tokens_before": agent.token_balance, "tokens_after": agent.token_balance,
                        "selection_policy": "discounted_eoe_v1",
                    })
                skill_id = forced_skill if forced_skill is not None else policy.best(round_index)
                pull_type = agent.pull_label(skill_id)
                copied, cue = agent.adopt(skill_id, round_index)
                total_copies += int(copied)
                selected[agent.agent_id] = dict(
                    skill_id=skill_id, pull_type=pull_type,
                    copied_this_pull=copied, copy_cue=cue, policy_tokens=0,
                    independently_acquired_this_pull=forced_skill is not None,
                    acquisition_source=agent.acquisition_source(skill_id))
                writer.write({
                    **common_event(config=config, seed=seed, round_index=round_index,
                                   agent_id=agent.agent_id),
                    "event": "decision", "stage": "policy", "wave": int(target is not None),
                    "parsed_action": "pull", "parsed_value": skill_id,
                    "learning_action": action, "target_id": target,
                    "observation": observation.visible_dict() if observation else None,
                    "valid": True, "error": None, "allocation": pull_type,
                    "completion_tokens": 0, "prompt_tokens": 0,
                    "tokens_before": agent.token_balance, "tokens_after": agent.token_balance,
                    "selection_seconds": time.perf_counter() - started,
                    "selection_policy": "discounted_eoe_v1",
                    **selected[agent.agent_id],
                })
        elif is_oracle:
            selected = {
                agent.agent_id: {
                    "skill_id": 7,
                    "pull_type": "exploit",
                    "copied_this_pull": False,
                    "copy_cue": None,
                    "policy_tokens": 0,
                }
                for agent in agents
            }
        else:
            selected: dict[int, dict[str, Any]] = {}
            active = set(range(env.num_agents))
            wave = 0
            while active:
                # Reserve guarantees an execution attempt. Exhausted selectors
                # use a predeclared personal-evidence fallback; record it so this
                # intervention is not mistaken for a pure change in token count.
                for agent_id in sorted(active):
                    agent = agents[agent_id]
                    if not execution_reserve or agent.token_balance > execution_reserve:
                        continue
                    forced_fresh = search_assigned[agent_id] and fresh_skills[agent_id]
                    if forced_fresh:
                        candidates = fresh_skills[agent_id]
                        skill_id = candidates[stable_seed(seed, "rung3_fallback", round_index, agent_id) % len(candidates)]
                        agent.acquire_independently(skill_id, round_index, env.num_arms)
                    else:
                        rewards = agent.personal_rewards()
                        skill_id = max(sorted(agent.owned_skill_ids),
                                       key=lambda k: np.mean(rewards[k]) if k in rewards else -1)
                    selected[agent_id] = dict(
                        skill_id=skill_id, pull_type=agent.pull_label(skill_id),
                        copied_this_pull=False, copy_cue=None, policy_tokens=0,
                        independently_acquired_this_pull=bool(forced_fresh),
                        acquisition_source=agent.acquisition_source(skill_id),
                        selection_fallback=True)
                    writer.write({
                        **common_event(config=config, seed=seed, round_index=round_index, agent_id=agent_id),
                        "event": "decision", "stage": "policy", "wave": wave,
                        "parsed_action": "pull", "parsed_value": skill_id,
                        "valid": True, "error": None, "completion_tokens": 0,
                        "tokens_before": agent.token_balance, "tokens_after": agent.token_balance,
                        "allocation": selected[agent_id]["pull_type"],
                        "execution_reserve_tokens": execution_reserve,
                        **selected[agent_id]})
                    active.remove(agent_id)
                active_ids = sorted(active)
                if not active_ids:
                    break
                requests = [
                    {
                        "system_prompt": system_prompt,
                        "state": agents[agent_id].render_state(
                            round_index=round_index,
                            num_agents=env.num_agents,
                            skills=env.skills,
                            social_enabled=config["social_info"] != "none",
                            independent_acquisition=independent_acquisition,
                        ) + (
                            "\nThis round requires independent search. Select PULL with one of "
                            f"these previously unacquired skill IDs: {fresh_skills[agent_id]}. "
                            "OBSERVE is unavailable this round.\n"
                            if search_assigned[agent_id] and fresh_skills[agent_id] else ""
                        ) + (
                            f"\nReserve {execution_reserve} tokens for task execution; "
                            "select a skill before the selection budget expires.\n"
                            if execution_reserve else ""
                        ),
                        "max_tokens": agents[agent_id].token_balance - execution_reserve,
                        "track_context_capacity": True,
                        "seed": stable_seed(
                            seed, config["condition"], round_index, wave, agent_id
                        ),
                    }
                    for agent_id in active_ids
                ]
                generation_started = time.perf_counter()
                generations = engine.generate(requests)
                batch_seconds = time.perf_counter() - generation_started
                inference_seconds["policy"] += batch_seconds
                for agent_id, generation in zip(active_ids, generations, strict=True):
                    agent = agents[agent_id]
                    tokens_before = agent.token_balance
                    agent.spend(generation.token_count)
                    context_limited_calls += int(generation.context_limited)
                    features = agent.decision_features(round_index)
                    action = generation.action
                    allocation = "invalid"
                    valid = False
                    error: str | None = None
                    extra: dict[str, Any] = {}
                    forced_fresh = search_assigned[agent_id] and fresh_skills[agent_id]
                    if action is not None and forced_fresh and (
                        action.kind != "pull" or action.value not in fresh_skills[agent_id]
                    ):
                        error = "this round requires pulling a previously unacquired skill"
                    elif action is None:
                        error = "missing final action"
                    elif action.kind == "observe":
                        if social_info == "none":
                            error = "OBSERVE is unavailable"
                        elif not 0 <= action.value < env.num_agents or action.value == agent_id:
                            error = "invalid observation target"
                        else:
                            observation = make_observation(
                                observer=agent,
                                target=agents[action.value],
                                latest_pull=latest_snapshot[action.value],
                                stats=stats_snapshot,
                                round_index=round_index,
                                social_info=social_info,
                            )
                            agent.add_observation(
                                round_index=round_index,
                                target_id=action.value,
                                observation=observation,
                            )
                            successful_observations += observation is not None
                            extra["observation"] = (
                                observation.visible_dict() if observation else None
                            )
                            allocation = "observe"
                            valid = True
                    elif action.kind == "pull":
                        newly_independent = False
                        if independent_acquisition and action.value in env.skills and action.value not in agent.available_skill_ids():
                            agent.acquire_independently(action.value, round_index, env.num_arms)
                            newly_independent = True
                        if action.value not in agent.available_skill_ids():
                            error = "skill is neither owned nor observed"
                        else:
                            pull_type = agent.pull_label(action.value)
                            copied, cue = agent.adopt(action.value, round_index)
                            total_copies += int(copied)
                            selected[agent_id] = {
                                "skill_id": action.value,
                                "pull_type": pull_type,
                                "copied_this_pull": copied,
                                "copy_cue": cue,
                                "policy_tokens": generation.token_count,
                                "independently_acquired_this_pull": newly_independent,
                                "acquisition_source": agent.acquisition_source(action.value),
                            }
                            allocation = pull_type
                            valid = True
                            active.remove(agent_id)
                            extra.update(selected[agent_id])
                    event = {
                        **common_event(
                            config=config,
                            seed=seed,
                            round_index=round_index,
                            agent_id=agent_id,
                        ),
                        "event": "decision",
                        "stage": "policy",
                        "wave": wave,
                        "final_output": completion_preview(generation.final_text),
                        "tokens_before": tokens_before,
                        "tokens_after": agent.token_balance,
                        "parsed_action": action.kind if action else None,
                        "parsed_value": action.value if action else None,
                        "valid": valid,
                        "error": error,
                        "allocation": allocation,
                        "independent_search_assigned": search_assigned[agent_id],
                        "independent_search_eligible": bool(fresh_skills[agent_id]),
                        "execution_reserve_tokens": execution_reserve,
                        "batch_generation_seconds": batch_seconds,
                        "batch_size": len(active_ids),
                        **generation_fields(generation),
                        **features,
                        **extra,
                    }
                    if not valid:
                        event["invalid_output_preview"] = completion_preview(generation.text)
                    writer.write(event)
                    allocations[agent_id][allocation] += generation.token_count
                    allocation_totals[allocation] += generation.token_count
                    if agent_id in active and agent.token_balance == 0:
                        active.remove(agent_id)
                wave += 1

        solve_ids = sorted(selected)
        solvable_ids = [
            agent_id for agent_id in solve_ids if agents[agent_id].token_balance > 0
        ]
        solve_requests = [
            {
                "system_prompt": (
                    "Solve the scheduling task using only the selected skill. "
                    "Return the required terminal schedule line."
                ),
                "state": solution_state(
                    env.skills[selected[agent_id]["skill_id"]],
                    env.task(round_index=round_index, agent_id=agent_id),
                ),
                "max_tokens": agents[agent_id].token_balance,
                "track_context_capacity": True,
                "seed": stable_seed(
                    config["model"]["name"],
                    seed,
                    "rung3_solution",
                    round_index,
                    agent_id,
                    selected[agent_id]["skill_id"],
                    *(["replay", execution_replicate] if execution_replicate else []),
                ),
            }
            for agent_id in solvable_ids
        ]
        generation_started = time.perf_counter()
        solve_generations = engine.generate(solve_requests) if solve_requests else []
        solve_seconds = time.perf_counter() - generation_started if solve_requests else 0.0
        inference_seconds["skill_execution"] += solve_seconds
        by_agent = dict(zip(solvable_ids, solve_generations, strict=True))
        requests_by_agent = dict(zip(solvable_ids, solve_requests, strict=True))

        for agent_id in solve_ids:
            agent = agents[agent_id]
            choice = selected[agent_id]
            skill_id = int(choice["skill_id"])
            pull_type = str(choice["pull_type"])
            task = env.task(round_index=round_index, agent_id=agent_id)
            generation = by_agent.get(agent_id)
            request = requests_by_agent.get(agent_id)
            tokens_before = agent.token_balance
            if generation is None:
                result = env.score_schedule(task, None)
                generation_payload = {
                    "completion_tokens": 0,
                    "prompt_tokens": None,
                    "requested_max_tokens": 0,
                    "effective_max_tokens": 0,
                    "context_limited": False,
                    "finish_reason": None,
                }
                final_output = ""
                invalid_preview = None
            else:
                agent.spend(generation.token_count)
                context_limited_calls += int(generation.context_limited)
                result = env.score_schedule(task, parse_schedule(generation.final_text))
                generation_payload = generation_fields(generation)
                final_output = completion_preview(generation.final_text)
                invalid_preview = (
                    None if result["valid"] else completion_preview(generation.text)
                )
                allocations[agent_id][pull_type] += generation.token_count
                allocation_totals[pull_type] += generation.token_count

            pull = Pull(round_index, agent_id, skill_id, float(result["reward"]))
            agent.add_pull(pull)
            if is_deoe:
                policies[agent_id].update(skill_id, pull.reward, round_index, agent_id)
            invalid_solutions += int(not result["valid"])
            total_reward += float(result["reward"])
            total_regret += float(result["expected_regret"])
            pulls_this_round[agent_id] = {
                "skill_id": skill_id,
                "pull_type": pull_type,
                "reward": result["reward"],
                "expected_regret": result["expected_regret"],
                "valid_solution": result["valid"],
                "copied_this_pull": choice["copied_this_pull"],
                "independently_acquired_this_pull": choice.get("independently_acquired_this_pull", False),
                "acquisition_source": choice.get("acquisition_source", "oracle"),
            }
            event = {
                **common_event(
                    config=config,
                    seed=seed,
                    round_index=round_index,
                    agent_id=agent_id,
                ),
                "event": "solution",
                "stage": "skill_execution",
                "executor_seed": request["seed"] if request else None,
                "executor_input_sha256": hashlib.sha256(json.dumps(
                    {key: request[key] for key in ("system_prompt", "state")},
                    sort_keys=True).encode()).hexdigest() if request else None,
                **({"source_run": config["source_run"], "source_events_sha256": source_hash,
                    "execution_replicate": execution_replicate,
                    "source_reward": replay[(round_index, agent_id)]["reward"],
                    "source_execution_tokens": replay[(round_index, agent_id)].get("solution_tokens", 0),
                    "source_acquisition_source": replay[(round_index, agent_id)].get("acquisition_source"),
                    } if is_replay else {}),
                "batch_generation_seconds": solve_seconds,
                "batch_size": len(solvable_ids),
                "skill_id": skill_id,
                "skill_intended_rank": env.skills[skill_id].intended_rank,
                "allocation": pull_type,
                "pull_type": pull_type,
                "copied_this_pull": choice["copied_this_pull"],
                "independently_acquired_this_pull": choice.get("independently_acquired_this_pull", False),
                "acquisition_source": choice.get("acquisition_source", "oracle"),
                "copy_cue": choice["copy_cue"],
                "policy_tokens": choice["policy_tokens"],
                "solution_tokens": generation_payload["completion_tokens"],
                "tokens_before": tokens_before,
                "tokens_after": agent.token_balance,
                "final_output": final_output,
                "invalid_output_preview": invalid_preview,
                **generation_payload,
                **result,
            }
            writer.write(event)

        for agent in agents:
            pull = pulls_this_round.get(agent.agent_id)
            if pull is None:
                missed_pulls += 1
                total_regret += 1.0
            per_agent = allocations[agent.agent_id]
            spent = sum(per_agent.values())
            unused_tokens = agent.token_balance
            if opening[agent.agent_id] != spent + unused_tokens:
                raise RuntimeError("rung 3 token budget conservation failed")
            tokens_expired = 0 if carry_over else unused_tokens
            if not carry_over:
                agent.token_balance = 0
            writer.write(
                {
                    **common_event(
                        config=config,
                        seed=seed,
                        round_index=round_index,
                        agent_id=agent.agent_id,
                    ),
                    "event": "round_end",
                    "opening_tokens": opening[agent.agent_id],
                    "fresh_tokens": tokens_per_round,
                    "tokens_observe": per_agent["observe"],
                    "tokens_explore": per_agent["explore"],
                    "tokens_exploit": per_agent["exploit"],
                    "tokens_invalid": per_agent["invalid"],
                    "tokens_pull": per_agent["explore"] + per_agent["exploit"],
                    "tokens_spent": spent,
                    "unused_tokens_end_of_round": unused_tokens,
                    "tokens_expired": tokens_expired,
                    "closing_tokens": agent.token_balance,
                    "pulled": pull is not None,
                    "independent_search_assigned": search_assigned[agent.agent_id],
                    "independent_search_eligible": bool(fresh_skills[agent.agent_id]),
                    "independent_search_completed": bool(pull and pull.get("independently_acquired_this_pull")),
                    "execution_reserve_tokens": execution_reserve,
                    "selection_fallback": selected.get(agent.agent_id, {}).get("selection_fallback", False),
                    "skill_id": pull["skill_id"] if pull else None,
                    "pull_type": pull["pull_type"] if pull else None,
                    "reward": pull["reward"] if pull else 0.0,
                    "expected_regret": pull["expected_regret"] if pull else 1.0,
                    "valid_solution": pull["valid_solution"] if pull else False,
                    "copied_this_pull": pull["copied_this_pull"] if pull else False,
                    "independently_acquired_this_pull": pull.get("independently_acquired_this_pull", False) if pull else False,
                    "acquisition_source": pull.get("acquisition_source") if pull else None,
                    "owned_skill_count": len(agent.owned_skill_ids),
                    "candidate_skill_count": len(agent.latest_candidates()),
                }
            )
        writer.flush()
        checkpoint_writer(
            {
                "rung": 3,
                "policy": config["policy"],
                "independent_skill_acquisition": independent_acquisition,
                "interventions": config.get("interventions", {}),
                **({"source_events_sha256": source_hash,
                    "execution_replicate": execution_replicate} if is_replay else {}),
                "next_round": round_index + 1,
                "agents": [agent.state_dict() for agent in agents],
                **({"deoe": [dict(
                    origins=policy.origins, estimates=policy.estimates,
                    seen_samples=sorted(policy.seen_samples),
                    last_observe=policy.last_observe, needs_check=policy.needs_check,
                ) for policy in policies]} if is_deoe else {}),
                "metrics": {
                    "total_reward": total_reward,
                    "total_regret": total_regret,
                    "missed_pulls": missed_pulls,
                    "invalid_solutions": invalid_solutions,
                    "successful_observations": successful_observations,
                    "total_copies": total_copies,
                    "context_limited_calls": context_limited_calls,
                    "allocation_totals": dict(allocation_totals),
                    "inference_seconds_by_stage": dict(inference_seconds),
                },
            }
        )
        if getattr(engine, "client", None) is not None:
            engine.client.clear_cache()

    denominator = env.num_agents * rounds
    return {
        "mean_reward_per_agent_round": total_reward / denominator,
        "mean_expected_regret_per_agent_round": total_regret / denominator,
        "missed_pulls": missed_pulls,
        "invalid_solutions": invalid_solutions,
        "successful_observations": successful_observations,
        "total_copies": total_copies,
        "total_independent_acquisitions": sum(len(agent.independent_acquisitions) for agent in agents),
        "context_limited_calls": context_limited_calls,
        "completion_tokens_by_allocation": dict(allocation_totals),
        "inference_seconds_by_stage": dict(inference_seconds),
        **({"source_run": config["source_run"], "source_events_sha256": source_hash,
            "execution_replicate": execution_replicate,
            "recorded_selections": sum(r < rounds for r, _ in replay)} if is_replay else {}),
    }
