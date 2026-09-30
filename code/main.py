"""Run one rung-1 condition and seed.

Examples:
    python main.py --config configs/rung1_social_action.json --seed 0
    python main.py --config configs/rung1_ucb.json --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, TextIO
from agent import (
    LLMAgent,
    UCBAgent,
    VLLMDecisionEngine,
    budget_mode,
    completion_preview,
    decision_token_limit,
    independent_search_assigned,
    intervention_settings,
    render_experiment_prompt,
    stable_seed,
)
from env import BanditEnvironment, observation_dict


ROOT = Path(__file__).resolve().parents[1]
LLM_POLICIES = {"social_llm", "solo_llm", "skill_oracle", "social_deoe", "solo_deoe", "skill_replay"}
CHECKPOINT_VERSION = 1
MODEL_PROFILES = {
    "Qwen/Qwen3-14B": {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": True},
    },
    "Qwen/Qwen3-32B": {
        "tensor_parallel_size": 2,
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": True},
    },
    "Qwen/Qwen3-8B": {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": True},
    },
    "openai/gpt-oss-20b": {
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "max_model_len": 32768,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "chat_template_kwargs": {"reasoning_effort": "medium"},
    },
    "mistralai/Ministral-3-14B-Reasoning-2512": {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {},
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "language_model_only": True,
    },
}
MODEL_LABEL_BY_NAME = {
    "Qwen/Qwen3-14B": "qwen3_14b",
    "Qwen/Qwen3-32B": "qwen3_32b",
    "Qwen/Qwen3-8B": "qwen3_8b",
    "openai/gpt-oss-20b": "gpt_oss_20b",
    "mistralai/Ministral-3-14B-Reasoning-2512": "ministral3_14b_reasoning",
}


class EventWriter:
    def __init__(self, path: Path, *, append: bool) -> None:
        self.handle: TextIO = path.open("a" if append else "w", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        self.handle.write(json.dumps(event, sort_keys=True) + "\n")

    def flush(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "checkpoint.json"
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version in {path}")
    return checkpoint


def save_checkpoint(run_dir: Path, checkpoint: dict[str, Any]) -> None:
    atomic_write_json(
        run_dir / "checkpoint.json",
        {"version": CHECKPOINT_VERSION, **checkpoint},
    )
    update_status(
        run_dir,
        state="running",
        next_round=int(checkpoint["next_round"]),
    )


def update_status(run_dir: Path, **updates: Any) -> None:
    """Atomically merge small run-health metadata used by launchers and analysis."""
    path = run_dir / "status.json"
    status: dict[str, Any] = {}
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
    status.update(updates)
    atomic_write_json(path, status)


def truncate_events_after_checkpoint(path: Path, next_round: int) -> None:
    """Discard an interrupted round while preserving checkpointed rounds."""
    if not path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with path.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                break
            if int(event["round"]) >= next_round:
                continue
            destination.write(json.dumps(event, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"condition", "policy", "environment", "output_root"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"config is missing: {', '.join(missing)}")
    if config["policy"] not in LLM_POLICIES | {"ucb", "oracle", "skill_oracle"}:
        raise ValueError(f"unknown policy: {config['policy']}")
    if config["policy"] in LLM_POLICIES or config["policy"] == "skill_oracle":
        for key in ("model", "budget", "prompt"):
            if key not in config:
                raise ValueError(f"LLM config is missing: {key}")
    rung = int(config.get("rung", 1))
    if rung not in {1, 2, 3}:
        raise ValueError(f"unknown rung: {rung}")
    if "budget" in config:
        intervention_settings(config)
    if config["policy"] in {"social_deoe", "solo_deoe"} and rung != 3:
        raise ValueError("hybrid DEOE policies require rung 3")
    if config["policy"] == "skill_replay" and (rung != 3 or not config.get("source_run")):
        raise ValueError("skill replay requires rung 3 and source_run")
    if rung == 2 and config["policy"] not in {"social_llm", "solo_llm"}:
        raise ValueError("rung 2 currently supports only LLM policies")
    if rung == 3:
        if config["policy"] not in LLM_POLICIES:
            raise ValueError("rung 3 supports social, solo, and skill-oracle policies")
        if config.get("social_info") not in {"none", "id", "payoff", "reputation", "full"}:
            raise ValueError("rung 3 config has an unknown social_info level")
    return config


def configure_model(
    config: dict[str, Any],
    *,
    model_name: str | None,
    model_label: str | None,
    tensor_parallel_size: int | None,
    model_provider: str | None = None,
    api_key_env: str | None = None,
    model_base_url: str | None = None,
) -> dict[str, Any]:
    """Apply a model override while keeping strategy and output IDs explicit."""
    config = copy.deepcopy(config)
    config["strategy"] = config.get("strategy", config["condition"])

    model_backed = config["policy"] in LLM_POLICIES or (
        int(config.get("rung", 1)) == 3 and config["policy"] == "skill_oracle"
    )
    if not model_backed:
        if model_name or model_label or tensor_parallel_size:
            raise ValueError("model overrides are only valid for LLM policies")
        config["model_label"] = "algorithmic"
        return config

    if model_name is not None:
        profile = MODEL_PROFILES.get(model_name, {})
        config["model"].update(profile)
        config["model"]["name"] = model_name
    if tensor_parallel_size is not None:
        config["model"]["tensor_parallel_size"] = tensor_parallel_size
    if model_provider is not None:
        config["model"]["provider"] = model_provider
    if api_key_env is not None:
        config["model"]["api_key_env"] = api_key_env
    if model_base_url is not None:
        config["model"]["base_url"] = model_base_url

    if model_label is None:
        model_label = MODEL_LABEL_BY_NAME.get(
            config["model"]["name"], config.get("model_label")
        )
    if model_label is None:
        model_label = re.sub(
            r"[^a-z0-9]+", "_", config["model"]["name"].lower()
        ).strip("_")
    if not re.fullmatch(r"[a-z0-9_]+", model_label):
        raise ValueError("model labels may contain only lowercase letters, digits, and _")

    config["model_label"] = model_label
    if model_name is not None:
        config["condition"] = f"{config['condition']}__{model_label}"
    return config


def configure_experiment(
    config: dict[str, Any],
    *,
    output_root: str | None,
    prompt: str | None,
    condition_prefix: str | None,
    num_agents: int | None,
    num_arms: int | None,
    budget_mode_override: str | None,
    guidance: str | None = None,
    reward_shape: float | None = None,
    spatial_length_scale: float | None = None,
    regime_period: int | None = None,
    coordinate_innovation: bool = False,
    independent_skill_acquisition: bool = False,
    independent_search_probability: float | None = None,
    execution_reserve_tokens: int | None = None,
) -> dict[str, Any]:
    """Apply explicit launch-time variants while retaining one canonical config."""
    config = copy.deepcopy(config)
    for key, value in (("independent_search_probability", independent_search_probability), ("execution_reserve_tokens", execution_reserve_tokens)):
        if value is not None:
            if config["policy"] not in {"solo_llm", "social_llm"}:
                raise ValueError("causal controls require an LLM selection policy")
            config.setdefault("interventions", {})[key] = value
            config["condition"] += f"__{key}_{str(value).replace('.', 'p')}"
    if "budget" in config:
        intervention_settings(config)
    if independent_skill_acquisition:
        if int(config.get("rung", 1)) != 3:
            raise ValueError("independent skill acquisition is rung-3 only")
        config["independent_skill_acquisition"] = True
        config["condition"] += "__independent_acquisition"
    if output_root is not None:
        config["output_root"] = output_root
    if prompt is not None:
        if "prompt" not in config:
            raise ValueError("prompt overrides are only valid for LLM policies")
        config["prompt"] = prompt
    if num_agents is not None:
        config["environment"]["num_agents"] = num_agents
    if num_arms is not None:
        if int(config.get("rung", 1)) != 1:
            raise ValueError("num-arms override is only valid for rung 1")
        config["environment"]["num_arms"] = num_arms
    if condition_prefix is not None:
        if not re.fullmatch(r"[a-z0-9_]+", condition_prefix):
            raise ValueError("condition prefixes may contain only lowercase letters, digits, and _")
        config["condition"] = f"{condition_prefix}{config['condition']}"
    if budget_mode_override is not None:
        if "budget" not in config:
            raise ValueError("budget modes are only valid for budgeted policies")
        config["budget"]["carry_over"] = budget_mode_override == "carry"
        config["condition"] = f"{config['condition']}__{budget_mode_override}"
    if guidance is not None:
        config["guidance"] = guidance
        if guidance != "neutral":
            config["condition"] = f"{config['condition']}__{guidance}"
    if reward_shape is not None:
        if reward_shape <= 0:
            raise ValueError("reward shape must be positive")
        config["environment"]["reward_shape"] = reward_shape
        if not math.isclose(reward_shape, 1.0):
            label = str(reward_shape).replace(".", "p")
            config["condition"] = f"{config['condition']}__shape_{label}"
    if spatial_length_scale is not None:
        if int(config.get("rung", 1)) != 2 or spatial_length_scale <= 0:
            raise ValueError("spatial length scale is positive and rung-2 only")
        config["environment"]["landscape"] = "structured"
        config["environment"]["structured_length_scale"] = spatial_length_scale
        if not math.isclose(spatial_length_scale, 0.15):
            label = str(spatial_length_scale).replace(".", "p")
            config["condition"] = f"{config['condition']}__corr_{label}"
    if regime_period is not None:
        if regime_period <= 0:
            raise ValueError("regime period must be positive")
        config["environment"]["regime_period"] = regime_period
        config["condition"] = f"{config['condition']}__regime_{regime_period}"
    if coordinate_innovation:
        if int(config.get("rung", 1)) != 2:
            raise ValueError("coordinate innovation is rung-2 only")
        config["environment"]["coordinate_innovation"] = True
    return config


def make_environment(config: dict[str, Any], seed: int) -> Any:
    rung = int(config.get("rung", 1))
    if rung == 2:
        from rung2 import make_environment as make_infinite_environment

        return make_infinite_environment(config, seed)
    if rung == 3:
        from rung3 import make_environment as make_skill_environment

        return make_skill_environment(config, seed)
    env = config["environment"]
    return BanditEnvironment(
        seed=seed,
        num_agents=env["num_agents"],
        num_arms=env["num_arms"],
        arm_mean_scale=env["arm_mean_scale"],
        reward_noise_std=env["reward_noise_std"],
        reward_shape=env.get("reward_shape", 1.0),
        regime_period=env.get("regime_period"),
    )


def common_event(
    *, config: dict[str, Any], seed: int, round_index: int, agent_id: int
) -> dict[str, Any]:
    environment = config["environment"]
    return {
        "condition": config["condition"],
        "strategy": config["strategy"],
        "model_label": config["model_label"],
        "model_name": config.get("model", {}).get("name"),
        "policy": config["policy"],
        "seed": seed,
        "round": round_index,
        "agent_id": agent_id,
        "budget_mode": budget_mode(config),
        "guidance": config.get("guidance", "neutral"),
        "reward_shape": environment.get("reward_shape", 1.0),
        "regime_period": environment.get("regime_period"),
        "regime_index": (
            round_index // environment["regime_period"]
            if environment.get("regime_period")
            else 0
        ),
    }


def prepare_run_directory(
    config: dict[str, Any], seed: int, overwrite: bool
) -> Path:
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_dir = output_root / config["condition"] / f"seed_{seed}"
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_metadata(
    *, run_dir: Path, config: dict[str, Any], seed: int, env: Any
) -> None:
    resolved = dict(config)
    resolved["seed"] = seed
    metadata = {
        run_dir / "config.json": resolved,
        run_dir / "environment.json": env.description(),
        run_dir / "prompt.json": {
            "system_prompt": (
                render_experiment_prompt(
                    (ROOT / config["prompt"]).read_text(encoding="utf-8").strip(),
                    config,
                )
                if config["policy"] in LLM_POLICIES
                or config["policy"] == "skill_oracle"
                else None
            )
        },
    }
    for path, expected in metadata.items():
        if path.exists():
            actual = json.loads(path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(
                    f"existing run metadata differs at {path}; "
                    "use --overwrite only if restarting is intentional"
                )
        else:
            atomic_write_json(path, expected)


def run_llm_condition(
    *,
    config: dict[str, Any],
    seed: int,
    env: BanditEnvironment,
    writer: EventWriter,
    run_dir: Path,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    num_agents = env.num_agents
    num_arms = env.num_arms
    rounds = config["environment"]["rounds"]
    tokens_per_round = config["budget"]["tokens_per_round"]
    max_tokens_per_decision = config["budget"]["max_tokens_per_decision"]
    carry_over = bool(config["budget"].get("carry_over", True))
    is_social = config["policy"] == "social_llm"
    reveal_payoff = bool(config.get("observe_payoff", False))
    _, execution_reserve = intervention_settings(config)
    if checkpoint is not None and checkpoint.get("interventions", {}) != config.get("interventions", {}):
        raise ValueError("checkpoint interventions do not match config")

    if checkpoint is None:
        start_round = 0
        agents = [LLMAgent(agent_id) for agent_id in range(num_agents)]
        total_reward = 0.0
        total_expected_regret = 0.0
        missed_pulls = 0
        successful_observations = 0
        allocation_totals: Counter[str] = Counter()
    else:
        if checkpoint.get("policy") != config["policy"]:
            raise ValueError("checkpoint policy does not match config")
        start_round = int(checkpoint["next_round"])
        agents = [
            LLMAgent.from_state_dict(state) for state in checkpoint["agents"]
        ]
        if [agent.agent_id for agent in agents] != list(range(num_agents)):
            raise ValueError("checkpoint agent count does not match config")
        env.restore_histories([agent.pulls for agent in agents])
        metrics = checkpoint["metrics"]
        total_reward = float(metrics["total_reward"])
        total_expected_regret = float(metrics["total_expected_regret"])
        missed_pulls = int(metrics["missed_pulls"])
        successful_observations = int(metrics["successful_observations"])
        allocation_totals = Counter(metrics["allocation_totals"])
        print(f"resuming {config['condition']} seed {seed} at round {start_round}")

    if not 0 <= start_round <= rounds:
        raise ValueError("checkpoint next round is outside the configured horizon")

    prompt_path = ROOT / config["prompt"]
    system_prompt = render_experiment_prompt(
        prompt_path.read_text(encoding="utf-8").strip(), config
    )
    engine = VLLMDecisionEngine(config["model"]) if start_round < rounds else None

    env.begin_round(start_round)
    for round_index in range(start_round, rounds):
        env.begin_round(round_index)
        assert engine is not None
        snapshot = env.snapshot(round_index)
        opening_tokens = {
            agent.agent_id: agent.begin_round(
                tokens_per_round, carry_over=carry_over
            )
            for agent in agents
        }
        round_allocations = {
            agent.agent_id: Counter() for agent in agents
        }
        round_pulls: dict[int, dict[str, Any]] = {}
        search_assigned = {a.agent_id: independent_search_assigned(config, seed, round_index, a.agent_id) for a in agents}
        fresh_arms = {a.agent_id: set(range(num_arms)) - {p.arm_id for p in a.pulls} - {o.arm_id for o in a.observations} for a in agents}
        active = set(range(num_agents))
        wave = 0

        while active:
            active_ids = sorted(active)
            requests = []
            pull_phase = {i: bool(execution_reserve and agents[i].token_balance <= execution_reserve) for i in active_ids}
            for agent_id in active_ids:
                agent = agents[agent_id]
                instruction = ""
                if search_assigned[agent_id] and fresh_arms[agent_id]:
                    instruction = f"\nIndependent-search intervention: PULL an arm in {sorted(fresh_arms[agent_id])}. OBSERVE and other pulls are unavailable this round."
                elif pull_phase[agent_id]:
                    instruction = "\nProtected execution phase: only PULL is available. Use your remaining budget to finish a pull."
                requests.append(
                    {
                        "system_prompt": system_prompt,
                        "state": agent.render_state(
                            round_index=round_index,
                            num_agents=num_agents,
                            num_arms=num_arms,
                            social_enabled=is_social,
                        ) + instruction,
                        "max_tokens": decision_token_limit(agent.token_balance, max_tokens_per_decision, execution_reserve),
                        "seed": stable_seed(
                            seed, config["condition"], round_index, wave, agent_id
                        ),
                    }
                )

            generation_started = time.perf_counter()
            generations = engine.generate(requests)
            batch_generation_seconds = time.perf_counter() - generation_started
            for agent_id, generation in zip(active_ids, generations, strict=True):
                agent = agents[agent_id]
                tokens_before = agent.token_balance
                agent.spend(generation.token_count)
                features = agent.decision_features(round_index)
                event = common_event(
                    config=config,
                    seed=seed,
                    round_index=round_index,
                    agent_id=agent_id,
                )
                event.update(
                    {
                        "event": "decision",
                        "wave": wave,
                        "batch_generation_seconds": batch_generation_seconds,
                        "batch_size": len(active_ids),
                        "final_output": completion_preview(generation.final_text),
                        "completion_tokens": generation.token_count,
                        "prompt_tokens": generation.prompt_tokens,
                        "cost_usd": generation.cost_usd,
                        "reasoning_tokens": generation.reasoning_tokens,
                        "finish_reason": generation.finish_reason,
                        "tokens_before": tokens_before,
                        "tokens_after": agent.token_balance,
                        **features,
                        "independent_search_assigned": search_assigned[agent_id],
                        "independent_search_eligible": bool(fresh_arms[agent_id]),
                        "execution_reserve_tokens": execution_reserve,
                        "protected_pull_phase": pull_phase[agent_id],
                    }
                )

                action = generation.action
                allocation = "invalid"
                valid = False
                error: str | None = None
                parsed_kind = action.kind if action else None
                parsed_value = action.value if action else None

                if action is None:
                    error = "missing final action"
                elif search_assigned[agent_id] and fresh_arms[agent_id] and (action.kind != "pull" or action.value not in fresh_arms[agent_id]):
                    error = "independent-search intervention requires a previously unknown arm"
                elif pull_phase[agent_id] and action.kind != "pull":
                    error = "protected execution phase requires PULL"
                elif action.kind == "observe":
                    if not is_social:
                        error = "OBSERVE is unavailable in the solo condition"
                    elif not 0 <= action.value < num_agents or action.value == agent_id:
                        error = "invalid observation target"
                    else:
                        observation = env.observe(
                            snapshot=snapshot,
                            observer_id=agent_id,
                            target_id=action.value,
                            round_index=round_index,
                            reveal_payoff=reveal_payoff,
                        )
                        agent.add_observation_attempt(
                            round_index=round_index,
                            target_id=action.value,
                            observation=observation,
                        )
                        if observation is not None:
                            successful_observations += 1
                        event["observation"] = observation_dict(observation)
                        allocation = "observe"
                        valid = True
                elif action.kind == "pull":
                    if not 0 <= action.value < num_arms:
                        error = "invalid arm"
                    else:
                        allocation = agent.pull_label(action.value)
                        copy_features = agent.copy_features(action.value, round_index)
                        pull = env.pull(
                            agent_id=agent_id,
                            arm_id=action.value,
                            round_index=round_index,
                        )
                        best_personal_arm = copy_features[
                            "best_personal_alternative_arm_id_before_pull"
                        ]
                        if copy_features["copy_any"] and best_personal_arm is not None:
                            selected_true_mean = float(env.arm_means[action.value])
                            best_personal_true_mean = float(
                                env.arm_means[best_personal_arm]
                            )
                            copy_features.update(
                                {
                                    "best_personal_alternative_arm_true_mean": best_personal_true_mean,
                                    "copied_arm_true_advantage_vs_best_personal_alternative": (
                                        selected_true_mean - best_personal_true_mean
                                    ),
                                    "copied_arm_better_than_best_personal_alternative": (
                                        selected_true_mean > best_personal_true_mean
                                    ),
                                    "copied_pull_reward_advantage_vs_best_personal_alternative_mean": (
                                        pull.reward
                                        - copy_features[
                                            "best_personal_alternative_mean_before_pull"
                                        ]
                                    ),
                                }
                            )
                        else:
                            copy_features.update(
                                {
                                    "best_personal_alternative_arm_true_mean": None,
                                    "copied_arm_true_advantage_vs_best_personal_alternative": None,
                                    "copied_arm_better_than_best_personal_alternative": None,
                                    "copied_pull_reward_advantage_vs_best_personal_alternative_mean": None,
                                }
                            )
                        agent.add_pull(pull)
                        expected_regret = env.best_mean - float(
                            env.arm_means[action.value]
                        )
                        round_pulls[agent_id] = {
                            "arm_id": action.value,
                            "reward": pull.reward,
                            "expected_regret": expected_regret,
                            "pulled_best_arm": action.value == env.best_arm,
                            "pull_type": allocation,
                            "copy_any": copy_features["copy_any"],
                        }
                        event.update(
                            {
                                "arm_id": action.value,
                                "reward": pull.reward,
                                "arm_mean": float(env.arm_means[action.value]),
                                "expected_regret": expected_regret,
                                "pulled_best_arm": action.value == env.best_arm,
                                "pull_type": allocation,
                                **copy_features,
                            }
                        )
                        valid = True
                        active.remove(agent_id)

                event.update(
                    {
                        "parsed_action": parsed_kind,
                        "parsed_value": parsed_value,
                        "valid": valid,
                        "error": error,
                        "allocation": allocation,
                    }
                )
                if not valid:
                    event["invalid_output_preview"] = completion_preview(
                        generation.text
                    )
                round_allocations[agent_id][allocation] += generation.token_count
                allocation_totals[allocation] += generation.token_count
                writer.write(event)

                if agent_id in active and agent.token_balance == 0:
                    active.remove(agent_id)
            wave += 1

        for agent in agents:
            pull = round_pulls.get(agent.agent_id)
            if pull is None:
                missed_pulls += 1
                reward = 0.0
                expected_regret = env.best_mean
                pulled_best = False
                pull_type = None
            else:
                reward = pull["reward"]
                expected_regret = pull["expected_regret"]
                pulled_best = pull["pulled_best_arm"]
                pull_type = pull["pull_type"]
            total_reward += reward
            total_expected_regret += expected_regret

            allocations = round_allocations[agent.agent_id]
            spent = sum(allocations.values())
            unused_tokens = agent.token_balance
            if opening_tokens[agent.agent_id] != spent + unused_tokens:
                raise RuntimeError("token budget conservation failed")
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
                    "independent_search_assigned": search_assigned[agent.agent_id],
                    "independent_search_eligible": bool(fresh_arms[agent.agent_id]),
                    "independent_search_completed": bool(search_assigned[agent.agent_id] and pull and pull["arm_id"] in fresh_arms[agent.agent_id]),
                    "execution_reserve_tokens": execution_reserve,
                    "opening_tokens": opening_tokens[agent.agent_id],
                    "fresh_tokens": tokens_per_round,
                    "tokens_observe": allocations["observe"],
                    "tokens_explore": allocations["explore"],
                    "tokens_exploit": allocations["exploit"],
                    "tokens_invalid": allocations["invalid"],
                    "tokens_spent": spent,
                    "unused_tokens_end_of_round": unused_tokens,
                    "tokens_expired": tokens_expired,
                    "closing_tokens": agent.token_balance,
                    "pulled": pull is not None,
                    "arm_id": pull["arm_id"] if pull else None,
                    "pull_type": pull_type,
                    "reward": reward,
                    "expected_regret": expected_regret,
                    "pulled_best_arm": pulled_best,
                    "copy_any": pull["copy_any"] if pull else False,
                }
            )
        writer.flush()
        save_checkpoint(
            run_dir,
            {
                "policy": config["policy"],
                "next_round": round_index + 1,
                "interventions": config.get("interventions", {}),
                "agents": [agent.state_dict() for agent in agents],
                "metrics": {
                    "total_reward": total_reward,
                    "total_expected_regret": total_expected_regret,
                    "missed_pulls": missed_pulls,
                    "successful_observations": successful_observations,
                    "allocation_totals": dict(allocation_totals),
                },
            },
        )
        if getattr(engine, "client", None) is not None:
            engine.client.clear_cache()
        if (round_index + 1) % 10 == 0 or round_index + 1 == rounds:
            print(f"completed round {round_index + 1}/{rounds}", flush=True)

    denominator = num_agents * rounds
    return {
        "mean_reward_per_agent_round": total_reward / denominator,
        "mean_expected_regret_per_agent_round": total_expected_regret / denominator,
        "missed_pulls": missed_pulls,
        "successful_observations": successful_observations,
        "completion_tokens_by_allocation": dict(allocation_totals),
    }


def run_algorithmic_condition(
    *,
    config: dict[str, Any],
    seed: int,
    env: BanditEnvironment,
    writer: EventWriter,
    run_dir: Path,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    rounds = config["environment"]["rounds"]
    is_oracle = config["policy"] == "oracle"
    if checkpoint is None:
        start_round = 0
        ucb_agents = (
            None
            if is_oracle
            else [
                UCBAgent(agent_id=i, num_arms=env.num_arms, seed=seed)
                for i in range(env.num_agents)
            ]
        )
        total_reward = 0.0
        total_expected_regret = 0.0
    else:
        if checkpoint.get("policy") != config["policy"]:
            raise ValueError("checkpoint policy does not match config")
        start_round = int(checkpoint["next_round"])
        total_reward = float(checkpoint["metrics"]["total_reward"])
        total_expected_regret = float(
            checkpoint["metrics"]["total_expected_regret"]
        )
        ucb_agents = (
            None
            if is_oracle
            else [
                UCBAgent.from_state_dict(
                    state, num_arms=env.num_arms, seed=seed
                )
                for state in checkpoint["agents"]
            ]
        )
        if ucb_agents is not None and [
            agent.agent_id for agent in ucb_agents
        ] != list(range(env.num_agents)):
            raise ValueError("checkpoint agent count does not match config")
        print(f"resuming {config['condition']} seed {seed} at round {start_round}")

    if not 0 <= start_round <= rounds:
        raise ValueError("checkpoint next round is outside the configured horizon")

    env.begin_round(start_round)
    if (
        checkpoint is not None
        and start_round > 0
        and env.regime_period is not None
        and start_round % env.regime_period == 0
        and ucb_agents is not None
    ):
        ucb_agents = [
            UCBAgent(agent_id=i, num_arms=env.num_arms, seed=seed)
            for i in range(env.num_agents)
        ]
    for round_index in range(start_round, rounds):
        if (
            round_index > start_round
            and env.regime_period is not None
            and round_index % env.regime_period == 0
            and ucb_agents is not None
        ):
            ucb_agents = [
                UCBAgent(agent_id=i, num_arms=env.num_arms, seed=seed)
                for i in range(env.num_agents)
            ]
        env.begin_round(round_index)
        for agent_id in range(env.num_agents):
            if is_oracle:
                arm_id = env.best_arm
                pull_type = "oracle"
            else:
                assert ucb_agents is not None
                arm_id = ucb_agents[agent_id].choose()
                pull_type = ucb_agents[agent_id].pull_label(arm_id)

            pull = env.pull(
                agent_id=agent_id, arm_id=arm_id, round_index=round_index
            )
            if not is_oracle:
                assert ucb_agents is not None
                ucb_agents[agent_id].update(arm_id, pull.reward)
            regret = env.best_mean - float(env.arm_means[arm_id])
            total_reward += pull.reward
            total_expected_regret += regret
            writer.write(
                {
                    **common_event(
                        config=config,
                        seed=seed,
                        round_index=round_index,
                        agent_id=agent_id,
                    ),
                    "event": "decision",
                    "wave": 0,
                    "valid": True,
                    "allocation": pull_type,
                    "parsed_action": "pull",
                    "parsed_value": arm_id,
                    "arm_id": arm_id,
                    "reward": pull.reward,
                    "arm_mean": float(env.arm_means[arm_id]),
                    "expected_regret": regret,
                    "pulled_best_arm": arm_id == env.best_arm,
                    "pull_type": pull_type,
                    "completion_tokens": 0,
                }
            )
            writer.write(
                {
                    **common_event(
                        config=config,
                        seed=seed,
                        round_index=round_index,
                        agent_id=agent_id,
                    ),
                    "event": "round_end",
                    "pulled": True,
                    "arm_id": arm_id,
                    "pull_type": pull_type,
                    "reward": pull.reward,
                    "expected_regret": regret,
                    "pulled_best_arm": arm_id == env.best_arm,
                }
            )
        writer.flush()
        save_checkpoint(
            run_dir,
            {
                "policy": config["policy"],
                "next_round": round_index + 1,
                "agents": (
                    []
                    if ucb_agents is None
                    else [agent.state_dict() for agent in ucb_agents]
                ),
                "metrics": {
                    "total_reward": total_reward,
                    "total_expected_regret": total_expected_regret,
                },
            },
        )
        if (round_index + 1) % 10 == 0 or round_index + 1 == rounds:
            print(f"completed round {round_index + 1}/{rounds}", flush=True)

    denominator = env.num_agents * rounds
    return {
        "mean_reward_per_agent_round": total_reward / denominator,
        "mean_expected_regret_per_agent_round": total_expected_regret / denominator,
        "missed_pulls": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--model",
        help="override the Hugging Face model for an LLM condition",
    )
    parser.add_argument(
        "--model-label",
        help="short lowercase label used in run IDs and plots",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="override the model config's tensor parallel size",
    )
    parser.add_argument("--model-provider", choices=("vllm", "openrouter"))
    parser.add_argument("--api-key-env")
    parser.add_argument("--model-base-url")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing run directory for this condition and seed",
    )
    parser.add_argument("--output-root", help="override the run output root")
    parser.add_argument("--prompt", help="override the policy prompt path")
    parser.add_argument(
        "--condition-prefix",
        help="prefix added to the condition identifier",
    )
    parser.add_argument("--num-agents", type=int, help="override population size")
    parser.add_argument("--num-arms", type=int, help="override rung-1 arm count")
    parser.add_argument(
        "--budget-mode",
        choices=("carry", "use_it_or_lose_it"),
        help="whether unused completion tokens carry across rounds",
    )
    parser.add_argument(
        "--guidance",
        choices=("neutral", "deliberate", "strategic"),
        help="optional V3 decision guidance",
    )
    parser.add_argument("--reward-shape", type=float)
    parser.add_argument("--spatial-length-scale", type=float)
    parser.add_argument("--regime-period", type=int)
    parser.add_argument("--coordinate-innovation", action="store_true")
    parser.add_argument("--independent-skill-acquisition", action="store_true")
    parser.add_argument("--independent-search-probability", type=float)
    parser.add_argument("--execution-reserve-tokens", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    config = configure_experiment(
        config,
        output_root=args.output_root,
        prompt=args.prompt,
        condition_prefix=args.condition_prefix,
        num_agents=args.num_agents,
        num_arms=args.num_arms,
        budget_mode_override=args.budget_mode,
        guidance=args.guidance,
        reward_shape=args.reward_shape,
        spatial_length_scale=args.spatial_length_scale,
        regime_period=args.regime_period,
        coordinate_innovation=args.coordinate_innovation,
        independent_skill_acquisition=args.independent_skill_acquisition,
        independent_search_probability=args.independent_search_probability,
        execution_reserve_tokens=args.execution_reserve_tokens,
    )
    config = configure_model(
        config,
        model_name=args.model,
        model_label=args.model_label,
        tensor_parallel_size=args.tensor_parallel_size,
        model_provider=args.model_provider,
        api_key_env=args.api_key_env,
        model_base_url=args.model_base_url,
    )
    run_dir = prepare_run_directory(config, args.seed, args.overwrite)
    os.environ["AGENT_MARKET_API_CACHE_DIR"] = str(run_dir / "api_batches")
    env = make_environment(config, args.seed)
    write_metadata(run_dir=run_dir, config=config, seed=args.seed, env=env)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("completed"):
            print(f"already completed: {run_dir}")
            return

    update_status(
        run_dir,
        state="initialized",
        completed=False,
        quality_gate_passed=False,
        condition=config["condition"],
        model_label=config["model_label"],
        seed=args.seed,
        next_round=0,
    )

    checkpoint = load_checkpoint(run_dir)
    next_round = int(checkpoint["next_round"]) if checkpoint else 0
    events_path = run_dir / "events.jsonl"
    truncate_events_after_checkpoint(events_path, next_round)

    writer = EventWriter(events_path, append=True)
    try:
        try:
            rung = int(config.get("rung", 1))
            if rung == 2:
                from rung2 import run_condition as run_rung2_condition

                results = run_rung2_condition(
                    config=config,
                    seed=args.seed,
                    env=env,
                    writer=writer,
                    checkpoint=checkpoint,
                    checkpoint_writer=lambda state: save_checkpoint(run_dir, state),
                )
            elif rung == 3:
                from rung3 import run_condition as run_rung3_condition

                results = run_rung3_condition(
                    config=config,
                    seed=args.seed,
                    env=env,
                    writer=writer,
                    checkpoint=checkpoint,
                    checkpoint_writer=lambda state: save_checkpoint(run_dir, state),
                )
            elif config["policy"] in LLM_POLICIES:
                results = run_llm_condition(
                    config=config,
                    seed=args.seed,
                    env=env,
                    writer=writer,
                    run_dir=run_dir,
                    checkpoint=checkpoint,
                )
            else:
                results = run_algorithmic_condition(
                    config=config,
                    seed=args.seed,
                    env=env,
                    writer=writer,
                    run_dir=run_dir,
                    checkpoint=checkpoint,
                )
        except Exception as exception:
            try:
                update_status(
                    run_dir,
                    state="failed",
                    completed=False,
                    quality_gate_passed=False,
                    error_type=type(exception).__name__,
                    error=str(exception)[:512],
                )
            except OSError:
                pass
            raise
    finally:
        writer.close()

    summary = {
        "completed": True,
        "rung": int(config.get("rung", 1)),
        "landscape": config["environment"].get("landscape"),
        "condition": config["condition"],
        "strategy": config["strategy"],
        "model_label": config["model_label"],
        "model_name": config.get("model", {}).get("name"),
        "policy": config["policy"],
        "budget_mode": budget_mode(config),
        "seed": args.seed,
        "num_agents": env.num_agents,
        "num_arms": getattr(env, "num_arms", None),
        "rounds": config["environment"]["rounds"],
        "quality_gate_passed": True,
        **results,
    }
    atomic_write_json(summary_path, summary)
    update_status(
        run_dir,
        state="completed",
        completed=True,
        quality_gate_passed=True,
        next_round=config["environment"]["rounds"],
        error_type=None,
        error=None,
    )
    (run_dir / "checkpoint.json").unlink(missing_ok=True)
    print(f"completed {config['condition']} seed {args.seed}: {run_dir}")


if __name__ == "__main__":
    main()
