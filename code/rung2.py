"""Rung-2 infinite-arm social-learning experiment loop."""

from __future__ import annotations

from collections import Counter
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from agent import (
    Probe,
    Rung2Agent,
    VLLMDecisionEngine,
    budget_mode,
    completion_preview,
    decision_token_limit,
    independent_search_assigned,
    intervention_settings,
    parse_rung2_action,
    render_experiment_prompt,
    stable_seed,
)
from env import InfiniteBanditEnvironment, observation_dict


ROOT = Path(__file__).resolve().parents[1]


class EventSink(Protocol):
    def write(self, event: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...


def make_environment(config: dict[str, Any], seed: int) -> InfiniteBanditEnvironment:
    environment = config["environment"]
    return InfiniteBanditEnvironment(
        seed=seed,
        num_agents=environment["num_agents"],
        landscape=environment["landscape"],
        arm_mean_scale=environment["arm_mean_scale"],
        reward_noise_std=environment["reward_noise_std"],
        coordinate_decimals=environment.get("coordinate_decimals", 9),
        structured_features=environment.get("structured_features", 32),
        structured_length_scale=environment.get(
            "structured_length_scale", 0.15
        ),
        reference_grid_size=environment.get("reference_grid_size", 100_001),
        reward_shape=environment.get("reward_shape", 1.0),
        regime_period=environment.get("regime_period"),
        coordinate_innovation=environment.get("coordinate_innovation", False),
    )


def common_event(
    *, config: dict[str, Any], seed: int, round_index: int, agent_id: int
) -> dict[str, Any]:
    environment = config["environment"]
    return {
        "rung": 2,
        "landscape": config["environment"]["landscape"],
        "condition": config["condition"],
        "strategy": config["strategy"],
        "model_label": config["model_label"],
        "model_name": config["model"]["name"],
        "policy": config["policy"],
        "seed": seed,
        "round": round_index,
        "agent_id": agent_id,
        "budget_mode": budget_mode(config),
        "guidance": config.get("guidance", "neutral"),
        "reward_shape": environment.get("reward_shape", 1.0),
        "spatial_length_scale": (
            environment.get("structured_length_scale")
            if environment["landscape"] == "structured"
            else 0.0
        ),
        "regime_period": environment.get("regime_period"),
        "regime_index": (
            round_index // environment["regime_period"]
            if environment.get("regime_period")
            else 0
        ),
    }


def _result(
    *,
    total_reward: float,
    total_latent_reward: float,
    total_expected_regret: float,
    regret_count: int,
    missed_pulls: int,
    successful_observations: int,
    total_innovations: int,
    allocation_totals: Counter[str],
    denominator: int,
) -> dict[str, Any]:
    return {
        "mean_reward_per_agent_round": total_reward / denominator,
        "mean_latent_reward_per_agent_round": total_latent_reward / denominator,
        "mean_expected_regret_per_pull": (
            total_expected_regret / regret_count if regret_count else None
        ),
        "missed_pulls": missed_pulls,
        "successful_observations": successful_observations,
        "total_innovations": total_innovations,
        "completion_tokens_by_allocation": dict(allocation_totals),
    }


def run_condition(
    *,
    config: dict[str, Any],
    seed: int,
    env: InfiniteBanditEnvironment,
    writer: EventSink,
    checkpoint: dict[str, Any] | None,
    checkpoint_writer: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    num_agents = env.num_agents
    rounds = config["environment"]["rounds"]
    tokens_per_round = config["budget"]["tokens_per_round"]
    max_tokens_per_decision = config["budget"]["max_tokens_per_decision"]
    carry_over = bool(config["budget"].get("carry_over", True))
    is_social = config["policy"] == "social_llm"
    reveal_payoff = bool(config.get("observe_payoff", False))
    if checkpoint is not None and checkpoint.get("interventions", {}) != config.get("interventions", {}):
        raise ValueError("checkpoint interventions do not match config")

    if checkpoint is None:
        start_round = 0
        agents = [Rung2Agent(agent_id) for agent_id in range(num_agents)]
        total_reward = 0.0
        total_latent_reward = 0.0
        total_expected_regret = 0.0
        regret_count = 0
        missed_pulls = 0
        successful_observations = 0
        total_innovations = 0
        allocation_totals: Counter[str] = Counter()
    else:
        if checkpoint.get("rung") != 2:
            raise ValueError("checkpoint rung does not match config")
        if checkpoint.get("policy") != config["policy"]:
            raise ValueError("checkpoint policy does not match config")
        start_round = int(checkpoint["next_round"])
        agents = [
            Rung2Agent.from_state_dict(state) for state in checkpoint["agents"]
        ]
        if [agent.agent_id for agent in agents] != list(range(num_agents)):
            raise ValueError("checkpoint agent IDs do not match config")
        env.restore_state(
            histories=[agent.pulls for agent in agents],
            innovation_arm_ids=[
                probe.arm_id for agent in agents for probe in agent.probes
            ],
        )
        metrics = checkpoint["metrics"]
        total_reward = float(metrics["total_reward"])
        total_latent_reward = float(metrics["total_latent_reward"])
        total_expected_regret = float(metrics["total_expected_regret"])
        regret_count = int(metrics["regret_count"])
        missed_pulls = int(metrics["missed_pulls"])
        successful_observations = int(metrics["successful_observations"])
        total_innovations = int(metrics["total_innovations"])
        allocation_totals = Counter(metrics["allocation_totals"])
        print(f"resuming {config['condition']} seed {seed} at round {start_round}")

    if not 0 <= start_round <= rounds:
        raise ValueError("checkpoint next round is outside the configured horizon")

    observe_rule = (
        "An observation reveals the arm and its realized scored payoff."
        if reveal_payoff
        else "An observation reveals the arm but never its payoff."
    )
    prompt = render_experiment_prompt(
        (ROOT / config["prompt"]).read_text(encoding="utf-8").strip(),
        config,
    )
    system_prompt = prompt.replace("{{OBSERVE_RULE}}", observe_rule)
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
        _, execution_reserve = intervention_settings(config)
        search_assigned = {a.agent_id: independent_search_assigned(config, seed, round_index, a.agent_id) for a in agents}
        search_completed = {a.agent_id: False for a in agents}
        active = set(range(num_agents))
        wave = 0

        while active:
            active_ids = sorted(active)
            search_required = {i: search_assigned[i] and not search_completed[i] for i in active_ids}
            pull_phase = {i: bool(execution_reserve and agents[i].token_balance <= execution_reserve and agents[i].known_arm_ids() and not search_required[i]) for i in active_ids}
            instructions = {i: (
                "\nIndependent-search intervention: your next valid action must be INNOVATE. Choose the innovation yourself; its usual inference cost applies. After innovation, PULL to earn reward."
                if search_required[i] else
                "\nProtected execution phase: only PULL is available. Use your remaining budget to finish a pull."
                if pull_phase[i] else ""
            ) for i in active_ids}
            requests = [
                {
                    "system_prompt": system_prompt,
                    "state": agents[agent_id].render_rung2_state(
                        round_index=round_index,
                        num_agents=num_agents,
                        landscape=env.landscape,
                        coordinate_scale=env.coordinate_scale,
                        coordinate_innovation=env.coordinate_innovation,
                        social_enabled=is_social,
                    ) + instructions[agent_id],
                    "max_tokens": decision_token_limit(
                        agents[agent_id].token_balance,
                        max_tokens_per_decision,
                        execution_reserve,
                    ),
                    "seed": stable_seed(
                        seed,
                        config["condition"],
                        round_index,
                        wave,
                        agent_id,
                    ),
                }
                for agent_id in active_ids
            ]
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
                        "independent_search_required": search_required[agent_id],
                        "execution_reserve_tokens": execution_reserve,
                        "protected_pull_phase": pull_phase[agent_id],
                    }
                )

                action = parse_rung2_action(generation.final_text)
                allocation = "invalid"
                valid = False
                error: str | None = None
                parsed_kind = action.kind if action else None
                parsed_value = action.value if action else None

                if action is None:
                    error = "missing final action"
                elif search_required[agent_id] and action.kind != "innovate":
                    error = "independent-search intervention requires INNOVATE first"
                elif pull_phase[agent_id] and action.kind != "pull":
                    error = "protected execution phase requires PULL"
                elif action.kind == "observe":
                    target_id = int(action.value) if action.value is not None else -1
                    if not is_social:
                        error = "OBSERVE is unavailable in the solo condition"
                    elif not 0 <= target_id < num_agents or target_id == agent_id:
                        error = "invalid observation target"
                    else:
                        observation = env.observe(
                            snapshot=snapshot,
                            observer_id=agent_id,
                            target_id=target_id,
                            round_index=round_index,
                            reveal_payoff=reveal_payoff,
                        )
                        agent.add_observation_attempt(
                            round_index=round_index,
                            target_id=target_id,
                            observation=observation,
                        )
                        successful_observations += observation is not None
                        event["observation"] = observation_dict(observation)
                        allocation = "observe"
                        valid = True
                elif action.kind == "innovate":
                    coordinate = (
                        float(action.value) if action.value is not None else None
                    )
                    if not env.coordinate_innovation and coordinate is not None:
                        error = "unstructured INNOVATE takes no coordinate"
                    elif env.coordinate_innovation and coordinate is None:
                        error = "coordinate INNOVATE requires a coordinate"
                    else:
                        try:
                            candidate_arm = (
                                env.coordinate_to_arm(coordinate)
                                if coordinate is not None
                                else None
                            )
                        except ValueError as exception:
                            error = str(exception)
                        else:
                            if (
                                candidate_arm is not None
                                and candidate_arm in agent.known_arm_ids()
                            ):
                                error = "coordinate is already in your repertoire"
                            else:
                                prior_true_best = max(
                                    (
                                        env.arm_mean(known_arm)
                                        for known_arm in agent.known_arm_ids()
                                    ),
                                    default=None,
                                )
                                result = env.innovate(
                                    agent_id=agent_id,
                                    innovation_index=len(agent.probes),
                                    round_index=round_index,
                                    coordinate=coordinate,
                                )
                                agent.add_probe(
                                    Probe(
                                        round=result.round,
                                        arm_id=result.arm_id,
                                        coordinate=result.coordinate,
                                        reward=result.reward,
                                    )
                                )
                                total_innovations += 1
                                social_parent = None
                                if env.coordinate_innovation and agent.observations:
                                    social_parent = min(
                                        agent.observations,
                                        key=lambda observation: abs(
                                            env.arm_to_coordinate(observation.arm_id)
                                            - float(result.coordinate)
                                        ),
                                    )
                                parent_mean = (
                                    env.arm_mean(social_parent.arm_id)
                                    if social_parent is not None
                                    else None
                                )
                                event.update(
                                    {
                                        "arm_id": result.arm_id,
                                        "coordinate": result.coordinate,
                                        "probe_reward": result.reward,
                                        "arm_mean": result.arm_mean,
                                        "novel_to_population": (
                                            result.novel_to_population
                                        ),
                                        "innovation_improved_true_frontier": (
                                            prior_true_best is None
                                            or result.arm_mean > prior_true_best
                                        ),
                                        "innovation_improved_evidence": (
                                            features["best_known_mean"] is None
                                            or result.reward
                                            > features["best_known_mean"]
                                        ),
                                        "innovation_parent_arm_id": (
                                            social_parent.arm_id
                                            if social_parent is not None
                                            else None
                                        ),
                                        "innovation_parent_agent_id": (
                                            social_parent.target_id
                                            if social_parent is not None
                                            else None
                                        ),
                                        "innovation_parent_distance": (
                                            abs(
                                                env.arm_to_coordinate(
                                                    social_parent.arm_id
                                                )
                                                - float(result.coordinate)
                                            )
                                            if social_parent is not None
                                            else None
                                        ),
                                        "innovation_improved_social_parent": (
                                            result.arm_mean > parent_mean
                                            if parent_mean is not None
                                            else None
                                        ),
                                    }
                                )
                                allocation = "innovate"
                                valid = True
                                search_completed[agent_id] = True
                elif action.kind == "pull":
                    arm_id = int(action.value) if action.value is not None else -1
                    if arm_id not in agent.known_arm_ids():
                        error = "PULL arm is not in your repertoire"
                    else:
                        allocation = agent.pull_label(arm_id)
                        copy_features = agent.copy_features(arm_id, round_index)
                        population_best = env.population_best_mean()
                        arm_mean = env.arm_mean(arm_id)
                        pull = env.pull(
                            agent_id=agent_id,
                            arm_id=arm_id,
                            round_index=round_index,
                        )
                        best_personal_arm = copy_features[
                            "best_personal_alternative_arm_id_before_pull"
                        ]
                        if copy_features["copy_any"] and best_personal_arm is not None:
                            best_personal_true_mean = env.arm_mean(best_personal_arm)
                            copy_features.update(
                                {
                                    "best_personal_alternative_arm_true_mean": best_personal_true_mean,
                                    "copied_arm_true_advantage_vs_best_personal_alternative": (
                                        arm_mean - best_personal_true_mean
                                    ),
                                    "copied_arm_better_than_best_personal_alternative": (
                                        arm_mean > best_personal_true_mean
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
                        expected_regret = (
                            max(env.reference_best_mean - arm_mean, 0.0)
                            if env.reference_best_mean is not None
                            else None
                        )
                        round_pulls[agent_id] = {
                            "arm_id": arm_id,
                            "coordinate": (
                                env.arm_to_coordinate(arm_id)
                                if env.coordinate_innovation
                                else None
                            ),
                            "reward": pull.reward,
                            "arm_mean": arm_mean,
                            "expected_regret": expected_regret,
                            "population_frontier_gap": (
                                population_best - arm_mean
                                if population_best is not None
                                else None
                            ),
                            "pull_type": allocation,
                            **copy_features,
                        }
                        event.update(round_pulls[agent_id])
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
                arm_mean = None
                expected_regret = None
            else:
                reward = float(pull["reward"])
                arm_mean = float(pull["arm_mean"])
                expected_regret = pull["expected_regret"]
                total_latent_reward += arm_mean
                if expected_regret is not None:
                    total_expected_regret += float(expected_regret)
                    regret_count += 1
            total_reward += reward

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
                    "independent_search_completed": search_assigned[agent.agent_id] and search_completed[agent.agent_id],
                    "execution_reserve_tokens": execution_reserve,
                    "opening_tokens": opening_tokens[agent.agent_id],
                    "fresh_tokens": tokens_per_round,
                    "tokens_observe": allocations["observe"],
                    "tokens_innovate": allocations["innovate"],
                    "tokens_explore": allocations["explore"],
                    "tokens_exploit": allocations["exploit"],
                    "tokens_invalid": allocations["invalid"],
                    "tokens_spent": spent,
                    "unused_tokens_end_of_round": unused_tokens,
                    "tokens_expired": tokens_expired,
                    "closing_tokens": agent.token_balance,
                    "pulled": pull is not None,
                    "arm_id": pull["arm_id"] if pull else None,
                    "coordinate": pull["coordinate"] if pull else None,
                    "pull_type": pull["pull_type"] if pull else None,
                    "reward": reward,
                    "arm_mean": arm_mean,
                    "expected_regret": expected_regret,
                    "population_frontier_gap": (
                        pull["population_frontier_gap"] if pull else None
                    ),
                    "copy_any": pull["copy_any"] if pull else False,
                }
            )
        writer.flush()
        checkpoint_writer(
            {
                "rung": 2,
                "policy": config["policy"],
                "next_round": round_index + 1,
                "interventions": config.get("interventions", {}),
                "agents": [agent.state_dict() for agent in agents],
                "metrics": {
                    "total_reward": total_reward,
                    "total_latent_reward": total_latent_reward,
                    "total_expected_regret": total_expected_regret,
                    "regret_count": regret_count,
                    "missed_pulls": missed_pulls,
                    "successful_observations": successful_observations,
                    "total_innovations": total_innovations,
                    "allocation_totals": dict(allocation_totals),
                },
            }
        )
        if getattr(engine, "client", None) is not None:
            engine.client.clear_cache()
        if (round_index + 1) % 10 == 0 or round_index + 1 == rounds:
            print(f"completed round {round_index + 1}/{rounds}", flush=True)

    return _result(
        total_reward=total_reward,
        total_latent_reward=total_latent_reward,
        total_expected_regret=total_expected_regret,
        regret_count=regret_count,
        missed_pulls=missed_pulls,
        successful_observations=successful_observations,
        total_innovations=total_innovations,
        allocation_totals=allocation_totals,
        denominator=num_agents * rounds,
    )
