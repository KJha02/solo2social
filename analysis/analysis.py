"""Analyze completed experiment rungs and produce paper-style figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STRATEGY_ORDER = [
    "social_action",
    "social_action_payoff",
    "social_action_payoff_deliberate",
    "social_action_payoff_strategic",
    "solo_llm",
    "ucb",
    "oracle",
]
MODEL_ORDER = [
    "qwen3_14b",
    "gpt_oss_20b",
    "ministral3_14b_reasoning",
    "qwen3_32b",
    "qwen3_8b",
    "algorithmic",
]
COLORS = {
    "social_action": "#0072B2",
    "social_action_payoff": "#009E73",
    "social_action_payoff_deliberate": "#56B4E9",
    "social_action_payoff_strategic": "#D55E00",
    "solo_llm": "#E69F00",
    "ucb": "#CC79A7",
    "oracle": "#333333",
    "social_full_strategic": "#D55E00",
}
STRATEGY_LABELS = {
    "social_action": "Social: Action",
    "social_action_payoff": "Social: Action + Payoff",
    "social_action_payoff_deliberate": "Social: Deliberation Prompt",
    "social_action_payoff_strategic": "Social: Strategy Prompt",
    "solo_llm": "Solo LLM",
    "ucb": "UCB",
    "oracle": "Oracle Ceiling",
    "social_full_strategic": "Social Full: Strategy Prompt",
}
MODEL_LABELS = {
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "qwen3_8b": "Qwen3-8B",
    "gpt_oss_20b": "GPT-OSS-20B",
    "ministral3_14b_reasoning": "Ministral 3 14B Reasoning",
    "algorithmic": "Algorithmic",
}
BUDGET_LABELS = {
    "carry": "Carry-over",
    "use_it_or_lose_it": "Use-it-or-lose-it",
    "not_applicable": "Not applicable",
}
BUDGET_COLORS = {
    "carry": "#0072B2",
    "use_it_or_lose_it": "#E69F00",
}
MODEL_LABEL_BY_NAME = {
    "Qwen/Qwen3-14B": "qwen3_14b",
    "Qwen/Qwen3-32B": "qwen3_32b",
    "Qwen/Qwen3-8B": "qwen3_8b",
    "openai/gpt-oss-20b": "gpt_oss_20b",
    "mistralai/Ministral-3-14B-Reasoning-2512": "ministral3_14b_reasoning",
}

ROUND_FIELDS = {
    "condition",
    "strategy",
    "model_label",
    "seed",
    "round",
    "agent_id",
    "budget_mode",
    "opening_tokens",
    "fresh_tokens",
    "tokens_observe",
    "tokens_explore",
    "tokens_exploit",
    "tokens_invalid",
    "tokens_spent",
    "unused_tokens_end_of_round",
    "tokens_expired",
    "closing_tokens",
    "pulled",
    "reward",
    "arm_mean",
    "expected_regret",
    "population_frontier_gap",
    "pulled_best_arm",
    "arm_id",
    "copy_any",
    "guidance",
    "reward_shape",
    "spatial_length_scale",
    "regime_period",
    "regime_index",
}
DECISION_FIELDS = {
    "condition",
    "strategy",
    "model_label",
    "seed",
    "round",
    "agent_id",
    "budget_mode",
    "valid",
    "allocation",
    "arm_id",
    "completion_tokens",
    "prompt_tokens",
    "observation",
    "guidance",
    "reward_shape",
    "spatial_length_scale",
    "regime_period",
    "regime_index",
    "last_reward_gap",
    "social_observation_count",
    "copy_any",
    "personally_novel_pull",
    "independent_exploration_pull",
    "copy_latest",
    "copy_support_count",
    "copy_source_agent_id",
    "copy_source_round",
    "best_personal_alternative_arm_id_before_pull",
    "best_personal_alternative_mean_before_pull",
    "selected_evidence_mean_before_pull",
    "selected_estimated_advantage_vs_best_personal_alternative",
    "best_personal_alternative_arm_true_mean",
    "copied_arm_true_advantage_vs_best_personal_alternative",
    "copied_arm_better_than_best_personal_alternative",
    "copied_pull_reward_advantage_vs_best_personal_alternative_mean",
}


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        font="sans-serif",
        font_scale=1.2,
        rc={
            "axes.grid": False,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        },
    )


def require_healthy_completed_run(summary_path: Path, summary: dict[str, Any]) -> None:
    """Refuse to analyze outputs that did not pass the inference health gate."""
    run_dir = summary_path.parent
    status_path = run_dir / "status.json"
    if not summary.get("completed") or not summary.get("quality_gate_passed"):
        raise ValueError(f"run is not complete and healthy: {run_dir}")
    if not status_path.exists():
        raise ValueError(f"run is missing status metadata: {run_dir}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed" or not status.get(
        "quality_gate_passed"
    ):
        raise ValueError(f"run failed its health gate: {run_dir}")


def resolved_budget_mode(
    summary: dict[str, Any], config: dict[str, Any]
) -> str:
    """Normalize budget labels across older summaries and V2 outputs."""
    if summary.get("policy") in {"ucb", "oracle"}:
        return "not_applicable"
    if summary.get("budget_mode"):
        return str(summary["budget_mode"])
    budget = config.get("budget", {})
    if "carry_over" not in budget:
        return "not_applicable"
    return "carry" if budget["carry_over"] else "use_it_or_lose_it"


def read_completed_runs(
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    round_events: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    environments: dict[int, list[float]] = {}

    for summary_path in sorted(runs_root.glob("*/seed_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require_healthy_completed_run(summary_path, summary)
        summaries.append(summary)
        run_dir = summary_path.parent
        resolved_config = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        summary["budget_mode"] = resolved_budget_mode(summary, resolved_config)
        strategy = summary.get(
            "strategy", resolved_config.get("strategy", summary["condition"])
        )
        model_name = summary.get(
            "model_name", resolved_config.get("model", {}).get("name")
        )
        if summary.get("policy") in {"ucb", "oracle"}:
            default_model_label = "algorithmic"
        else:
            default_model_label = MODEL_LABEL_BY_NAME.get(
                model_name, "qwen3_32b"
            )
        if summary.get("model_label"):
            model_label = summary["model_label"]
        elif model_name in MODEL_LABEL_BY_NAME:
            model_label = MODEL_LABEL_BY_NAME[model_name]
        else:
            model_label = resolved_config.get(
                "model_label", default_model_label
            )
        guidance = resolved_config.get("guidance", "neutral")
        if guidance != "neutral":
            strategy = f"{strategy}_{guidance}"
        summary["strategy"] = strategy
        summary["model_label"] = model_label
        summary["model_name"] = model_name
        summary["guidance"] = guidance
        summary["reward_shape"] = resolved_config["environment"].get(
            "reward_shape", 1.0
        )
        summary["regime_period"] = resolved_config["environment"].get(
            "regime_period"
        )

        environment = json.loads(
            (run_dir / "environment.json").read_text(encoding="utf-8")
        )
        seed = int(summary["seed"])
        means = environment["arm_means"]
        if seed in environments and environments[seed] != means:
            raise ValueError(f"environment mismatch across conditions for seed {seed}")
        environments[seed] = means

        with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event["event"] == "round_end":
                    selected = {key: event.get(key) for key in ROUND_FIELDS}
                    selected["strategy"] = strategy
                    selected["model_label"] = event.get(
                        "model_label", model_label
                    )
                    round_events.append(selected)
                elif event["event"] == "decision":
                    selected = {key: event.get(key) for key in DECISION_FIELDS}
                    selected["strategy"] = strategy
                    selected["model_label"] = event.get(
                        "model_label", model_label
                    )
                    decisions.append(selected)

    if not summaries:
        raise FileNotFoundError(f"no completed runs under {runs_root}")
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(round_events),
        pd.DataFrame(decisions),
    )


def relabel_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        display = {**STRATEGY_LABELS, **MODEL_LABELS, **BUDGET_LABELS}
        display.update({"strategy": "strategy", "model_label": "model"})
        ax.legend(handles, [display.get(label, label) for label in labels], frameon=False)


def legend_below(fig: plt.Figure, ax: plt.Axes) -> None:
    """Put the shared strategy/model legend below a single-panel figure."""
    handles, labels = ax.get_legend_handles_labels()
    if ax.legend_ is not None:
        ax.legend_.remove()
    display = {**STRATEGY_LABELS, **MODEL_LABELS, **BUDGET_LABELS}
    entries = [
        (handle, display.get(label, label))
        for handle, label in zip(handles, labels, strict=True)
        if label not in {"strategy", "model_label", "budget_mode"}
    ]
    if entries:
        legend_handles, legend_labels = zip(*entries, strict=True)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0.22, 1, 1))


def shared_legend_below(fig: plt.Figure, axes: Any) -> None:
    """Move the first populated legend below a multi-panel figure."""
    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes.flat:
        if not handles:
            handles, labels = ax.get_legend_handles_labels()
        if ax.legend_ is not None:
            ax.legend_.remove()
    display = {**STRATEGY_LABELS, **MODEL_LABELS, **BUDGET_LABELS}
    entries = [
        (handle, display.get(label, label))
        for handle, label in zip(handles, labels, strict=True)
        if label not in {"strategy", "model_label", "budget_mode"}
    ]
    if entries:
        legend_handles, legend_labels = zip(*entries, strict=True)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(5, len(legend_labels)),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0.10, 1, 1))


def completion_columns(rounds: pd.DataFrame) -> pd.DataFrame:
    """Add explicit population and completion-conditional outcomes."""
    values = rounds.copy()
    values["pulled"] = values["pulled"].fillna(False).astype(bool)
    values["reward_population"] = pd.to_numeric(
        values["reward"], errors="coerce"
    ).fillna(0.0)
    values["reward_given_pull"] = values["reward_population"].where(
        values["pulled"]
    )
    values["observed"] = (
        pd.to_numeric(values["tokens_observe"], errors="coerce").fillna(0)
        > 0
    )
    if "budget_mode" not in values:
        values["budget_mode"] = "not_applicable"
    values["budget_mode"] = values["budget_mode"].fillna("not_applicable")
    if "landscape" not in values:
        values["landscape"] = "all"
    values["landscape"] = values["landscape"].fillna("all")
    if "valid_solution" in values:
        values["valid_solution"] = (
            values["valid_solution"].fillna(False).astype(bool)
        )
        values["reward_given_valid_solution"] = values[
            "reward_population"
        ].where(values["valid_solution"])
    return values


def write_v3_mechanism_outputs(
    rounds: pd.DataFrame,
    decisions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write compact V3 compute, diversity, and endogenous-network measures."""
    identity = ["condition", "strategy", "model_label", "seed"]
    calls = decisions.copy()
    for column in ("completion_tokens", "prompt_tokens"):
        calls[column] = pd.to_numeric(calls.get(column), errors="coerce").fillna(0)
    calls["inference_tokens"] = calls["completion_tokens"] + calls["prompt_tokens"]
    compute = calls.groupby(identity, dropna=False).agg(
        decision_calls=("allocation", "size"),
        completion_tokens=("completion_tokens", "sum"),
        prompt_tokens=("prompt_tokens", "sum"),
        inference_tokens=("inference_tokens", "sum"),
    ).reset_index()
    reward = completion_columns(rounds).groupby(identity, dropna=False).agg(
        total_reward=("reward_population", "sum"),
        mean_reward=("reward_population", "mean"),
        pull_completion_rate=("pulled", "mean"),
    ).reset_index()
    compute_reward = compute.merge(reward, on=identity, validate="one_to_one")
    compute_reward["reward_per_million_inference_tokens"] = np.where(
        compute_reward["inference_tokens"] > 0,
        compute_reward["total_reward"]
        / compute_reward["inference_tokens"]
        * 1_000_000,
        np.nan,
    )
    compute_reward.to_csv(output_dir / "reward_vs_tokens_by_seed.csv", index=False)

    calls["valid_action"] = calls.get("valid", False).fillna(False).astype(int)
    calls["observed_action"] = (
        calls.get("allocation").eq("observe") & calls["valid_action"].astype(bool)
    ).astype(int)
    calls["innovated_action"] = (
        calls.get("allocation").eq("innovate") & calls["valid_action"].astype(bool)
    ).astype(int)
    calls["copied_pull"] = calls.get("copy_any", False).fillna(False).astype(int)
    calls["independent_exploration_pull"] = calls.get(
        "independent_exploration_pull", False
    ).fillna(False).astype(int)
    calls.groupby(identity, dropna=False).agg(
        valid_actions=("valid_action", "sum"),
        observations=("observed_action", "sum"),
        innovations=("innovated_action", "sum"),
        copied_pulls=("copied_pull", "sum"),
        independent_exploration_pulls=("independent_exploration_pull", "sum"),
    ).reset_index().to_csv(
        output_dir / "endogenous_actions_by_seed.csv", index=False
    )

    allocation_tokens = calls.groupby(
        identity + ["allocation"], dropna=False
    ).agg(
        calls=("allocation", "size"),
        mean_completion_tokens_when_chosen=("completion_tokens", "mean"),
        mean_prompt_tokens_when_chosen=("prompt_tokens", "mean"),
    ).reset_index()
    allocation_tokens.to_csv(
        output_dir / "tokens_conditional_on_action_by_seed.csv", index=False
    )

    pulls = rounds[rounds.get("pulled", False).fillna(False)].copy()
    pulls = pulls.dropna(subset=["arm_id"])
    diversity_rows: list[dict[str, Any]] = []
    for keys, group in pulls.groupby(identity + ["round"], dropna=False):
        counts = group["arm_id"].value_counts()
        probabilities = counts.to_numpy(dtype=float) / counts.sum()
        modal_arm = counts.index[0]
        modal_rows = group[group["arm_id"] == modal_arm]
        diversity_rows.append(
            {
                **dict(zip(identity + ["round"], keys, strict=True)),
                "unique_pulled_arms": int(len(counts)),
                "effective_arm_diversity": float(
                    np.exp(-np.sum(probabilities * np.log(probabilities)))
                ),
                "modal_arm_id": modal_arm,
                "modal_arm_share": float(counts.iloc[0] / counts.sum()),
                "modal_expected_regret": pd.to_numeric(
                    modal_rows.get("expected_regret"), errors="coerce"
                ).mean(),
                "modal_population_frontier_gap": pd.to_numeric(
                    modal_rows.get("population_frontier_gap"), errors="coerce"
                ).mean(),
            }
        )
    pd.DataFrame(diversity_rows).to_csv(
        output_dir / "population_arm_diversity_by_seed_round.csv", index=False
    )

    observation_rows: list[dict[str, Any]] = []
    for row in decisions.itertuples(index=False):
        observation = getattr(row, "observation", None)
        if isinstance(observation, dict):
            observation_rows.append(
                {
                    **{key: getattr(row, key) for key in identity},
                    "round": row.round,
                    "observer_agent_id": row.agent_id,
                    "source_agent_id": observation.get("target_id"),
                    "source_round": observation.get("source_round"),
                    "arm_id": observation.get("arm_id"),
                    "observed_reward": observation.get("reward"),
                }
            )
    pd.DataFrame(observation_rows).to_csv(
        output_dir / "observation_edges.csv", index=False
    )

    adoption_rows: list[dict[str, Any]] = []
    pull_decisions = decisions[
        decisions.get("valid", False).fillna(False)
        & decisions.get("arm_id").notna()
        & decisions.get("allocation").isin(["explore", "exploit"])
    ].copy()
    for run_keys, group in pull_decisions.groupby(identity, dropna=False):
        depths: dict[tuple[int, int, int], int] = {}
        for row in group.sort_values(["round", "agent_id"]).itertuples(index=False):
            arm_id = int(row.arm_id)
            source_agent = getattr(row, "copy_source_agent_id", None)
            source_round = getattr(row, "copy_source_round", None)
            copied = pd.notna(source_agent) and pd.notna(source_round)
            if copied:
                source_key = (int(source_agent), int(source_round), arm_id)
                copy_depth = depths.get(source_key, 0) + 1
            else:
                source_key = None
                copy_depth = 0
            depths[(int(row.agent_id), int(row.round), arm_id)] = copy_depth
            if copied:
                adoption_rows.append(
                    {
                        **dict(zip(identity, run_keys, strict=True)),
                        "round": int(row.round),
                        "copier_agent_id": int(row.agent_id),
                        "source_agent_id": int(source_agent),
                        "source_round": int(source_round),
                        "arm_id": arm_id,
                        "copy_depth": copy_depth,
                        "source_pull_found": source_key in depths,
                    }
                )
    pd.DataFrame(adoption_rows).to_csv(output_dir / "adoption_edges.csv", index=False)

    if "innovation_parent_arm_id" in decisions:
        build_on = decisions[
            decisions["innovation_parent_arm_id"].notna()
        ][
            identity
            + [
                "round",
                "agent_id",
                "arm_id",
                "innovation_parent_arm_id",
                "innovation_parent_agent_id",
                "innovation_parent_distance",
                "innovation_improved_social_parent",
            ]
        ]
        build_on.to_csv(output_dir / "socially_seeded_innovations.csv", index=False)

    models = model_facet_order(compute_reward)
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 3.3), squeeze=False, sharey=True)
    for ax, model in zip(axes.flat, models, strict=True):
        subset = compute_reward[compute_reward["model_label"] == model]
        sns.scatterplot(
            data=subset,
            x="completion_tokens",
            y="mean_reward",
            hue="strategy",
            style="strategy",
            ax=ax,
        )
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Charged Completion Tokens")
        ax.set_ylabel("Mean Reward" if ax is axes.flat[0] else "")
        ax.grid(False)
    shared_legend_below(fig, axes)
    fig.savefig(output_dir / "reward_vs_tokens.pdf")
    plt.close(fig)


def write_v3_paired_effects(rounds: pd.DataFrame, output_dir: Path) -> None:
    """Population-level paired contrasts and regime-change event studies."""
    values = completion_columns(rounds)
    for column, default in (
        ("guidance", "neutral"),
        ("reward_shape", 1.0),
        ("spatial_length_scale", 0.0),
    ):
        if column not in values:
            values[column] = default
        values[column] = values[column].fillna(default)
    if "regime_period" not in values:
        values["regime_period"] = np.nan
    if "copy_any" not in values:
        values["copy_any"] = False
    values["copy_any"] = values["copy_any"].fillna(False).astype(float)
    values["reward_on_copied_pull"] = values["reward_population"].where(
        values["copy_any"].astype(bool)
    )
    values["tokens_observe"] = pd.to_numeric(
        values.get("tokens_observe"), errors="coerce"
    ).fillna(0)
    run_keys = [
        "condition",
        "strategy",
        "guidance",
        "reward_shape",
        "spatial_length_scale",
        "regime_period",
        "model_label",
        "budget_mode",
        "seed",
    ]
    run_metrics = values.groupby(run_keys, dropna=False, as_index=False).agg(
        mean_reward=("reward_population", "mean"),
        pull_completion_rate=("pulled", "mean"),
        copy_rate=("copy_any", "mean"),
        mean_copied_pull_reward=("reward_on_copied_pull", "mean"),
        mean_observation_tokens=("tokens_observe", "mean"),
    )
    match_keys = [
        "reward_shape",
        "spatial_length_scale",
        "regime_period",
        "model_label",
        "budget_mode",
        "seed",
    ]
    solo = run_metrics[run_metrics["strategy"] == "solo_llm"][
        match_keys
        + ["mean_reward", "pull_completion_rate", "mean_observation_tokens"]
    ].rename(
        columns={
            "mean_reward": "solo_mean_reward",
            "pull_completion_rate": "solo_pull_completion_rate",
            "mean_observation_tokens": "solo_mean_observation_tokens",
        }
    )
    social = run_metrics[
        run_metrics["strategy"].str.startswith("social_action", na=False)
    ].merge(solo, on=match_keys, how="left", validate="many_to_one")
    for metric in ("mean_reward", "pull_completion_rate", "mean_observation_tokens"):
        social[f"{metric}_delta_vs_solo"] = (
            social[metric] - social[f"solo_{metric}"]
        )
    social["copied_pull_reward_delta_vs_solo"] = (
        social["mean_copied_pull_reward"] - social["solo_mean_reward"]
    )
    social.to_csv(output_dir / "v3_social_vs_solo_by_seed.csv", index=False)

    plot_values = social.dropna(
        subset=["copy_rate", "copied_pull_reward_delta_vs_solo"]
    )
    models = model_facet_order(plot_values)
    if models:
        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(4.1 * len(models), 3.3),
            squeeze=False,
            sharey=True,
        )
        for ax, model in zip(axes.flat, models, strict=True):
            sns.scatterplot(
                data=plot_values[plot_values["model_label"] == model],
                x="copy_rate",
                y="copied_pull_reward_delta_vs_solo",
                hue="strategy",
                style="strategy",
                ax=ax,
            )
            ax.set_title(MODEL_LABELS.get(model, model))
            ax.set_xlabel("Population Copying Rate")
            ax.set_ylabel(
                "Copied-Pull Reward Minus Matched Solo Reward"
                if ax is axes.flat[0]
                else ""
            )
        shared_legend_below(fig, axes)
        fig.savefig(output_dir / "copier_payoff_vs_copying_intensity.pdf")
        plt.close(fig)

    seed_round = values.groupby(
        run_keys + ["round"], dropna=False, as_index=False
    ).agg(reward=("reward_population", "mean"), copy_rate=("copy_any", "mean"))
    dynamic = seed_round[seed_round["regime_period"].notna()].copy()
    static = seed_round[seed_round["regime_period"].isna()].copy()
    if dynamic.empty or static.empty:
        return
    static_keys = [
        "strategy",
        "guidance",
        "reward_shape",
        "spatial_length_scale",
        "model_label",
        "budget_mode",
        "seed",
        "round",
    ]
    static = static[static_keys + ["reward", "copy_rate"]].rename(
        columns={"reward": "static_reward", "copy_rate": "static_copy_rate"}
    )
    event = dynamic.merge(static, on=static_keys, how="left", validate="many_to_one")
    event["reward_dynamic_minus_static"] = event["reward"] - event["static_reward"]
    event["copy_dynamic_minus_static"] = event["copy_rate"] - event["static_copy_rate"]
    event["round_since_change"] = event["round"] % event["regime_period"]
    baseline_keys = [
        "condition",
        "model_label",
        "seed",
        "regime_period",
    ]
    baseline = event[event["round"] < event["regime_period"]].groupby(
        baseline_keys, as_index=False
    ).agg(
        pre_reward_difference=("reward_dynamic_minus_static", "mean"),
        pre_copy_difference=("copy_dynamic_minus_static", "mean"),
    )
    event = event.merge(baseline, on=baseline_keys, validate="many_to_one")
    event["reward_did"] = (
        event["reward_dynamic_minus_static"] - event["pre_reward_difference"]
    )
    event["copy_rate_did"] = (
        event["copy_dynamic_minus_static"] - event["pre_copy_difference"]
    )
    event.to_csv(output_dir / "v3_regime_event_study_by_seed_round.csv", index=False)
    models = model_facet_order(event)
    if models:
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), squeeze=False, sharey=True)
        for ax, model in zip(axes.flat, models, strict=True):
            sns.lineplot(
                data=event[
                    (event["model_label"] == model)
                    & (event["round"] >= event["regime_period"])
                ],
                x="round_since_change",
                y="reward_did",
                hue="strategy",
                style="regime_period",
                estimator="mean",
                errorbar="se",
                ax=ax,
            )
            ax.set_title(MODEL_LABELS.get(model, model))
            ax.set_xlabel("Rounds Since Payoff Change")
            ax.set_ylabel("Reward Difference-in-Differences" if ax is axes.flat[0] else "")
            ax.grid(False)
        shared_legend_below(fig, axes)
        fig.savefig(output_dir / "regime_change_event_study.pdf")
        plt.close(fig)


def plot_v3_factor_sweeps(rounds: pd.DataFrame, output_dir: Path) -> None:
    values = completion_columns(rounds)
    values["guidance"] = values.get("guidance", "neutral").fillna("neutral")
    values["reward_shape"] = pd.to_numeric(
        values.get("reward_shape", 1.0), errors="coerce"
    ).fillna(1.0)
    values["spatial_length_scale"] = pd.to_numeric(
        values.get("spatial_length_scale", 0.0), errors="coerce"
    ).fillna(0.0)
    base = values[
        values["regime_period"].isna()
        & (values["budget_mode"] == "use_it_or_lose_it")
        & (values["guidance"] == "neutral")
        & values["strategy"].isin(["solo_llm", "social_action_payoff"])
    ]
    specs = [
        (
            base[base["spatial_length_scale"] == 0.0],
            "reward_shape",
            "Reward Scarcity Shape",
            "reward_by_scarcity.pdf",
        ),
        (
            base[base["reward_shape"] == 1.0],
            "spatial_length_scale",
            "Spatial Correlation Length",
            "reward_by_spatial_correlation.pdf",
        ),
    ]
    for subset, x, xlabel, filename in specs:
        if subset.empty or subset[x].nunique() < 2:
            continue
        seed_values = subset.groupby(
            ["strategy", "model_label", "seed", x], as_index=False
        )["reward_population"].mean()
        models = model_facet_order(seed_values)
        fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 3.3), squeeze=False, sharey=True)
        for ax, model in zip(axes.flat, models, strict=True):
            sns.lineplot(
                data=seed_values[seed_values["model_label"] == model],
                x=x,
                y="reward_population",
                hue="strategy",
                marker="o",
                estimator="mean",
                errorbar="se",
                ax=ax,
            )
            ax.set_title(MODEL_LABELS.get(model, model))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Mean Reward" if ax is axes.flat[0] else "")
            ax.grid(False)
        shared_legend_below(fig, axes)
        fig.savefig(output_dir / filename)
        plt.close(fig)


def write_v3_benchmark_guide(output_dir: Path, *, rung: int) -> None:
    text = f"""# V3 Rung {rung} analysis guide

V3 keeps observation, copying, exploration, innovation, and exploitation fully
endogenous. Conditions are randomized across matched environment seeds; agents
within a population are not treated as independent replicates.

- `reward_vs_tokens.pdf` compares performance with charged completion tokens;
  `reward_vs_tokens_by_seed.csv` also records prompt and total inference tokens.
- `tokens_conditional_on_action_by_seed.csv` measures token use only on calls
  where each action was actually selected, avoiding an observation floor effect.
- `endogenous_actions_by_seed.csv` records observation, copying, innovation, and
  independent-exploration totals.
- `population_arm_diversity_by_seed_round.csv` records unique arms, effective
  diversity, modal-arm share, and the modal arm's regret/frontier gap.
- `observation_edges.csv` is the who-observed-whom graph. `adoption_edges.csv` is
  the who-copied-whom graph and includes reconstructed copying depth.
- `v3_social_vs_solo_by_seed.csv` contains matched-seed population contrasts.
- `copier_payoff_vs_copying_intensity.pdf` shows whether the payoff on copied
  pulls falls relative to matched solo populations as endogenous copying rises.
- `v3_regime_event_study_by_seed_round.csv` and
  `regime_change_event_study.pdf` report pre-period-adjusted changes relative to
  matched static populations.

Raw relationships between naturally occurring copying and reward are mechanism
evidence, not standalone causal estimates. Causal claims should be attached to
the randomized information, budget, prompt, scarcity, correlation, and
volatility conditions.
"""
    if rung == 2:
        text += """
- `socially_seeded_innovations.csv` identifies innovations near socially learned
  arms and whether they improved on the current true value of that parent arm.
- `reward_by_scarcity.pdf` and `reward_by_spatial_correlation.pdf` show the two
  one-factor sweeps; they do not collapse scarcity and structure into one axis.
"""
    (output_dir / "V3_ANALYSIS_GUIDE.md").write_text(text, encoding="utf-8")


def expand_shared_baselines(values: pd.DataFrame) -> pd.DataFrame:
    """Repeat budget-free algorithmic baselines in each budget comparison."""
    budget_modes = sorted(
        set(values.loc[values["budget_mode"] != "not_applicable", "budget_mode"])
    )
    if not budget_modes:
        values = values.copy()
        values["budget_mode"] = "all"
        return values
    regular = values[values["budget_mode"] != "not_applicable"]
    shared = values[values["budget_mode"] == "not_applicable"]
    copies = []
    for budget_mode in budget_modes:
        copy = shared.copy()
        copy["budget_mode"] = budget_mode
        copies.append(copy)
    return pd.concat([regular, *copies], ignore_index=True)


def model_facet_order(values: pd.DataFrame) -> list[str]:
    """Return real model families in a stable left-to-right order."""
    present = set(values["model_label"].dropna())
    models = [
        model
        for model in MODEL_ORDER
        if model != "algorithmic" and model in present
    ]
    if models:
        return models
    return [model for model in MODEL_ORDER if model in present]


def expand_algorithmic_model_facets(values: pd.DataFrame) -> pd.DataFrame:
    """Repeat shared algorithmic baselines inside every LLM model panel."""
    models = model_facet_order(values)
    algorithmic = values[values["model_label"] == "algorithmic"]
    if algorithmic.empty or not models:
        return values
    model_specific = values[values["model_label"] != "algorithmic"]
    copies = []
    for model in models:
        copy = algorithmic.copy()
        copy["model_label"] = model
        copies.append(copy)
    return pd.concat([model_specific, *copies], ignore_index=True)


def plot_completion_aware_performance(
    rounds: pd.DataFrame,
    output_dir: Path,
    *,
    strategy_order: list[str],
    include_valid_solution: bool = False,
) -> None:
    """Plot performance for everyone and conditional on completing a pull."""
    values = expand_algorithmic_model_facets(
        expand_shared_baselines(completion_columns(rounds))
    )
    group_columns = [
        "landscape",
        "budget_mode",
        "strategy",
        "model_label",
        "seed",
        "round",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "pull_completion_rate": ("pulled", "mean"),
        "reward_population": ("reward_population", "mean"),
        "reward_given_pull": ("reward_given_pull", "mean"),
    }
    metrics = [
        ("pull_completion_rate", "Valid Pull Rate"),
        ("reward_population", "Reward (Non-Pulls = 0)"),
        ("reward_given_pull", "Reward | Valid Pull"),
    ]
    if include_valid_solution:
        aggregations.update(
            {
                "valid_solution_rate": ("valid_solution", "mean"),
                "reward_given_valid_solution": (
                    "reward_given_valid_solution",
                    "mean",
                ),
            }
        )
        metrics.extend(
            [
                ("valid_solution_rate", "Valid Solution Rate"),
                (
                    "reward_given_valid_solution",
                    "Reward | Valid Solution",
                ),
            ]
        )
    seed_round = (
        values.groupby(group_columns, as_index=False, dropna=False)
        .agg(**aggregations)
        .sort_values(group_columns)
    )
    seed_round.to_csv(
        output_dir / "completion_aware_performance_by_seed_round.csv",
        index=False,
    )

    contexts = list(
        seed_round[["landscape", "budget_mode"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    model_order = model_facet_order(seed_round)
    row_count = len(contexts) * len(metrics)
    fig, axes = plt.subplots(
        row_count,
        len(model_order),
        figsize=(5.0 * len(model_order), 3.2 * row_count + 1.0),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    present_strategies = [
        value for value in strategy_order if value in set(seed_round["strategy"])
    ]
    palette = COLORS if set(present_strategies).issubset(COLORS) else None
    for context_index, (landscape, budget_mode) in enumerate(contexts):
        subset = seed_round[
            (seed_round["landscape"] == landscape)
            & (seed_round["budget_mode"] == budget_mode)
        ]
        context = BUDGET_LABELS.get(budget_mode, budget_mode)
        if landscape != "all":
            context = f"{str(landscape).capitalize()}; {context}"
        for metric_index, (metric, ylabel) in enumerate(metrics):
            row = context_index * len(metrics) + metric_index
            for column, model in enumerate(model_order):
                ax = axes[row, column]
                sns.lineplot(
                    data=subset[subset["model_label"] == model],
                    x="round",
                    y=metric,
                    hue="strategy",
                    hue_order=present_strategies,
                    palette=palette,
                    estimator="mean",
                    errorbar="se",
                    ax=ax,
                )
                ax.set(
                    title=MODEL_LABELS.get(model, model) if row == 0 else "",
                    xlabel="Round" if row == row_count - 1 else "",
                    ylabel=(f"{ylabel}\n{context}" if column == 0 else ""),
                )
                if metric.endswith("rate"):
                    ax.set_ylim(0, 1)
    shared_legend_below(fig, axes)
    sns.despine(fig)
    fig.savefig(output_dir / "completion_aware_performance.pdf")
    plt.close(fig)


def plot_observation_tradeoffs(
    rounds: pd.DataFrame,
    output_dir: Path,
    *,
    strategy_order: list[str],
) -> None:
    """Contrast pull completion/payoff by observation and one round later."""
    values = completion_columns(rounds)
    observe_capable = values.groupby("strategy")["observed"].any()
    observe_capable = list(observe_capable[observe_capable].index)
    values = values[values["strategy"].isin(observe_capable)].copy()
    if values.empty:
        return
    values["observation_status"] = values["observed"].map(
        {False: "Did Not Observe", True: "Observed"}
    )
    group_base = [
        "landscape",
        "budget_mode",
        "strategy",
        "model_label",
        "seed",
    ]
    same_round = values.groupby(
        [*group_base, "observation_status"], as_index=False, dropna=False
    ).agg(
        pull_completion_rate=("pulled", "mean"),
        reward_population=("reward_population", "mean"),
        reward_given_pull=("reward_given_pull", "mean"),
        agent_rounds=("pulled", "size"),
    )
    same_round.to_csv(
        output_dir / "outcomes_by_observation_by_seed.csv", index=False
    )

    sort_columns = ["condition", "model_label", "seed", "agent_id", "round"]
    values = values.sort_values(sort_columns)
    agent_group = values.groupby(
        ["condition", "model_label", "seed", "agent_id"], sort=False
    )
    values["next_round"] = agent_group["round"].shift(-1)
    values["next_pulled"] = agent_group["pulled"].shift(-1)
    values["next_reward_population"] = agent_group[
        "reward_population"
    ].shift(-1)
    values["next_reward_given_pull"] = agent_group[
        "reward_given_pull"
    ].shift(-1)
    values = values[values["next_round"] == values["round"] + 1].copy()
    values["prior_observation_status"] = "Did Not Observe"
    values.loc[
        values["observed"] & values["pulled"], "prior_observation_status"
    ] = "Observed + Pulled"
    values.loc[
        values["observed"] & ~values["pulled"], "prior_observation_status"
    ] = "Observed + No Pull"
    next_round = values.groupby(
        [*group_base, "prior_observation_status"],
        as_index=False,
        dropna=False,
    ).agg(
        next_pull_completion_rate=("next_pulled", "mean"),
        next_reward_population=("next_reward_population", "mean"),
        next_reward_given_pull=("next_reward_given_pull", "mean"),
        agent_rounds=("next_pulled", "size"),
    )
    next_round.to_csv(
        output_dir / "next_round_outcomes_by_prior_observation_by_seed.csv",
        index=False,
    )

    contexts = list(
        same_round[["landscape", "budget_mode"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    panels = [
        (
            same_round,
            "observation_status",
            "pull_completion_rate",
            "Same-Round Valid Pull Rate",
        ),
        (
            same_round,
            "observation_status",
            "reward_population",
            "Same-Round Reward (Non-Pulls = 0)",
        ),
        (
            same_round,
            "observation_status",
            "reward_given_pull",
            "Same-Round Reward | Valid Pull",
        ),
        (
            next_round,
            "prior_observation_status",
            "next_pull_completion_rate",
            "Next-Round Valid Pull Rate",
        ),
        (
            next_round,
            "prior_observation_status",
            "next_reward_population",
            "Next-Round Reward (Non-Pulls = 0)",
        ),
        (
            next_round,
            "prior_observation_status",
            "next_reward_given_pull",
            "Next-Round Reward | Valid Pull",
        ),
    ]
    model_order = model_facet_order(values)
    row_count = len(contexts) * len(panels)
    fig, axes = plt.subplots(
        row_count,
        len(model_order),
        figsize=(5.0 * len(model_order), 3.2 * row_count + 1.0),
        squeeze=False,
        sharey="row",
    )
    present_strategies = [
        value for value in strategy_order if value in set(values["strategy"])
    ]
    palette = COLORS if set(present_strategies).issubset(COLORS) else None
    for context_index, (landscape, budget_mode) in enumerate(contexts):
        context = BUDGET_LABELS.get(budget_mode, budget_mode)
        if landscape != "all":
            context = f"{str(landscape).capitalize()}; {context}"
        for metric_index, (table, x, metric, ylabel) in enumerate(panels):
            subset = table[
                (table["landscape"] == landscape)
                & (table["budget_mode"] == budget_mode)
            ]
            row = context_index * len(panels) + metric_index
            for column, model in enumerate(model_order):
                ax = axes[row, column]
                sns.lineplot(
                    data=subset[subset["model_label"] == model],
                    x=x,
                    y=metric,
                    hue="strategy",
                    hue_order=present_strategies,
                    palette=palette,
                    estimator="mean",
                    errorbar="se",
                    markers=True,
                    dashes=False,
                    ax=ax,
                )
                ax.set(
                    title=MODEL_LABELS.get(model, model) if row == 0 else "",
                    xlabel="",
                    ylabel=(f"{ylabel}\n{context}" if column == 0 else ""),
                )
                ax.tick_params(axis="x", rotation=25)
                if metric.endswith("rate"):
                    ax.set_ylim(0, 1)
    shared_legend_below(fig, axes)
    sns.despine(fig)
    fig.savefig(output_dir / "observation_tradeoffs.pdf")
    plt.close(fig)


def write_completion_guide(output_dir: Path, *, rung: int) -> None:
    valid_solution_note = ""
    if rung == 3:
        valid_solution_note = (
            " Rung 3 additionally reports the rate of valid task solutions and "
            "reward conditional on a valid solution; selecting a skill and "
            "finishing its execution are distinct stages."
        )
    text = f"""# Rung {rung} completion and failure metrics

## [Completion-aware performance](completion_aware_performance.pdf)

- **Valid pull rate:** fraction of all agent-rounds ending in a parseable,
  permitted pull action.
- **Reward (non-pulls = 0):** population-level realized performance. Every
  agent-round is retained and a failure to pull receives zero.
- **Reward | valid pull:** decision quality among agents that completed a pull.
  This excludes non-pulls and must always be read alongside the valid pull rate.
{valid_solution_note}

Rows keep carry-over and use-it-or-lose-it budgets separate. Ribbons are SE
across seeds; agents within a seed are not treated as independent replicates.

## [Observation tradeoffs](observation_tradeoffs.pdf)

The first three panels compare agents that did and did not take an OBSERVE
action in the same round. The final three compare the next-round outcomes after
no observation, observation followed by a pull, and observation followed by no
pull. Both population reward and reward conditional on a valid pull are shown.

These are descriptive associations, not causal effects: agents may choose to
observe because their histories were already worse. A causal claim would need
to adjust for prior payoff, round, seed, model, strategy, and budget condition.
The underlying seed-level values are in
`outcomes_by_observation_by_seed.csv` and
`next_round_outcomes_by_prior_observation_by_seed.csv`.
"""
    (output_dir / "COMPLETION_METRICS.md").write_text(text, encoding="utf-8")


def plot_performance(rounds: pd.DataFrame, output_dir: Path) -> None:
    values = expand_algorithmic_model_facets(
        expand_shared_baselines(completion_columns(rounds))
    )
    values["pulled_best_arm"] = (
        values["pulled_best_arm"].fillna(False).astype(float)
    )
    seed_round = (
        values.groupby(
            [
                "condition",
                "strategy",
                "model_label",
                "budget_mode",
                "seed",
                "round",
            ],
            as_index=False,
        )
        .agg(
            reward=("reward_population", "mean"),
            expected_regret=("expected_regret", "mean"),
            best_arm_rate=("pulled_best_arm", "mean"),
            missed_rate=("pulled", lambda values: 1.0 - values.mean()),
        )
        .sort_values(
            ["strategy", "model_label", "budget_mode", "seed", "round"]
        )
    )
    seed_round["cumulative_expected_regret"] = seed_round.groupby(
        ["strategy", "model_label", "budget_mode", "seed"]
    )["expected_regret"].cumsum()

    present_strategies = [
        value for value in STRATEGY_ORDER if value in set(seed_round["strategy"])
    ]
    present_models = model_facet_order(seed_round)
    budget_modes = list(seed_round["budget_mode"].drop_duplicates())
    panels = [
        (
            "reward",
            "Mean Realized Reward",
            "How does realized reward change over time?",
            "reward_over_time.pdf",
        ),
        (
            "cumulative_expected_regret",
            "Cumulative Expected Regret",
            "How quickly does expected regret accumulate?",
            "cumulative_regret_over_time.pdf",
        ),
        (
            "best_arm_rate",
            "Fraction Pulling Best Arm",
            "How often do agents pull the best arm?",
            "best_arm_rate_over_time.pdf",
        ),
    ]
    for metric, ylabel, title, filename in panels:
        fig, axes = plt.subplots(
            len(budget_modes),
            len(present_models),
            figsize=(5.0 * len(present_models), 4.0 * len(budget_modes) + 1.0),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for row, budget_mode in enumerate(budget_modes):
            for column, model in enumerate(present_models):
                ax = axes[row, column]
                subset = seed_round[
                    (seed_round["budget_mode"] == budget_mode)
                    & (seed_round["model_label"] == model)
                ]
                sns.lineplot(
                    data=subset,
                    x="round",
                    y=metric,
                    hue="strategy",
                    hue_order=present_strategies,
                    palette=COLORS,
                    estimator="mean",
                    errorbar="se",
                    ax=ax,
                )
                ax.set(
                    title=(MODEL_LABELS.get(model, model) if row == 0 else ""),
                    xlabel="Round" if row == len(budget_modes) - 1 else "",
                    ylabel=(
                        f"{ylabel}\n{BUDGET_LABELS.get(budget_mode, budget_mode)}"
                        if column == 0
                        else ""
                    ),
                )
        fig.suptitle(title)
        shared_legend_below(fig, axes)
        sns.despine(fig)
        fig.savefig(output_dir / filename)
        plt.close(fig)
    seed_round.to_csv(output_dir / "performance_by_seed_round.csv", index=False)


def plot_budget(rounds: pd.DataFrame, output_dir: Path) -> None:
    budget = rounds.dropna(subset=["opening_tokens"]).copy()
    if budget.empty:
        return
    budget["budget_mode"] = budget["budget_mode"].fillna("all")
    totals = (
        budget.groupby(
            ["strategy", "model_label", "budget_mode", "seed", "round"],
            as_index=False,
        )[
            [
                "opening_tokens",
                "tokens_observe",
                "tokens_explore",
                "tokens_exploit",
                "tokens_invalid",
                "closing_tokens",
            ]
        ]
        .sum()
    )
    for category in ("observe", "explore", "exploit", "invalid"):
        totals[category] = totals[f"tokens_{category}"] / totals["opening_tokens"]
    totals["carry"] = totals["closing_tokens"] / totals["opening_tokens"]
    long = totals.melt(
        id_vars=["strategy", "model_label", "budget_mode", "seed", "round"],
        value_vars=["observe", "explore", "exploit", "invalid", "carry"],
        var_name="allocation",
        value_name="share",
    )

    present_strategies = [
        value for value in STRATEGY_ORDER if value in set(long["strategy"])
    ]
    present_models = model_facet_order(long)
    budget_modes = list(long["budget_mode"].drop_duplicates())
    questions = {
        "observe": "What share of the token budget is spent observing others?",
        "explore": "What share of the token budget is spent exploring arms?",
        "exploit": "What share of the token budget is spent exploiting known arms?",
        "invalid": "What share of the token budget is lost to invalid decisions?",
        "carry": "What share of the token budget is carried into the next round?",
    }
    for category in ["observe", "explore", "exploit", "invalid", "carry"]:
        fig, axes = plt.subplots(
            len(budget_modes),
            len(present_models),
            figsize=(5.0 * len(present_models), 4.0 * len(budget_modes) + 1.0),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for row, budget_mode in enumerate(budget_modes):
            for column, model in enumerate(present_models):
                ax = axes[row, column]
                subset = long[
                    (long["allocation"] == category)
                    & (long["budget_mode"] == budget_mode)
                    & (long["model_label"] == model)
                ]
                sns.lineplot(
                    data=subset,
                    x="round",
                    y="share",
                    hue="strategy",
                    hue_order=present_strategies,
                    palette=COLORS,
                    estimator="mean",
                    errorbar="se",
                    ax=ax,
                )
                ax.set(
                    title=(MODEL_LABELS.get(model, model) if row == 0 else ""),
                    xlabel="Round" if row == len(budget_modes) - 1 else "",
                    ylabel=(
                        "Share of Opening Budget\n"
                        f"{BUDGET_LABELS.get(budget_mode, budget_mode)}"
                        if column == 0
                        else ""
                    ),
                    ylim=(0, 1),
                )
        fig.suptitle(questions[category])
        shared_legend_below(fig, axes)
        sns.despine(fig)
        fig.savefig(output_dir / f"token_{category}_over_time.pdf")
        plt.close(fig)
    long.to_csv(output_dir / "token_allocation_by_seed_round.csv", index=False)


def plot_social_cues(decisions: pd.DataFrame, output_dir: Path) -> None:
    social = decisions[
        decisions["strategy"].str.startswith("social_action", na=False)
        & decisions["valid"].fillna(False)
    ].copy()
    if social.empty:
        return

    action_decisions = social[social["allocation"].isin(["observe", "explore", "exploit"])].copy()
    action_decisions["chose_observe"] = action_decisions["allocation"] == "observe"
    action_decisions["reward_gap_bin"] = pd.cut(
        action_decisions["last_reward_gap"],
        bins=[-float("inf"), -1.0, -0.25, 0.0, float("inf")],
        labels=["≤ -1", "-1 to -.25", "-.25 to 0", "≥ 0"],
    ).astype("object")
    action_decisions["reward_gap_bin"] = action_decisions[
        "reward_gap_bin"
    ].fillna("no prior reward")
    observe_seed = (
        action_decisions.groupby(
            [
                "strategy",
                "model_label",
                "budget_mode",
                "seed",
                "reward_gap_bin",
            ],
            as_index=False,
        )["chose_observe"]
        .mean()
    )

    pulls = social[social["allocation"].isin(["explore", "exploit"])].copy()
    pulls["observation_bin"] = pd.cut(
        pulls["social_observation_count"].fillna(0),
        bins=[-0.1, 0.5, 1.5, 3.5, float("inf")],
        labels=["0", "1", "2–3", "4+"],
    )
    copy_seed = (
        pulls.groupby(
            [
                "strategy",
                "model_label",
                "budget_mode",
                "seed",
                "observation_bin",
            ],
            as_index=False,
            observed=True,
        )[
            "copy_any"
        ]
        .mean()
    )
    copy_round = (
        pulls.groupby(
            ["strategy", "model_label", "budget_mode", "seed", "round"],
            as_index=False,
        )["copy_any"]
        .mean()
    )

    present_strategies = [
        value for value in STRATEGY_ORDER if value in set(social["strategy"])
    ]
    present_models = model_facet_order(social)

    def plot_model_facets(
        data: pd.DataFrame,
        *,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        filename: str,
        rotate_x: bool = False,
    ) -> None:
        budget_modes = list(data["budget_mode"].drop_duplicates())
        fig, axes = plt.subplots(
            len(budget_modes),
            len(present_models),
            figsize=(5.0 * len(present_models), 4.0 * len(budget_modes) + 1.0),
            squeeze=False,
            sharey=True,
        )
        for row, budget_mode in enumerate(budget_modes):
            for column, model in enumerate(present_models):
                ax = axes[row, column]
                subset = data[
                    (data["budget_mode"] == budget_mode)
                    & (data["model_label"] == model)
                ]
                sns.lineplot(
                    data=subset,
                    x=x,
                    y=y,
                    hue="strategy",
                    hue_order=present_strategies,
                    palette=COLORS,
                    errorbar="se",
                    markers=True,
                    ax=ax,
                )
                ax.set(
                    title=(MODEL_LABELS.get(model, model) if row == 0 else ""),
                    xlabel=xlabel if row == len(budget_modes) - 1 else "",
                    ylabel=(
                        f"{ylabel}\n{BUDGET_LABELS.get(budget_mode, budget_mode)}"
                        if column == 0
                        else ""
                    ),
                    ylim=(0, 1),
                )
                if rotate_x:
                    ax.tick_params(axis="x", rotation=25)
        fig.suptitle(title)
        shared_legend_below(fig, axes)
        sns.despine(fig)
        fig.savefig(output_dir / filename)
        plt.close(fig)

    plot_model_facets(
        observe_seed,
        x="reward_gap_bin",
        y="chose_observe",
        title="Does poor recent performance trigger observation?",
        xlabel="Last Reward − Best Known Mean",
        ylabel="P(Observe)",
        filename="observe_response_to_reward_gap.pdf",
        rotate_x=True,
    )
    plot_model_facets(
        copy_seed,
        x="observation_bin",
        y="copy_any",
        title="Does more social evidence increase copying?",
        xlabel="Successful Observations Held",
        ylabel="P(Pull Observed Arm)",
        filename="copy_rate_by_observations.pdf",
    )
    budget_order = [
        budget
        for budget in ["carry", "use_it_or_lose_it"]
        if budget in set(copy_round["budget_mode"])
    ]
    fig, axes = plt.subplots(
        1,
        len(present_models),
        figsize=(5.0 * len(present_models), 4.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for column, model in enumerate(present_models):
        ax = axes[0, column]
        sns.lineplot(
            data=copy_round[copy_round["model_label"] == model],
            x="round",
            y="copy_any",
            hue="strategy",
            hue_order=present_strategies,
            style="budget_mode",
            style_order=budget_order,
            dashes={"carry": "", "use_it_or_lose_it": (4, 2)},
            palette=COLORS,
            estimator="mean",
            errorbar="se",
            ax=ax,
        )
        ax.set(
            title=MODEL_LABELS.get(model, model),
            xlabel="Round",
            ylabel="P(Pull Observed Arm)" if column == 0 else "",
            ylim=(0, 1),
        )
    fig.suptitle("How Does Copying Change over the Tournament?")
    shared_legend_below(fig, axes)
    sns.despine(fig)
    fig.savefig(output_dir / "copy_rate_over_time.pdf")
    plt.close(fig)
    observe_seed.to_csv(output_dir / "observe_cue_by_seed.csv", index=False)
    copy_seed.to_csv(output_dir / "copy_cue_by_seed.csv", index=False)
    copy_round.to_csv(output_dir / "copy_rate_by_seed_round.csv", index=False)


def plot_copy_quality_vs_personal_alternative(
    decisions: pd.DataFrame,
    output_dir: Path,
    *,
    context_column: str | None = None,
) -> None:
    """Measure copied-arm quality against the agent's best personal option."""
    copied = decisions[
        decisions["strategy"].str.startswith("social_action", na=False)
        & decisions["valid"].fillna(False)
        & decisions["allocation"].isin(["explore", "exploit"])
        & decisions["copy_any"].fillna(False)
    ].copy()
    if copied.empty:
        return
    copied["budget_mode"] = copied["budget_mode"].fillna("not_applicable")
    copied["has_personal_comparator"] = copied[
        "copied_arm_true_advantage_vs_best_personal_alternative"
    ].notna()
    group_columns = [
        *([context_column] if context_column else []),
        "strategy",
        "model_label",
        "budget_mode",
        "seed",
    ]
    coverage = copied.groupby(group_columns, as_index=False).agg(
        copied_pulls=("copy_any", "size"),
        copied_pulls_with_personal_comparator=(
            "has_personal_comparator",
            "sum",
        ),
        comparator_coverage=("has_personal_comparator", "mean"),
    )
    comparable = copied[copied["has_personal_comparator"]].copy()
    comparable["copied_arm_better_rate"] = comparable[
        "copied_arm_better_than_best_personal_alternative"
    ].astype(float)
    quality = comparable.groupby(group_columns, as_index=False).agg(
        mean_true_advantage=(
            "copied_arm_true_advantage_vs_best_personal_alternative",
            "mean",
        ),
        copied_arm_better_rate=("copied_arm_better_rate", "mean"),
        mean_realized_reward_advantage=(
            "copied_pull_reward_advantage_vs_best_personal_alternative_mean",
            "mean",
        ),
    )
    seed_quality = coverage.merge(
        quality, on=group_columns, how="left", validate="one_to_one"
    )
    seed_quality.to_csv(
        output_dir / "copy_quality_vs_personal_alternative_by_seed.csv",
        index=False,
    )
    summary_metrics = [
        "copied_pulls",
        "copied_pulls_with_personal_comparator",
        "comparator_coverage",
        "mean_true_advantage",
        "copied_arm_better_rate",
        "mean_realized_reward_advantage",
    ]
    summary = seed_quality.groupby(
        [
            *([context_column] if context_column else []),
            "strategy",
            "model_label",
            "budget_mode",
        ],
        dropna=False,
    )[summary_metrics].agg(["mean", "sem"])
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary.reset_index().to_csv(
        output_dir / "copy_quality_vs_personal_alternative_summary.csv",
        index=False,
    )

    models = model_facet_order(seed_quality)
    strategies = [
        strategy
        for strategy in STRATEGY_ORDER
        if strategy in set(seed_quality["strategy"])
    ]
    budget_order = [
        budget
        for budget in ["carry", "use_it_or_lose_it"]
        if budget in set(seed_quality["budget_mode"])
    ]
    contexts = (
        list(seed_quality[context_column].drop_duplicates())
        if context_column
        else [None]
    )
    fig, axes = plt.subplots(
        len(contexts),
        len(models),
        figsize=(5.0 * len(models), 4.0 * len(contexts) + 1.0),
        squeeze=False,
        sharey=True,
    )
    for row, context in enumerate(contexts):
        for column, model in enumerate(models):
            ax = axes[row, column]
            subset = seed_quality[seed_quality["model_label"] == model]
            if context_column:
                subset = subset[subset[context_column] == context]
            sns.barplot(
                data=subset,
                x="strategy",
                y="copied_arm_better_rate",
                order=strategies,
                hue="budget_mode",
                hue_order=budget_order,
                palette=BUDGET_COLORS,
                estimator="mean",
                errorbar="se",
                capsize=0.12,
                ax=ax,
            )
            ylabel = "P(Copied Arm Better Than Best Personally Discovered Arm)"
            if context is not None:
                ylabel = f"{ylabel}\n{str(context).capitalize()}"
            ax.set(
                title=MODEL_LABELS.get(model, model) if row == 0 else "",
                xlabel=(
                    "Social Information Condition"
                    if row == len(contexts) - 1
                    else ""
                ),
                ylabel=ylabel if column == 0 else "",
                ylim=(0, 1),
            )
            ax.set_xticks(
                range(len(strategies)),
                [
                    STRATEGY_LABELS.get(strategy, strategy)
                    for strategy in strategies
                ],
            )
            ax.grid(False)
    fig.suptitle(
        "Copied Arm Quality Relative to Personal Discovery"
    )
    shared_legend_below(fig, axes)
    sns.despine(fig)
    fig.savefig(output_dir / "copy_quality_vs_personal_alternative.pdf")
    plt.close(fig)


def write_rung1_report_tables(
    summaries: pd.DataFrame,
    rounds: pd.DataFrame,
    decisions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write compact mean/SE tables used by the Rung 1 interpretation."""
    run_performance = summaries[
        [
            "strategy",
            "model_label",
            "seed",
            "budget_mode",
            "rounds",
            "num_agents",
            "mean_reward_per_agent_round",
            "mean_expected_regret_per_agent_round",
            "missed_pulls",
        ]
    ].copy()
    run_performance["budget_mode"] = run_performance["budget_mode"].fillna(
        "not_applicable"
    )
    best_arm = rounds.copy()
    best_arm["budget_mode"] = best_arm["budget_mode"].fillna("not_applicable")
    best_arm["pulled_best_arm"] = best_arm["pulled_best_arm"].fillna(False).astype(float)
    best_arm = best_arm.groupby(
        ["strategy", "model_label", "seed", "budget_mode"], as_index=False
    )["pulled_best_arm"].mean().rename(columns={"pulled_best_arm": "best_arm_rate"})
    run_performance = run_performance.merge(
        best_arm,
        on=["strategy", "model_label", "seed", "budget_mode"],
        validate="one_to_one",
    )
    run_performance["cumulative_expected_regret"] = (
        run_performance["mean_expected_regret_per_agent_round"]
        * run_performance["rounds"]
    )
    run_performance["missed_pull_rate"] = run_performance["missed_pulls"] / (
        run_performance["rounds"] * run_performance["num_agents"]
    )
    oracle = run_performance[run_performance["strategy"] == "oracle"][
        ["seed", "mean_reward_per_agent_round"]
    ].rename(columns={"mean_reward_per_agent_round": "oracle_reward"})
    run_performance = run_performance.merge(oracle, on="seed", validate="many_to_one")
    run_performance["reward_gap_to_oracle"] = (
        run_performance["mean_reward_per_agent_round"]
        - run_performance["oracle_reward"]
    )
    run_performance["reward_fraction_of_oracle"] = (
        run_performance["mean_reward_per_agent_round"]
        / run_performance["oracle_reward"]
    )
    performance_metrics = [
        "mean_reward_per_agent_round",
        "cumulative_expected_regret",
        "best_arm_rate",
        "missed_pull_rate",
        "reward_gap_to_oracle",
        "reward_fraction_of_oracle",
    ]
    performance_summary = run_performance.groupby(
        ["strategy", "model_label", "budget_mode"]
    )[performance_metrics].agg(["mean", "sem"])
    performance_summary.columns = [
        f"{metric}_{stat}" for metric, stat in performance_summary.columns
    ]
    performance_summary.reset_index().to_csv(
        output_dir / "performance_summary.csv", index=False
    )
    solo = run_performance[run_performance["strategy"] == "solo_llm"][
        [
            "model_label",
            "seed",
            "budget_mode",
            "mean_reward_per_agent_round",
            "cumulative_expected_regret",
            "best_arm_rate",
        ]
    ].rename(
        columns={
            "mean_reward_per_agent_round": "solo_reward",
            "cumulative_expected_regret": "solo_cumulative_regret",
            "best_arm_rate": "solo_best_arm_rate",
        }
    )
    social_comparison = run_performance[
        run_performance["strategy"].isin(
            ["social_action", "social_action_payoff"]
        )
    ].merge(
        solo,
        on=["model_label", "seed", "budget_mode"],
        validate="many_to_one",
    )
    social_comparison["reward_delta_vs_solo"] = (
        social_comparison["mean_reward_per_agent_round"]
        - social_comparison["solo_reward"]
    )
    social_comparison["cumulative_regret_delta_vs_solo"] = (
        social_comparison["cumulative_expected_regret"]
        - social_comparison["solo_cumulative_regret"]
    )
    social_comparison["best_arm_rate_delta_vs_solo"] = (
        social_comparison["best_arm_rate"]
        - social_comparison["solo_best_arm_rate"]
    )
    comparison_metrics = [
        "reward_delta_vs_solo",
        "cumulative_regret_delta_vs_solo",
        "best_arm_rate_delta_vs_solo",
    ]
    comparison_summary = social_comparison.groupby(
        ["strategy", "model_label", "budget_mode"]
    )[comparison_metrics].agg(["mean", "sem"])
    comparison_summary.columns = [
        f"{metric}_{stat}" for metric, stat in comparison_summary.columns
    ]
    comparison_summary.reset_index().to_csv(
        output_dir / "social_vs_solo_summary.csv", index=False
    )

    budget = rounds.dropna(subset=["opening_tokens"]).groupby(
        ["strategy", "model_label", "seed", "budget_mode", "round"],
        as_index=False,
        dropna=False,
    )[
        [
            "opening_tokens",
            "tokens_observe",
            "tokens_explore",
            "tokens_exploit",
            "tokens_invalid",
            "closing_tokens",
        ]
    ].sum()
    for category in ("observe", "explore", "exploit", "invalid"):
        budget[category] = budget[f"tokens_{category}"] / budget["opening_tokens"]
    budget["carry"] = budget["closing_tokens"] / budget["opening_tokens"]
    budget_seed = budget.groupby(
        ["strategy", "model_label", "seed", "budget_mode"],
        as_index=False,
        dropna=False,
    )[["observe", "explore", "exploit", "invalid", "carry"]].mean()
    budget_long = budget_seed.melt(
        id_vars=["strategy", "model_label", "seed", "budget_mode"],
        var_name="allocation",
        value_name="share",
    )
    budget_summary = budget_long.groupby(
        ["strategy", "model_label", "budget_mode", "allocation"],
        dropna=False,
    )["share"].agg(["mean", "sem"]).reset_index()
    budget_summary.to_csv(output_dir / "token_allocation_summary.csv", index=False)

    social = decisions[
        decisions["strategy"].str.startswith("social_action", na=False)
        & decisions["valid"].fillna(False)
    ].copy()
    social["budget_mode"] = social["budget_mode"].fillna("not_applicable")
    social["observed"] = (social["allocation"] == "observe").astype(float)
    observe_seed = social.groupby(
        ["strategy", "model_label", "seed", "budget_mode"], as_index=False
    )["observed"].mean()
    pulls = social[social["allocation"].isin(["explore", "exploit"])].copy()
    pulls["copied"] = pulls["copy_any"].fillna(False).astype(float)
    copy_seed = pulls.groupby(
        ["strategy", "model_label", "seed", "budget_mode"], as_index=False
    )["copied"].mean()
    behavior_seed = observe_seed.merge(
        copy_seed,
        on=["strategy", "model_label", "seed", "budget_mode"],
        validate="one_to_one",
    )
    behavior_seed["not_copied"] = 1.0 - behavior_seed["copied"]
    observations = summaries[
        summaries["strategy"].str.startswith("social_action", na=False)
    ][
        [
            "strategy",
            "model_label",
            "seed",
            "budget_mode",
            "successful_observations",
            "num_agents",
        ]
    ].copy()
    observations["budget_mode"] = observations["budget_mode"].fillna(
        "not_applicable"
    )
    observations["observations_per_agent"] = (
        observations["successful_observations"] / observations["num_agents"]
    )
    behavior_seed = behavior_seed.merge(
        observations[
            [
                "strategy",
                "model_label",
                "seed",
                "budget_mode",
                "observations_per_agent",
            ]
        ],
        on=["strategy", "model_label", "seed", "budget_mode"],
        validate="one_to_one",
    )
    behavior_summary = behavior_seed.groupby(
        ["strategy", "model_label", "budget_mode"]
    )[
        ["observed", "copied", "not_copied", "observations_per_agent"]
    ].agg(["mean", "sem"])
    behavior_summary.columns = [
        f"{metric}_{stat}" for metric, stat in behavior_summary.columns
    ]
    behavior_summary.reset_index().to_csv(
        output_dir / "social_behavior_summary.csv", index=False
    )

    cue_specs = [
        ("observe_cue_by_seed.csv", "reward_gap_bin", "chose_observe"),
        ("copy_cue_by_seed.csv", "observation_bin", "copy_any"),
    ]
    for filename, bin_column, metric in cue_specs:
        values = pd.read_csv(output_dir / filename)
        cue_summary = values.groupby(
            ["strategy", "model_label", bin_column], observed=True
        )[metric].agg(["mean", "sem"]).reset_index()
        cue_summary.to_csv(
            output_dir / filename.replace("_by_seed.csv", "_summary.csv"),
            index=False,
        )
    copy_round = pd.read_csv(output_dir / "copy_rate_by_seed_round.csv")
    copy_round.groupby(["strategy", "model_label", "round"])["copy_any"].agg(
        ["mean", "sem"]
    ).reset_index().to_csv(output_dir / "copy_rate_by_round_summary.csv", index=False)


def read_completed_rung2(
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    discovery: list[dict[str, Any]] = []
    environments: dict[tuple[Any, ...], dict[str, Any]] = {}

    for summary_path in sorted(runs_root.glob("*/seed_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("rung", 1)) != 2:
            continue
        require_healthy_completed_run(summary_path, summary)
        run_dir = summary_path.parent
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        environment = json.loads(
            (run_dir / "environment.json").read_text(encoding="utf-8")
        )
        strategy = summary.get("strategy", config["strategy"])
        guidance = config.get("guidance", "neutral")
        if guidance != "neutral":
            strategy = f"{strategy}_{guidance}"
        model_label = summary.get("model_label", config["model_label"])
        landscape = summary.get("landscape", environment["landscape"])
        summary.update(
            {
                "strategy": strategy,
                "model_label": model_label,
                "landscape": landscape,
                "budget_mode": resolved_budget_mode(summary, config),
                "guidance": guidance,
                "reward_shape": config["environment"].get("reward_shape", 1.0),
                "spatial_length_scale": (
                    config["environment"].get("structured_length_scale", 0.15)
                    if landscape == "structured"
                    else 0.0
                ),
                "regime_period": config["environment"].get("regime_period"),
            }
        )
        summaries.append(summary)

        environment_key = (
            int(summary["seed"]),
            landscape,
            float(environment.get("reward_shape", 1.0)),
            float(environment.get("structured_length_scale", 0.0)),
            environment.get("regime_period"),
            bool(environment.get("coordinate_innovation", False)),
        )
        if (
            environment_key in environments
            and environments[environment_key] != environment
        ):
            raise ValueError(
                f"environment mismatch for seed {summary['seed']} {landscape}"
            )
        environments[environment_key] = environment

        run_decisions = [
            [] for _ in range(int(summary["rounds"]))
        ]
        arm_means: dict[int, float] = {}
        with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
            event_stream = (json.loads(line) for line in handle)
            for event in event_stream:
                base = {
                    "condition": summary["condition"],
                    "strategy": strategy,
                    "model_label": model_label,
                    "landscape": landscape,
                    "seed": int(summary["seed"]),
                    "round": int(event["round"]),
                    "agent_id": int(event["agent_id"]),
                    "budget_mode": event.get("budget_mode"),
                    "guidance": summary["guidance"],
                    "reward_shape": summary["reward_shape"],
                    "spatial_length_scale": summary["spatial_length_scale"],
                    "regime_period": summary["regime_period"],
                }
                if event["event"] == "round_end":
                    rounds.append(
                        {
                            **base,
                            "opening_tokens": event.get("opening_tokens"),
                            "fresh_tokens": event.get("fresh_tokens"),
                            "tokens_observe": event.get("tokens_observe"),
                            "tokens_innovate": event.get("tokens_innovate"),
                            "tokens_explore": event.get("tokens_explore"),
                            "tokens_exploit": event.get("tokens_exploit"),
                            "tokens_invalid": event.get("tokens_invalid"),
                            "tokens_spent": event.get("tokens_spent"),
                            "unused_tokens_end_of_round": event.get(
                                "unused_tokens_end_of_round"
                            ),
                            "tokens_expired": event.get("tokens_expired"),
                            "closing_tokens": event.get("closing_tokens"),
                            "pulled": event.get("pulled"),
                            "reward": event.get("reward"),
                            "arm_mean": event.get("arm_mean"),
                            "expected_regret": event.get("expected_regret"),
                            "population_frontier_gap": event.get(
                                "population_frontier_gap"
                            ),
                            "copy_any": event.get("copy_any"),
                            "arm_id": event.get("arm_id"),
                            "guidance": event.get("guidance", "neutral"),
                            "reward_shape": event.get("reward_shape", 1.0),
                            "spatial_length_scale": event.get(
                                "spatial_length_scale", 0.0
                            ),
                            "regime_period": event.get("regime_period"),
                            "regime_index": event.get("regime_index", 0),
                        }
                    )
                    continue
                if event["event"] != "decision":
                    continue
                compact = {
                    **base,
                    "valid": event.get("valid"),
                    "allocation": event.get("allocation"),
                    "arm_id": event.get("arm_id"),
                    "coordinate": event.get("coordinate"),
                    "arm_mean": event.get("arm_mean"),
                    "probe_reward": event.get("probe_reward"),
                    "completion_tokens": event.get("completion_tokens"),
                    "prompt_tokens": event.get("prompt_tokens"),
                    "guidance": event.get("guidance", "neutral"),
                    "reward_shape": event.get("reward_shape", 1.0),
                    "spatial_length_scale": event.get(
                        "spatial_length_scale", 0.0
                    ),
                    "regime_period": event.get("regime_period"),
                    "regime_index": event.get("regime_index", 0),
                    "novel_to_population": event.get("novel_to_population"),
                    "innovation_improved_true_frontier": event.get(
                        "innovation_improved_true_frontier"
                    ),
                    "innovation_parent_arm_id": event.get(
                        "innovation_parent_arm_id"
                    ),
                    "innovation_parent_agent_id": event.get(
                        "innovation_parent_agent_id"
                    ),
                    "innovation_parent_distance": event.get(
                        "innovation_parent_distance"
                    ),
                    "innovation_improved_social_parent": event.get(
                        "innovation_improved_social_parent"
                    ),
                    "best_known_mean": event.get("best_known_mean"),
                    "last_reward_gap": event.get("last_reward_gap"),
                    "rounds_since_improvement": event.get(
                        "rounds_since_improvement"
                    ),
                    "known_arm_count": event.get("known_arm_count"),
                    "social_observation_count": event.get(
                        "social_observation_count"
                    ),
                    "copy_any": event.get("copy_any"),
                    "personally_novel_pull": event.get("personally_novel_pull"),
                    "independent_exploration_pull": event.get(
                        "independent_exploration_pull"
                    ),
                    "copy_source_agent_id": event.get("copy_source_agent_id"),
                    "copy_source_round": event.get("copy_source_round"),
                    "best_personal_alternative_arm_id_before_pull": event.get(
                        "best_personal_alternative_arm_id_before_pull"
                    ),
                    "best_personal_alternative_mean_before_pull": event.get(
                        "best_personal_alternative_mean_before_pull"
                    ),
                    "selected_evidence_mean_before_pull": event.get(
                        "selected_evidence_mean_before_pull"
                    ),
                    "selected_estimated_advantage_vs_best_personal_alternative": event.get(
                        "selected_estimated_advantage_vs_best_personal_alternative"
                    ),
                    "copied_arm_true_advantage_vs_best_personal_alternative": event.get(
                        "copied_arm_true_advantage_vs_best_personal_alternative"
                    ),
                    "copied_arm_better_than_best_personal_alternative": event.get(
                        "copied_arm_better_than_best_personal_alternative"
                    ),
                    "copied_pull_reward_advantage_vs_best_personal_alternative_mean": event.get(
                        "copied_pull_reward_advantage_vs_best_personal_alternative_mean"
                    ),
                    "observation": event.get("observation"),
                }
                decisions.append(compact)
                run_decisions[int(compact["round"])].append(compact)
                if compact["valid"] and compact["allocation"] == "innovate":
                    arm_means[int(compact["arm_id"])] = float(
                        compact["arm_mean"]
                    )

        known_by_agent = [set() for _ in range(int(summary["num_agents"]))]
        population_arms: set[int] = set()
        harmonic = 0.0
        previous_innovation_count = 0
        for round_index in range(int(summary["rounds"])):
            round_innovations = 0
            for event in run_decisions[round_index]:
                if not event.get("valid"):
                    continue
                agent_id = int(event["agent_id"])
                if event.get("allocation") == "innovate":
                    arm_id = int(event["arm_id"])
                    population_arms.add(arm_id)
                    known_by_agent[agent_id].add(arm_id)
                    round_innovations += 1
                elif event.get("allocation") == "observe":
                    observation = event.get("observation")
                    if observation is not None:
                        known_by_agent[agent_id].add(int(observation["arm_id"]))

            innovation_count = len(population_arms)
            for index in range(previous_innovation_count + 1, innovation_count + 1):
                harmonic += 1.0 / index
            previous_innovation_count = innovation_count
            population_frontier = max(
                (arm_means[arm_id] for arm_id in population_arms), default=0.0
            )
            agent_frontiers = [
                max((arm_means[arm_id] for arm_id in known), default=0.0)
                for known in known_by_agent
            ]
            reference_best = environment.get("reference_best_mean")
            discovery.append(
                {
                    "condition": summary["condition"],
                    "strategy": strategy,
                    "model_label": model_label,
                    "landscape": landscape,
                    "budget_mode": summary["budget_mode"],
                    "guidance": summary["guidance"],
                    "reward_shape": summary["reward_shape"],
                    "spatial_length_scale": summary["spatial_length_scale"],
                    "regime_period": summary["regime_period"],
                    "seed": int(summary["seed"]),
                    "round": round_index,
                    "innovations_per_agent": round_innovations
                    / int(summary["num_agents"]),
                    "cumulative_unique_arms": innovation_count,
                    "population_frontier": population_frontier,
                    "mean_agent_frontier": sum(agent_frontiers)
                    / len(agent_frontiers),
                    "benchmark_gap": (
                        harmonic * environment["arm_mean_scale"]
                        - population_frontier
                        if landscape == "unstructured"
                        else max(float(reference_best) - population_frontier, 0.0)
                    ),
                }
            )

    if not summaries:
        raise FileNotFoundError(f"no completed rung-2 runs under {runs_root}")
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(rounds),
        pd.DataFrame(decisions),
        pd.DataFrame(discovery),
    )


def _orders(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    strategies = [
        value for value in STRATEGY_ORDER if value in set(data["strategy"])
    ]
    models = [value for value in MODEL_ORDER if value in set(data["model_label"])]
    return strategies, models


def _rung2_lineplot(
    *, data: pd.DataFrame, x: str, y: str, ax: plt.Axes, errorbar: str = "se"
) -> None:
    strategies, _ = _orders(data)
    budget_order = [
        budget
        for budget in ["carry", "use_it_or_lose_it"]
        if budget in set(data["budget_mode"])
    ]
    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue="strategy",
        hue_order=strategies,
        style="budget_mode",
        style_order=budget_order,
        dashes={"carry": "", "use_it_or_lose_it": (4, 2)},
        palette=COLORS,
        estimator="mean",
        errorbar=errorbar,
        ax=ax,
    )


def plot_rung2_performance(rounds: pd.DataFrame, output_dir: Path) -> None:
    values = rounds.copy()
    values["reward_population"] = pd.to_numeric(
        values["reward"], errors="coerce"
    ).fillna(0.0)
    values["arm_mean"] = values["arm_mean"].fillna(0.0)
    seed_round = (
        values.groupby(
            [
                "landscape",
                "strategy",
                "model_label",
                "budget_mode",
                "seed",
                "round",
            ],
            as_index=False,
        )
        .agg(
            reward=("reward_population", "mean"),
            latent_reward=("arm_mean", "mean"),
            missed_rate=("pulled", lambda pulled: 1.0 - pulled.mean()),
        )
        .sort_values(
            [
                "landscape",
                "strategy",
                "model_label",
                "budget_mode",
                "seed",
                "round",
            ]
        )
    )
    seed_round["cumulative_reward"] = seed_round.groupby(
        ["landscape", "strategy", "model_label", "budget_mode", "seed"]
    )["reward"].cumsum()

    landscapes = ["unstructured", "structured"]
    panels = [
        ("reward", "Mean Realized Reward"),
        ("cumulative_reward", "Cumulative Mean Reward"),
        ("latent_reward", "Mean Latent Reward"),
        ("missed_rate", "Fraction Without a Pull"),
    ]
    models = model_facet_order(seed_round)
    row_count = len(landscapes) * len(panels)
    fig, axes = plt.subplots(
        row_count,
        len(models),
        figsize=(5.0 * len(models), 3.1 * row_count + 1.0),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for landscape_index, landscape in enumerate(landscapes):
        for metric_index, (metric, ylabel) in enumerate(panels):
            row = landscape_index * len(panels) + metric_index
            for column, model in enumerate(models):
                ax = axes[row, column]
                subset = seed_round[
                    (seed_round["landscape"] == landscape)
                    & (seed_round["model_label"] == model)
                ]
                _rung2_lineplot(data=subset, x="round", y=metric, ax=ax)
                ax.set(
                    title=MODEL_LABELS.get(model, model) if row == 0 else "",
                    xlabel="Round" if row == row_count - 1 else "",
                    ylabel=(
                        f"{ylabel}\n{landscape.capitalize()}"
                        if column == 0
                        else ""
                    ),
                )
                if metric == "missed_rate":
                    ax.set_ylim(0, 1)
    fig.suptitle("Rung 2 Performance")
    shared_legend_below(fig, axes)
    sns.despine(fig)
    fig.savefig(output_dir / "performance.pdf")
    plt.close(fig)
    seed_round.to_csv(output_dir / "performance_by_seed_round.csv", index=False)


def plot_rung2_budget(rounds: pd.DataFrame, output_dir: Path) -> None:
    totals = (
        rounds.groupby(
            [
                "landscape",
                "strategy",
                "model_label",
                "budget_mode",
                "seed",
                "round",
            ],
            as_index=False,
        )[
            [
                "opening_tokens",
                "tokens_observe",
                "tokens_innovate",
                "tokens_explore",
                "tokens_exploit",
                "closing_tokens",
            ]
        ]
        .sum()
    )
    categories = ["observe", "innovate", "explore", "exploit", "carry"]
    for category in categories[:-1]:
        totals[category] = (
            totals[f"tokens_{category}"] / totals["opening_tokens"]
        )
    totals["carry"] = totals["closing_tokens"] / totals["opening_tokens"]
    long = totals.melt(
        id_vars=[
            "landscape",
            "strategy",
            "model_label",
            "budget_mode",
            "seed",
            "round",
        ],
        value_vars=categories,
        var_name="allocation",
        value_name="share",
    )

    landscapes = ["unstructured", "structured"]
    models = model_facet_order(long)
    row_count = len(landscapes) * len(categories)
    fig, axes = plt.subplots(
        row_count,
        len(models),
        figsize=(5.0 * len(models), 3.0 * row_count + 1.0),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for landscape_index, landscape in enumerate(landscapes):
        for category_index, category in enumerate(categories):
            row = landscape_index * len(categories) + category_index
            for column, model in enumerate(models):
                ax = axes[row, column]
                subset = long[
                    (long["landscape"] == landscape)
                    & (long["allocation"] == category)
                    & (long["model_label"] == model)
                ]
                _rung2_lineplot(data=subset, x="round", y="share", ax=ax)
                ax.set(
                    title=MODEL_LABELS.get(model, model) if row == 0 else "",
                    xlabel="Round" if row == row_count - 1 else "",
                    ylabel=(
                        f"{category.capitalize()} Token Share\n"
                        f"{landscape.capitalize()}"
                        if column == 0
                        else ""
                    ),
                    ylim=(0, 1),
                )
    fig.suptitle("Rung 2 Token Allocation")
    shared_legend_below(fig, axes)
    sns.despine(fig)
    fig.savefig(output_dir / "token_allocation.pdf")
    plt.close(fig)
    long.to_csv(output_dir / "token_allocation_by_seed_round.csv", index=False)


def plot_rung2_discovery(discovery: pd.DataFrame, output_dir: Path) -> None:
    panels = [
        ("innovations_per_agent", "Innovations per Agent"),
        ("cumulative_unique_arms", "Population Unique Arms"),
        ("population_frontier", "Population Latent Frontier"),
        ("mean_agent_frontier", "Mean Agent Latent Frontier"),
        ("benchmark_gap", "Frontier Benchmark Gap"),
    ]
    landscapes = ["unstructured", "structured"]
    models = model_facet_order(discovery)
    row_count = len(landscapes) * len(panels)
    fig, axes = plt.subplots(
        row_count,
        len(models),
        figsize=(5.0 * len(models), 3.0 * row_count + 1.0),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for landscape_index, landscape in enumerate(landscapes):
        for metric_index, (metric, ylabel) in enumerate(panels):
            row = landscape_index * len(panels) + metric_index
            for column, model in enumerate(models):
                ax = axes[row, column]
                subset = discovery[
                    (discovery["landscape"] == landscape)
                    & (discovery["model_label"] == model)
                ]
                _rung2_lineplot(data=subset, x="round", y=metric, ax=ax)
                ax.set(
                    title=MODEL_LABELS.get(model, model) if row == 0 else "",
                    xlabel="Round" if row == row_count - 1 else "",
                    ylabel=(
                        f"{ylabel}\n{landscape.capitalize()}"
                        if column == 0
                        else ""
                    ),
                )
    fig.suptitle("Rung 2 Discovery Dynamics")
    shared_legend_below(fig, axes)
    sns.despine(fig)
    fig.savefig(output_dir / "discovery.pdf")
    plt.close(fig)
    discovery.to_csv(output_dir / "discovery_by_seed_round.csv", index=False)


def plot_rung2_cues(decisions: pd.DataFrame, output_dir: Path) -> None:
    valid = decisions[decisions["valid"].fillna(False)].copy()
    valid["chose_innovate"] = valid["allocation"] == "innovate"
    valid["known_bin"] = pd.cut(
        valid["known_arm_count"].fillna(0),
        bins=[-0.1, 0.5, 1.5, 4.5, 9.5, float("inf")],
        labels=["0", "1", "2–4", "5–9", "10+"],
    )
    valid["quality_bin"] = pd.cut(
        valid["best_known_mean"],
        bins=[-float("inf"), 0.0, 0.5, 1.0, 2.0, float("inf")],
        labels=["≤0", "0–.5", ".5–1", "1–2", "2+"],
    ).astype("object")
    valid["quality_bin"] = valid["quality_bin"].fillna("None")
    valid["stagnation_bin"] = pd.cut(
        valid["rounds_since_improvement"],
        bins=[-0.1, 0.5, 2.5, 5.5, 10.5, float("inf")],
        labels=["0", "1–2", "3–5", "6–10", "11+"],
    ).astype("object")
    valid["stagnation_bin"] = valid["stagnation_bin"].fillna("Never")

    social = valid[
        valid["strategy"].str.startswith("social_action", na=False)
    ].copy()
    social["chose_observe"] = social["allocation"] == "observe"
    social["reward_gap_bin"] = pd.cut(
        social["last_reward_gap"],
        bins=[-float("inf"), -1.0, -0.25, 0.0, float("inf")],
        labels=["≤-1", "-1–-.25", "-.25–0", "≥0"],
    ).astype("object")
    social["reward_gap_bin"] = social["reward_gap_bin"].fillna("None")
    pulls = social[social["allocation"].isin(["explore", "exploit"])].copy()
    pulls["observation_bin"] = pd.cut(
        pulls["social_observation_count"].fillna(0),
        bins=[-0.1, 0.5, 1.5, 3.5, float("inf")],
        labels=["0", "1", "2–3", "4+"],
    )

    cue_specs = [
        (valid, "known_bin", "chose_innovate", "Known Arms", "P(Innovate)"),
        (valid, "quality_bin", "chose_innovate", "Best Evidence", "P(Innovate)"),
        (
            valid,
            "stagnation_bin",
            "chose_innovate",
            "Rounds Since Improvement",
            "P(Innovate)",
        ),
        (social, "reward_gap_bin", "chose_observe", "Last Reward Gap", "P(Observe)"),
        (pulls, "observation_bin", "copy_any", "Observations Held", "P(Copy)"),
    ]
    seed_tables = []
    landscapes = ["unstructured", "structured"]
    models = model_facet_order(valid)
    row_count = len(landscapes) * len(cue_specs)
    fig, axes = plt.subplots(
        row_count,
        len(models),
        figsize=(5.0 * len(models), 3.2 * row_count + 1.0),
        squeeze=False,
        sharey="row",
    )
    for landscape_index, landscape in enumerate(landscapes):
        for cue_index, (data, x, y, xlabel, ylabel) in enumerate(cue_specs):
            subset = data[data["landscape"] == landscape]
            seed_table = (
                subset.groupby(
                    [
                        "landscape",
                        "strategy",
                        "model_label",
                        "budget_mode",
                        "seed",
                        x,
                    ],
                    as_index=False,
                    observed=True,
                )[y]
                .mean()
            )
            seed_table["cue"] = x
            seed_tables.append(seed_table)
            row = landscape_index * len(cue_specs) + cue_index
            for column, model in enumerate(models):
                ax = axes[row, column]
                _rung2_lineplot(
                    data=seed_table[seed_table["model_label"] == model],
                    x=x,
                    y=y,
                    ax=ax,
                )
                ax.set(
                    title=MODEL_LABELS.get(model, model) if row == 0 else "",
                    xlabel=xlabel,
                    ylabel=(
                        f"{ylabel}\n{landscape.capitalize()}"
                        if column == 0
                        else ""
                    ),
                    ylim=(0, 1),
                )
                ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Rung 2 Learning Cues")
    shared_legend_below(fig, axes)
    sns.despine(fig)
    fig.savefig(output_dir / "learning_cues.pdf")
    plt.close(fig)
    pd.concat(seed_tables, ignore_index=True).to_csv(
        output_dir / "learning_cues_by_seed.csv", index=False
    )


def analyze_rung2(
    runs_root: Path,
    output_dir: Path,
    *,
    models: list[str] | None = None,
) -> None:
    summaries, rounds, decisions, discovery = read_completed_rung2(runs_root)
    if models:
        selected_models = set(models)
        summaries = summaries[summaries["model_label"].isin(selected_models)]
        rounds = rounds[rounds["model_label"].isin(selected_models)]
        decisions = decisions[decisions["model_label"].isin(selected_models)]
        discovery = discovery[discovery["model_label"].isin(selected_models)]
        missing = selected_models - set(summaries["model_label"])
        if missing:
            raise ValueError(f"no healthy rung-2 runs for models: {sorted(missing)}")
    model_count = len(models) if models else 3
    expected_per_model = (
        208 if "v3" in runs_root.parts else 60 if "v2" in runs_root.parts else 30
    )
    expected = expected_per_model * model_count
    if len(summaries) != expected:
        raise ValueError(
            "rung 2 is incomplete: "
            f"found {len(summaries)} healthy runs, expected {expected}"
        )
    summaries.to_csv(output_dir / "condition_summary_by_seed.csv", index=False)
    numeric = [
        "mean_reward_per_agent_round",
        "mean_latent_reward_per_agent_round",
        "mean_expected_regret_per_pull",
        "missed_pulls",
        "total_innovations",
    ]
    summaries[numeric] = summaries[numeric].apply(
        pd.to_numeric, errors="coerce"
    )
    summary_groups = (
        [
            "condition",
            "landscape",
            "strategy",
            "guidance",
            "reward_shape",
            "spatial_length_scale",
            "regime_period",
            "model_label",
            "budget_mode",
        ]
        if "v3" in runs_root.parts
        else ["landscape", "strategy", "model_label", "budget_mode"]
    )
    condition_summary = summaries.groupby(
        summary_groups, dropna=False
    )[numeric].agg(["mean", "sem"])
    condition_summary.columns = [
        f"{metric}_{stat}" for metric, stat in condition_summary.columns
    ]
    condition_summary.reset_index().to_csv(
        output_dir / "condition_summary.csv", index=False
    )
    run_metrics = summaries[
        [
            "landscape",
            "strategy",
            "model_label",
            "budget_mode",
            "guidance",
            "reward_shape",
            "spatial_length_scale",
            "regime_period",
            "seed",
            "num_agents",
            "rounds",
            *numeric,
        ]
    ].copy()
    run_metrics["missed_pull_rate"] = run_metrics["missed_pulls"] / (
        run_metrics["num_agents"] * run_metrics["rounds"]
    )
    paired_metrics = [
        "mean_reward_per_agent_round",
        "mean_latent_reward_per_agent_round",
        "mean_expected_regret_per_pull",
        "total_innovations",
        "missed_pull_rate",
    ]
    comparison_keys = ["landscape", "model_label", "budget_mode", "seed"]
    if "v3" in runs_root.parts:
        comparison_keys += ["reward_shape", "spatial_length_scale", "regime_period"]
    solo = run_metrics[
        (run_metrics["strategy"] == "solo_llm")
        & (run_metrics["guidance"] == "neutral")
    ][
        [*comparison_keys, *paired_metrics]
    ].rename(columns={metric: f"solo_{metric}" for metric in paired_metrics})
    social_vs_solo = run_metrics[
        run_metrics["strategy"].str.startswith("social_action", na=False)
    ].merge(
        solo,
        on=comparison_keys,
        validate="many_to_one",
    )
    delta_metrics = []
    for metric in paired_metrics:
        delta = f"{metric}_delta_vs_solo"
        social_vs_solo[delta] = (
            social_vs_solo[metric] - social_vs_solo[f"solo_{metric}"]
        )
        delta_metrics.append(delta)
    social_vs_solo.to_csv(
        output_dir / "social_vs_solo_by_seed.csv", index=False
    )
    paired_summary_groups = ["landscape", "strategy", "model_label", "budget_mode"]
    if "v3" in runs_root.parts:
        paired_summary_groups += [
            "guidance",
            "reward_shape",
            "spatial_length_scale",
            "regime_period",
        ]
    paired_summary = social_vs_solo.groupby(
        paired_summary_groups,
        dropna=False,
    )[delta_metrics].agg(["mean", "sem"])
    paired_summary.columns = [
        f"{metric}_{stat}" for metric, stat in paired_summary.columns
    ]
    paired_summary.reset_index().to_csv(
        output_dir / "social_vs_solo_summary.csv", index=False
    )
    if "v3" in runs_root.parts:
        anchor_rounds = rounds[
            (pd.to_numeric(rounds["reward_shape"]) == 1.0)
            & (pd.to_numeric(rounds["spatial_length_scale"]).isin([0.0, 0.15]))
            & (rounds["regime_period"].isna())
            & (rounds["budget_mode"] == "use_it_or_lose_it")
        ]
        anchor_decisions = decisions[
            (pd.to_numeric(decisions["reward_shape"]) == 1.0)
            & (pd.to_numeric(decisions["spatial_length_scale"]).isin([0.0, 0.15]))
            & (decisions["regime_period"].isna())
            & (decisions["budget_mode"] == "use_it_or_lose_it")
        ]
        anchor_discovery = discovery[
            (pd.to_numeric(discovery["reward_shape"]) == 1.0)
            & (pd.to_numeric(discovery["spatial_length_scale"]).isin([0.0, 0.15]))
            & (discovery["regime_period"].isna())
            & (discovery["budget_mode"] == "use_it_or_lose_it")
        ]
    else:
        anchor_rounds, anchor_decisions, anchor_discovery = rounds, decisions, discovery
    plot_rung2_performance(anchor_rounds, output_dir)
    plot_rung2_budget(anchor_rounds, output_dir)
    plot_rung2_discovery(anchor_discovery, output_dir)
    plot_rung2_cues(anchor_decisions, output_dir)
    plot_copy_quality_vs_personal_alternative(
        anchor_decisions, output_dir, context_column="landscape"
    )
    if "v3" in runs_root.parts:
        write_v3_mechanism_outputs(rounds, decisions, output_dir)
        write_v3_paired_effects(rounds, output_dir)
        plot_v3_factor_sweeps(rounds, output_dir)
        write_v3_benchmark_guide(output_dir, rung=2)
    plot_completion_aware_performance(
        anchor_rounds, output_dir, strategy_order=STRATEGY_ORDER
    )
    plot_observation_tradeoffs(
        anchor_rounds, output_dir, strategy_order=STRATEGY_ORDER
    )
    write_completion_guide(output_dir, rung=2)


RUNG3_STRATEGY_ORDER = [
    "solo",
    "social_id",
    "social_payoff",
    "social_reputation",
    "social_full",
    "social_full_strategic",
    "skill_oracle",
]


def read_completed_rung3(
    runs_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    solutions: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    exposures: list[dict[str, Any]] = []
    copies: dict[tuple[str, int, int, int], tuple[int, str | None]] = {}

    for summary_path in sorted(runs_root.glob("*/seed_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("rung", 1)) != 3:
            continue
        require_healthy_completed_run(summary_path, summary)
        summaries.append(summary)
        run_dir = summary_path.parent
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summary["budget_mode"] = resolved_budget_mode(summary, config)
        guidance = config.get("guidance", "neutral")
        summary["guidance"] = guidance
        strategy = summary.get("strategy", config["strategy"])
        if guidance != "neutral":
            strategy = f"{strategy}_{guidance}"
        summary["strategy"] = strategy
        base_run = {
            "condition": summary["condition"],
            "strategy": strategy,
            "guidance": guidance,
            "social_info": config["social_info"],
            "model_label": summary["model_label"],
            "seed": int(summary["seed"]),
        }
        run_key = str(run_dir)
        with (run_dir / "events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                base = {
                    **base_run,
                    "round": int(event["round"]),
                    "agent_id": int(event["agent_id"]),
                    "budget_mode": event.get("budget_mode"),
                }
                if event["event"] == "round_end":
                    rounds.append(
                        {
                            **base,
                            **{
                                key: event.get(key)
                                for key in (
                                    "opening_tokens",
                                    "tokens_observe",
                                    "tokens_explore",
                                    "tokens_exploit",
                                    "tokens_invalid",
                                    "tokens_pull",
                                    "closing_tokens",
                                    "pulled",
                                    "skill_id",
                                    "pull_type",
                                    "reward",
                                    "expected_regret",
                                    "valid_solution",
                                    "copied_this_pull",
                                    "independently_acquired_this_pull",
                                    "acquisition_source",
                                    "owned_skill_count",
                                    "candidate_skill_count",
                                )
                            },
                        }
                    )
                    continue
                if event["event"] in {"decision", "solution"}:
                    calls.append(
                        {
                            **base,
                            "stage": event.get("stage"),
                            "allocation": event.get("allocation"),
                            "completion_tokens": event.get("completion_tokens", 0),
                            "prompt_tokens": event.get("prompt_tokens"),
                            "requested_max_tokens": event.get("requested_max_tokens"),
                            "effective_max_tokens": event.get("effective_max_tokens"),
                            "context_limited": event.get("context_limited", False),
                        }
                    )
                if event["event"] == "decision" and event.get("observation"):
                    observation = event["observation"]
                    exposures.append(
                        {
                            **base,
                            "run_key": run_key,
                            **observation,
                        }
                    )
                if event["event"] != "solution":
                    continue
                cue = event.get("copy_cue") or {}
                row = {
                    **base,
                    "skill_id": event.get("skill_id"),
                    "skill_intended_rank": event.get("skill_intended_rank"),
                    "pull_type": event.get("pull_type"),
                    "reward": event.get("reward"),
                    "expected_regret": event.get("expected_regret"),
                    "valid_solution": event.get("valid"),
                    "achieved_profit": event.get("achieved_profit"),
                    "optimal_profit": event.get("optimal_profit"),
                    "policy_tokens": event.get("policy_tokens", 0),
                    "solution_tokens": event.get("solution_tokens", 0),
                    "copied_this_pull": event.get("copied_this_pull", False),
                    "independently_acquired_this_pull": event.get("independently_acquired_this_pull", False),
                    "acquisition_source": event.get("acquisition_source"),
                    "copy_recency": cue.get("copy_recency"),
                    "copy_support_count": cue.get("copy_support_count"),
                    "copy_source_count": cue.get("copy_source_count"),
                    "copy_source_agent_id": cue.get("target_id"),
                    "copy_source_round": cue.get("source_round"),
                    "cue_latest_reward": cue.get("latest_reward"),
                    "cue_global_mean_reward": cue.get("global_mean_reward"),
                    "cue_global_reward_count": cue.get("global_reward_count"),
                    "cue_adoption_count": cue.get("adoption_count"),
                    "cue_holder_count": cue.get("holder_count"),
                    "cue_description_visible": cue.get("description") is not None,
                    "cue_body_visible": cue.get("skill_body") is not None,
                }
                solutions.append(row)
                if row["copied_this_pull"]:
                    copies[(run_key, base["agent_id"], int(row["skill_id"]), base["round"])] = (
                        base["round"],
                        cue.get("observation_id"),
                    )

        # Link every observed candidate to the first later adoption in this run.
        run_exposures = [row for row in exposures if row["run_key"] == run_key]
        run_copies = [
            (agent_id, skill_id, copy_round, observation_id)
            for (key, agent_id, skill_id, _), (copy_round, observation_id) in copies.items()
            if key == run_key
        ]
        for exposure in run_exposures:
            matching = [
                value
                for value in run_copies
                if value[0] == exposure["agent_id"]
                and value[1] == int(exposure["skill_id"])
                and value[2] >= exposure["round"]
            ]
            first = min(matching, key=lambda value: value[2]) if matching else None
            exposure["eventually_adopted"] = first is not None
            exposure["adoption_lag"] = first[2] - exposure["round"] if first else None
            exposure["proximal_cue"] = (
                first is not None and first[3] == exposure.get("observation_id")
            )

    if not summaries:
        raise FileNotFoundError(f"no completed rung-3 runs under {runs_root}")
    exposure_frame = pd.DataFrame(exposures)
    if not exposure_frame.empty:
        exposure_frame = exposure_frame.drop(columns=["run_key"])
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(rounds),
        pd.DataFrame(solutions),
        pd.DataFrame(calls),
        exposure_frame,
    )


def analyze_rung3(runs_root: Path, output_dir: Path) -> None:
    summaries, rounds, solutions, calls, exposures = read_completed_rung3(runs_root)
    expected = 96 if "v3_acquisition" in runs_root.parts else 120 if "v3" in runs_root.parts else 180 if "v2" in runs_root.parts else 90
    if len(summaries) != expected:
        raise ValueError(
            "rung 3 is incomplete: "
            f"found {len(summaries)} healthy runs, expected {expected}"
        )
    summaries.to_csv(output_dir / "condition_summary_by_seed.csv", index=False)
    summary_metrics = [
        "mean_reward_per_agent_round",
        "mean_expected_regret_per_agent_round",
        "missed_pulls",
        "invalid_solutions",
        "successful_observations",
        "total_copies",
        "context_limited_calls",
    ]
    if "total_independent_acquisitions" in summaries:
        summary_metrics.append("total_independent_acquisitions")
    condition_summary = summaries.groupby(
        ["strategy", "model_label", "budget_mode"], dropna=False
    )[
        summary_metrics
    ].agg(["mean", "sem"])
    condition_summary.columns = [
        f"{metric}_{stat}" for metric, stat in condition_summary.columns
    ]
    condition_summary.reset_index().to_csv(
        output_dir / "condition_summary.csv", index=False
    )

    seed_round = (
        rounds.groupby(["strategy", "model_label", "seed", "round"], as_index=False)
        .agg(
            reward=("reward", "mean"),
            regret=("expected_regret", "mean"),
            valid_rate=("valid_solution", "mean"),
            copy_rate=("copied_this_pull", "mean"),
            owned_skills=("owned_skill_count", "mean"),
        )
    )
    seed_round.to_csv(output_dir / "performance_by_seed_round.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, metric, ylabel in zip(
        axes,
        ["reward", "valid_rate", "copy_rate"],
        ["normalized reward", "valid schedule rate", "copy rate"],
        strict=True,
    ):
        sns.lineplot(
            data=seed_round,
            x="round",
            y=metric,
            hue="strategy",
            hue_order=RUNG3_STRATEGY_ORDER,
            style="model_label",
            estimator="mean",
            errorbar="se",
            ax=ax,
        )
        ax.set(xlabel="round", ylabel=ylabel)
        if ax is not axes[0] and ax.legend_ is not None:
            ax.legend_.remove()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output_dir / "performance.pdf")
    plt.close(fig)

    budget = rounds.groupby(
        ["strategy", "model_label", "seed", "round"], as_index=False
    )[
        [
            "opening_tokens",
            "tokens_observe",
            "tokens_explore",
            "tokens_exploit",
            "tokens_invalid",
            "closing_tokens",
        ]
    ].sum()
    categories = ["observe", "explore", "exploit", "invalid", "carry"]
    for category in categories[:-1]:
        budget[category] = budget[f"tokens_{category}"] / budget["opening_tokens"]
    budget["carry"] = budget["closing_tokens"] / budget["opening_tokens"]
    budget_long = budget.melt(
        id_vars=["strategy", "model_label", "seed", "round"],
        value_vars=categories,
        var_name="allocation",
        value_name="share",
    )
    budget_long.to_csv(output_dir / "token_allocation_by_seed_round.csv", index=False)
    fig, axes = plt.subplots(1, 5, figsize=(19, 4), sharex=True, sharey=True)
    for ax, category in zip(axes, categories, strict=True):
        sns.lineplot(
            data=budget_long[budget_long["allocation"] == category],
            x="round",
            y="share",
            hue="strategy",
            hue_order=RUNG3_STRATEGY_ORDER,
            style="model_label",
            estimator="mean",
            errorbar="se",
            ax=ax,
        )
        ax.set(title=category, xlabel="round", ylabel="share", ylim=(0, 1))
        if ax is not axes[0] and ax.legend_ is not None:
            ax.legend_.remove()
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output_dir / "token_allocation.pdf")
    plt.close(fig)

    skill_quality = (
        solutions.groupby(
            ["strategy", "model_label", "seed", "skill_id", "skill_intended_rank"],
            as_index=False,
        )
        .agg(
            mean_reward=("reward", "mean"),
            pulls=("reward", "size"),
            valid_rate=("valid_solution", "mean"),
            mean_solution_tokens=("solution_tokens", "mean"),
        )
    )
    skill_quality.to_csv(output_dir / "skill_quality_by_seed.csv", index=False)
    skill_use = (
        solutions.groupby(
            [
                "condition",
                "strategy",
                "model_label",
                "seed",
                "round",
                "skill_id",
                "skill_intended_rank",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "agents_using_skill"})
    )
    population_sizes = summaries[
        ["condition", "model_label", "seed", "num_agents"]
    ].drop_duplicates()
    skill_use = skill_use.merge(
        population_sizes,
        on=["condition", "model_label", "seed"],
        validate="many_to_one",
    )
    skill_use["population_share"] = (
        skill_use["agents_using_skill"] / skill_use["num_agents"]
    )
    skill_use.to_csv(output_dir / "skill_use_by_seed_round.csv", index=False)
    panel_rows = math.ceil(len(RUNG3_STRATEGY_ORDER) / 3)
    fig, axes = plt.subplots(panel_rows, 3, figsize=(15, 4 * panel_rows), sharex=True, sharey=True, squeeze=False)
    for ax, strategy in zip(axes.flat, RUNG3_STRATEGY_ORDER):
        sns.lineplot(
            data=skill_use[skill_use["strategy"] == strategy],
            x="round",
            y="population_share",
            hue="skill_id",
            palette="viridis",
            estimator="mean",
            errorbar="se",
            ax=ax,
        )
        ax.set(title=strategy, xlabel="round", ylabel="population use share", ylim=(0, 1))
        if ax is not axes.flat[0] and ax.legend_ is not None:
            ax.legend_.remove()
    for ax in axes.flat[len(RUNG3_STRATEGY_ORDER):]:
        ax.set_visible(False)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(output_dir / "skill_diffusion.pdf")
    plt.close(fig)
    calls.to_csv(output_dir / "generation_calls.csv", index=False)
    solutions.to_csv(output_dir / "solutions.csv", index=False)
    exposures.to_csv(output_dir / "social_candidate_exposures.csv", index=False)

    if "v3" in runs_root.parts or "v3_acquisition" in runs_root.parts:
        identity = ["condition", "strategy", "model_label", "seed"]
        compute = calls.copy()
        compute["completion_tokens"] = pd.to_numeric(
            compute["completion_tokens"], errors="coerce"
        ).fillna(0)
        compute["prompt_tokens"] = pd.to_numeric(
            compute["prompt_tokens"], errors="coerce"
        ).fillna(0)
        compute["inference_tokens"] = (
            compute["completion_tokens"] + compute["prompt_tokens"]
        )
        compute = compute.groupby(identity, as_index=False).agg(
            generation_calls=("allocation", "size"),
            completion_tokens=("completion_tokens", "sum"),
            prompt_tokens=("prompt_tokens", "sum"),
            inference_tokens=("inference_tokens", "sum"),
        )
        compute = compute.merge(
            summaries[identity + ["mean_reward_per_agent_round"]],
            on=identity,
            validate="one_to_one",
        )
        compute.to_csv(output_dir / "reward_vs_tokens_by_seed.csv", index=False)
        calls.groupby(identity + ["stage", "allocation"], dropna=False).agg(
            calls=("allocation", "size"),
            mean_completion_tokens_when_chosen=("completion_tokens", "mean"),
            mean_prompt_tokens_when_chosen=("prompt_tokens", "mean"),
        ).reset_index().to_csv(
            output_dir / "tokens_conditional_on_action_by_seed.csv", index=False
        )
        models = model_facet_order(compute)
        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(4.1 * len(models), 3.3),
            squeeze=False,
            sharey=True,
        )
        for ax, model in zip(axes.flat, models, strict=True):
            sns.scatterplot(
                data=compute[compute["model_label"] == model],
                x="completion_tokens",
                y="mean_reward_per_agent_round",
                hue="strategy",
                style="strategy",
                ax=ax,
            )
            ax.set_title(MODEL_LABELS.get(model, model))
            ax.set_xlabel("Charged Completion Tokens")
            ax.set_ylabel("Mean Reward" if ax is axes.flat[0] else "")
        shared_legend_below(fig, axes)
        fig.savefig(output_dir / "reward_vs_tokens.pdf")
        plt.close(fig)

        if not exposures.empty:
            exposures.rename(
                columns={"agent_id": "observer_agent_id", "target_id": "source_agent_id"}
            ).to_csv(output_dir / "observation_edges.csv", index=False)
        adoption_columns = identity + [
            "round",
            "agent_id",
            "copy_source_agent_id",
            "copy_source_round",
            "skill_id",
        ]
        adoptions = solutions[solutions["copied_this_pull"].fillna(False)][
            adoption_columns
        ].rename(
            columns={
                "agent_id": "copier_agent_id",
                "copy_source_agent_id": "source_agent_id",
                "copy_source_round": "source_round",
            }
        )
        adoptions.to_csv(output_dir / "adoption_edges.csv", index=False)

    if not exposures.empty:
        exposure_summary = (
            exposures.groupby(["strategy", "model_label", "seed"], as_index=False)
            .agg(
                adoption_rate=("eventually_adopted", "mean"),
                proximal_rate=("proximal_cue", "mean"),
                mean_adoption_lag=("adoption_lag", "mean"),
            )
        )
        exposure_summary.to_csv(
            output_dir / "copy_cues_by_seed.csv", index=False
        )
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        sns.pointplot(
            data=exposure_summary,
            x="strategy",
            y="adoption_rate",
            hue="model_label",
            order=RUNG3_STRATEGY_ORDER,
            errorbar="se",
            ax=axes[0],
        )
        sns.pointplot(
            data=exposure_summary,
            x="strategy",
            y="mean_adoption_lag",
            hue="model_label",
            order=RUNG3_STRATEGY_ORDER,
            errorbar="se",
            ax=axes[1],
        )
        axes[0].set(xlabel="information condition", ylabel="candidate adoption rate", ylim=(0, 1))
        axes[1].set(xlabel="information condition", ylabel="rounds to adoption")
        for ax in axes:
            ax.tick_params(axis="x", rotation=30)
        if axes[1].legend_ is not None:
            axes[1].legend_.remove()
        sns.despine(fig)
        fig.tight_layout()
        fig.savefig(output_dir / "copy_cues.pdf")
        plt.close(fig)

    plot_completion_aware_performance(
        rounds,
        output_dir,
        strategy_order=RUNG3_STRATEGY_ORDER,
        include_valid_solution=True,
    )
    plot_observation_tradeoffs(
        rounds, output_dir, strategy_order=RUNG3_STRATEGY_ORDER
    )
    write_completion_guide(output_dir, rung=3)


def analyze_rung3_replay(runs_root: Path, output_dir: Path) -> None:
    """Decompose recorded selection coverage from fixed-budget execution."""
    rows = []
    for path in sorted(runs_root.glob("*/seed_*/summary.json")):
        summary = json.loads(path.read_text())
        status_path = path.parent / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if not summary.get("completed") or not status.get("quality_gate_passed"):
            continue
        config = json.loads((path.parent / "config.json").read_text())
        if config.get("policy") != "skill_replay":
            continue
        source = Path(config["source_run"])
        source_config = json.loads((source / "config.json").read_text())
        source_summary = json.loads((source / "summary.json").read_text())
        if not source_summary.get("completed"):
            raise ValueError(f"incomplete replay source: {source}")

        def solutions(directory):
            result = {}
            with (directory / "events.jsonl").open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get("event") != "solution":
                        continue
                    key = (int(event["round"]), int(event["agent_id"]))
                    if key in result:
                        raise ValueError(f"duplicate solution event: {directory}, {key}")
                    result[key] = event
            return result

        original, replay = solutions(source), solutions(path.parent)
        if set(original) != set(replay):
            raise ValueError(f"replay did not preserve recorded selections: {path.parent}")
        for key, event in replay.items():
            prior = original[key]
            if not np.isclose(event["source_reward"], prior["reward"]):
                raise ValueError(f"replay source reward mismatch: {path.parent}, {key}")
            if config.get("execution_replicate", 0) == 0 and (
                event.get("executor_seed") != prior.get("executor_seed")
                or event.get("executor_input_sha256") != prior.get("executor_input_sha256")
            ):
                raise ValueError(f"replay executor pairing mismatch: {path.parent}, {key}")

        opportunities = int(config["environment"]["num_agents"]) * int(
            config["environment"]["rounds"]
        )
        selected = len(original)
        source_reward = sum(float(event["reward"]) for event in original.values())
        replay_reward = sum(float(event["reward"]) for event in replay.values())
        rows.append(
            {
                "run": str(path.parent),
                "source_run": str(source),
                "model": config["model_label"],
                "seed": int(config["seed"]),
                "selection_condition": source_config["condition"],
                "selection_strategy": source_config["strategy"],
                "guidance": source_config.get("guidance", "neutral"),
                "opportunities": opportunities,
                "recorded_selections": selected,
                "selection_coverage": selected / opportunities,
                "source_population_reward": source_reward / opportunities,
                "fixed_budget_population_reward": replay_reward / opportunities,
                "fixed_budget_reward_effect": (replay_reward - source_reward) / opportunities,
                "source_execution_completion": sum(
                    bool(event["valid_solution"]) for event in original.values()
                ) / opportunities,
                "fixed_budget_execution_completion": sum(
                    bool(event["valid_solution"]) for event in replay.values()
                ) / opportunities,
                "source_reward_given_selection": source_reward / selected if selected else np.nan,
                "fixed_budget_reward_given_selection": replay_reward / selected if selected else np.nan,
                "source_execution_tokens": sum(
                    int(event.get("solution_tokens", 0)) for event in original.values()
                ),
                "fixed_budget_execution_tokens": sum(
                    int(event.get("solution_tokens", 0)) for event in replay.values()
                ),
            }
        )

    result = pd.DataFrame(rows)
    if len(result) != 96:
        raise ValueError(f"rung-3 replay is incomplete: found {len(result)} healthy runs, expected 96")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "selection_execution_by_seed.csv", index=False)
    metrics = [
        "selection_coverage",
        "source_population_reward",
        "fixed_budget_population_reward",
        "fixed_budget_reward_effect",
        "source_execution_completion",
        "fixed_budget_execution_completion",
        "source_reward_given_selection",
        "fixed_budget_reward_given_selection",
        "source_execution_tokens",
        "fixed_budget_execution_tokens",
    ]
    grouped = result.groupby(
        ["model", "selection_strategy", "guidance"], dropna=False
    )[metrics].agg(["mean", "sem", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.to_csv(output_dir / "selection_execution_summary.csv")

    contrasts = []
    for (model, seed), group in result.groupby(["model", "seed"]):
        solo = group[group.selection_strategy == "solo"]
        if len(solo) != 1:
            raise ValueError(f"expected one solo replay for {model}, seed {seed}")
        solo = solo.iloc[0]
        for _, social in group[group.selection_strategy != "solo"].iterrows():
            original_gap = social.source_population_reward - solo.source_population_reward
            fixed_gap = social.fixed_budget_population_reward - solo.fixed_budget_population_reward
            contrasts.append(
                {
                    "model": model,
                    "seed": seed,
                    "social_condition": social.selection_condition,
                    "social_strategy": social.selection_strategy,
                    "guidance": social.guidance,
                    "original_social_gap": original_gap,
                    "fixed_budget_social_gap": fixed_gap,
                    "execution_budget_did": fixed_gap - original_gap,
                }
            )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output_dir / "social_gap_execution_did_by_seed.csv", index=False)
    contrast_summary = contrast_frame.groupby(
        ["model", "social_strategy", "guidance"], dropna=False
    )[["original_social_gap", "fixed_budget_social_gap", "execution_budget_did"]].agg(
        ["mean", "sem", "count"]
    )
    contrast_summary.columns = ["_".join(column) for column in contrast_summary.columns]
    contrast_summary.to_csv(output_dir / "social_gap_execution_did_summary.csv")
    (output_dir / "PLOT_GUIDE.md").write_text(
        "# Selection–execution decomposition\n\n"
        "`selection_execution_by_seed.csv` is the primary population-level evidence. "
        "Selection coverage is the fraction of all agent-rounds for which the original "
        "policy recorded a skill choice. Source outcomes use the original residual budget; "
        "fixed-budget outcomes replay exactly those choices with an 8192-token execution "
        "grant, paired by model, task, skill, seed and executor input. Missing original "
        "choices remain missing and count as zero population reward.\n\n"
        "`social_gap_execution_did_by_seed.csv` asks whether equalizing execution budget "
        "changes each matched social-minus-solo reward gap. It isolates execution conditional "
        "on the endogenous recorded selection trajectory; it does not repair a missed choice "
        "or estimate the closed-loop behavior that a different budget would have induced. "
        "Aggregate only across the eight seed/population rows, never agent-rounds.\n"
    )


def independent_copy_comparisons(events, mean_at, *, include_all_pulls=False):
    """Replay acquisition provenance; compare against the prior independent frontier.

    R1 discovery is a first pull before observing that arm. R2 discovery is
    a first INNOVATE before observing that arm (a probe need not be pulled).
    Later observation never erases independent provenance. Social-first and
    independent-first sets are disjoint. Legacy ever-observed pulls remain in
    the returned records for audit; social-first quality filters social_first.
    """
    known, independent, social = {}, {}, {}
    for event in events:
        if event.get("event") != "decision" or not event.get("valid"):
            continue
        agent = event["agent_id"]
        seen = known.setdefault(agent, set())
        found = independent.setdefault(agent, set())
        shared = social.setdefault(agent, set())
        action = event.get("parsed_action")
        if action == "observe":
            observation = event.get("observation")
            if observation is not None:
                observed_arm = int(observation["arm_id"])
                if observed_arm not in seen:
                    shared.add(observed_arm)
                seen.add(observed_arm)
            continue
        if action not in {"pull", "innovate"}:
            continue
        arm, t = int(event["arm_id"]), event["round"]
        discovery_action = "innovate" if event.get("rung", 1) == 2 else "pull"
        assert not (found & shared)
        if action == "pull" and (include_all_pulls or event.get("copy_any")):
            social_first = arm in shared
            independent_first = arm in found or (discovery_action == "pull" and arm not in seen)
            assert social_first != independent_first
            assert not social_first or event.get("copy_any")
            best = max((mean_at(t, candidate) for candidate in found), default=None)
            value = mean_at(t, arm)
            yield {
                "legacy_copy": bool(event.get("copy_any")),
                "social_first": social_first,
                "independent_first": independent_first,
                "has_comparator": best is not None,
                "better": value > best if best is not None else np.nan,
                "tie": value == best if best is not None else np.nan,
                "worse": value < best if best is not None else np.nan,
                "selected_independently_discovered": arm in found,
                "advantage": value - best if best is not None else np.nan,
                "old_has_comparator": event.get(
                    "copied_arm_true_advantage_vs_best_personal_alternative"
                ) is not None,
                "old_better": event.get(
                    "copied_arm_better_than_best_personal_alternative"
                ),
            }
        if action == discovery_action and arm not in seen:
            # A copied pull must never be relabeled as an independent discovery.
            assert not (action == "pull" and event.get("copy_any"))
            found.add(arm)
        seen.add(arm)


def write_independent_copy_quality(runs_root: Path, output: Path, rung: int) -> None:
    """Recompute the corrected metric from immutable logs, for every social cell."""
    from functools import lru_cache
    from env import BanditEnvironment, InfiniteBanditEnvironment

    rows = []
    for path in sorted(runs_root.glob("*/seed_*/summary.json")):
        config = json.loads((path.parent / "config.json").read_text())
        if not config["strategy"].startswith("social_action"):
            continue
        parameters = dict(config["environment"])
        parameters.pop("rounds")
        if rung == 2:
            # This grid only finds the diagnostic oracle; it does not set arm means.
            parameters["reference_grid_size"] = 2
        env = (BanditEnvironment if rung == 1 else InfiniteBanditEnvironment)(
            seed=config["seed"], **parameters
        )

        @lru_cache(maxsize=None)
        def mean_for_regime(regime, arm):
            env.begin_round(regime * (env.regime_period or 1))
            return float(env.arm_means[arm]) if rung == 1 else env.arm_mean(arm)

        def mean_at(t, arm):
            return mean_for_regime(t // (env.regime_period or 10**12), arm)

        def events():
            with (path.parent / "events.jsonl").open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get("event") == "decision" and event.get("valid") and "arm_mean" in event:
                        assert math.isclose(mean_at(event["round"], int(event["arm_id"])), event["arm_mean"], rel_tol=1e-10, abs_tol=1e-10), path
                    yield event

        pulls = pd.DataFrame(independent_copy_comparisons(events(), mean_at, include_all_pulls=True))
        comparisons = pulls[pulls.legacy_copy] if len(pulls) else pulls
        social_first = pulls[pulls.social_first] if len(pulls) else pulls
        opportunities = config["environment"]["num_agents"] * config["environment"]["rounds"]
        base = dict(condition=config["condition"], seed=config["seed"],
                    model=config["model_label"], strategy=config["strategy"])
        rows.append(base | dict(
            valid_pulls=len(pulls),
            social_first_pulls=len(social_first),
            independent_first_pulls=int(pulls.independent_first.sum()) if len(pulls) else 0,
            social_first_rate=len(social_first) / opportunities,
            independent_first_rate=float(pulls.independent_first.sum()) / opportunities if len(pulls) else 0,
            social_first_comparable_pulls=int(social_first.has_comparator.sum()) if len(social_first) else 0,
            social_first_coverage=social_first.has_comparator.mean() if len(social_first) else np.nan,
            social_first_better=social_first.better.mean() if len(social_first) else np.nan,
            social_first_tie=social_first.tie.mean() if len(social_first) else np.nan,
            social_first_worse=social_first.worse.mean() if len(social_first) else np.nan,
            social_first_advantage=social_first.advantage.mean() if len(social_first) else np.nan,
            copied_pulls=len(comparisons),
            copied_pulls_with_independent_comparator=int(comparisons.has_comparator.sum()) if len(comparisons) else 0,
            independent_copy_coverage=comparisons.has_comparator.mean() if len(comparisons) else np.nan,
            independent_copy_better=comparisons.better.mean() if len(comparisons) else np.nan,
            independent_copy_tie=comparisons.tie.mean() if len(comparisons) else np.nan,
            independent_copy_worse=comparisons.worse.mean() if len(comparisons) else np.nan,
            selected_independently_discovered_rate=comparisons.selected_independently_discovered.mean() if len(comparisons) else np.nan,
            independent_copy_advantage=comparisons.advantage.mean() if len(comparisons) else np.nan,
            previous_copy_coverage=comparisons.old_has_comparator.mean() if len(comparisons) else np.nan,
            previous_copy_better=comparisons.old_better.mean() if len(comparisons) else np.nan,
        ))
    result = pd.DataFrame(rows)
    result.to_csv(output / "copy_quality_vs_independent_discoveries_by_seed.csv", index=False)
    metrics = [c for c in result if c.startswith(("independent_", "social_first_"))]
    summary = result.groupby(["condition", "model", "strategy"])[metrics].agg(["mean", "sem", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(output / "copy_quality_vs_independent_discoveries_summary.csv")
    (output / "INDEPENDENT_COPY_QUALITY.md").write_text(
        "# Social-first versus independent-first arms\n\n"
        "Use social_first_* columns. Supersedes both the old tried-alternative comparison and "
        "the overlapping ever-observed versus independent comparison (independent_copy_* columns). "
        "Raw runs and legacy outputs are preserved.\n\n"
        "An arm is social-first if this agent first acquired it by observation; it is independent-first "
        "if first acquired by its own discovery. The sets are disjoint and provenance never changes. "
        "For each valid SOCIAL-FIRST pull (including reuse), compare its current true mean with the MAXIMUM "
        "current true mean among arms that this agent independently discovered BEFORE that decision. "
        "R1: first acquired by pulling, before observing that arm. R2: first acquired by INNOVATE, "
        "before observing that arm; probes count without requiring a later pull. An innovation can "
        "be informed by nearby social examples; this definition concerns acquisition of the exact arm, "
        "not psychological independence. Independently discovered arms subsequently observed are never "
        "social-first; same-arm comparisons are impossible. Equal means of distinct arms, if any, are ties. "
        "Revalue all candidates after shocks. "
        "This is an analyst-known quality frontier, not the agent's noisy estimate or the value of another search action.\n\n"
        "Coverage = social-first pulls with any prior independent-first discovery / all valid social-first pulls. "
        "Social-first frequency divides by ALL agent-rounds, including missed pulls. Independent-first frequency "
        "uses the same denominator; their sum equals completion. Zero social-first pulls gives zero frequency "
        "and undefined quality/coverage, not a failed comparison. "
        "Quality and advantage exclude missing comparators. Compute ratios within population seed, then "
        "mean and SE across seeds. Socially learned arms are never made independent merely by trying them. "
        "Legacy fields are retained only for audit; never label them social-first.\n\n"
        f"Replayed {len(result)} social runs; reconstructed means checked against every logged valid pull/probe.\n"
    )
    print(f"Corrected independent copy quality: {len(result)} rung-{rung} social runs", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/rung1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rung", type=int, choices=[1, 2, 3])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--independent-copy-quality-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.runs_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    rung = args.rung
    if rung is None:
        completed = sorted(args.runs_root.glob("*/seed_*/summary.json"))
        if not completed:
            raise FileNotFoundError(f"no completed runs under {args.runs_root}")
        rung = int(json.loads(completed[0].read_text(encoding="utf-8")).get("rung", 1))
    if rung in {1, 2}:
        write_independent_copy_quality(args.runs_root, output, rung)
    if args.independent_copy_quality_only:
        if rung not in {1, 2}:
            raise ValueError("independent discovery comparison applies to rungs 1 and 2")
        return
    if rung == 2:
        analyze_rung2(args.runs_root, output, models=args.models)
        print(f"wrote rung-2 analysis to {output}")
        return
    if rung == 3:
        if "v3_replay" in args.runs_root.parts:
            analyze_rung3_replay(args.runs_root, output)
        else:
            analyze_rung3(args.runs_root, output)
        print(f"wrote rung-3 analysis to {output}")
        return

    summaries, rounds, decisions = read_completed_runs(args.runs_root)
    expected = 384 if "v3" in args.runs_root.parts else 100 if "v2" in args.runs_root.parts else 55
    if len(summaries) != expected:
        raise ValueError(
            "rung 1 is incomplete: "
            f"found {len(summaries)} healthy runs, expected {expected}"
        )
    summaries.to_csv(output / "condition_summary_by_seed.csv", index=False)
    numeric = [
        "mean_reward_per_agent_round",
        "mean_expected_regret_per_agent_round",
        "missed_pulls",
    ]
    summary_groups = (
        [
            "condition",
            "strategy",
            "guidance",
            "regime_period",
            "model_label",
            "budget_mode",
        ]
        if "v3" in args.runs_root.parts
        else ["strategy", "model_label", "budget_mode"]
    )
    condition_summary = summaries.groupby(
        summary_groups, dropna=False
    )[numeric].agg(["mean", "sem"])
    condition_summary.columns = [
        f"{metric}_{stat}" for metric, stat in condition_summary.columns
    ]
    condition_summary.reset_index().to_csv(
        output / "condition_summary.csv", index=False
    )
    for legacy_plot in (
        "performance.pdf",
        "token_allocation.pdf",
        "social_cues.pdf",
    ):
        (output / legacy_plot).unlink(missing_ok=True)
    if "v3" in args.runs_root.parts:
        anchor_summaries = summaries[summaries["regime_period"].isna()]
        anchor_rounds = rounds[rounds["regime_period"].isna()]
        anchor_decisions = decisions[decisions["regime_period"].isna()]
    else:
        anchor_summaries, anchor_rounds, anchor_decisions = (
            summaries,
            rounds,
            decisions,
        )
    plot_performance(anchor_rounds, output)
    plot_budget(anchor_rounds, output)
    plot_social_cues(anchor_decisions, output)
    plot_copy_quality_vs_personal_alternative(anchor_decisions, output)
    if "v3" in args.runs_root.parts:
        write_v3_mechanism_outputs(rounds, decisions, output)
        write_v3_paired_effects(rounds, output)
        plot_v3_factor_sweeps(rounds, output)
        write_v3_benchmark_guide(output, rung=1)
    write_rung1_report_tables(
        anchor_summaries, anchor_rounds, anchor_decisions, output
    )
    plot_completion_aware_performance(
        anchor_rounds, output, strategy_order=STRATEGY_ORDER
    )
    plot_observation_tradeoffs(
        anchor_rounds, output, strategy_order=STRATEGY_ORDER
    )
    write_completion_guide(output, rung=1)
    print(f"wrote analysis to {output}")


if __name__ == "__main__":
    main()
