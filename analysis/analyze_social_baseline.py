"""Analyze the matched CPU social baseline and save presentation-ready PNGs."""
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v3/social_baseline"
MODELS = ["qwen3_14b", "gpt_oss_20b", "ministral3_14b_reasoning"]
LABELS = ["Qwen3-14B", "GPT-OSS-20B", "Ministral-14B"]


def environment_label(config):
    e = config["environment"]
    if config["rung"] == 1:
        return "Stationary" if not e.get("regime_period") else f"Change every {e['regime_period']}"
    landscape = e["landscape"]
    if e.get("regime_period"):
        return f"Structured; change {e['regime_period']}"
    if landscape == "structured":
        return f"Structured; range {e.get('structured_length_scale', .15):g}"
    return {0.6: "Unstructured; heavy tail", 1.: "Unstructured; exponential", 2.: "Unstructured; thin tail"}[e.get("reward_shape", 1.)]


def hierarchical_comparison(d, spec):
    """Build matched R1 comparisons requested for hierarchical social UCB."""
    r1 = d[d.rung == 1].copy()
    labels = {
        "hierarchical_payoff": "H-SUCB arm + payoff",
        "social_action": "DEOE social arm only",
        "social_payoff": "DEOE social arm + payoff",
        "solo": "DEOE solo",
    }
    fields = ["reward", "early_reward", "late_reward", "diversity",
              "unique_discovered", "observation_rate"]
    reference = r1[r1["mode"].isin(labels)].copy()
    reference["policy"] = reference["mode"].map(labels)
    reference = reference[["environment_id", "seed", "policy"] + fields]

    period_to_environment = {
        int(value["environment"].get("regime_period") or 0): key
        for key, value in spec["environments"].items() if value["rung"] == 1
    }
    assert len(period_to_environment) == 3
    ucb = pd.read_csv(ROOT / "runs/v3/rung1/analysis/report_metrics_by_seed.csv")
    ucb = ucb[(ucb.model == "algorithmic") & (ucb.strategy == "ucb")].copy()
    ucb["environment_id"] = ucb.period.astype(int).map(period_to_environment)
    assert ucb.environment_id.notna().all() and len(ucb) == 24
    ucb["policy"] = "UCB solo"
    reference = pd.concat(
        [reference, ucb[["environment_id", "seed", "policy"] + fields]],
        ignore_index=True,
    )
    assert not reference.duplicated(["environment_id", "seed", "policy"]).any()
    reference.to_csv(OUT / "hierarchical_ucb_comparison_by_seed.csv", index=False)

    absolute = reference.groupby(["environment_id", "policy"])[fields].agg(["mean", "sem"])
    absolute.columns = ["_".join(column) for column in absolute.columns]
    absolute = absolute.reset_index()
    absolute["environment"] = absolute.environment_id.map(
        lambda key: environment_label(spec["environments"][key])
    )
    absolute.to_csv(OUT / "hierarchical_ucb_comparison_summary.csv", index=False)

    reward = reference.pivot(
        index=["environment_id", "seed"], columns="policy", values="reward"
    )
    contrasts = []
    primaries = ["H-SUCB arm + payoff"]
    comparators = ["DEOE social arm + payoff", "UCB solo", "DEOE solo"]
    for primary in primaries:
        for comparator in comparators:
            values = reward[primary] - reward[comparator]
            for (environment_id, seed), value in values.items():
                contrasts.append(dict(environment_id=environment_id, seed=seed,
                                      primary=primary, comparator=comparator,
                                      reward_difference=value))
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(OUT / "hierarchical_ucb_contrasts_by_seed.csv", index=False)
    paired = contrasts.groupby(["environment_id", "primary", "comparator"]).reward_difference.agg(
        ["mean", "sem", "count"]
    ).reset_index()
    paired["ci_low"] = paired["mean"] - 2.364624 * paired["sem"]
    paired["ci_high"] = paired["mean"] + 2.364624 * paired["sem"]
    paired["environment"] = paired.environment_id.map(
        lambda key: environment_label(spec["environments"][key])
    )
    paired.to_csv(OUT / "hierarchical_ucb_contrasts_summary.csv", index=False)

    order = ["H-SUCB arm + payoff", "DEOE social arm + payoff", "UCB solo", "DEOE solo"]
    colors = ["#D34B8A", "#327AB7", "#E18727", "#9454A1"]
    environments = sorted(r1.environment_id.unique(),
                          key=lambda key: environment_label(spec["environments"][key]))
    fig, axes = plt.subplots(1, len(environments), figsize=(17, 6), sharey=True,
                             layout="constrained")
    for ax, environment_id in zip(axes, environments):
        values = absolute[absolute.environment_id == environment_id].set_index("policy").loc[order]
        ax.bar(range(len(order)), values.reward_mean, yerr=values.reward_sem,
               color=colors, capsize=4)
        ax.set(title=environment_label(spec["environments"][environment_id]),
               xticks=range(len(order)), xticklabels=["H-SUCB", "DEOE\nsocial", "UCB\nsolo", "DEOE\nsolo"])
    axes[0].set_ylabel("Reward per agent-round")
    fig.suptitle("Rung 1 algorithmic policies; matched seeds; mean ± seed SE")
    fig.savefig(OUT / "rung1_hierarchical_ucb_comparison.png", dpi=160)
    plt.close(fig)
    return absolute, paired


def main():
    d = pd.read_csv(OUT / "metrics_by_seed.csv")
    times = pd.read_csv(OUT / "performance_by_seed_round.csv")
    mapping = pd.read_csv(OUT / "condition_mapping.csv")
    spec = json.loads((OUT / "specification.json").read_text())
    expected = sum(
        8 * (4 if value["rung"] == 1 else 3)
        for value in spec["environments"].values()
    )
    assert len(d) == expected and not d.duplicated(["environment_id", "seed", "mode"]).any()
    assert d.groupby(["environment_id", "mode"]).seed.apply(set).map(lambda s: s == set(range(8))).all()
    assert len(times) == expected * 100 and (d.completion == 1).all()
    fields = ["reward", "latent_reward", "unique_discovered", "social_first_rate", "observation_rate", "early_reward", "late_reward"]
    solo = d[d["mode"] == "solo"]
    deoe_social = d[d["mode"].isin(["social_action", "social_payoff"])]
    paired = deoe_social.merge(solo, on=["environment_id", "rung", "seed"], suffixes=("", "_solo"), validate="many_to_one")
    for field in fields:
        paired[field] = paired[field] - paired[field + "_solo"]
    paired = paired[["environment_id", "rung", "seed", "mode"] + fields]
    paired.to_csv(OUT / "social_minus_solo_by_seed.csv", index=False)
    aggregate = paired.groupby(["environment_id", "rung", "mode"])[fields].agg(["mean", "sem"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["environment"] = aggregate.environment_id.map(lambda k: environment_label(spec["environments"][k]))
    aggregate["reward_ci_low"] = aggregate.reward_mean - 2.364624 * aggregate.reward_sem
    aggregate["reward_ci_high"] = aggregate.reward_mean + 2.364624 * aggregate.reward_sem
    aggregate.to_csv(OUT / "social_minus_solo_summary.csv", index=False)
    gaps = []
    for rung in (1, 2):
        llm = pd.read_csv(ROOT / f"runs/v3/rung{rung}/analysis/report_metrics_by_seed.csv")
        llm = llm[llm.model != "algorithmic"].merge(mapping[mapping.rung == rung], on="condition", validate="many_to_one")
        llm["mode"] = llm.strategy.map({"solo_llm": "solo", "social_action": "social_action", "social_action_payoff": "social_payoff"})
        assert llm["mode"].notna().all()
        merged = llm.merge(d[d["mode"].isin(["solo", "social_action", "social_payoff"])], on=["environment_id", "rung", "seed", "mode"], suffixes=("_llm", "_baseline"), validate="many_to_one")
        assert len(merged) == len(llm)
        merged["baseline_minus_llm_reward"] = merged.reward_baseline - merged.reward_llm
        merged["baseline_minus_llm_early_reward"] = merged.early_reward_baseline - merged.early_reward_llm
        gaps.append(merged)
    gaps = pd.concat(gaps)
    gaps.to_csv(OUT / "baseline_minus_llm_by_seed.csv", index=False)
    g = gaps.groupby(["rung", "condition", "model", "mode"])[["baseline_minus_llm_reward", "baseline_minus_llm_early_reward"]].agg(["mean", "sem"])
    g.columns = ["_".join(c) for c in g.columns]
    g.to_csv(OUT / "baseline_minus_llm_summary.csv")
    hierarchical_absolute, hierarchical_paired = hierarchical_comparison(d, spec)
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})
    colors = {"solo": "#9454A1", "social_payoff": "#D34B8A", "social_action": "#327AB7"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")
    for rung, ax in zip((1, 2), axes):
        sub = aggregate[aggregate.rung == rung]
        keys = sorted(sub.environment_id.unique(), key=lambda k: environment_label(spec["environments"][k]))
        for mode, offset in [("social_action", -.13), ("social_payoff", .13)]:
            z = sub[sub["mode"] == mode].set_index("environment_id").loc[keys]
            ax.errorbar(z.reward_mean, np.arange(len(keys)) + offset, xerr=z.reward_sem, fmt="o", capsize=4,
                        color=colors[mode], label="Arm only" if mode == "social_action" else "Arm + payoff")
        ax.axvline(0, color="grey", lw=1)
        ax.set(yticks=range(len(keys)), yticklabels=[environment_label(spec["environments"][k]) for k in keys],
               xlabel="Algorithmic social − algorithmic solo reward", title=f"Rung {rung}")
        ax.invert_yaxis()
    axes[1].legend(loc="best")
    fig.suptitle("Does the same simple policy benefit from social access?\nAll agents use DEOE; matched populations; mean ± seed SE")
    fig.savefig(OUT / "social_advantage_across_environments.png", dpi=160)
    plt.close(fig)
    anchor_tables = []
    for rung in (1, 2):
        sub = gaps[(gaps.rung == rung) & (gaps.period == 0) & (gaps["shape"] == 1.) &
                   (gaps.length == (0. if rung == 1 else .15)) & (gaps.guidance == "neutral") &
                   (gaps.budget == "use_it_or_lose_it")]
        keys = sub.environment_id.unique()
        assert len(keys) == 1
        anchor_tables.append(sub)
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True, layout="constrained")
        for model, label, ax in zip(MODELS, LABELS, axes):
            z = sub[sub.model == model]
            for i, mode in enumerate(("solo", "social_action", "social_payoff")):
                a = z[z["mode"] == mode]
                assert len(a) == 8
                for offset, field, col, name in [(-.16, "reward_llm", "#008978", "LLM"), (.16, "reward_baseline", "#9454A1", "DEOE")]:
                    ax.bar(i + offset, a[field].mean(), yerr=a[field].sem(), width=.30, color=col, capsize=4, label=name if i == 0 else None)
            ax.set(title=label, xticks=range(3), xticklabels=["Solo", "Social\narm only", "Social\n+ payoff"])
        axes[0].set_ylabel("Reward per agent-round")
        axes[-1].legend()
        fig.suptitle(f"Rung {rung}: attainable algorithmic performance versus LLMs\nStationary {'finite' if rung == 1 else 'structured'} task; neutral prompts; mean ± seed SE; algorithms are not token-matched")
        fig.savefig(OUT / f"rung{rung}_llm_comparison.png", dpi=160)
        plt.close(fig)
    anchors = pd.concat(anchor_tables)
    anchor_summary = anchors.groupby(["rung", "model", "mode"])[["reward_llm", "reward_baseline", "baseline_minus_llm_reward", "baseline_minus_llm_early_reward"]].agg(["mean", "sem"])
    anchor_summary.columns = ["_".join(c) for c in anchor_summary.columns]
    anchor_summary.to_csv(OUT / "anchor_comparisons_summary.csv")
    wins = aggregate[(aggregate["mode"] == "social_payoff") & (aggregate.reward_ci_low > 0)]
    validation = dict(populations=len(d), environments=len(spec["environments"]), matched_llm_runs=len(gaps),
                      runtime_seconds_sum=float(d.seconds.sum()), max_population_seconds=float(d.seconds.max()),
                      positive_paired_95ci_rungs=sorted(map(int, wins.rung.unique())),
                      homogeneous_within_population=True, parameters_selected_on_v3=False,
                      hierarchical_rung1_populations=int((d.rung.eq(1) & d["mode"].str.startswith("hierarchical_")).sum()))
    (OUT / "VALIDATION.json").write_text(json.dumps(validation, indent=2))
    report = ["# Algorithmic social-learning reference", "DEOE is discountmachine-inspired, not the tournament winner reproduced. All agents use the same policy. See SOCIAL_BASELINE.md for the frozen rule and information limits.",
              "## Social versus its own solo control", "Mean ± SE over eight matched seeds; 95% intervals below are unadjusted, exploratory intervals across the environment sweep."]
    for row in aggregate.itertuples():
        report.append(f"- R{row.rung}, {row.environment}, {row.mode}: {row.reward_mean:+.3f} ± {row.reward_sem:.3f}; 95% interval [{row.reward_ci_low:+.3f}, {row.reward_ci_high:+.3f}].")
    report.extend(["## Reading the comparisons", "The paired DEOE contrast isolates enabling observation for that rule, including the resulting changes in population histories. H-SUCB is a separate source-selection policy, not a causal treatment on DEOE. The algorithm–LLM gap combines policy and execution differences. Zero model tokens are not evidence for a token-matched LLM speedup. No baseline was added here to rung 3 or rung 4.",
                   "## Runtime and coverage", json.dumps(validation, indent=2)])
    report.extend(["## Stationary task comparison with language models", "R1 uses the finite stationary task; R2 uses the stationary medium-range structured task. Neutral prompts, expiring budget. The following are matched baseline-minus-LLM reward differences, mean ± seed SE."])
    for (rung, model, mode), row in anchor_summary.iterrows():
        report.append(f"- R{rung}, {model}, {mode}: DEOE {row.reward_baseline_mean:.3f}; LLM {row.reward_llm_mean:.3f}; gap {row.baseline_minus_llm_reward_mean:+.3f} ± {row.baseline_minus_llm_reward_sem:.3f}.")
    report.extend(["## Rung 1 hierarchical social UCB", "H-SUCB first uses UCB to choose self versus a peer. Selecting self supplies the lower arm UCB with the agent's own earned reward. Selecting a peer updates that source's outer-UCB value from the peer's displayed payoff and adds the peer's arm/payoff to the lower arm UCB with weight one; the lower UCB then independently chooses the arm that the observer executes."])
    stationary_id = next(key for key, value in spec["environments"].items()
                         if value["rung"] == 1 and not value["environment"].get("regime_period"))
    for row in hierarchical_absolute[hierarchical_absolute.environment_id == stationary_id].itertuples():
        report.append(f"- Stationary, {row.policy}: reward {row.reward_mean:.3f} ± {row.reward_sem:.3f}; late reward {row.late_reward_mean:.3f} ± {row.late_reward_sem:.3f}.")
    for row in hierarchical_paired[(hierarchical_paired.environment_id == stationary_id) &
                                   (hierarchical_paired.primary == "H-SUCB arm + payoff")].itertuples():
        report.append(f"- Stationary paired H-SUCB arm + payoff minus {row.comparator}: {row.mean:+.3f} ± {row.sem:.3f}; 95% interval [{row.ci_low:+.3f}, {row.ci_high:+.3f}].")
    report.append("A higher algorithmic social reward does not imply that its social-minus-solo gain exceeds every LLM's gain. In particular, compare GPT-OSS's R2 gain against its own solo control separately: an LLM can have a larger social treatment effect while still achieving a lower absolute reward. This reference does not justify labeling all model copying excessive or establishing a token-matched speed advantage.")
    (OUT / "REPORT.md").write_text("\n\n".join(report))
    (OUT / "PLOT_GUIDE.md").write_text("# Figure guide\n\n`social_advantage_across_environments.png`: paired social minus solo reward for the same DEOE policy, across every unique V3 environment; positive values favor observation. Error bars are seed SE, not confidence intervals.\n\n`rung1_hierarchical_ucb_comparison.png`: absolute Rung 1 reward for payoff-visible hierarchical social UCB, payoff-visible DEOE social, UCB solo and DEOE solo. Panels are matched stationary and changing environments; error bars are seed SE. Use `hierarchical_ucb_contrasts_summary.csv` for paired uncertainty.\n\n`rung1_llm_comparison.png` and `rung2_llm_comparison.png`: absolute reward for DEOE versus each model, separately under solo, action-only and payoff-visible social access. Stationary neutral conditions; R2 uses the medium-range structured task. Each algorithmic reference is shared across models, not independently rerun.\n\nAll seed values and all-condition gaps are provided in CSV files. Algorithms use the original environments but no language-model computation; these are task-performance references, not token-matched agents.\n")
    shutil.copyfile(ROOT / "docs/algorithmic_baselines.md", OUT / "SOCIAL_BASELINE.md")
    shutil.copyfile(ROOT / "code/social_baseline.py", OUT / "policy_source.py")
    shutil.copyfile(ROOT / "tests/test_social_baseline.py", OUT / "policy_tests.py")
    print(json.dumps(validation, indent=2), flush=True)
    print(aggregate[["rung", "environment", "mode", "reward_mean", "reward_sem"]].to_string(index=False), flush=True)
    print(hierarchical_absolute[["environment", "policy", "reward_mean", "reward_sem"]].to_string(index=False), flush=True)
    print(hierarchical_paired[["environment", "primary", "comparator", "mean", "sem", "ci_low", "ci_high"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
