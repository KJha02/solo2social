#!/usr/bin/env python3
"""Build the compact Overleaf report figures from saved seed-level analyses."""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "llm_social_learning_update" / "figures"
PAPER_ROOT = ROOT / "outputs"
PAPER_OUT = PAPER_ROOT / "figures"
PAPER_RESULTS = PAPER_ROOT / "results"
MODELS = ["qwen3_14b", "gpt_oss_20b", "ministral3_14b_reasoning"]
NAMES = {"qwen3_14b": "Qwen", "gpt_oss_20b": "GPT-OSS", "ministral3_14b_reasoning": "Ministral"}
COLORS = {"solo": "#4C78A8", "solo_llm": "#4C78A8", "social_action_payoff": "#E69F00", "frozen": "#777777", "rsi": "#008F7A"}


def setup():
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams.update({"font.family": "monospace", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "figure.dpi": 140, "savefig.bbox": "tight"})
    OUT.mkdir(parents=True, exist_ok=True)


def save_paper_figure(fig, stem):
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png"]:
        fig.savefig(PAPER_OUT / f"{stem}.{extension}", dpi=300)


def paired_endpoint(frame, value, strategy_a, strategy_b, label, round_index=99):
    endpoint = frame[frame["round"] == round_index]
    wide = endpoint.pivot(index="seed", columns="strategy", values=value).dropna()
    difference = wide[strategy_a] - wide[strategy_b]
    return dict(
        contrast=label,
        round=round_index + 1,
        mean=float(difference.mean()),
        sem=float(difference.sem()),
        count=int(difference.count()),
    )


def paper_ucb_precision():
    """Fixed 100-seed precision audit; never stop or tune against significance."""
    from agent import UCBAgent
    from main import make_environment
    from social_baseline import simulate_hierarchical_ucb
    config = json.loads((RUNS / 'v3/rung1/v3_solo_llm__use_it_or_lose_it__gpt_oss_20b/seed_0/config.json').read_text())
    assert config['environment']['num_agents'] == 10 and config['environment']['num_arms'] == 25
    assert config['environment']['rounds'] == 100
    destination = PAPER_RESULTS / 'rung1_ucb_100seeds.csv'
    if destination.exists():
        saved = pd.read_csv(destination)
        assert len(saved) == 30000 and saved.seed.nunique() == 100
        return
    rows = []
    for seed in range(100):
        _, social = simulate_hierarchical_ucb(config, seed, 'hierarchical_payoff')
        env = make_environment(config, seed)
        best = env.best_mean
        curve = social.groupby('round')[['reward', 'latent_reward']].mean().reset_index()
        for row in curve.to_dict('records'):
            rows.append(dict(seed=seed, strategy='Hierarchical UCB', round=row['round'],
                             reward=row['reward'], expected_regret=best-row['latent_reward']))
        for policy in ('Solo UCB', 'Oracle'):
            env = make_environment(config, seed)
            agents = [UCBAgent(agent_id=i, num_arms=25, seed=seed) for i in range(10)]
            for r in range(100):
                env.begin_round(r)
                rewards, regrets = [], []
                for agent in agents:
                    arm = env.best_arm if policy == 'Oracle' else agent.choose()
                    pull = env.pull(agent_id=agent.agent_id, arm_id=arm, round_index=r)
                    agent.update(arm, pull.reward)
                    rewards.append(pull.reward); regrets.append(env.best_mean-env.arm_means[arm])
                rows.append(dict(seed=seed, strategy=policy, round=r, reward=np.mean(rewards), expected_regret=np.mean(regrets)))
    frame = pd.DataFrame(rows).sort_values(['strategy', 'seed', 'round'])
    for metric in ('reward', 'expected_regret'):
        frame['cumulative_reward' if metric == 'reward' else 'cumulative_regret'] = frame.groupby(['strategy', 'seed'])[metric].cumsum()
    # Independently reproduce the original eight-seed controls before expanding.
    old = pd.read_csv(PAPER_RESULTS / 'rung1_algorithm_curves_by_seed.csv')
    matched = frame[frame.seed < 8].merge(old, on=['strategy', 'seed', 'round'], suffixes=('_new', '_old'))
    assert len(matched) == 2400
    assert np.allclose(matched.reward_new, matched.reward_old, atol=1e-10)
    frame.to_csv(destination, index=False)
    (PAPER_RESULTS / 'rung1_ucb_precision_protocol.json').write_text(json.dumps(dict(
        seeds=list(range(100)), original_matched_seeds=list(range(8)),
        fixed_before_simulation=True, matched_original_trajectories=True,
        environment=config['environment'], selection='No tuning or significance-based stopping',
        extra_llm_calls=0), indent=2))
    print('Validated and saved 100 matched UCB/oracle seeds', flush=True)


def paper_rung1_curves(only_stem=None):
    """Main cumulative-reward figure and the matched cumulative-regret appendix."""
    llm = pd.read_csv(OUT / "rung1_cumulative_by_seed.csv")
    baseline = pd.read_csv(RUNS / "v3/social_baseline/performance_by_seed_round.csv")
    baseline = baseline[(baseline.rung == 1) &
                        (baseline.environment_id == "r1_e96ca53d112f") &
                        baseline["mode"].isin(["solo", "social_payoff", "hierarchical_payoff"])].copy()
    baseline["strategy"] = baseline["mode"].map({
        "solo": "DEOE solo",
        "social_payoff": "DEOE social",
        "hierarchical_payoff": "Hierarchical UCB",
    })
    baseline = (baseline.groupby(["seed", "round", "strategy"], as_index=False)
                .agg(reward=("reward", "mean"), latent_reward=("latent_reward", "mean")))

    standard = pd.read_csv(RUNS / "v3/rung1/analysis/performance_by_seed_round.csv")
    standard = standard[(standard.condition.isin(["v3_ucb", "v3_oracle"])) &
                        (standard.model_label == "gpt_oss_20b") &
                        (standard.budget_mode == "use_it_or_lose_it")].copy()
    standard["strategy"] = standard.condition.map({"v3_ucb": "Solo UCB", "v3_oracle": "Oracle"})
    standard = standard[["seed", "round", "strategy", "reward", "expected_regret"]]

    environment_means = []
    for seed in range(8):
        path = RUNS / "v3/rung1/v3_solo_llm__use_it_or_lose_it__gpt_oss_20b" / f"seed_{seed}/environment.json"
        metadata = json.loads(path.read_text())
        environment_means.append(dict(seed=seed, best_mean=float(metadata["best_mean"])))
    best = pd.DataFrame(environment_means)
    baseline = baseline.merge(best, on="seed", validate="many_to_one")
    baseline["expected_regret"] = baseline.best_mean - baseline.latent_reward

    algorithms = pd.concat([
        baseline[["seed", "round", "strategy", "reward", "expected_regret"]],
        standard,
    ], ignore_index=True)
    algorithms = algorithms.sort_values(["strategy", "seed", "round"])
    algorithms["cumulative_reward"] = algorithms.groupby(["strategy", "seed"]).reward.cumsum()
    algorithms["cumulative_regret"] = algorithms.groupby(["strategy", "seed"]).expected_regret.cumsum()
    PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
    algorithms.to_csv(PAPER_RESULTS / "rung1_algorithm_curves_by_seed.csv", index=False)

    palette = {
        "Solo": "#4C78A8", "Social": "#E69F00", "Solo UCB": "#4C78A8",
        "Hierarchical UCB": "#E69F00", "DEOE solo": "#4C78A8",
        "DEOE social": "#E69F00", "Oracle": "#333333",
    }
    styles = {
        "Solo": "-", "Social": "-", "Solo UCB": "-", "Hierarchical UCB": "-",
        "DEOE solo": "--", "DEOE social": "--", "Oracle": "--",
    }
    stats = []
    paper_ucb_precision()
    expanded = pd.read_csv(PAPER_RESULTS / 'rung1_ucb_100seeds.csv')
    # Raw reward comparisons must use the same payoff landscapes in every panel.
    matched = expanded[expanded.seed.isin(llm.seed.unique())].copy()
    assert set(matched.seed) == set(range(8))
    matched.to_csv(PAPER_RESULTS / 'rung1_matched_ucb_curves.csv', index=False)
    for metric, ylabel, stem in [
        ("reward", "Mean reward per agent", "rung1_reward_per_round_paper"),
        ("cumulative_reward", "Cumulative reward", "rung1_cumulative_reward_paper"),
        ("cumulative_regret", "Cumulative expected regret", "rung1_cumulative_regret_paper"),
    ]:
        if only_stem is not None and stem != only_stem:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9), sharex=True, sharey=True)
        axes = axes.ravel()
        model_order = ["qwen3_14b", "ministral3_14b_reasoning", "gpt_oss_20b"]
        model_names = {
            "qwen3_14b": "Qwen3-14B",
            "ministral3_14b_reasoning": "Ministral-3-14B",
            "gpt_oss_20b": "GPT-OSS-20B",
        }
        for col, model in enumerate(model_order):
            for strategy, label in [("solo_llm", "Solo"), ("social_action_payoff", "Social")]:
                subset = llm[(llm.model == model) & (llm.strategy == strategy)]
                result = mean_se(subset, metric, ["round"])
                x = result["round"].to_numpy() + 1
                y, se = result["mean"].to_numpy(), result["sem"].to_numpy()
                axes[col].plot(x, y, color=palette[label], ls=styles[label], label=label, lw=2)
                axes[col].fill_between(x, y-se, y+se, color=palette[label], alpha=.16, linewidth=0)
            axes[col].set_title(f"({'abc'[col]}) {model_names[model]}", fontweight="bold")
            oracle = mean_se(matched[matched.strategy == 'Oracle'], metric, ['round'])
            ox = oracle['round'].to_numpy()+1
            oy, ose = oracle['mean'].to_numpy(), oracle['sem'].to_numpy()
            axes[col].plot(ox, oy, color='black', ls='--', lw=1.5, label='Oracle')
            axes[col].fill_between(ox, oy-ose, oy+ose, color='gray', alpha=.12, linewidth=0, zorder=0)
        order = ["Solo UCB", "Hierarchical UCB", "Oracle"]
        for label in order:
            subset = matched[matched.strategy == label]
            result = mean_se(subset, metric, ["round"])
            x = result["round"].to_numpy() + 1
            y, se = result["mean"].to_numpy(), result["sem"].to_numpy()
            if label == "Oracle":
                axes[3].plot(x, y, color='black', ls='--', label=label, lw=1.5)
                axes[3].fill_between(x, y-se, y+se, color='gray', alpha=.12, linewidth=0, zorder=0)
            else:
                axes[3].plot(x, y, color=palette[label], ls=styles[label],
                             label='Solo' if label == 'Solo UCB' else 'Social', lw=1.9)
                axes[3].fill_between(x, y-se, y+se, color=palette[label], alpha=.10, linewidth=0)
        axes[3].set_title("(d) UCB policies", fontweight="bold")
        for i, ax in enumerate(axes):
            ax.set_xlabel("Round")
            ax.tick_params(labelsize=11)
            ax.set_xticks([1, 50, 100])
        fig.supylabel(ylabel, x=.02, fontsize=12)
        handles, labels = [], []
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle); labels.append(label)
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(.53, -.02), fontsize=12)
        fig.subplots_adjust(left=.085, right=.995, top=.85, bottom=.31, wspace=.13)
        save_paper_figure(fig, stem)
        plt.close(fig)

    for model in MODELS:
        subset = llm[llm.model == model]
        for endpoint in [49, 99]:
            stats.append(paired_endpoint(subset, "cumulative_reward",
                                         "social_action_payoff", "solo_llm",
                                         f"{model}: social minus solo cumulative reward", endpoint))
            stats.append(paired_endpoint(subset, "cumulative_regret",
                                         "social_action_payoff", "solo_llm",
                                         f"{model}: social minus solo cumulative regret", endpoint))
    for endpoint in [49, 99]:
        stats.append(paired_endpoint(algorithms, "cumulative_reward", "Hierarchical UCB", "Solo UCB",
                                     "hierarchical UCB minus solo UCB cumulative reward", endpoint))
        stats.append(paired_endpoint(algorithms, "cumulative_regret", "Hierarchical UCB", "Solo UCB",
                                     "hierarchical UCB minus solo UCB cumulative regret", endpoint))
        stats.append(paired_endpoint(algorithms, "cumulative_reward", "DEOE social", "DEOE solo",
                                     "DEOE social minus DEOE solo cumulative reward", endpoint))
    pd.DataFrame(stats).to_csv(PAPER_RESULTS / "rung1_endpoint_contrasts.csv", index=False)
    precision = []
    for n in (8, 100):
        precision.append(dict(seeds=n, **paired_endpoint(expanded[expanded.seed < n], 'cumulative_reward',
            'Hierarchical UCB', 'Solo UCB', 'hierarchical minus solo cumulative reward')))
    pd.DataFrame(precision).to_csv(PAPER_RESULTS / 'rung1_ucb_precision.csv', index=False)
    audit = []
    for n in (8, 100):
        sub = expanded[expanded.seed < n]
        for strategy, group in sub.groupby('strategy'):
            per_seed = group.groupby('seed').reward.mean()
            endpoint = group[group['round'] == 99].set_index('seed').reward
            audit.append(dict(seeds=n, strategy=strategy, mean_reward=per_seed.mean(),
                mean_reward_se=per_seed.sem(), final_round_reward=endpoint.mean(), final_round_se=endpoint.sem()))
    for (model, strategy), group in llm.groupby(['model', 'strategy']):
        per_seed = group.groupby('seed').reward.mean()
        endpoint = group[group['round'] == 99].set_index('seed').reward
        audit.append(dict(seeds=8, strategy=model+': '+strategy, mean_reward=per_seed.mean(),
            mean_reward_se=per_seed.sem(), final_round_reward=endpoint.mean(), final_round_se=endpoint.sem()))
    pd.DataFrame(audit).to_csv(PAPER_RESULTS/'rung1_matched_comparison_audit.csv', index=False)
    print(pd.DataFrame(audit).to_string(index=False), flush=True)
    last = expanded[expanded['round'] == 99].groupby('strategy').cumulative_reward.mean()/100
    (PAPER_RESULTS/'paper_numbers.tex').write_text(
        '\\newcommand{\\UCBGain}{'+f"{precision[-1]['mean']/100:.3f}"+'}\n'+
        '\\newcommand{\\UCBGainSE}{'+f"{precision[-1]['sem']/100:.3f}"+'}\n'+
        '\\newcommand{\\UCBSoloMean}{'+f"{last['Solo UCB']:.3f}"+'}\n'+
        '\\newcommand{\\UCBSocialMean}{'+f"{last['Hierarchical UCB']:.3f}"+'}\n')
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    for i, row in enumerate(precision):
        ax.errorbar(row['mean']/100, i, xerr=row['sem']/100, fmt='o', capsize=4, color='#008F7A')
    ax.axvline(0, color='gray', lw=1); ax.set_yticks([0, 1], ['Matched 8 seeds', 'Fixed 100 seeds'])
    ax.set_xlabel('Social minus solo UCB\nmean reward per round', fontsize=11); fig.tight_layout()
    save_paper_figure(fig, 'rung1_ucb_precision'); plt.close(fig)


def paper_rung1_interventions():
    data = pd.read_csv(OUT / "rung1_intervention_raw_performance_summary.csv")
    label_map = {"Default": "Default", "Token intervention": "PE",
                 "Forced search intervention": "IS"}
    data["short"] = data.intervention.map(label_map)
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.75), sharey=True)
    x = np.arange(3); width = .35
    for col, model in enumerate(MODELS):
        for offset, access in enumerate(["Solo", "Social"]):
            subset = data[(data.model == model) & (data.access == access)].set_index("short").reindex(["Default", "PE", "IS"])
            axes[col].bar(x + (offset-.5)*width, subset["mean"], width,
                          yerr=subset["sem"], capsize=3,
                          color=COLORS["solo"] if access == "Solo" else COLORS["social_action_payoff"],
                          label=access)
        axes[col].axhline(2.911256, color="#555555", ls=":", lw=1.5, label="Solo UCB")
        axes[col].set_xticks(x, ["Default", "PE", "IS"])
        axes[col].set_title(f"({'abc'[col]}) {NAMES[model]}", fontweight="bold")
    fig.supylabel("Mean reward per agent", x=.025, fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(.53, -.015), fontsize=13)
    fig.subplots_adjust(left=.12, bottom=.27, top=.86, wspace=.12)
    save_paper_figure(fig, "rung1_interventions_paper")
    plt.close(fig)

    sep = pd.read_csv(RUNS / "paper_analysis/analysis/causal_rung1/causal_separate_effects_summary.csv")
    did = pd.read_csv(RUNS / "paper_analysis/analysis/causal_rung1/causal_did_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8))
    specs = [
        (axes[0,0], sep[(sep.p_independent == 0) & (sep.execution_reserve == 2048) & sep.social],
         "completion_rate_effect", "completion_rate_effect_se", "(a) PE: social completion", 100),
        (axes[0,1], did[(did.p_independent == 0) & (did.execution_reserve == 2048)],
         "completion_rate_did", "completion_rate_did_se", "(b) PE: completion interaction", 100),
        (axes[1,0], sep[(sep.p_independent == .15) & (sep.execution_reserve == 0) & sep.social],
         "effective_selected_diversity_effect", "effective_selected_diversity_effect_se", "(c) IS: social diversity", 1),
        (axes[1,1], did[(did.p_independent == .15) & (did.execution_reserve == 0)],
         "reward_did", "reward_did_se", "(d) IS: reward interaction", 1),
    ]
    for ax, subset, value, error, title, scale in specs:
        subset = subset.set_index("model").reindex(MODELS)
        ax.bar(np.arange(3), scale*subset[value], yerr=scale*subset[error],
               color=["#6BAED6", "#F28E2B", "#59A14F"], capsize=3)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_xticks(np.arange(3), [NAMES[m] for m in MODELS])
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel({
            "completion_rate_effect": "Percentage-point effect",
            "completion_rate_did": "Percentage-point interaction",
            "effective_selected_diversity_effect": "Effective-diversity effect",
            "reward_did": "Reward interaction",
        }[value], fontsize=9)
    fig.subplots_adjust(left=.11, hspace=.46, wspace=.42)
    save_paper_figure(fig, "rung1_intervention_mechanisms_paper")
    plt.close(fig)

    selected = pd.concat([
        sep[((sep.p_independent == 0) & (sep.execution_reserve == 2048) & sep.social)].assign(estimand="PE social effect"),
        did[((did.p_independent == 0) & (did.execution_reserve == 2048))].assign(estimand="PE interaction"),
        sep[((sep.p_independent == .15) & (sep.execution_reserve == 0) & sep.social)].assign(estimand="IS social effect"),
        did[((did.p_independent == .15) & (did.execution_reserve == 0))].assign(estimand="IS interaction"),
    ], ignore_index=True, sort=False)
    selected.to_csv(PAPER_RESULTS / "rung1_intervention_effects.csv", index=False)


def paper_rung1_arm_audit():
    rows, profiles = [], []
    for seed in range(8):
        path = RUNS / "v3/rung1/v3_solo_llm__use_it_or_lose_it__gpt_oss_20b" / f"seed_{seed}/environment.json"
        metadata = json.loads(path.read_text())
        means = np.sort(np.asarray(metadata["arm_means"], dtype=float))
        for rank, value in enumerate(means, start=1):
            profiles.append(dict(seed=seed, rank=rank, arm_mean=value))
        rows.append(dict(seed=seed, best=means[-1], second=means[-2], median=float(np.median(means)),
                         best_second_gap=means[-1]-means[-2], best_median_gap=means[-1]-np.median(means),
                         noise_std=float(metadata["reward_noise_std"])))
    audit, profile = pd.DataFrame(rows), pd.DataFrame(profiles)
    audit.to_csv(PAPER_RESULTS / "rung1_arm_gap_audit.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    summary = mean_se(profile, "arm_mean", ["rank"])
    axes[0].plot(summary["rank"], summary["mean"], color="#4C78A8", lw=2)
    axes[0].fill_between(summary["rank"], summary["mean"]-summary["sem"],
                         summary["mean"]+summary["sem"], color="#4C78A8", alpha=.18)
    axes[0].set_xlabel("Arm rank within seed")
    axes[0].set_ylabel("Expected payoff")
    axes[0].set_title("(a) Arm payoffs", fontweight="bold", fontsize=12)
    long = audit.melt(id_vars="seed", value_vars=["best_second_gap", "best_median_gap"],
                      var_name="comparison", value_name="gap")
    names = {"best_second_gap": "Best - second", "best_median_gap": "Best - median"}
    long["comparison"] = long.comparison.map(names)
    for i, name in enumerate(names.values()):
        values = long.loc[long.comparison == name, "gap"]
        axes[1].scatter(np.full(len(values), i), values, color=["#E69F00", "#008F7A"][i], alpha=.75)
        axes[1].errorbar(i, values.mean(), yerr=values.sem(), color="#222222", marker="o", capsize=4)
    axes[1].axhline(audit.noise_std.mean(), color="#777777", ls=":", label="One-draw noise SD")
    axes[1].set_xticks([0,1], list(names.values()))
    axes[1].set_ylabel("Expected-payoff gap")
    axes[1].set_title("(b) Payoff gaps", fontweight="bold", fontsize=12)
    axes[1].legend(frameon=False, fontsize=8)
    fig.subplots_adjust(wspace=.34)
    save_paper_figure(fig, "rung1_arm_gap_audit")
    plt.close(fig)


def paper_rung1_copy_quality():
    data = pd.read_csv(RUNS / "v3/rung1/analysis/copy_quality_vs_personal_alternative_by_seed.csv")
    data = data[(data.strategy == "social_action_payoff") &
                (data.budget_mode == "use_it_or_lose_it") &
                data.model_label.isin(MODELS)].copy()
    data.to_csv(PAPER_RESULTS / "rung1_copy_quality_by_seed.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, metric, title in [
        (axes[0], "copied_arm_better_rate", "(a) Copied arm beats personal best"),
        (axes[1], "comparator_coverage", "(b) Personal comparator exists"),
    ]:
        summary = mean_se(data, metric, ["model_label"]).set_index("model_label").reindex(MODELS)
        ax.bar(np.arange(3), 100*summary["mean"], yerr=100*summary["sem"],
               color=["#6BAED6", "#F28E2B", "#59A14F"], capsize=3)
        ax.set_xticks(np.arange(3), [NAMES[m] for m in MODELS])
        ax.set_ylim(0, 105)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].axhline(50, color="#777777", ls=":", lw=1)
    axes[0].set_ylabel("Percent of comparable pulls", fontsize=9)
    axes[1].set_ylabel("Percent of socially learned pulls", fontsize=9)
    fig.subplots_adjust(left=.10, wspace=.30)
    save_paper_figure(fig, "rung1_copy_quality_paper")
    plt.close(fig)


def paper_figures():
    import shutil
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
    cumulative_bandits(1)
    paper_rung1_curves()
    causal_raw_performance(1)
    for suffix in ('by_seed', 'summary'):
        name = f'rung1_intervention_raw_performance_{suffix}.csv'
        shutil.copy2(OUT/name, PAPER_RESULTS/name)
    paper_rung1_interventions()
    paper_rung1_arm_audit()
    paper_rung1_copy_quality()
    cumulative_bandits(2)
    for rung in (1, 2):
        trajectories(rung, primary=True)
        shutil.copy2(OUT/f'rung{rung}_trajectories.png', PAPER_OUT/f'rung{rung}_trajectories.png')
        shutil.copy2(OUT/f'rung{rung}_cumulative_by_seed.csv', PAPER_RESULTS/f'rung{rung}_primary_trajectories_by_seed.csv')
    for metric in ('cumulative_reward', 'cumulative_regret'):
        for extension in ('pdf', 'png'):
            shutil.copy2(OUT/f'rung2_{metric}.{extension}', PAPER_OUT/f'rung2_{metric}.{extension}')


def mean_se(frame, value, group):
    x = frame.groupby(group)[value].agg(["mean", "sem"]).reset_index()
    return x


def cumulative_bandits(rung):
    """Stationary V3 anchor; aggregate complete populations before seed SE."""
    records = []
    for path in sorted((RUNS / f"v3/rung{rung}").glob("*/seed_*/config.json")):
        config = json.loads(path.read_text())
        env = config["environment"]
        if (config.get("model_label") not in MODELS or
            config.get("strategy") not in {"solo_llm", "social_action_payoff"} or
            config.get("guidance", "neutral") != "neutral" or
            config["budget"].get("carry_over", True) or
            env.get("regime_period") or env.get("reward_shape", 1) != 1):
            continue
        if rung == 2 and (env["landscape"] != "structured" or
                          env.get("structured_length_scale") != .15):
            continue
        directory = path.parent
        metadata = json.loads((directory / "environment.json").read_text())
        oracle = metadata["best_mean" if rung == 1 else "reference_best_mean"]
        rows = []
        with (directory / "events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event") != "round_end":
                    continue
                if rung == 1:
                    regret = event["expected_regret"]
                else:
                    regret = oracle - (event["arm_mean"] if event.get("pulled") else 0.)
                rows.append(dict(round=event["round"], agent=event["agent_id"],
                                 reward=event["reward"], regret=regret))
        frame = pd.DataFrame(rows)
        if frame.empty or frame.duplicated(["round", "agent"]).any():
            raise ValueError(f"Missing or duplicated events: {directory}")
        sizes = frame.groupby("round").size()
        if len(sizes) != env["rounds"] or not (sizes == env["num_agents"]).all():
            continue
        curve = frame.groupby("round")[["reward", "regret"]].mean().sort_index()
        curve["cumulative_reward"] = curve.reward.cumsum()
        curve["cumulative_regret"] = curve.regret.cumsum()
        curve["mean_lifetime_payoff"] = curve.cumulative_reward / np.arange(1, len(curve)+1)
        curve["model"] = config["model_label"]
        curve["strategy"] = config["strategy"]
        curve["seed"] = int(directory.name.split("_")[-1])
        records.append(curve.reset_index())
    data = pd.concat(records, ignore_index=True)
    data.to_csv(OUT / f"rung{rung}_cumulative_by_seed.csv", index=False)
    for metric, ylabel in [("cumulative_reward", "Cumulative reward per agent"),
                           ("cumulative_regret", "Cumulative expected regret"),
                           ("mean_lifetime_payoff", "Mean lifetime payoff")]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for col, model in enumerate(MODELS):
            for strategy in ["solo_llm", "social_action_payoff"]:
                subset = data[(data.model == model) & (data.strategy == strategy)]
                result = mean_se(subset, metric, ["round"])
                t = result["round"].to_numpy()+1
                mean, se = result["mean"].to_numpy(), result["sem"].to_numpy()
                axes[col].plot(t, mean, color=COLORS[strategy],
                               label="Solo" if strategy == "solo_llm" else "Social")
                axes[col].fill_between(t, mean-se, mean+se, color=COLORS[strategy], alpha=.18)
            axes[col].set_title(f"({'abc'[col]}) {NAMES[model]}")
            axes[col].set_xlabel("Round")
        axes[0].set_ylabel(ylabel.replace('Cumulative ', 'Cumulative\n'))
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(bottom=.30, wspace=.15)
        for extension in ["png", "pdf"]:
            fig.savefig(OUT / f"rung{rung}_{metric}.{extension}", dpi=300, bbox_inches='tight')
        plt.close(fig)


def main_r1():
    base = pd.read_csv(RUNS / "v3/rung1/analysis/condition_summary_by_seed.csv")
    base = base[(base.guidance == "neutral") & (base.budget_mode == "carry") &
                (base.reward_shape == 1.0) & base.regime_period.isna() &
                base.strategy.isin(["solo_llm", "social_action_payoff"]) & base.model_label.isin(MODELS)]
    summ = mean_se(base, "mean_reward_per_agent_round", ["model_label", "strategy"])
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
    x = np.arange(3); width = .36
    for j, strategy in enumerate(["solo_llm", "social_action_payoff"]):
        s = summ[summ.strategy == strategy].set_index("model_label").reindex(MODELS)
        ax[0].bar(x + (j-.5)*width, s["mean"], width, yerr=s["sem"], color=COLORS[strategy],
                  label="Solo" if strategy == "solo_llm" else "Social: action + payoff", capsize=3)
    ax[0].set_xticks(x, [NAMES[m] for m in MODELS]); ax[0].set_ylabel("Mean reward per agent-round")
    ax[0].set_title("a  Absolute performance")
    ax[1].bar(x, [95.2, 93.2, 74.9], color="#008F7A")
    ax[1].axhline(50, color="#777777", ls="--", lw=1)
    ax[1].set_xticks(x, [NAMES[m] for m in MODELS]); ax[1].set_ylim(0, 105)
    ax[1].set_ylabel("Copied arm beats personal best (%)"); ax[1].set_title("b  Local value of copying")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=.24, wspace=.32); fig.savefig(OUT / "rung1_main.png"); plt.close(fig)


def trajectories(rung, primary=False):
    mode = 'use_it_or_lose_it' if primary or rung == 2 else 'carry'
    perf = (pd.read_csv(OUT/f'rung{rung}_cumulative_by_seed.csv').rename(columns={'model': 'model_label'})
            if primary else pd.read_csv(RUNS / f"v3/rung{rung}/analysis/performance_by_seed_round.csv"))
    perf = perf[(perf.model_label.isin(MODELS)) & perf.strategy.isin(["solo_llm", "social_action_payoff"])]
    if "budget_mode" in perf:
        perf = perf[perf.budget_mode == mode]
    if rung == 2 and not primary: perf = perf[perf.landscape == "structured"]
    if rung == 1:
        keep = pd.read_csv(RUNS / "v3/rung1/analysis/condition_summary_by_seed.csv")
        keep = keep[(keep.guidance == "neutral") & (keep.budget_mode == mode) &
                    (keep.reward_shape == 1.0) & keep.regime_period.isna()]
        if not primary:
            perf = perf.merge(keep[["condition", "seed"]].drop_duplicates(), on=["condition", "seed"])
    metric_file = "population_arm_diversity_by_seed_round.csv" if rung == 1 else "discovery_by_seed_round.csv"
    disc = pd.read_csv(RUNS / f"v3/rung{rung}/analysis/{metric_file}")
    disc = disc[(disc.model_label.isin(MODELS)) & disc.strategy.isin(["solo_llm", "social_action_payoff"])]
    if "budget_mode" in disc:
        disc = disc[disc.budget_mode == mode]
    if rung == 2:
        disc = disc[(disc.landscape == "structured") & (disc.guidance == "neutral") &
                    (disc.reward_shape == 1.0) & (disc.spatial_length_scale == .15) & disc.regime_period.isna()]
    else:
        disc = disc.merge(keep[["condition", "seed"]].drop_duplicates(), on=["condition", "seed"])
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.3), sharex=True, sharey='row')
    dmetric = "unique_pulled_arms" if rung == 1 else "cumulative_unique_arms"
    for col, model in enumerate(MODELS):
        for strategy in ["solo_llm", "social_action_payoff"]:
            p = perf[(perf.model_label == model) & (perf.strategy == strategy)]
            p = mean_se(p, "reward", ["round"])
            axes[0,col].plot(p["round"], p["mean"], color=COLORS[strategy], label="Solo" if strategy=="solo_llm" else "Social")
            axes[0,col].fill_between(p["round"], p["mean"]-p["sem"], p["mean"]+p["sem"], color=COLORS[strategy], alpha=.15)
            d = disc[(disc.model_label == model) & (disc.strategy == strategy)]
            d = mean_se(d, dmetric, ["round"])
            axes[1,col].plot(d["round"], d["mean"], color=COLORS[strategy])
            axes[1,col].fill_between(d["round"], d["mean"]-d["sem"], d["mean"]+d["sem"], color=COLORS[strategy], alpha=.15)
        axes[0,col].set_title(f"{'abc'[col]}  {NAMES[model]}")
        axes[1,col].set_title(f"{'def'[col]}  {NAMES[model]}")
        axes[1,col].set_xlabel("Round")
    axes[0,0].set_ylabel("Reward")
    axes[1,0].set_ylabel('Distinct arms pulled' if rung == 1 else 'Cumulative arms\ndiscovered')
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=.22, hspace=.45, wspace=.15); fig.savefig(OUT / f"rung{rung}_trajectories.png"); plt.close(fig)


def rung3_trajectory():
    d = pd.read_csv(RUNS / "v3/rung3/analysis/report_timeseries_by_seed.csv")
    d = d[(d.model.isin(MODELS)) & d.strategy.isin(["solo", "social_full"]) &
          ~d.condition.str.contains("strategic")]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.3), sharex=True, sharey="row")
    for col, model in enumerate(MODELS):
        for strategy in ["solo", "social_full"]:
            s = d[(d.model == model) & (d.strategy == strategy)]
            for row, metric in enumerate(["reward", "social_use_rate"]):
                z = mean_se(s, metric, ["round"])
                color = COLORS["solo"] if strategy == "solo" else COLORS["social_action_payoff"]
                axes[row,col].plot(z["round"], z["mean"], color=color, label="Solo" if strategy=="solo" else "Social")
        axes[0,col].set_title(f"{'abc'[col]}  {NAMES[model]}"); axes[1,col].set_title(f"{'def'[col]}  {NAMES[model]}")
        axes[1,col].set_xlabel("Round")
    axes[0,0].set_ylabel("Reward"); axes[1,0].set_ylabel("Social-skill use rate")
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=.14, hspace=.35, wspace=.2); fig.savefig(OUT / "rung3_trajectories.png"); plt.close(fig)


def rung1_controls():
    root = RUNS / "paper_analysis/analysis/causal_rung1"
    sep = pd.read_csv(root / "causal_separate_effects_summary.csv")
    did = pd.read_csv(root / "causal_did_summary.csv")
    models = ["qwen3_14b", "gpt_oss_20b"]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6))
    specs = [
        (axes[0,0], sep[(sep.p_independent == 0) & (sep.execution_reserve == 2048) & (sep.social)],
         "completion_rate_effect", "completion_rate_effect_se", "a  Reserve: social completion", 100),
        (axes[0,1], did[(did.p_independent == 0) & (did.execution_reserve == 2048)],
         "completion_rate_did", "completion_rate_did_se", "b  Reserve: completion DID", 100),
        (axes[1,0], sep[(sep.p_independent == .15) & (sep.execution_reserve == 0) & (sep.social)],
         "effective_selected_diversity_effect", "effective_selected_diversity_effect_se", "c  Forced search: social diversity", 1),
        (axes[1,1], did[(did.p_independent == .15) & (did.execution_reserve == 0)],
         "reward_did", "reward_did_se", "d  Forced search: reward DID", 1),
    ]
    for ax, data, val, err, title, scale in specs:
        s = data.set_index("model").reindex(models)
        ax.bar(np.arange(2), scale*s[val], yerr=scale*s[err], color=["#4C78A8", "#E69F00"], capsize=3)
        ax.axhline(0, color="#555555", lw=.8); ax.set_xticks(np.arange(2), [NAMES[m] for m in models]); ax.set_title(title)
    axes[0,0].set_ylabel("Percentage-point effect"); axes[0,1].set_ylabel("Percentage-point DID")
    axes[1,0].set_ylabel("Effective-diversity effect"); axes[1,1].set_ylabel("Reward DID")
    fig.subplots_adjust(hspace=.4, wspace=.35); fig.savefig(OUT / "rung1_controls_readable.png"); plt.close(fig)


def causal_raw_performance(rung):
    """Plot raw solo/social reward under each randomized intervention."""
    source = RUNS / f"paper_analysis/analysis/causal_rung{rung}/target_outcomes.csv"
    data = pd.read_csv(source)
    data = data[(data.model.isin(MODELS)) & (data.observed_rounds == 100)].copy()
    data["replication"] = "causal"
    data["pair_id"] = "causal-" + data.seed.astype(str)

    # The zero-intervention bar pools the original V3 run and its causal-suite
    # replication. Match the scientific configuration used by the causal suite:
    # stationary, neutral, payoff-visible social access, and expiring budgets.
    original = pd.read_csv(RUNS / f"v3/rung{rung}/analysis/report_metrics_by_seed.csv")
    original = original[(original.model.isin(MODELS)) &
                        (original.strategy.isin(["solo_llm", "social_action_payoff"])) &
                        (original.guidance == "neutral") &
                        (original.budget == "use_it_or_lose_it") &
                        np.isclose(original["shape"], 1.0) &
                        (original["period"] == 0)].copy()
    if rung == 1:
        original = original[np.isclose(original["length"], 0.0)]
    else:
        original = original[original.condition.str.contains("rung2_structured_") &
                            np.isclose(original["length"], 0.15)]
    original = original.rename(columns={"reward": "mean_reward"})
    original["social"] = original.strategy == "social_action_payoff"
    original["social_info"] = np.where(original.social, "payoff", "none")
    original["p_independent"] = 0.0
    original["execution_reserve"] = 0
    original["observed_rounds"] = 100
    original["replication"] = "v3"
    original["pair_id"] = "v3-" + original.seed.astype(str)
    conditions = [
        ("Default", 0.0, 0),
        ("Token intervention", 0.0, 2048),
        ("Forced search intervention", 0.15, 0),
    ]
    paired = []
    for model in MODELS:
        for label, independent, reserve in conditions:
            cell = data[(data.model == model) &
                        np.isclose(data.p_independent, independent) &
                        (data.execution_reserve == reserve) &
                        (((data.social) & (data.social_info == "payoff")) |
                         ((~data.social) & (data.social_info == "none")))]
            if label == "Default":
                cell = pd.concat([cell, original[original.model == model]], ignore_index=True)
            solo_pairs = set(cell.loc[~cell.social, "pair_id"])
            social_pairs = set(cell.loc[cell.social, "pair_id"])
            matched = solo_pairs & social_pairs
            cell = cell[cell.pair_id.isin(matched)].copy()
            cell["intervention"] = label
            cell["access"] = np.where(cell.social, "Social", "Solo")
            paired.append(cell)
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(OUT / f"rung{rung}_intervention_raw_performance_by_seed.csv", index=False)
    summary = (paired.groupby(["model", "intervention", "access"], observed=True)
               .mean_reward.agg(["mean", "sem", "count"]).reset_index())
    summary.to_csv(OUT / f"rung{rung}_intervention_raw_performance_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), sharey=True)
    x = np.arange(len(conditions)); width = 0.36
    labels = ["Default", "Token\nintervention", "Forced search\nintervention"]
    for col, model in enumerate(MODELS):
        ax = axes[col]
        for offset, access in enumerate(["Solo", "Social"]):
            values, errors, counts = [], [], []
            for label, _, _ in conditions:
                row = summary[(summary.model == model) &
                              (summary.intervention == label) &
                              (summary.access == access)]
                values.append(row["mean"].iloc[0] if len(row) else np.nan)
                errors.append(row["sem"].iloc[0] if len(row) and pd.notna(row["sem"].iloc[0]) else 0)
                counts.append(int(row["count"].iloc[0]) if len(row) else 0)
            positions = x + (offset - .5) * width
            bars = ax.bar(positions, values, width, yerr=errors, capsize=4,
                          color=COLORS["solo"] if access == "Solo" else COLORS["social_action_payoff"],
                          label=access)
            for bar, count in zip(bars, counts):
                if count:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"n={count}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_title(f"({'abc'[col]}) {NAMES[model]}", fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
    axes[0].set_ylabel("Mean reward per agent-round")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=.24, wspace=.12)
    fig.savefig(OUT / f"rung{rung}_intervention_raw_performance.png", dpi=300)
    plt.close(fig)


def rung3_deoe():
    rows = []
    for design in ('v3_acquisition', 'v3_deoe'):
        for path in (RUNS / design / 'rung3').glob('*/seed_*/summary.json'):
            row = json.loads(path.read_text())
            config = json.loads((path.parent / 'config.json').read_text())
            if config.get('guidance', 'neutral') != 'neutral':
                continue
            if row['strategy'] not in ('solo', 'social_full', 'deoe_solo', 'deoe_social'):
                continue
            if not row.get('completed'):
                raise ValueError(f'Incomplete run: {path.parent}')
            rows.append(dict(model=row['model_label'], strategy=row['strategy'],
                             seed=row['seed'], reward=row['mean_reward_per_agent_round']))
    table = pd.DataFrame(rows)
    order = ['solo', 'deoe_solo', 'social_full', 'deoe_social']
    summary = table.groupby(['model', 'strategy']).reward.agg(['mean', 'sem', 'count'])
    if len(summary) != 12 or not (summary['count'] == 8).all():
        raise ValueError('Expected eight seeds for each fixed-skill model/condition')
    means = np.array([summary.loc[m].reindex(order)['mean'] for m in MODELS])
    ses = np.array([summary.loc[m].reindex(order)['sem'] for m in MODELS])
    table.to_csv(OUT / 'rung3_deoe_by_seed.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8), sharey=True)
    x = np.arange(2); w = .34
    for i, (ax, model) in enumerate(zip(axes, MODELS)):
        ax.bar(x-w/2, means[i,[0,2]], w, yerr=ses[i,[0,2]], color="#4C78A8", label="LLM selects", capsize=3)
        ax.bar(x+w/2, means[i,[1,3]], w, yerr=ses[i,[1,3]], color="#008F7A", label="DEOE selects", capsize=3)
        ax.set_xticks(x, ["Solo", "Social"]); ax.set_title(f"{'abc'[i]}  {NAMES[model]}"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mean scheduling reward")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=.25, wspace=.15); fig.savefig(OUT / "rung3_deoe.png"); plt.close(fig)


def rung2_volatility():
    data = pd.read_csv(RUNS / 'v3/rung2/analysis/report_metrics_by_seed.csv')
    data = data[(data.guidance == 'neutral') & (data.budget == 'use_it_or_lose_it')
                & (data['length'] == .15) & (data['shape'] == 1)
                & data.strategy.isin(['solo_llm', 'social_action_payoff'])]
    means, ses = [], []
    for model in MODELS:
        model_means, model_ses = [], []
        for period in (0, 25, 10):
            wide = data[(data.model == model) & (data.period == period)].pivot(
                index='seed', columns='strategy', values='reward').dropna()
            if len(wide) != 8:
                raise ValueError('Expected eight matched volatility seeds')
            delta = wide.social_action_payoff - wide.solo_llm
            model_means.append(delta.mean()); model_ses.append(delta.sem())
        means.append(model_means); ses.append(model_ses)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7), sharey=True)
    labels = ["Stationary", "Change / 25", "Change / 10"]
    for i, (ax, model) in enumerate(zip(axes, MODELS)):
        ax.errorbar(means[i], np.arange(3), xerr=ses[i], fmt="o", color="#008F7A", capsize=3)
        ax.axvline(0, color="#666666", lw=.8); ax.set_yticks(np.arange(3), labels); ax.invert_yaxis()
        ax.set_title(f"{'abc'[i]}  {NAMES[model]}"); ax.set_xlabel("Social minus solo reward")
    fig.subplots_adjust(wspace=.18); fig.savefig(OUT / "rung2_volatility.png"); plt.close(fig)


def rung3_access():
    sources = [("Original restricted acquisition", RUNS / "v3/rung3/analysis/condition_summary_by_seed.csv"),
               ("Independent acquisition enabled", RUNS / "v3_acquisition/rung3/analysis/condition_summary_by_seed.csv")]
    strategies = ["solo", "social_id", "social_full", "social_full_strategic"]
    labels = ["Solo", "Social ID", "Social full", "Full strategic"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
    x = np.arange(4); w = .36
    for j, (access, path) in enumerate(sources):
        d = pd.read_csv(path)
        d = d[(d.model_label.isin(MODELS)) & d.strategy.isin(strategies) & (d.budget_mode == "use_it_or_lose_it")]
        s = mean_se(d, "mean_reward_per_agent_round", ["model_label", "strategy"])
        for i, (ax, model) in enumerate(zip(axes, MODELS)):
            z = s[s.model_label == model].set_index("strategy").reindex(strategies)
            ax.bar(x + (j-.5)*w, z["mean"], w, yerr=z["sem"], color=["#999999", "#087DB5"][j], label=access, capsize=2)
            ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_title(f"{'abc'[i]}  {NAMES[model]}"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Population reward")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(bottom=.31, wspace=.12); fig.savefig(OUT / "rung3_access_control.png"); plt.close(fig)


def paper_offline_skill_check():
    """Audit the earlier single-agent capability check without making API calls."""
    roots = [
        ROOT / 'runs/offline/gpt_oss/ifbench',
        ROOT / 'runs/offline/glm/ifbench',
    ]
    rows = []
    common_tasks = None
    for root in roots:
        for condition in ['minimal_frozen', 'minimal_openevolve_full']:
            for seed in range(8):
                directory = root/condition/f'seed_{seed}'
                assert (directory/'completed.json').exists(), directory
                cfg = json.loads((directory/'config.json').read_text())
                selection = json.loads((directory/'evaluation_selection.json').read_text())
                tasks = set(selection['task_ids'])
                assert len(tasks) == 200
                if common_tasks is None: common_tasks = tasks
                assert tasks == common_tasks
                last = cfg['rounds']-1
                selected = json.loads((directory/'rounds'/f'{last:04d}.json').read_text())['results']
                assert len(selected) == 1
                version = selected[0]['version']
                values = {}
                for path in (directory/'holdout').glob(f'{version}-*-0.json'):
                    record = json.loads(path.read_text())
                    identity = record['identity']
                    if identity['task'] not in tasks: continue
                    assert identity['task'] not in values
                    values[identity['task']] = record['result']
                assert set(values) == tasks, (directory, len(values))
                rows.append(dict(model=cfg['model']['name'], condition=condition, seed=seed,
                    version=version, tasks=len(tasks),
                    prompt_accuracy=np.mean([v['prompt_level_reward'] for v in values.values()]),
                    constraint_accuracy=np.mean([v['constraint_accuracy'] for v in values.values()])))
                if condition == 'minimal_openevolve_full':
                    checkpoint = json.loads((directory/'checkpoint.json').read_text())
                    initial = [v for v, a in checkpoint['learners'][0]['acquisitions'].items() if a['origin'] == 'initial']
                    assert len(initial) == 1
                    replay = {}
                    for path in (directory/'holdout').glob(f'{initial[0]}-*-0.json'):
                        record = json.loads(path.read_text())
                        if record['identity']['task'] in tasks:
                            replay[record['identity']['task']] = record['result']
                    rows.append(dict(model=cfg['model']['name'], condition='initial_replay', seed=seed,
                        version=initial[0], tasks=len(replay),
                        prompt_accuracy=np.mean([v['prompt_level_reward'] for v in replay.values()]) if len(replay) == 200 else np.nan,
                        constraint_accuracy=np.mean([v['constraint_accuracy'] for v in replay.values()]) if len(replay) == 200 else np.nan))
    frame = pd.DataFrame(rows)
    frame.to_csv(PAPER_RESULTS/'r4_offline_capability_by_seed.csv', index=False)
    models = list(frame.model.unique())
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3), sharey=True)
    for mi, model in enumerate(models):
        for j, metric in enumerate(['prompt_accuracy', 'constraint_accuracy']):
            ax = axes[2*j+mi]
            subset = frame[frame.model == model]
            order = ['minimal_frozen', 'initial_replay', 'minimal_openevolve_full']
            z = mean_se(subset, metric, ['condition']).set_index('condition').reindex(order)
            ax.bar(range(3), 100*z['mean'], yerr=100*z['sem'], color=['#777777', '#4C78A8', '#008F7A'], capsize=3)
            ax.set_xticks([0, 1, 2], ['Frozen', 'Replay', 'Evolved'], fontsize=9, rotation=30, ha='right')
            ax.set_ylim(0, 85)
            ax.set_title(f'({"abcd"[2*j+mi]}) '+['GPT-OSS-120B', 'GLM-5.3-Flash'][mi], fontsize=11)
            if mi == 0: ax.set_ylabel(['Whole-answer\naccuracy (%)', 'Constraint\naccuracy (%)'][j], fontsize=11)
    fig.subplots_adjust(left=.09, bottom=.2, top=.84, wspace=.3)
    save_paper_figure(fig, 'r4_offline_capability'); plt.close(fig)
    stats = []
    for metric in ['prompt_accuracy', 'constraint_accuracy']:
        stats.extend(mean_se(frame, metric, ['model', 'condition']).assign(metric=metric).to_dict('records'))
        for model in models:
            wide = frame[frame.model == model].pivot(index='seed', columns='condition', values=metric)
            delta = wide.minimal_openevolve_full-wide.minimal_frozen
            stats.append(dict(model=model, condition='paired_gain', metric=metric, mean=delta.mean(), sem=delta.sem()))
            delta = wide.minimal_openevolve_full-wide.initial_replay
            stats.append(dict(model=model, condition='paired_replay_gain', metric=metric, mean=delta.mean(), sem=delta.sem(), n=delta.count()))
    pd.DataFrame(stats).to_csv(PAPER_RESULTS/'r4_offline_capability_statistics.csv', index=False)
    print('Audited 32 single-agent endpoints, each on the same 200 held-out tasks', flush=True)


def paper_matched_token_results():
    """Total-token learning curves and paired interpolation sensitivity; no API calls."""
    curves = pd.read_csv(PAPER_RESULTS / 'r4_total_token_checkpoints.csv')
    models = ['openai/gpt-oss-120b', 'z-ai/glm-5.3-flash']
    names = ['GPT-OSS-120B', 'GLM-5.3-Flash']
    policies = ['llm_solo', 'llm_social_payoff', 'openevolve_solo', 'ucb_social', 'uniform_social']
    labels = ['LLM solo', 'LLM social', 'OE solo', 'UCB social', 'Uniform social']
    colors = ['#4C78A8', '#E69F00', '#008F7A', '#B65C9A', '#777777']
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for mi, model in enumerate(models):
        ax = axes[mi]
        for condition, label, color in zip(policies, labels, colors):
            g = curves[(curves.model == model) & (curves.condition == condition)]
            z = g.groupby('round').agg(x=('cumulative_learning_total_tokens','mean'), y=('prompt_accuracy','mean'), se=('prompt_accuracy','sem'))
            x, y, se = z.x.to_numpy()/1e6, z.y.to_numpy()*100, z.se.to_numpy()*100
            ax.plot(x, y, marker='o', ms=4, color=color, label=label)
            ax.fill_between(x, y-se, y+se, color=color, alpha=.14, linewidth=0)
        ax.set_title(f'({"ab"[mi]}) {names[mi]}')
        ax.set_xlabel('Learning tokens (M/population)', fontsize=10)
        ax.set_ylabel('Test accuracy (%)', fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.085, right=.985, bottom=.31, top=.87, wspace=.3)
    save_paper_figure(fig, 'r4_accuracy_vs_tokens'); plt.close(fig)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for mi, model in enumerate(models):
        values = [[], [], []]
        for seed in range(6):
            a = curves[(curves.model == model) & (curves.condition == 'llm_solo') & (curves.seed == seed)].sort_values('round')
            b = curves[(curves.model == model) & (curves.condition == 'llm_social_payoff') & (curves.seed == seed)].sort_values('round')
            x, z = a.cumulative_learning_total_tokens.to_numpy(), b.cumulative_learning_total_tokens.to_numpy()
            target = min(x[-1], z[-1])
            social = np.interp(target, z, b.prompt_accuracy)
            right = min(np.searchsorted(x, target, side='right'), len(x)-1)
            left = right-1
            comparisons = [a.prompt_accuracy.iloc[left], np.interp(target, x, a.prompt_accuracy), a.prompt_accuracy.iloc[right]]
            for j, accuracy in enumerate(comparisons):
                delta = 100*(social-accuracy)
                values[j].append(delta)
                rows.append(dict(model=model, seed=seed, comparison=['preceding','interpolated','following'][j], gain_pp=delta, budget_tokens=target))
        ax = axes[mi]
        for j, v in enumerate(values):
            ax.scatter(j+np.linspace(-.10,.10,6), v, color=colors[mi], s=24, alpha=.75)
            ax.errorbar(j, np.mean(v), yerr=np.std(v,ddof=1)/np.sqrt(6), fmt='o', color='black', capsize=4)
        ax.axhline(0, color='.5', lw=1, ls='--')
        ax.set_xticks(range(3), ['Solo before', 'Matched\ntokens', 'Solo after'], fontsize=10)
        ax.set_title(f'({"ab"[mi]}) {names[mi]}')
        ax.set_ylabel('Social − solo accuracy (pp)', fontsize=11)
    fig.subplots_adjust(left=.09, right=.98, bottom=.23, top=.87, wspace=.3)
    save_paper_figure(fig, 'r4_matched_spending'); plt.close(fig)
    pd.DataFrame(rows).to_csv(PAPER_RESULTS / 'r4_matched_total_tokens_sensitivity.csv', index=False)


def paper_skill_results():
    """Completed corrected populations only; no API calls or pending timing runs."""
    root = ROOT / 'runs/population'
    analysis = root / 'analysis'
    models = ['openai/gpt-oss-120b', 'z-ai/glm-5.3-flash']
    names = ['GPT-OSS-120B', 'GLM-5.3-Flash']
    policies = ['llm_solo', 'llm_social_payoff', 'openevolve_solo', 'ucb_social', 'uniform_social']
    labels = ['LLM solo', 'LLM social', 'OE solo', 'UCB social', 'Uniform social']
    short = ['Solo', 'Social', 'OpenEvolve', 'Source UCB', 'Uniform']
    colors = dict(zip(policies, ['#4C78A8', '#E69F00', '#008F7A', '#B65C9A', '#777777']))
    final = pd.read_csv(analysis / 'final_by_seed.csv')
    tests = pd.read_csv(analysis / 'test_by_seed.csv')
    learning = pd.read_csv(analysis / 'learning_by_seed.csv')
    spending = pd.read_csv(analysis / 'spending_by_phase.csv')
    behavior = pd.DataFrame(json.loads((analysis / 'copy_behavior.json').read_text())['seed_level']).drop(columns='raw')
    timing = pd.DataFrame(json.loads((analysis / 'observation_timing.json').read_text())['seed_level']).drop(columns='round_counts')
    assert len(final) == 60 and len(tests) == 240
    assert (final.groupby(['model', 'condition']).seed.nunique() == 6).all()
    for stem, frame in [('r4_final', final), ('r4_test', tests), ('r4_learning', learning),
                        ('r4_spending', spending), ('r4_behavior', behavior), ('r4_timing', timing)]:
        frame.to_csv(PAPER_RESULTS / (stem + '_by_seed.csv'), index=False)

    def curve(ax, frame, metric, condition, scale=100, x='round'):
        z = mean_se(frame[frame.condition == condition], metric, [x])
        ax.plot(z[x], scale*z['mean'], color=colors[condition], label=labels[policies.index(condition)], lw=2)
        ax.fill_between(z[x].to_numpy(), scale*(z['mean']-z['sem']).to_numpy(),
                        scale*(z['mean']+z['sem']).to_numpy(), color=colors[condition], alpha=.15, linewidth=0)

    def finish(fig, axes, stem, legend=True, bottom=.29, left=.095):
        if legend:
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='lower center',
                       ncol=4 if len(handles) == 4 else 3, frameon=False, fontsize=11,
                       bbox_to_anchor=(.54, -.08))
        fig.subplots_adjust(left=left, right=.985, bottom=bottom, top=.85,
                            wspace=.65 if len(axes) == 4 else .22)
        if len(axes) == 4:
            for ax in axes:
                ax.title.set_fontsize(10)
                ax.tick_params(axis='y', labelsize=10)
        save_paper_figure(fig, stem); plt.close(fig)

    for metric, stem in [('prompt_accuracy', 'r4_learning'), ('constraint_accuracy', 'r4_constraint_learning')]:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=False)
        for i, model in enumerate(models):
            for condition in policies[:3]:
                curve(axes[i], tests[tests.model == model], metric, condition)
            initial = final[(final.model == model) & (final.condition == 'llm_solo')][metric+'_initial'].mean()*100
            axes[i].axhline(initial, color='black', ls=':', lw=1.3, label='Initial skill (LLM solo)')
            axes[i].set_title(f'({"ab"[i]}) {names[i]}'); axes[i].set_xlabel('Learning round'); axes[i].set_xticks([0, 10, 25, 49])
            axes[i].margins(y=.18)
            axes[i].set_ylabel('Test accuracy (%)')
        finish(fig, axes, stem, bottom=.33, left=.125)

    costs = spending.groupby(['model', 'condition', 'seed'], as_index=False)[['reported_cost', 'conservative_exposure']].sum()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    for mi, model in enumerate(models):
        for condition in policies:
            curve(axes[mi], learning[learning.model == model], 'deployed_training_fitness', condition)
        axes[mi].set_title(f'({"ab"[mi]}) {names[mi]}', fontsize=12)
        axes[mi].set_xlabel('Learning round'); axes[mi].set_ylim(45, 90)
    axes[0].set_ylabel('Recorded training\nfitness (%)', fontsize=11)
    finish(fig, axes, 'r4_training', bottom=.33)

    resources = tests.copy()
    resources.to_csv(PAPER_RESULTS/'r4_checkpoint_resources_by_seed.csv', index=False)
    for resource, xlabel, scale, stem in [
        ('cumulative_attributed_learning_cost', 'Attributed learning cost ($/population)', 1, 'r4_accuracy_vs_cost')]:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=(stem != 'r4_accuracy_vs_cost'))
        for mi, model in enumerate(models):
            for condition, label in zip(policies, labels):
                sub = resources[(resources.model == model) & (resources.condition == condition)]
                z = sub.groupby('round').agg(x=(resource, 'mean'), y=('prompt_accuracy', 'mean'), se=('prompt_accuracy', 'sem'))
                axes[mi].errorbar(scale*z.x, 100*z.y, yerr=100*z.se, marker='o', markersize=3,
                    color=colors[condition], label=label, capsize=2)
            axes[mi].set_title(f'({"ab"[mi]}) {names[mi]}', fontsize=12)
            axes[mi].set_xlabel(xlabel, fontsize=10)
            if stem == 'r4_accuracy_vs_cost':
                axes[mi].set_ylim((65, 71) if mi == 0 else (51, 64))
            else:
                axes[mi].set_ylim(48, 74)
        axes[0].set_ylabel('Test accuracy (%)', fontsize=11)
        finish(fig, axes, stem, bottom=.33)
    paper_matched_token_results()
    all_tokens = (spending.groupby(['model', 'condition', 'seed'], as_index=False)
                  [['prompt_tokens', 'completion_tokens']].sum())
    all_tokens['all_tokens_millions'] = (
        all_tokens['prompt_tokens'] + all_tokens['completion_tokens']) / 1e6
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9))
    for mi, model in enumerate(models):
        for col, frame, metric, scale in [
            (mi, final, 'prompt_accuracy', 100),
            (mi+2, all_tokens, 'all_tokens_millions', 1),
        ]:
            z = mean_se(frame[frame.model == model], metric, ['condition']).set_index('condition').reindex(policies)
            axes[col].bar(range(5), scale*z['mean'], yerr=scale*z['sem'], capsize=2, color=[colors[c] for c in policies])
            axes[col].set_xticks(range(5), short, rotation=35, ha='right', fontsize=8)
            axes[col].set_title(f'({"abcd"[col]}) {names[mi]}', fontsize=12)
        axes[mi].set_ylim(50, 80)
        model_tokens = all_tokens[all_tokens.model == model].groupby('condition').all_tokens_millions.mean()
        axes[mi+2].set_ylim(0, 1.12*model_tokens.max())
    axes[0].set_ylabel('Final test\naccuracy (%)', fontsize=11)
    axes[2].set_ylabel('All tokens\n(millions/population)', fontsize=11)
    finish(fig, axes, 'r4_controllers', legend=False, bottom=.32)

    contrasts = [('llm_social_payoff', 'llm_solo'), ('ucb_social', 'openevolve_solo'),
                 ('uniform_social', 'openevolve_solo'), ('ucb_social', 'uniform_social')]
    contrast_labels = ['LLM social − solo', 'UCB − OE solo', 'Uniform − OE solo', 'UCB − Uniform']
    paired = []
    for model in models:
        for (t, c), label in zip(contrasts, contrast_labels):
            trows = final[(final.model == model) & (final.condition == t)].set_index('seed')
            crows = final[(final.model == model) & (final.condition == c)].set_index('seed')
            for metric in ('prompt_accuracy', 'constraint_accuracy'):
                for suffix in ('', '_auc_per_round', '_gain'):
                    for seed, value in (trows[metric+suffix]-crows[metric+suffix]).items():
                        paired.append(dict(model=model, contrast=label, metric=metric+suffix, seed=seed, effect=100*value))
                adjusted = ((trows[metric+'_auc_per_round']-trows[metric+'_initial']) -
                            (crows[metric+'_auc_per_round']-crows[metric+'_initial']))
                for seed, value in adjusted.items():
                    paired.append(dict(model=model, contrast=label, metric=metric+'_auc_gain', seed=seed, effect=100*value))
    paired = pd.DataFrame(paired)
    paired.to_csv(PAPER_RESULTS / 'r4_paired_effects_by_seed.csv', index=False)
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), sharey=True)
    for mi, model in enumerate(models):
        for j, metric in enumerate(['prompt_accuracy', 'prompt_accuracy_auc_per_round']):
            ax = axes[2*mi+j]
            z = mean_se(paired[(paired.model == model) & (paired.metric == metric)], 'effect', ['contrast']).set_index('contrast').reindex(contrast_labels)
            ax.errorbar(z['mean'], np.arange(4), xerr=z['sem'], fmt='o', color='#008F7A', capsize=3)
            for k, label in enumerate(contrast_labels):
                values = paired[(paired.model == model) & (paired.metric == metric) & (paired.contrast == label)].effect
                ax.scatter(values, k + np.linspace(-.12, .12, len(values)), s=10, alpha=.5, color='#4C78A8')
            ax.axvline(0, color='gray', lw=1); ax.set_yticks(range(4), contrast_labels, fontsize=10)
            ax.set_title(f'({"abcd"[2*mi+j]}) {names[mi]}\n'+('Final accuracy' if j == 0 else 'Average over rounds'), fontsize=11)
            ax.set_xlabel('Difference (pp)', fontsize=10)
            ax.tick_params(axis='x', labelsize=10)
    fig.subplots_adjust(left=.18, bottom=.22, top=.78, wspace=.25)
    save_paper_figure(fig, 'r4_paired_effects'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    for mi, model in enumerate(models):
        subset = paired[(paired.model == model) & (paired.metric == 'prompt_accuracy_auc_gain')]
        z = mean_se(subset, 'effect', ['contrast']).set_index('contrast').reindex(contrast_labels)
        axes[mi].errorbar(z['mean'], range(4), xerr=z['sem'], fmt='o', color='#008F7A', capsize=3)
        axes[mi].axvline(0, color='gray', lw=1)
        axes[mi].set_yticks(range(4), contrast_labels, fontsize=10)
        axes[mi].set_title(f'({"ab"[mi]}) {names[mi]}', fontsize=11)
        axes[mi].set_xlabel('Initial-adjusted\ncurve difference (pp)', fontsize=10)
    fig.subplots_adjust(left=.27, bottom=.22, top=.84, wspace=.28)
    save_paper_figure(fig, 'r4_initial_adjusted_curve'); plt.close(fig)

    # Read per-round deployment origin, not a proxy based on observation counts.
    trajectory, alternatives = [], []
    for slot in json.loads((root/'allocation.json').read_text())['slots']:
        cfg = slot['config']
        if cfg['smoke']: continue
        directory = root/slot['label']
        assert (directory/'completed.json').exists()
        meta = dict(model=cfg['model']['name'], condition=cfg['condition'], seed=cfg['seed'])
        learners = [json.loads((directory/'agents'/str(i)/'checkpoint.json').read_text())['state']['learner'] for i in range(5)]
        for r in range(1, 50):
            rows = json.loads((directory/'rounds'/f'{r:04d}.json').read_text())
            copied = sum(learners[x['agent']]['acquisitions'][x['version']]['origin'] == 'social' and
                         learners[x['agent']]['acquisitions'][x['version']]['round'] <= r for x in rows)
            observed = sum(e['kind'] == 'observe' for x in rows for e in x['events'])
            trajectory.append(dict(**meta, round=r, block=(r-1)//5, observe_rate=observed/5, social_deploy=copied/5))
        alt = [json.loads((directory/'test_summary'/f'alternative-{i}.json').read_text()) for i in range(5)]
        selected = [json.loads((directory/'test_summary'/f'0049-{i}.json').read_text()) for i in range(5)]
        alternatives.append(dict(**meta, selected=np.mean([a['prompt_accuracy'] for a in selected]),
            alternative=np.mean([a['prompt_accuracy'] for a in alt]),
            difference=np.mean([a['prompt_accuracy']-s['prompt_accuracy'] for a,s in zip(alt, selected)])))
    trajectory = pd.DataFrame(trajectory); alternatives = pd.DataFrame(alternatives)
    trajectory.to_csv(PAPER_RESULTS/'r4_behavior_trajectories_by_seed.csv', index=False)
    alternatives.to_csv(PAPER_RESULTS/'r4_training_ranked_alternative_by_seed.csv', index=False)
    blocks = trajectory.groupby(['model', 'condition', 'seed', 'block'], as_index=False)[['round', 'observe_rate', 'social_deploy']].mean()
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.8), sharey=True)
    for mi, model in enumerate(models):
        for j, metric in enumerate(['observe_rate', 'social_deploy']):
            ax = axes[2*j+mi]
            for condition in ('llm_social_payoff', 'ucb_social', 'uniform_social'):
                curve(ax, blocks[blocks.model == model], metric, condition)
            ax.set_title(f'({"abcd"[2*j+mi]}) {names[mi]}', fontsize=12); ax.set_xlabel('Learning round')
            ax.set_ylim(0, 100); ax.set_xticks([1, 25, 49])
    axes[0].set_ylabel('Rounds\nobserving (%)', fontsize=11)
    axes[2].set_ylabel('Using a copied\nskill (%)', fontsize=11)
    finish(fig, axes, 'r4_social_behavior')

    for stem, frame, metrics, titles in [
        ('r4_behavior_summary', behavior, ['observe_pct', 'social_deploy_pct'], ['Rounds observing (%)', 'Direct copied deployments (%)']),
        ('r4_timing_summary', timing, ['mean_round', 'fraction_first10'], ['Mean observation round', 'Observations in rounds 1–10 (%)']),
        ('r4_search_supply', behavior, ['changed_revisions_per_agent', 'new_social_acquisitions_per_agent'], ['Successful revisions / agent', 'New copied skills / agent']),
        ('r4_early_late_observation', behavior, ['early_observe_pct', 'late_observe_pct'], ['Early rounds observing (%)', 'Later rounds observing (%)']),
        ('r4_initial_gains', final, ['prompt_accuracy_gain', 'constraint_accuracy_gain'], ['Whole-answer improvement (pp)', 'Constraint improvement (pp)']),
        ('r4_selection_diagnostic', alternatives, ['selected', 'alternative'], ['Selected test accuracy (%)', 'Training-ranked test accuracy (%)'])]:
        fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.95))
        for mi, model in enumerate(models):
            for j, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[2*j+mi]
                subset = frame[frame.model == model]
                order = [c for c in policies if c in set(subset.condition)]
                z = mean_se(subset, metric, ['condition']).set_index('condition').reindex(order)
                scale = 100 if frame is final or frame is alternatives else 1
                ax.bar(range(len(order)), scale*z['mean'], yerr=scale*z['sem'], capsize=2, color=[colors[c] for c in order])
                ax.set_xticks(range(len(order)), [short[policies.index(c)] for c in order], rotation=35, ha='right', fontsize=10)
                ax.set_title(f'({"abcd"[2*j+mi]}) {names[mi]}', fontsize=11)
                if mi == 0:
                    words = title.split()
                    midpoint = len(words)//2
                    ax.set_ylabel(' '.join(words[:midpoint])+'\n'+' '.join(words[midpoint:]), fontsize=10)
                if frame is final: ax.axhline(0, color='gray', lw=.8)
        for j in range(2):
            pair = axes[2*j:2*j+2]
            low = min(ax.get_ylim()[0] for ax in pair)
            high = max(ax.get_ylim()[1] for ax in pair)
            for ax in pair: ax.set_ylim(low, high)
        finish(fig, axes, stem, legend=False, bottom=.33)

    # Historical implementation comparison is optional, never mixed into current results.
    if (ROOT / 'runs/original_population/analysis/final_by_seed.csv').exists():
        old = pd.read_csv(ROOT / 'runs/original_population/analysis/final_by_seed.csv')
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.8), sharey=True)
        for mi, model in enumerate(models):
            for offset, (frame, label, color) in enumerate([(old, 'Original loop', '#999999'), (final, 'Corrected loop', '#008F7A')]):
                z = mean_se(frame[frame.model == model], 'prompt_accuracy', ['condition']).set_index('condition').reindex(policies)
                axes[mi].bar(np.arange(5)+(offset-.5)*.36, 100*z['mean'], .36, yerr=100*z['sem'], capsize=2, color=color, label=label)
            axes[mi].set_xticks(range(5), short, rotation=25, ha='right'); axes[mi].set_title(f'({"ab"[mi]}) {names[mi]}')
            axes[mi].set_ylim(0, 80)
        axes[0].set_ylabel('Final test accuracy (%)'); finish(fig, axes, 'r4_original_corrected', bottom=.34)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)
    phases = [('development_candidate', 'Candidate evaluation', '#4C78A8'),
              ('development_parent', 'Parent evaluation', '#9ECAE1'),
              ('revision', 'Revision', '#E69F00'), ('holdout', 'Hidden evaluation', '#777777')]
    for mi, model in enumerate(models):
        bottom = np.zeros(5)
        for phase, label, color in phases + [('other', 'Other calls', '#008F7A')]:
            ss = spending[spending.model == model]
            ss = ss[~ss.phase.isin([p[0] for p in phases])] if phase == 'other' else ss[ss.phase == phase]
            index = pd.MultiIndex.from_product([policies, sorted(final[final.model == model].seed.unique())], names=['condition', 'seed'])
            z = (ss.groupby(['condition', 'seed']).reported_cost.sum().reindex(index, fill_value=0)
                 .groupby('condition').mean().reindex(policies))
            axes[mi].bar(range(5), z, bottom=bottom, label=label, color=color); bottom += z.to_numpy()
        total = mean_se(costs[costs.model == model], 'reported_cost', ['condition']).set_index('condition').reindex(policies)
        assert np.allclose(bottom, total['mean'])
        axes[mi].errorbar(range(5), bottom, yerr=total['sem'], fmt='none', ecolor='black', capsize=2, lw=1)
        axes[mi].set_xticks(range(5), short, rotation=25, ha='right'); axes[mi].set_title(f'({"ab"[mi]}) {names[mi]}')
    axes[0].set_ylabel('Reported cost\n($/population)', fontsize=11)
    finish(fig, axes, 'r4_cost_breakdown', bottom=.39)

    # Machine-readable means/SE for auditing every numerical statement.
    stats = []
    for name, frame in [('final', final), ('behavior', behavior), ('timing', timing), ('alternative', alternatives), ('cost', costs)]:
        for metric in frame.select_dtypes('number').columns.difference(['seed']):
            z = mean_se(frame, metric, ['model', 'condition']).assign(source=name, metric=metric)
            stats.extend(z.to_dict('records'))
    pd.DataFrame(stats).to_csv(PAPER_RESULTS/'r4_statistics.csv', index=False)
    mean_se(paired, 'effect', ['model', 'contrast', 'metric']).to_csv(PAPER_RESULTS/'r4_paired_statistics.csv', index=False)
    print('Saved corrected population figures, seed-level data and paired statistics', flush=True)


def paper_feedback_bandits():
    """Reconstruct intervention integrals and changing-payoff controls from logs."""
    data = pd.read_csv(OUT/'rung1_intervention_raw_performance_by_seed.csv')
    curves, endpoints = [], []
    for row in data.to_dict('records'):
        directory = Path(row['run']) if isinstance(row['run'], str) else RUNS/'v3/rung1'/row['condition']/f"seed_{row['seed']}"
        rows = []
        with (directory/'events.jsonl').open() as stream:
            for line in stream:
                event = json.loads(line)
                if event.get('event') == 'round_end':
                    rows.append(dict(round=event['round'], agent=event['agent_id'],
                                     reward=event.get('reward') or 0., regret=event['expected_regret']))
        frame = pd.DataFrame(rows)
        assert len(frame) == 1000 and not frame.duplicated(['round', 'agent']).any(), directory
        assert frame.regret.notna().all()
        averaged = frame.groupby('round')[['reward', 'regret']].mean().sort_index()
        assert len(averaged) == 100 and np.isclose(averaged.reward.mean(), row['mean_reward'])
        meta = {k:row[k] for k in ['model', 'seed', 'replication', 'pair_id', 'intervention', 'access']}
        averaged['cumulative_reward'] = averaged.reward.cumsum()
        averaged['cumulative_regret'] = averaged.regret.cumsum()
        curves.extend(averaged.reset_index().assign(**meta).to_dict('records'))
        endpoints.append(dict(**meta, mean_reward=averaged.reward.mean(),
            cumulative_reward=averaged.cumulative_reward.iloc[-1],
            cumulative_regret=averaged.cumulative_regret.iloc[-1]))
    endpoints, curves = pd.DataFrame(endpoints), pd.DataFrame(curves)
    endpoints.to_csv(PAPER_RESULTS/'rung1_intervention_integrals_by_seed.csv', index=False)
    curves.to_csv(PAPER_RESULTS/'rung1_intervention_integral_curves.csv', index=False)
    order = ['Default', 'Token intervention', 'Forced search intervention']
    for metric, ylabel, stem in [('cumulative_reward', 'Cumulative reward\nthrough round 100', 'rung1_interventions_cumulative'),
                                 ('cumulative_regret', 'Cumulative regret\nthrough round 100', 'rung1_interventions_regret')]:
        fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.8), sharey=True)
        summary = mean_se(endpoints, metric, ['model', 'intervention', 'access'])
        summary.to_csv(PAPER_RESULTS/f'{stem}_statistics.csv', index=False)
        for mi, model in enumerate(MODELS):
            for ai, access in enumerate(['Solo', 'Social']):
                z = summary[(summary.model == model) & (summary.access == access)].set_index('intervention').reindex(order)
                axes[mi].bar(np.arange(3)+(ai-.5)*.36, z['mean'], .36, yerr=z['sem'], capsize=3,
                             color=['#4C78A8', '#E69F00'][ai], label=access)
            axes[mi].set_xticks(range(3), ['Default', 'PE', 'IS'])
            axes[mi].set_title(f'({"abc"[mi]}) {NAMES[model]}')
        axes[0].set_ylabel(ylabel, fontsize=11)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
        fig.subplots_adjust(left=.11, bottom=.27, top=.84, wspace=.15)
        save_paper_figure(fig, stem); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(9, 7), sharex=True, sharey=True)
    for ri, condition in enumerate(order):
        for mi, model in enumerate(MODELS):
            ax = axes[ri, mi]
            for access, color in [('Solo', '#4C78A8'), ('Social', '#E69F00')]:
                z = mean_se(curves[(curves.model == model) & (curves.intervention == condition) & (curves.access == access)], 'cumulative_reward', ['round'])
                ax.plot(z['round']+1, z['mean'], color=color, label=access)
                ax.fill_between(z['round']+1, z['mean']-z['sem'], z['mean']+z['sem'], color=color, alpha=.15)
            ax.set_title(f'({"abcdefghi"[3*ri+mi]}) {NAMES[model]} / '+['Default', 'PE', 'IS'][ri], fontsize=11)
            if mi == 0: ax.set_ylabel('Cumulative reward', fontsize=10)
            if ri == 2: ax.set_xlabel('Round')
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=.10, bottom=.16, top=.94, hspace=.4, wspace=.15)
    save_paper_figure(fig, 'rung1_intervention_learning'); plt.close(fig)
    # Reuse the saved restless controls; never silently launch new simulations.
    base = pd.read_csv(RUNS/'v3/social_baseline/hierarchical_ucb_comparison_by_seed.csv')
    labels = {'r1_e96ca53d112f':'Stationary', 'r1_f6f0aefb3fcc':'Change every 25', 'r1_120402ecc72d':'Change every 10'}
    base = base[base.policy.isin(['UCB solo', 'H-SUCB arm + payoff'])].copy()
    base['environment'] = base.environment_id.map(labels)
    base['cumulative_reward'] = 100*base.reward
    assert len(base) == 48 and base.environment.notna().all()
    base.to_csv(PAPER_RESULTS/'rung1_ucb_volatility_by_seed.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.9))
    differences = []
    for ei, env in enumerate(labels.values()):
        for pi, policy in enumerate(['UCB solo', 'H-SUCB arm + payoff']):
            values = base[(base.environment == env) & (base.policy == policy)].cumulative_reward
            axes[0].bar(ei+(pi-.5)*.36, values.mean(), .36, yerr=values.sem(), capsize=3,
                        color=['#4C78A8', '#E69F00'][pi], label=['Solo', 'Social'][pi] if ei == 0 else None)
        wide = base[base.environment == env].pivot(index='seed', columns='policy', values='cumulative_reward')
        delta = wide['H-SUCB arm + payoff']-wide['UCB solo']
        differences.extend(dict(environment=env, seed=s, effect=v) for s,v in delta.items())
        axes[1].bar(ei, delta.mean(), yerr=delta.sem(), capsize=3, color='#008F7A')
    for i, ax in enumerate(axes):
        ax.set_xticks(range(3), ['Stationary', 'Every 25', 'Every 10'], fontsize=10); ax.set_xlabel('Payoff changes')
        ax.set_title(['(a) Cumulative reward', '(b) Social minus solo'][i])
    axes[0].set_ylabel('Cumulative reward', fontsize=11); axes[1].axhline(0, color='black', lw=.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=.10, right=.98, bottom=.3, top=.84, wspace=.4)
    save_paper_figure(fig, 'rung1_ucb_volatility'); plt.close(fig)
    pd.DataFrame(differences).to_csv(PAPER_RESULTS/'rung1_ucb_volatility_contrasts.csv', index=False)
    print('Verified cumulative reward = 100 × all-round mean for every intervention run.', flush=True)


def paper_rung1_diagnosis_and_compact_intervention():
    """Compact main-text mechanism and intervention figures from saved results."""
    diagnostics = pd.read_csv(PAPER_RESULTS / 'rung1_primary_diagnostics_by_seed.csv')
    diagnostics = diagnostics[(diagnostics.model.isin(MODELS)) &
                              diagnostics.strategy.isin(['solo_llm', 'social_action_payoff'])]
    model_order = ['qwen3_14b', 'ministral3_14b_reasoning', 'gpt_oss_20b']
    short_names = {'qwen3_14b': 'Qwen', 'ministral3_14b_reasoning': 'Ministral',
                   'gpt_oss_20b': 'GPT-OSS'}

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.75))
    summary = mean_se(diagnostics, 'completion', ['model', 'strategy'])
    positions = np.arange(3)
    for offset, (strategy, label) in enumerate([('solo_llm', 'Solo'),
                                                ('social_action_payoff', 'Social')]):
        values = summary[summary.strategy == strategy].set_index('model').reindex(model_order)
        axes[0].bar(positions + (offset-.5)*.34, 100*values['mean'], .34,
                    yerr=100*values['sem'], capsize=2.5, color=COLORS[strategy], label=label)
    axes[0].set_xticks(positions, [short_names[m] for m in model_order], rotation=18, ha='right')
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Completed pulls (%)', fontsize=12)
    axes[0].set_title('(a) Execution', fontweight='bold', fontsize=13)

    diversity = pd.read_csv(RUNS / 'v3/rung1/analysis/population_arm_diversity_by_seed_round.csv')
    keep = pd.read_csv(RUNS / 'v3/rung1/analysis/condition_summary_by_seed.csv')
    keep = keep[(keep.guidance == 'neutral') &
                (keep.budget_mode == 'use_it_or_lose_it') &
                np.isclose(keep.reward_shape, 1.0) & keep.regime_period.isna()]
    diversity = diversity[(diversity.model_label == 'gpt_oss_20b') &
                          diversity.strategy.isin(['solo_llm', 'social_action_payoff'])]
    diversity = diversity.merge(keep[['condition', 'seed']].drop_duplicates(),
                                on=['condition', 'seed'])
    for strategy, label in [('solo_llm', 'Solo'), ('social_action_payoff', 'Social')]:
        values = mean_se(diversity[diversity.strategy == strategy], 'unique_pulled_arms', ['round'])
        x, y, se = values['round'] + 1, values['mean'], values['sem']
        axes[1].plot(x, y, color=COLORS[strategy], lw=2, label=label)
        axes[1].fill_between(x, y-se, y+se, color=COLORS[strategy], alpha=.16, linewidth=0)
    axes[1].set_xticks([1, 50, 100])
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Distinct arms pulled', fontsize=12)
    axes[1].set_title('(b) GPT-OSS search', fontweight='bold', fontsize=13)
    for ax in axes:
        ax.tick_params(labelsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(.52, -.02), fontsize=11)
    fig.subplots_adjust(left=.12, right=.99, bottom=.30, top=.84, wspace=.40)
    save_paper_figure(fig, 'rung1_diagnosis'); plt.close(fig)

    endpoints = pd.read_csv(PAPER_RESULTS / 'rung1_intervention_integrals_by_seed.csv')
    order = ['Default', 'Token intervention', 'Forced search intervention']
    intervention = mean_se(endpoints, 'cumulative_reward', ['model', 'intervention', 'access'])
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=True)
    for column, model in enumerate(model_order):
        for offset, access in enumerate(['Solo', 'Social']):
            values = intervention[(intervention.model == model) &
                                  (intervention.access == access)].set_index('intervention').reindex(order)
            axes[column].bar(np.arange(3)+(offset-.5)*.36, values['mean'], .36,
                             yerr=values['sem'], capsize=2.5,
                             color=['#4C78A8', '#E69F00'][offset], label=access)
        axes[column].set_xticks(range(3), ['Default', 'PE', 'IS'])
        axes[column].set_title(f'({"abc"[column]}) {short_names[model]}',
                               fontweight='bold', fontsize=12)
        axes[column].tick_params(labelsize=10)
    axes[0].set_ylabel('Cumulative reward', fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, bbox_to_anchor=(.52, -.02), fontsize=11)
    fig.subplots_adjust(left=.11, right=.995, bottom=.29, top=.84, wspace=.12)
    save_paper_figure(fig, 'rung1_interventions_cumulative'); plt.close(fig)

    audit = []
    for model in model_order:
        wide = diagnostics[diagnostics.model == model].pivot(
            index='seed', columns='strategy', values=['completion', 'completion_tokens'])
        audit.append(dict(
            model=model,
            solo_completion=wide['completion']['solo_llm'].mean(),
            social_completion=wide['completion']['social_action_payoff'].mean(),
            social_failed_pull_rate=1-wide['completion']['social_action_payoff'].mean(),
            solo_completion_tokens=wide['completion_tokens']['solo_llm'].mean(),
            social_completion_tokens=wide['completion_tokens']['social_action_payoff'].mean(),
            paired_social_minus_solo_tokens=(wide['completion_tokens']['social_action_payoff']-
                                             wide['completion_tokens']['solo_llm']).mean(),
        ))
    ucb = pd.read_csv(PAPER_RESULTS / 'rung1_matched_ucb_curves.csv')
    final_ucb = (ucb[(ucb['round'] == 99) &
                     ucb.strategy.isin(['Solo UCB', 'Hierarchical UCB', 'Oracle'])]
                 .groupby('strategy').agg(reward=('reward', 'mean'),
                                          expected_regret=('expected_regret', 'mean'),
                                          cumulative_regret=('cumulative_regret', 'mean')).reset_index())
    pd.DataFrame(audit).to_csv(PAPER_RESULTS / 'rung1_todo_audit.csv', index=False)
    final_ucb.to_csv(PAPER_RESULTS / 'rung1_ucb_endpoint_audit.csv', index=False)
    print(pd.DataFrame(audit).to_string(index=False), flush=True)
    print(final_ucb.to_string(index=False), flush=True)


def paper_skill_origin_costs():
    """Conditional execution/selection cost by acquisition origin, using saved rounds."""
    root = ROOT / 'runs/population'
    models = ['openai/gpt-oss-120b', 'z-ai/glm-5.3-flash']
    policies = ['llm_solo', 'llm_social_payoff', 'openevolve_solo', 'ucb_social', 'uniform_social']
    names = ['GPT-OSS-120B', 'GLM-5.3-Flash']
    rows, budgets = [], []
    for mi, model in enumerate(models):
        for policy in policies:
            for seed in range(6):
                directory = root/'full'/f'model_{mi}'/policy/f'seed_{seed}'
                cfg = json.loads((directory/'config.json').read_text())
                budgets.append(dict(model=model, condition=policy, seed=seed, **{k:cfg[k] for k in
                    ['tokens_per_round', 'controller_tokens', 'execution_tokens', 'revision_tokens', 'training_task_tokens']}))
                acquisitions = [json.loads((directory/'agents'/str(i)/'checkpoint.json').read_text())['state']['learner']['acquisitions'] for i in range(5)]
                assert (directory/'completed.json').exists()
                for r in range(1, 50):
                    records = json.loads((directory/'rounds'/f'{r:04d}.json').read_text())
                    assert len(records) == 5
                    for event in records:
                        a = acquisitions[event['agent']][event['version']]
                        assert a['round'] <= r
                        origin = 'Social' if a['origin'] == 'social' else 'Private/initial'
                        row = dict(model=model, condition=policy, seed=seed, round=r, agent=event['agent'],
                            origin=origin, original_origin=a['origin'], period='Early' if r <= 10 else 'Late')
                        for phase in ['selection', 'deployment', 'execution']:
                            p = event['budget']['phases'].get(phase, {})
                            row[phase+'_tokens'] = p.get('completion_tokens', 0)
                            row[phase+'_cost'] = p.get('cost_usd', 0)
                        rows.append(row)
                print(f'Audited origin/cost {mi} {policy} seed {seed}', flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == 60*49*5
    budgets = pd.DataFrame(budgets)
    for key in ['tokens_per_round', 'controller_tokens', 'execution_tokens', 'revision_tokens', 'training_task_tokens']:
        assert budgets[key].nunique() == 1, key
    budgets.to_csv(PAPER_RESULTS/'r4_matched_budget_audit.csv', index=False)
    frame.to_csv(PAPER_RESULTS/'r4_origin_token_events.csv', index=False)
    keys = ['model', 'condition', 'seed', 'period', 'origin']
    seed = frame.groupby(keys, as_index=False).agg(execution_tokens=('execution_tokens', 'mean'),
        deployment_tokens=('deployment_tokens', 'mean'), selection_tokens=('selection_tokens', 'mean'),
        execution_cost=('execution_cost', 'mean'), count=('agent', 'size'))
    seed.to_csv(PAPER_RESULTS/'r4_origin_tokens_by_seed.csv', index=False)
    stats = []
    for metric in ['execution_tokens', 'deployment_tokens', 'selection_tokens', 'execution_cost']:
        z = seed.groupby(['model', 'condition', 'period', 'origin'])[metric].agg(['mean', 'sem', 'count']).reset_index()
        stats.extend(z.assign(metric=metric).to_dict('records'))
    stats = pd.DataFrame(stats)
    stats.to_csv(PAPER_RESULTS/'r4_origin_token_statistics.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.4), sharey='row')
    social = ['llm_social_payoff', 'ucb_social', 'uniform_social']
    for mi, model in enumerate(models):
        for ci, policy in enumerate(social):
            ax = axes[mi, ci]
            for oi, origin in enumerate(['Private/initial', 'Social']):
                z = stats[(stats.model == model) & (stats.condition == policy) & (stats.origin == origin) & (stats.metric == 'execution_tokens')].set_index('period').reindex(['Early', 'Late'])
                ax.bar(np.arange(2)+(oi-.5)*.36, z['mean'], .36, yerr=z['sem'], capsize=3,
                       color=['#4C78A8', '#E69F00'][oi], label=origin)
            ax.set_xticks([0, 1], ['Early', 'Late']); ax.set_ylim(bottom=0)
            ax.set_title(f'({"abcdef"[mi*3+ci]}) '+['LLM social', 'Source UCB', 'Uniform source'][ci], fontsize=11)
            if ci == 0: ax.set_ylabel(names[mi]+'\nExecution tokens', fontsize=10)
        row_stats = stats[(stats.model == model) & stats.condition.isin(social) & (stats.metric == 'execution_tokens')]
        axes[mi, 0].set_ylim(0, 1.12 * (row_stats['mean'] + row_stats['sem']).max())
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=.12, bottom=.13, top=.93, hspace=.45, wspace=.16)
    save_paper_figure(fig, 'r4_execution_tokens_by_origin'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    for mi, model in enumerate(models):
        for oi, origin in enumerate(['Private/initial', 'Social']):
            z = stats[(stats.model == model) & (stats.condition == 'llm_social_payoff') & (stats.origin == origin) & (stats.metric == 'deployment_tokens')].set_index('period').reindex(['Early', 'Late'])
            axes[mi].bar(np.arange(2)+(oi-.5)*.36, z['mean'], .36, yerr=z['sem'], capsize=3,
                         color=['#4C78A8', '#E69F00'][oi], label=origin)
        axes[mi].set_xticks([0, 1], ['Early', 'Late']); axes[mi].set_title(f'({"ab"[mi]}) {names[mi]}', fontsize=12)
        axes[mi].set_ylabel('Deployment-choice tokens', fontsize=11); axes[mi].set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=.1, bottom=.26, top=.84, wspace=.38)
    save_paper_figure(fig, 'r4_deployment_tokens_by_origin'); plt.close(fig)
    learning = pd.read_csv(PAPER_RESULTS/'r4_learning_by_seed.csv')
    tests = pd.read_csv(PAPER_RESULTS/'r4_test_by_seed.csv')
    paired = []
    for metric, column, data in [('Recorded training', 'deployed_training_fitness', learning), ('Held-out constraints', 'constraint_accuracy', tests)]:
        for (model, policy, s), sub in data.groupby(['model', 'condition', 'seed']):
            z = sub.set_index('round')[column]
            paired.append(dict(model=model, condition=policy, seed=s, metric=metric,
                               initial=100*z.loc[0], final=100*z.loc[49], gain=100*(z.loc[49]-z.loc[0])))
    paired = pd.DataFrame(paired)
    paired.to_csv(PAPER_RESULTS/'r4_training_test_comparison_by_seed.csv', index=False)
    z = mean_se(paired, 'gain', ['model', 'condition', 'metric'])
    z.to_csv(PAPER_RESULTS/'r4_training_test_gain_statistics.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True)
    for mi, model in enumerate(models):
        for j, metric in enumerate(['Recorded training', 'Held-out constraints']):
            sub = z[(z.model == model) & (z.metric == metric)].set_index('condition').reindex(policies)
            axes[mi].bar(np.arange(5)+(j-.5)*.36, sub['mean'], .36, yerr=sub['sem'], capsize=2,
                         color=['#008F7A', '#4C78A8'][j], label=metric)
        axes[mi].set_xticks(range(5), ['Solo', 'Social', 'Open-\nEvolve', 'Source\nUCB', 'Uniform'], fontsize=9)
        axes[mi].set_title(f'({"ab"[mi]}) {names[mi]}', fontsize=12)
    axes[0].set_ylabel('Final minus initial\nconstraint score (pp)', fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=.11, bottom=.32, top=.84, wspace=.17)
    save_paper_figure(fig, 'r4_training_vs_test_gain'); plt.close(fig)
    print(z.to_string(index=False), flush=True)
    print(stats[(stats.condition.isin(social)) & stats.metric.isin(['execution_tokens', 'deployment_tokens'])].to_string(index=False), flush=True)


def paper_metric_motivation():
    """Appendix-only conceptual comparison: population controls and metric scope."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Rectangle
    fig, ax = plt.subplots(figsize=(12, 8.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9.3); ax.axis('off')
    ink, blue, green, gold = '#182838', '#356B99', '#08765A', '#B57817'

    def box(x, y, w, h, color, fill='white', radius=.1):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
            boxstyle=f'round,pad=.02,rounding_size={radius}',
            edgecolor=color, facecolor=fill, linewidth=1.4))

    def arrow(a,b,color,curve=0,style='-|>',width=1.5):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=13,
            connectionstyle=f'arc3,rad={curve}',color=color,linewidth=width))

    def learner(x, y, color):
        box(x,y,.58,.47,color,fill='#F2F6FA')
        for dx in [-.13,.13]:
            ax.add_patch(Rectangle((x+dx-.03,y+.035),.06,.065,color=color))
        ax.add_patch(Arc((x,y-.025),.27,.18,theta1=200,theta2=340,color=color,lw=1.4))
        box(x,y-.83,.62,.49,color)
        ax.text(x,y-.83,'skill',ha='center',va='center',fontsize=10,color=ink)
        arrow((x-.22,y-.30),(x-.22,y-.53),color)
        arrow((x+.32,y-.90),(x+.30,y+.03),color,curve=.55,width=1.1)

    titles=['(a) Single learner','(b) Independent learners','(c) Interacting learners']
    for i, (cx,color) in enumerate([(1.95,blue),(6,blue),(10.05,green)]):
        box(cx,7.30,3.72,3.42,color,fill='#FBFCFD')
        ax.text(cx,8.70,titles[i],ha='center',va='center',fontsize=13,weight='bold',color=ink)
        xs=[cx] if i==0 else [cx-1.12,cx,cx+1.12]
        for x in xs: learner(x,8.05,color)
        if i==2:
            for a,b in zip(xs,xs[1:]):
                arrow((a+.37,7.20),(b-.37,7.20),gold,style='<->',width=1.8)
        ax.text(cx,6.54,['One search history','Separate search histories','Copy + revise peers\' skills'][i],
            ha='center',va='center',fontsize=11,color=ink)
        ax.text(cx,6.18,'1 learner' if i==0 else 'N learners (3 shown)',ha='center',va='center',fontsize=11,color=ink)
        ax.text(cx,5.83,'Total learning tokens: C',ha='center',va='center',fontsize=11,weight='bold',color=color)
    ax.text(.15,5.26,'(b) − (a): value of parallel search at the same total C',
        ha='left',va='center',fontsize=12,weight='bold',color=blue)
    ax.text(.15,4.88,r'(c) − (b): $\Delta J(C)$ tests gains from exchange beyond independent search',
        ha='left',va='center',fontsize=12,weight='bold',color=green)
    ax.text(.13,4.42,'(d) What does each reported metric tell us?',fontsize=14,weight='bold',color=ink)

    widths=[3.55,1.85,1.70,1.90,2.75]
    bounds=np.cumsum([.1]+widths)
    centers=(bounds[:-1]+bounds[1:])/2
    headers=['Metric (Section 3)','Who performs\nbetter at the\ncomparison point?','Who spends\nfewer\ntokens?','Who earns more\nreward per\ntoken?','Is the whole more\nthan its parts\nat the same C?*']
    for x,label in zip(centers,headers):
        ax.text(x,3.76,label,ha='center',va='center',fontsize=9.7,weight='bold',color=ink)
    ax.plot([bounds[0],bounds[-1]],[3.30,3.30],color=ink,lw=1.3)
    labels=[r'Final performance: $J_T(\boldsymbol{\pi})$',r'Token use: $C_T(\boldsymbol{\pi})$',r'Efficiency: $E(\boldsymbol{\pi})$',r'Matched-cost gain: $\Delta J(C)$']
    checks=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,0,0,1]]
    ys=[2.99,2.38,1.77,1.16]
    box(6.0,1.16,11.8,.57,green,fill='#EAF5EF',radius=.04)
    for label,values,y in zip(labels,checks,ys):
        ax.text(.25,y,label,ha='left',va='center',fontsize=10.5,color=ink,
                weight='bold' if y==ys[-1] else 'normal')
        for x,yes in zip(centers[1:],values):
            if yes:
                ax.plot([x-.115,x-.025,x+.14],[y,y-.095,y+.12],color=green,lw=2.5,solid_capstyle='round')
            else:
                for sign in [-1,1]:
                    ax.plot([x-.10,x+.10],[y-sign*.10,y+sign*.10],color='#A55757',lw=2,solid_capstyle='round')
    ax.text(.15,.53,'*Same N, total C, and evaluation: better than the tested independent population, not every solo algorithm.',fontsize=9.5,color=ink)
    ax.text(.15,.19,'Checks show questions answered, not results. Recursive improvement also requires revision + reuse.',fontsize=9.7,color=ink)
    fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.99)
    save_paper_figure(fig,'metric_motivation'); plt.close(fig)


def paper_intro_figure():
    """Three decision stages, with private storage and a population of peers."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Arc
    from matplotlib.path import Path as DrawingPath
    fig, ax = plt.subplots(figsize=(9, 3.85))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6.3); ax.axis('off')
    ink, blue, gold, green = '#142331', '#285889', '#835500', '#006454'

    def box(x, y, w, h, text='', color=blue, size=13, fill='white', zorder=3):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle='round,pad=.04,rounding_size=.12',
                     lw=1.5, ec=color, fc=fill, zorder=zorder))
        if text:
            ax.text(x, y, text, ha='center', va='center', fontsize=size,
                    color=ink, weight='bold', zorder=4)

    def arrow(a, b, color=ink, dashed=False, curve=0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>',
                     connectionstyle=f'arc3,rad={curve}', mutation_scale=12,
                     lw=1.35, color=color, ls='--' if dashed else '-', zorder=2))

    def face(x, y, scale=1, color=blue):
        box(x, y, .62*scale, .52*scale, color=color, fill=color+'12')
        for dx in [-.14, .14]:
            ax.add_patch(Rectangle((x+(dx-.04)*scale, y+.05*scale),
                         .08*scale, .08*scale, fc=color, zorder=4))
        ax.add_patch(Arc((x,y-.025*scale), .30*scale, .22*scale,
                     theta1=200, theta2=340, color=color, lw=1.4, zorder=4))

    def file(x, y, color, label='.md', scale=1):
        box(x, y, .61*scale, .76*scale, color=color, fill='white')
        ax.text(x, y+.11*scale, label, ha='center', va='center',
                fontsize=11*scale, weight='bold', color=ink, zorder=4)
        for dy in [-.10, -.23]:
            ax.plot([x-.19*scale,x+.19*scale], [y+dy*scale]*2,
                    color=color, lw=1, zorder=4)

    box(7.5,5.88,14.55,.60,
        'One finite token budget: decide + learn + select + solve',
        color=ink, size=13, fill='#EAF0F6')
    for left, width, title in [(0.15,6.0,'1  CHOOSE HOW TO LEARN'),
                               (6.4,3.8,'2  SELECT SKILL'),
                               (10.45,4.4,'3  SOLVE TASK')]:
        box(left+width/2, 3.04, width, 4.64, color='#D6DEE5', fill='#FAFBFC', zorder=0)
        ax.text(left+.16, 5.05, title, fontsize=13, weight='bold', color=ink, zorder=4)

    # A smaller population runs the same loop independently.
    for x in [2.8, 3.9, 5.0]:
        face(x-.13, 4.48, .63, green)
        file(x+.27, 4.16, green, scale=.52)
        # A continuous curve ends at the file, so the arrowhead follows the loop.
        loop = DrawingPath([(x+.36,4.64), (x-.13,5.02), (x-.65,4.71),
                            (x-.48,4.12), (x-.33,3.78), (x+.27,3.75),
                            (x+.27,3.92)],
                           [DrawingPath.MOVETO] + [DrawingPath.CURVE4]*6)
        ax.add_patch(FancyArrowPatch(path=loop, arrowstyle='-|>',
                     mutation_scale=11, lw=1.2, color=green, zorder=4))
    ax.text(3.9, 3.48, 'Peers revise skills', ha='center', fontsize=12, weight='bold', color=ink, zorder=4)
    face(.95, 3.0, 1.15)
    ax.text(.95, 2.31, 'Agent', ha='center', fontsize=12, weight='bold', color=ink, zorder=4)
    # Copying/revision add files; skipping leaves the library unchanged.
    for y, label, color in [(2.85,'Copy a peer',gold),
                            (1.97,'Revise skill',blue),
                            (1.09,'Skip learning',ink)]:
        box(3.85,y,2.55,.53,label,color)
        arrow((1.35,3.0),(2.52,y),color)
        if y > 1.5:
            lane = 5.65 if y > 2.5 else 6.02
            end_y = 3.82 if y > 2.5 else 3.37
            ax.plot([5.18,lane,lane],[y,y,end_y],color=color,lw=1.35,zorder=2)
            arrow((lane,end_y),(6.91,end_y),color)
        else:
            ax.plot([5.18,6.50,6.50],[y,y,2.42],color=color,lw=1.35,zorder=2)
            arrow((6.50,2.42),(6.84,2.42),color)
    arrow((3.85,3.42),(3.85,3.17),gold,True)
    ax.text(8.3,4.55,'Own library of\nskill files',ha='center',va='center',fontsize=12,weight='bold',color=ink,zorder=4)
    for x, color in [(7.38,blue),(8.3,gold),(9.22,blue)]:
        file(x,3.61,color)
        arrow((x,3.16),(8.3,2.79),color)
    box(8.3,2.42,2.8,.62,'Choose skill',blue)
    ax.text(8.3,1.57,'Choose a new\nor earlier skill',
            ha='center',va='center',fontsize=12,weight='bold',color=ink,zorder=4,linespacing=1.4)
    arrow((9.75,2.42),(10.78,2.42),blue)

    ax.text(12.65,4.42,'Task distribution',ha='center',fontsize=12,weight='bold',color=ink,zorder=4)
    for x,y in [(11.7,3.65),(12.65,3.8),(13.6,3.65)]:
        file(x,y,green,label='task',scale=.8)
    arrow((12.65,3.34),(12.65,2.81),green)
    box(12.65,2.42,3.55,.62,'Use skill on task',green)
    arrow((12.65,2.05),(12.65,1.77),green)
    box(12.65,1.40,3.10,.55,'Task feedback',green)
    # Training feedback closes the loop; held-out evaluation is not in it.
    ax.plot([12.65,12.65,.95,.95],[1.08,.29,.29,1.70],color=blue,lw=1.4,zorder=2)
    arrow((.95,1.70),(.95,2.02),blue)
    ax.text(7.6,.29,'Next round: choose again using feedback',
            ha='center',va='center',fontsize=12,weight='bold',color=ink,
            bbox=dict(fc='white',ec='none',pad=3),zorder=4)
    fig.subplots_adjust(left=.005, right=.995, top=.99, bottom=.01)
    save_paper_figure(fig, 'recursive_social_improvement_overview'); plt.close(fig)


def paper_copy_timing():
    """Assigned-timing and early-access results from completed Rung 4 populations."""
    run_root = (ROOT / 'runs/timing')
    root = run_root / 'analysis'
    tests = pd.read_csv(root / 'test_by_seed.csv')
    final = pd.read_csv(root / 'final_by_seed.csv')
    paired = pd.read_csv(root / 'paired_by_seed.csv')
    learning = pd.read_csv(root / 'learning_by_seed.csv')
    spending = pd.read_csv(root / 'spending_by_phase.csv')
    models = ['openai/gpt-oss-120b', 'z-ai/glm-5.3-flash']
    names = ['GPT-OSS-120B', 'GLM-5.3-Flash']
    common = [0, 10, 25, 49]
    colors = {
        'llm_solo': '#4C78A8', 'llm_social_payoff': '#C76D00',
        'forced_early': '#E69F00', 'forced_distributed': '#8E63B0',
        'optional_early': '#008F7A',
    }
    labels = {
        'llm_solo': 'Solo', 'llm_social_payoff': 'Unrestricted social',
        'forced_early': 'Forced early',
        'forced_distributed': 'Forced distributed',
        'optional_early': 'Early access',
    }

    def curve(ax, frame, model, condition, metric='prompt_accuracy', x='round'):
        sub = frame[(frame.model == model) & (frame.condition == condition)]
        z = sub.groupby(x)[metric].agg(['mean', 'sem']).reset_index()
        xx, yy, ss = z[x].to_numpy(), 100*z['mean'].to_numpy(), 100*z['sem'].to_numpy()
        ax.plot(xx, yy, color=colors[condition], lw=2, label=labels[condition])
        ax.fill_between(xx, yy-ss, yy+ss, color=colors[condition], alpha=.16, linewidth=0)

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.5), sharex=True, sharey='col')
    rows = [
        ['llm_solo', 'forced_early', 'forced_distributed'],
        ['llm_solo', 'llm_social_payoff', 'optional_early'],
    ]
    common_tests = tests[tests['round'].isin(common)]
    for ri, conditions in enumerate(rows):
        for mi, model in enumerate(models):
            ax = axes[ri, mi]
            for condition in conditions:
                curve(ax, common_tests, model, condition)
            ax.set_title(f'({"abcd"[2*ri+mi]}) {names[mi]}', fontsize=11)
            ax.set_xticks(common)
            ax.set_xlabel('Learning round')
            if mi == 0:
                ax.set_ylabel('Test accuracy (%)')
            ax.margins(y=.18)
    handles, legend_labels = [], []
    for ax in axes.ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in legend_labels:
                handles.append(handle); legend_labels.append(label)
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.52, -.005),
               ncol=5, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.10, right=.99, top=.93, bottom=.14, hspace=.42, wspace=.16)
    save_paper_figure(fig, 'r4_copy_timing_performance'); plt.close(fig)

    endpoints = learning.sort_values('round').groupby(['model', 'condition', 'seed'], as_index=False).tail(1)
    allocation = json.loads((run_root / 'allocation.json').read_text())
    novelty = []
    for slot in allocation['slots'] + allocation.get('reused_controls', []):
        cfg = slot['config']
        if cfg.get('smoke'):
            continue
        directory = Path(slot['directory']) if slot.get('reused') else run_root / slot['label']
        initial_rows = json.loads((directory / 'rounds' / '0000.json').read_text())
        initial_versions = {row['agent']: row['version'] for row in initial_rows}
        observations = []
        for path in sorted((directory / 'rounds').glob('*.json')):
            for row in json.loads(path.read_text()):
                for event in row['events']:
                    if event['kind'] == 'observe':
                        observations.append(event['version'] != initial_versions[row['agent']])
        novelty.append(dict(model=cfg['model']['name'], condition=cfg['condition'], seed=cfg['seed'],
                            noninitial_observation_pct=100*np.mean(observations) if observations else np.nan))
    endpoints = endpoints.merge(pd.DataFrame(novelty), on=['model', 'condition', 'seed'], how='left', validate='one_to_one')
    metrics = [('cumulative_observations_per_agent', 'Observations\nper agent'),
               ('noninitial_observation_pct', 'Non-initial\nobservations (%)'),
               ('cumulative_valid_revisions_per_agent', 'Successful revisions\nper agent')]
    conditions = ['llm_solo', 'llm_social_payoff', 'optional_early', 'forced_early', 'forced_distributed']
    short = ['Solo', 'Unrestricted', 'Early access', 'Early forced', 'Distributed']
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 7.3), sharex=True)
    for ri, (metric, ylabel) in enumerate(metrics):
        for mi, model in enumerate(models):
            ax = axes[ri, mi]
            z = mean_se(endpoints[endpoints.model == model], metric, ['condition']).set_index('condition').reindex(conditions)
            ax.bar(range(len(conditions)), z['mean'], yerr=z['sem'], capsize=3,
                   color=[colors[c] for c in conditions])
            ax.set_title(f'({"abcdef"[2*ri+mi]}) {names[mi]}', fontsize=11)
            ax.set_xticks(range(len(conditions)), short, rotation=25, ha='right', fontsize=8)
            if mi == 0:
                ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
    fig.subplots_adjust(left=.13, right=.99, top=.95, bottom=.17, hspace=.42, wspace=.18)
    save_paper_figure(fig, 'r4_copy_timing_behavior'); plt.close(fig)

    paired_summary = (paired.groupby(['model', 'contrast', 'metric'])
                      .agg(final_gain=('final_gain', 'mean'), final_gain_se=('final_gain', 'sem'),
                           learning_curve_gain=('learning_curve_gain', 'mean'),
                           learning_curve_gain_se=('learning_curve_gain', 'sem'),
                           baseline_adjusted_gain=('baseline_adjusted_learning_curve_gain', 'mean'),
                           baseline_adjusted_gain_se=('baseline_adjusted_learning_curve_gain', 'sem'),
                           seeds=('seed', 'nunique')).reset_index())
    for column in ['final_gain', 'final_gain_se', 'learning_curve_gain', 'learning_curve_gain_se',
                   'baseline_adjusted_gain', 'baseline_adjusted_gain_se']:
        paired_summary[column] *= 100
    paired_summary.to_csv(PAPER_RESULTS / 'r4_copy_timing_paired_summary.csv', index=False)
    endpoint_summary = []
    for metric in ['prompt_accuracy', 'prompt_accuracy_common_auc', 'prompt_accuracy_gain',
                   'constraint_accuracy', 'constraint_accuracy_common_auc', 'constraint_accuracy_gain']:
        z = mean_se(final, metric, ['model', 'condition'])
        z['metric'] = metric
        endpoint_summary.append(z)
    pd.concat(endpoint_summary, ignore_index=True).to_csv(
        PAPER_RESULTS / 'r4_copy_timing_endpoint_summary.csv', index=False)
    behavior_summary = []
    for metric, _ in metrics:
        z = mean_se(endpoints, metric, ['model', 'condition'])
        z['metric'] = metric
        behavior_summary.append(z)
    pd.concat(behavior_summary, ignore_index=True).to_csv(
        PAPER_RESULTS / 'r4_copy_timing_behavior_summary.csv', index=False)
    new_spending = spending[~spending.reused_control]
    spend = (new_spending.groupby(['model', 'condition', 'seed'], as_index=False)
             [['reported_cost', 'conservative_exposure']].sum())
    spend.to_csv(PAPER_RESULTS / 'r4_copy_timing_spending_by_seed.csv', index=False)
    print(paired_summary.to_string(index=False), flush=True)
    print('\nBehavior endpoints:', flush=True)
    print(pd.concat(behavior_summary, ignore_index=True).to_string(index=False), flush=True)
    print(f'\nNew production reported cost: ${new_spending.reported_cost.sum():.2f}', flush=True)
    print(f'New production conservative exposure: ${new_spending.conservative_exposure.sum():.2f}', flush=True)


def paper_observation_networks():
    """Describe saved directed observations and controller effort, without API calls.

    Source repetition excludes each observer's first observation. Familiarity
    means previously observed at any earlier round, not a claim about the prompt's
    memory. All uncertainty is across populations, never individual edges/calls.
    """
    root = ROOT / 'runs/population'
    models = ['openai/gpt-oss-120b', 'z-ai/glm-5.3-flash']
    names = ['GPT-OSS-120B', 'GLM-5.3-Flash']
    social = ['llm_social_payoff', 'ucb_social', 'uniform_social']
    colors = dict(zip(social, ['#E69F00', '#B65C9A', '#777777']))
    labels = dict(zip(social, ['LLM social', 'UCB social', 'Uniform social']))
    records = []
    for mi, model in enumerate(models):
        for condition in social + ['llm_solo']:
            for seed in range(6):
                directory = root / 'full' / f'model_{mi}' / condition / f'seed_{seed}'
                assert (directory / 'completed.json').exists(), directory
                paths = sorted((directory / 'rounds').glob('*.json'))
                assert len(paths) == 50, directory
                seen = [set() for _ in range(5)]
                seen_versions = [set() for _ in range(5)]
                previous = [None] * 5
                for r, path in enumerate(paths):
                    rows = json.loads(path.read_text())
                    assert len(rows) == 5 and {x['agent'] for x in rows} == set(range(5))
                    for row in rows:
                        assert row['round'] == r
                        agent = row['agent']
                        # Version IDs denote content-addressed skills, including
                        # copies already imported or privately produced.
                        before = seen_versions[agent].copy()
                        observations = [e for e in row['events'] if e['kind'] == 'observe']
                        assert len(observations) <= 1
                        event = observations[0] if observations else None
                        target = int(event['target']) if event else None
                        phases = row['budget']['phases']
                        record = dict(model=model, condition=condition, seed=seed,
                            agent=agent, round=r, action=row['action']['action'],
                            observed=event is not None, target=target,
                            previous_target=previous[agent],
                            repeat_previous=(target == previous[agent]) if event and previous[agent] is not None else np.nan,
                            familiar_source=(target in seen[agent]) if event else np.nan,
                            new_version=(event['version'] not in before) if event else np.nan)
                        for phase in ['selection', 'deployment']:
                            values = phases.get(phase, {})
                            for metric in ['completion_tokens', 'prompt_tokens', 'cost_usd', 'calls']:
                                record[f'{phase}_{metric}'] = values.get(metric, 0)
                        records.append(record)
                        if event:
                            assert target != agent and 0 <= target < 5
                            seen[agent].add(target)
                            previous[agent] = target
                            seen_versions[agent].add(event['version'])
                        seen_versions[agent].add(row['version'])
                        if row.get('candidate_version'):
                            seen_versions[agent].add(row['candidate_version'])
    frame = pd.DataFrame(records)
    frame = frame[frame['round'] > 0].copy()
    assert len(frame) == 2 * 4 * 6 * 5 * 49
    assert not frame[frame.condition == 'llm_solo'].observed.any()
    frame['block'] = ((frame['round']-1)//5)*5 + 1
    frame['period'] = np.where(frame['round'] <= 10, 'Early (1-10)', 'Late (11-49)')
    frame.to_csv(PAPER_RESULTS/'r4_observation_events.csv', index=False)
    keys = ['model', 'condition', 'seed']
    blocks, periods, edges, familiar = [], [], [], []

    def describe(sub):
        obs = sub[sub.observed]
        targets = obs.target.value_counts()
        return dict(observations=len(obs), available_rounds=len(sub),
            eligible_repeats=int(obs.repeat_previous.count()),
            repeat_previous=obs.repeat_previous.mean(), familiar_source=obs.familiar_source.mean(),
            new_version=obs.new_version.mean(),
            top_source_share=targets.max()/len(obs) if len(obs) else np.nan,
            directed_edges=obs[['agent', 'target']].drop_duplicates().shape[0],
            selection_tokens_observe=obs.selection_completion_tokens.mean(),
            deployment_tokens_observe=obs.deployment_completion_tokens.mean(),
            selection_tokens_nonobserve=sub[~sub.observed].selection_completion_tokens.mean(),
            selection_tokens_all=sub.selection_completion_tokens.mean(),
            deployment_tokens_all=sub.deployment_completion_tokens.mean(),
            observers=int(obs.agent.nunique()))

    for values, sub in frame.groupby(keys):
        meta = dict(zip(keys, values))
        for block, group in sub.groupby('block'):
            row = dict(**meta, block=block, **describe(group))
            cumulative = sub[(sub['round'] <= group['round'].max()) & sub.observed]
            row['cumulative_edge_fraction'] = cumulative[['agent', 'target']].drop_duplicates().shape[0]/20
            blocks.append(row)
        for period, group in list(sub.groupby('period')) + [('All (1-49)', sub)]:
            periods.append(dict(**meta, period=period, **describe(group)))
            obs = group[group.observed]
            for observer in range(5):
                for target in range(5):
                    if observer == target:
                        continue
                    n = int(((obs.agent == observer) & (obs.target == target)).sum())
                    edges.append(dict(**meta, period=period, observer=observer, target=target,
                        count=n, total_observations=len(obs), fraction=n/len(obs) if len(obs) else np.nan))
        # Within-observer, within-five-round-block contrasts avoid comparing
        # different observers or distant learning stages. They remain descriptive.
        for (agent, block), group in sub[sub.observed].groupby(['agent', 'block']):
            novel = group[group.familiar_source == False]
            known = group[group.familiar_source == True]
            if len(novel) and len(known):
                familiar.append(dict(**meta, agent=agent, block=block,
                    novel_observations=len(novel), familiar_observations=len(known),
                    selection_token_difference=known.selection_completion_tokens.mean()-novel.selection_completion_tokens.mean(),
                    deployment_token_difference=known.deployment_completion_tokens.mean()-novel.deployment_completion_tokens.mean()))
    blocks = pd.DataFrame(blocks)
    periods = pd.DataFrame(periods)
    edges = pd.DataFrame(edges)
    familiar = pd.DataFrame(familiar)
    for name, data in [('network_blocks_by_seed', blocks), ('network_periods_by_seed', periods),
                       ('network_edges_by_seed', edges), ('familiarity_matched_blocks', familiar)]:
        data.to_csv(PAPER_RESULTS/f'r4_{name}.csv', index=False)
    familiar_seed = familiar.groupby(keys, as_index=False).agg(
        selection_token_difference=('selection_token_difference', 'mean'),
        deployment_token_difference=('deployment_token_difference', 'mean'),
        matched_blocks=('block', 'size'))
    familiar_seed.to_csv(PAPER_RESULTS/'r4_familiarity_by_seed.csv', index=False)
    stats = []
    for metric in periods.select_dtypes('number').columns.difference(['seed']):
        stats.extend(mean_se(periods, metric, ['model', 'condition', 'period']).assign(metric=metric).to_dict('records'))
    pd.DataFrame(stats).to_csv(PAPER_RESULTS/'r4_network_statistics.csv', index=False)

    def line(ax, data, metric, label, color, scale=1, style='-'):
        z = mean_se(data, metric, ['block'])
        x = z.block.to_numpy()+2
        y, se = scale*z['mean'].to_numpy(), scale*z['sem'].to_numpy()
        ax.plot(x, y, lw=1.8, color=color, label=label, ls=style)
        ax.fill_between(x, y-se, y+se, color=color, alpha=.15, linewidth=0)

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
    for mi, model in enumerate(models):
        for condition in social:
            sub = blocks[(blocks.model == model) & (blocks.condition == condition)]
            line(axes[mi], sub, 'repeat_previous', labels[condition], colors[condition], 100)
        axes[mi].axhline(25, ls=':', color='black', lw=1)
        axes[mi].set_ylim(0, 105)
        sub = blocks[(blocks.model == model) & (blocks.condition == 'llm_social_payoff')]
        line(axes[mi+2], sub, 'selection_tokens_observe', 'Social: observe', '#E69F00')
        line(axes[mi+2], sub, 'selection_tokens_nonobserve', 'Social: other action', '#008F7A')
        sub = blocks[(blocks.model == model) & (blocks.condition == 'llm_solo')]
        line(axes[mi+2], sub, 'selection_tokens_all', 'Solo: any action', '#4C78A8', style='--')
        axes[mi+2].set_ylim(bottom=0)
    for i, ax in enumerate(axes):
        ax.set_title(f'({"abcd"[i]}) {names[i%2]}', fontsize=11)
        ax.set_xlabel('Learning round'); ax.set_xticks([1, 25, 49])
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel('Repeat previous\nsource (%)', fontsize=11)
    axes[2].set_ylabel('Learning-decision\ncompletion tokens', fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.02, -.06), ncol=3, fontsize=9, frameon=False)
    fig.legend(*axes[2].get_legend_handles_labels(), loc='lower right', bbox_to_anchor=(1.01, -.06), ncol=3, fontsize=9, frameon=False)
    fig.subplots_adjust(left=.075, right=.99, bottom=.27, top=.84, wspace=.7)
    save_paper_figure(fig, 'r4_source_persistence'); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3))
    for mi, model in enumerate(models):
        for condition in social:
            sub = blocks[(blocks.model == model) & (blocks.condition == condition)]
            line(axes[mi], sub, 'cumulative_edge_fraction', labels[condition], colors[condition], 100)
            line(axes[mi+2], sub, 'top_source_share', labels[condition], colors[condition], 100)
    for i, ax in enumerate(axes):
        ax.set_title(f'({"abcd"[i]}) {names[i%2]}', fontsize=11)
        ax.set_xlabel('Learning round'); ax.set_xticks([1, 25, 49]); ax.set_ylim(0, 105)
    axes[0].set_ylabel('Directed links\never used (%)', fontsize=11)
    axes[2].set_ylabel('Most-observed\npeer share (%)', fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=11, frameon=False)
    fig.subplots_adjust(left=.075, right=.99, bottom=.28, top=.84, wspace=.7)
    save_paper_figure(fig, 'r4_network_coverage'); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.1))
    for pi, period in enumerate(['Early (1-10)', 'Late (11-49)']):
        for mi, model in enumerate(models):
            ax = axes[2*pi+mi]
            sub = edges[(edges.model == model) & (edges.condition == 'llm_social_payoff') & (edges.period == period)]
            matrix = sub.groupby(['observer', 'target']).fraction.mean().unstack().reindex(index=range(5), columns=range(5))*100
            sns.heatmap(matrix, ax=ax, cmap='Blues', vmin=0, vmax=20, square=True, cbar=False,
                        annot=True, fmt='.0f', annot_kws={'size': 8})
            ax.set_title(f'({"abcd"[2*pi+mi]}) {names[mi]}\n{period}', fontsize=10)
            ax.set_xlabel('Observed peer ID', fontsize=10)
            ax.set_ylabel('Observer ID' if mi == 0 else '', fontsize=10)
    fig.subplots_adjust(left=.065, right=.99, bottom=.19, top=.80, wspace=.43)
    save_paper_figure(fig, 'r4_observation_graphs'); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(11.1, 3.0))
    for mi, model in enumerate(models):
        data = frame[(frame.model == model) & (frame.condition == 'llm_social_payoff') & frame.observed]
        for phase, color, label in [('selection', '#E69F00', 'Before observation'), ('deployment', '#4C78A8', 'After observation')]:
            grouped = data.groupby(['seed', 'period', 'familiar_source'], as_index=False)[phase+'_completion_tokens'].mean()
            # Keep period on the x-axis; average seeds independently in each cell.
            for known, style in [(False, '--'), (True, '-')]:
                z = mean_se(grouped[grouped.familiar_source == known], phase+'_completion_tokens', ['period']).set_index('period').reindex(['Early (1-10)', 'Late (11-49)'])
                axes[mi].errorbar([0, 1], z['mean'], yerr=z['sem'], color=color, ls=style,
                                 marker='o', capsize=3, label=label+(': familiar' if known else ': new'))
        axes[mi].set_xticks([0, 1], ['Early', 'Late']); axes[mi].set_ylim(bottom=0)
        sub = familiar_seed[(familiar_seed.model == model) & (familiar_seed.condition == 'llm_social_payoff')]
        for j, metric in enumerate(['selection_token_difference', 'deployment_token_difference']):
            values = sub[metric]
            axes[mi+2].bar(j, values.mean(), yerr=values.sem(), color=['#E69F00', '#4C78A8'][j], capsize=3)
            axes[mi+2].scatter(j+np.linspace(-.07, .07, len(values)), values, color='black', s=12, zorder=3)
        axes[mi+2].axhline(0, color='black', lw=.8)
        axes[mi+2].set_xticks([0, 1], ['Before', 'After']); axes[mi+2].set_xlabel('Observation', fontsize=11)
    for i, ax in enumerate(axes):
        ax.set_title(f'({"abcd"[i]}) {names[i%2]}', fontsize=11)
    axes[0].set_ylabel('Completion tokens\nper observation', fontsize=11)
    axes[2].set_ylabel('Familiar minus new\ncompletion tokens', fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(.53, -.1))
    fig.subplots_adjust(left=.08, right=.99, bottom=.29, top=.84, wspace=.7)
    save_paper_figure(fig, 'r4_familiarity_compute'); plt.close(fig)
    print('Saved network/effort diagnostics from 48 completed populations; no model calls.', flush=True)
    print(pd.DataFrame(stats).query("condition == 'llm_social_payoff' and metric in ['repeat_previous', 'selection_tokens_observe', 'deployment_tokens_observe', 'observations', 'directed_edges', 'top_source_share']").to_string(index=False), flush=True)
    print('Within-observer/block familiarity contrasts:', flush=True)
    print(familiar_seed[familiar_seed.condition == 'llm_social_payoff'].to_string(index=False), flush=True)
    paper_network_summary()


def paper_network_summary():
    """Compact main-text view and coverage audit of the saved network analysis."""
    edges = pd.read_csv(PAPER_RESULTS/'r4_network_edges_by_seed.csv')
    periods = pd.read_csv(PAPER_RESULTS/'r4_network_periods_by_seed.csv')
    events = pd.read_csv(PAPER_RESULTS/'r4_observation_events.csv')
    models = ['openai/gpt-oss-120b', 'z-ai/glm-5.3-flash']
    names = ['GPT-OSS-120B', 'GLM-5.3-Flash']
    intervals = ['Early (1-10)', 'Late (11-49)']
    attention = edges.groupby(['model', 'condition', 'seed', 'period', 'target'], as_index=False).fraction.sum(min_count=1)
    attention.to_csv(PAPER_RESULTS/'r4_peer_attention_by_seed.csv', index=False)
    stats = mean_se(attention, 'fraction', ['model', 'condition', 'period', 'target'])
    stats.to_csv(PAPER_RESULTS/'r4_peer_attention_statistics.csv', index=False)
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0))
    for mi, model in enumerate(models):
        for pi, (period, color) in enumerate(zip(intervals, ['#4C78A8', '#E69F00'])):
            z = stats[(stats.model == model) & (stats.condition == 'llm_social_payoff') & (stats.period == period)].set_index('target').reindex(range(5))
            axes[mi].bar(np.arange(5)+(pi-.5)*.38, 100*z['mean'], .38,
                         yerr=100*z['sem'], capsize=2, color=color, label=period)
            sub = periods[(periods.model == model) & (periods.condition == 'llm_social_payoff') & (periods.period == period)]
            columns = ['selection_tokens_observe', 'deployment_tokens_observe']
            axes[mi+2].bar(np.arange(2)+(pi-.5)*.38, sub[columns].mean(), .38,
                           yerr=sub[columns].sem(), capsize=2, color=color, label=period)
        axes[mi].set_xticks(range(5)); axes[mi].set_xlabel('Observed peer ID'); axes[mi].set_ylim(0, 55)
        axes[mi+2].set_xticks([0, 1], ['Before', 'After']); axes[mi+2].set_xlabel('Observation')
        axes[mi+2].set_ylim(bottom=0)
    for i, ax in enumerate(axes):
        ax.set_title(f'({"abcd"[i]}) {names[i%2]}', fontsize=11)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel('Share of\nobservations (%)', fontsize=11)
    axes[2].set_ylabel('Decision-call\ncompletion tokens', fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.075, right=.99, bottom=.28, top=.84, wspace=.65)
    save_paper_figure(fig, 'r4_social_attention'); plt.close(fig)
    # Explicitly retain conditional coverage, including empty late blocks.
    blocks = pd.read_csv(PAPER_RESULTS/'r4_network_blocks_by_seed.csv')
    metrics = ['repeat_previous', 'top_source_share', 'selection_tokens_observe', 'selection_tokens_nonobserve']
    coverage = []
    for metric in metrics:
        z = blocks.groupby(['model', 'condition', 'block'])[metric].agg(['mean', 'sem', 'count']).reset_index()
        coverage.extend(z.assign(metric=metric).to_dict('records'))
    pd.DataFrame(coverage).to_csv(PAPER_RESULTS/'r4_network_block_statistics.csv', index=False)
    familiarity = pd.read_csv(PAPER_RESULTS/'r4_familiarity_by_seed.csv')
    for metric in ['selection_token_difference', 'deployment_token_difference']:
        print(familiarity.groupby(['model', 'condition'])[metric].agg(['mean', 'sem', 'count']).to_string(), flush=True)
    # Long calls are retained in means; save their frequency so a noisy mean
    # is not mistaken for every decision taking that much computation.
    obs = events[(events.condition == 'llm_social_payoff') & events.observed].copy()
    obs['selection_over_4k'] = obs.selection_completion_tokens > 4096
    token_audit = obs.groupby(['model', 'seed']).agg(observations=('observed', 'size'),
        selection_over_4k=('selection_over_4k', 'sum'),
        median_selection_tokens=('selection_completion_tokens', 'median'),
        max_selection_tokens=('selection_completion_tokens', 'max'))
    token_audit.to_csv(PAPER_RESULTS/'r4_observation_token_tail_by_seed.csv')
    print('Observation decision token-tail audit:', flush=True)
    print(token_audit.to_string(), flush=True)
    # Independent reconstruction cross-check against the prior behavior audit.
    behavior = pd.read_csv(PAPER_RESULTS/'r4_behavior_by_seed.csv')
    all_rounds = periods[periods.period == 'All (1-49)'].merge(behavior, on=['model', 'condition', 'seed'], validate='one_to_one')
    assert len(all_rounds) == 48
    assert np.allclose(all_rounds.observations/5, all_rounds.observations_per_agent)
    assert np.allclose(100*all_rounds.observations/all_rounds.available_rounds, all_rounds.observe_pct)
    print('Verified all 48 observation totals against the prior independent audit.', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-raw-only", action="store_true")
    parser.add_argument("--cumulative-only", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--ucb-precision", action="store_true")
    parser.add_argument("--paper-skills", action="store_true")
    parser.add_argument("--paper-offline", action="store_true")
    parser.add_argument("--paper-networks", action="store_true")
    parser.add_argument("--paper-feedback", action="store_true")
    parser.add_argument("--paper-timing", action="store_true")
    parser.add_argument("--paper-metric-motivation", action="store_true")
    args = parser.parse_args()
    setup()
    if args.paper_metric_motivation:
        paper_metric_motivation()
    elif args.paper_timing:
        PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
        paper_copy_timing()
    elif args.paper_feedback:
        PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
        paper_intro_figure()
        paper_feedback_bandits()
        paper_skill_origin_costs()
    elif args.paper_networks:
        PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
        paper_observation_networks()
    elif args.paper_offline:
        PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
        paper_offline_skill_check()
    elif args.paper_skills:
        PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
        paper_skill_results()
    elif args.ucb_precision:
        PAPER_RESULTS.mkdir(parents=True, exist_ok=True)
        paper_ucb_precision()
    elif args.paper:
        paper_figures()
    elif args.cumulative_only:
        cumulative_bandits(1); cumulative_bandits(2)
    elif args.causal_raw_only:
        causal_raw_performance(1); causal_raw_performance(2)
    else:
        main_r1(); trajectories(1); trajectories(2); rung3_trajectory(); rung1_controls()
        rung3_deoe(); rung2_volatility(); rung3_access(); causal_raw_performance(1); causal_raw_performance(2)
