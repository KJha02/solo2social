"""Population-level efficiency curves and censored targets; run through Slurm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CAUSAL_METRICS = [
    'mean_reward', 'completion_rate', 'independent_search_assignment_rate',
    'independent_search_completion_rate', 'innovations_per_round',
    'unique_selected_options', 'effective_selected_diversity',
]


def selection_diversity(values: list[dict]) -> tuple[int, float]:
    selected = [v.get('arm_id', v.get('skill_id')) for v in values
                if v.get('pulled', v.get('valid_solution', v.get('submitted', False)))
                and v.get('arm_id', v.get('skill_id')) is not None]
    if not selected:
        return 0, 0.0
    counts = pd.Series(selected).value_counts(normalize=True)
    return len(counts), float(1 / np.square(counts).sum())


def read_population(directory: Path) -> tuple[dict, list[dict]]:
    config = json.loads((directory / 'config.json').read_text())
    rung = int(config.get('rung', 4 if (directory / 'rounds').is_dir() else 1))
    count = int(config.get('num_agents', config.get('environment', {}).get('num_agents', 0)))
    rows = []
    if rung == 4:
        for path in sorted((directory / 'rounds').glob('*.json')):
            payload = json.loads(path.read_text())
            values = payload['results']
            if len(values) != count or len({v['agent'] for v in values}) != count:
                raise ValueError(f'incomplete population round: {path}')
            unique, diversity = selection_diversity(values)
            events = payload.get('events', [])
            rows.append(dict(round=values[0]['round'], reward=np.mean([v['reward'] for v in values]),
                completion_tokens=sum(v['budget']['completion'] for v in values),
                prompt_tokens=sum(v['budget']['prompt'] for v in values),
                seconds=sum(v['wall_seconds_round'] for v in values),
                timing_kind='sequential_agent_round_wall',
                completion_rate=np.mean([v['submitted'] for v in values]),
                independent_search_assignment_rate=np.mean([v.get('independent_search_assigned', False) for v in values]),
                independent_search_completion_rate=np.mean([v.get('independent_search_completed', False) for v in values]),
                innovations_per_round=sum(e.get('tool') == 'write_skill' and not e.get('error', False) for e in events),
                unique_selected_options=unique, effective_selected_diversity=diversity))
    else:
        events: dict[int, list] = {}
        with (directory / 'events.jsonl').open() as handle:
            for line in handle:
                event = json.loads(line)
                events.setdefault(int(event['round']), []).append(event)
        for round_index, values in sorted(events.items()):
            ends = [v for v in values if v['event'] == 'round_end']
            if not ends:
                continue  # interrupted, uncheckpointed round
            if len(ends) != count or len({v['agent_id'] for v in ends}) != count:
                raise ValueError(f'incomplete or duplicate population round: {directory}/{round_index}')
            calls = [v for v in values if v['event'] in {'decision', 'solution'} and 'completion_tokens' in v]
            batches = {}
            for v in calls:
                if 'batch_generation_seconds' in v:
                    key = (v['event'], v.get('stage', ''), v.get('wave', 0))
                    seconds = float(v['batch_generation_seconds'])
                    if key in batches and not np.isclose(seconds, batches[key]):
                        raise ValueError(f'inconsistent duplicated batch time: {directory}/{round_index}')
                    batches[key] = seconds
            timed = all('batch_generation_seconds' in v for v in calls if v['completion_tokens'] > 0)
            unique, diversity = selection_diversity(ends)
            decisions = [v for v in values if v['event'] == 'decision']
            rows.append(dict(round=round_index, reward=np.mean([v['reward'] for v in ends]),
                completion_tokens=sum(v.get('tokens_spent', 0) for v in ends),
                prompt_tokens=(sum(v['prompt_tokens'] for v in calls)
                    if all(v.get('prompt_tokens') is not None for v in calls) else np.nan),
                seconds=sum(batches.values()) if timed and batches else np.nan,
                timing_kind='batched_generation_wall',
                completion_rate=np.mean([v.get('valid_solution', v.get('pulled', False)) for v in ends]),
                independent_search_assignment_rate=np.mean([v.get('independent_search_assigned', False) for v in ends]),
                independent_search_completion_rate=np.mean([v.get('independent_search_completed', False) for v in ends]),
                innovations_per_round=sum(v.get('allocation') == 'innovate' and v.get('valid', False) for v in decisions),
                unique_selected_options=unique, effective_selected_diversity=diversity))
    social = config.get('policy', '').startswith('social') or config.get('social_info', 'none') != 'none'
    # Match scientific setup, not condition labels, social prompts, output directories or API URLs.
    environment = config.get('environment', {})
    match = dict(rung=rung, environment=environment, budget=config.get('budget', config.get('tokens_per_round')),
        guidance=config.get('guidance', 'neutral'), acquisition=config.get('independent_skill_acquisition', False),
        interventions=config.get('interventions', {k: config.get(k, 0) for k in ('independent_search_probability', 'execution_reserve_tokens')}),
        policy_family=config.get('policy', 'endogenous').replace('solo_', '').replace('social_', ''),
        dataset=config.get('dataset'), num_agents=count,
        rounds=config.get('rounds'), model_config={k: v for k, v in config.get('model', {}).items()
            if k not in {'base_url', 'api_key', 'endpoint'}})
    # Empty and explicitly zero controls are equivalent for matching.
    match['interventions'] = {k: v for k, v in match['interventions'].items() if v}
    if rung == 4:
        provenance = json.loads((directory / 'provenance.json').read_text()) if (directory / 'provenance.json').exists() else {}
        match['dataset_provenance'] = {k: provenance.get(k) for k in ('dataset', 'development_ids', 'online_ids', 'holdout_ids')}
        match['source_sha256'] = provenance.get('source_sha256')
        match['development_tokens'] = config.get('development_tokens', config.get('tokens_per_round', 0) // 4)
        match['development_cohort_size'] = config.get('development_cohort_size', min(4, len(provenance.get('development_ids', []))))
        match['development_cascade'] = bool(config.get('development_cascade', False))
        match['initial_skill'] = config.get('initial_skill', 'strong')
        match['execution_tokens'] = config.get('execution_tokens', config.get('tokens_per_round'))
    model = config.get('model_label', config.get('model', {}).get('name', config.get('model', {}).get('model', 'unknown')))
    match['model'] = model
    key = hashlib.sha256(json.dumps(match, sort_keys=True).encode()).hexdigest()[:16]
    base_key = hashlib.sha256(json.dumps({k: v for k, v in match.items() if k != 'interventions'}, sort_keys=True).encode()).hexdigest()[:16]
    info = config.get('social_info', 'payoff' if config.get('observe_payoff') else 'action') if social else 'none'
    metadata = dict(run=str(directory), rung=rung, condition=config.get('condition', f"{config.get('policy', 'endogenous')}_{info}_{key[:6]}"),
        seed=int(config.get('seed', directory.name.removeprefix('seed_'))), model=model,
        policy=config.get('policy', 'endogenous'), initial_skill=config.get('initial_skill', 'strong'),
        social=social, social_info=info, match_key=key, base_match_key=base_key, num_agents=count,
        p_independent=float(match['interventions'].get('independent_search_probability', 0)),
        execution_reserve=int(match['interventions'].get('execution_reserve_tokens', 0)))
    return metadata, rows


def summarize(curves: pd.DataFrame, target: float, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    output, targets = [], []
    for _, frame in curves.groupby('run', sort=False):
        frame = frame.sort_values('round').copy()
        if frame['round'].tolist() != list(range(len(frame))):
            raise ValueError(f"missing or repeated rounds: {frame.iloc[0]['run']}")
        frame['rolling_reward'] = frame.reward.rolling(window, min_periods=window).mean()
        frame['cumulative_reward'] = frame.reward.cumsum()
        for cost in ('completion_tokens', 'prompt_tokens', 'seconds'):
            frame[f'cumulative_{cost}'] = frame[cost].cumsum(skipna=False)
        frame['cumulative_total_tokens'] = frame.cumulative_completion_tokens + frame.cumulative_prompt_tokens
        reached = frame[frame.rolling_reward >= target]
        stop = reached.iloc[0] if len(reached) else frame.iloc[-1]
        row = {k: stop[k] for k in ('run', 'rung', 'condition', 'seed', 'model', 'social', 'social_info', 'match_key', 'timing_kind')}
        row.update({k: stop[k] for k in ('base_match_key', 'p_independent', 'execution_reserve') if k in stop})
        row.update(target=target, window=window, reached=bool(len(reached)), censored=not bool(len(reached)),
            stop_round=int(stop['round']) + 1, observed_rounds=len(frame), mean_reward=frame.reward.mean())
        for metric in CAUSAL_METRICS[1:]:
            row[metric] = frame[metric].mean()
        for cost in ('completion_tokens', 'prompt_tokens', 'total_tokens', 'seconds'):
            row[f'observed_{cost}'] = stop[f'cumulative_{cost}']
            row[f'to_target_{cost}'] = stop[f'cumulative_{cost}'] if len(reached) else np.nan
        targets.append(row)
        output.append(frame)
    return pd.concat(output, ignore_index=True), pd.DataFrame(targets)


def causal_contrasts(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matched factorial main/joint effects; every endpoint retains nonreachers."""
    effects, differences = [], []
    for (base, seed, horizon), group in targets.groupby(['base_match_key', 'seed', 'observed_rounds']):
        indexed = {}
        for _, row in group.iterrows():
            key = (row.p_independent, row.execution_reserve, bool(row.social), row.social_info)
            if key in indexed:
                raise ValueError(f'ambiguous causal comparator: {base}, seed {seed}, {key}')
            indexed[key] = row
        for (probability, reserve, social, info), treated in indexed.items():
            if probability == 0 and reserve == 0:
                continue
            control = indexed.get((0, 0, social, info))
            if control is None:
                continue
            common = dict(base_match_key=base, seed=seed, observed_rounds=horizon,
                rung=treated.rung, model=treated.model, p_independent=probability,
                execution_reserve=reserve, social_info=info)
            effects.append(dict(**common, social=social, treated_run=treated.run, control_run=control.run,
                **{f'{metric}_effect': treated[metric]-control[metric] for metric in CAUSAL_METRICS},
                reward_effect=treated.mean_reward-control.mean_reward,
                reach_effect=int(treated.reached)-int(control.reached),
                treated_unreached=int(not treated.reached), control_unreached=int(not control.reached)))
            if not social:
                continue
            solo = indexed.get((probability, reserve, False, 'none'))
            solo_control = indexed.get((0, 0, False, 'none'))
            if solo is None or solo_control is None:
                continue
            differences.append(dict(**common, social_treated_run=treated.run, social_control_run=control.run,
                solo_treated_run=solo.run, solo_control_run=solo_control.run,
                **{f'{metric}_did': ((treated[metric]-solo[metric])-
                    (control[metric]-solo_control[metric])) for metric in CAUSAL_METRICS},
                reward_did=((treated.mean_reward-solo.mean_reward)-
                    (control.mean_reward-solo_control.mean_reward)),
                reach_did=(int(treated.reached)-int(solo.reached))-(int(control.reached)-int(solo_control.reached)),
                social_treated_unreached=int(not treated.reached), social_control_unreached=int(not control.reached),
                solo_treated_unreached=int(not solo.reached), solo_control_unreached=int(not solo_control.reached)))
    return pd.DataFrame(differences), pd.DataFrame(effects)


def write_causal_contrasts(targets: pd.DataFrame, output: Path) -> None:
    differences, effects = causal_contrasts(targets)
    keys = ['base_match_key', 'rung', 'model', 'p_independent', 'execution_reserve', 'social_info', 'observed_rounds']
    for name, frame, metrics, grouping in (
            ('causal_did', differences, ('reward_did',) + tuple(f'{m}_did' for m in CAUSAL_METRICS[1:]) + ('reach_did',), keys),
            ('causal_separate_effects', effects, ('reward_effect',) + tuple(f'{m}_effect' for m in CAUSAL_METRICS[1:]) + ('reach_effect',), keys + ['social'])):
        if frame.empty:
            frame = pd.DataFrame(columns=grouping + ['seed'] + list(metrics))
        frame.to_csv(output / f'{name}_seeds.csv', index=False)
        aggregates = dict(seeds=('seed', 'nunique'))
        for metric in metrics:
            aggregates[metric] = (metric, 'mean')
            aggregates[f'{metric}_se'] = (metric, 'sem')
        for column in frame:
            if column.endswith('_unreached'):
                aggregates[column] = (column, 'sum')
        frame.groupby(grouping).agg(**aggregates).to_csv(output / f'{name}_summary.csv')

    if not effects.empty:
        shown = ['mean_reward_effect', 'completion_rate_effect',
                 'independent_search_completion_rate_effect',
                 'innovations_per_round_effect', 'effective_selected_diversity_effect']
        models = sorted(effects.model.unique())
        fig, axes = plt.subplots(len(models), len(shown), figsize=(18, 3.8 * len(models)), squeeze=False)
        for row, model in enumerate(models):
            data = effects[effects.model == model].copy()
            data['intervention'] = np.where(data.execution_reserve > 0, 'Reserve', 'Independent search')
            data['access'] = np.where(data.social, 'Social', 'Solo')
            groups = [(i, a) for i in ('Reserve', 'Independent search') for a in ('Solo', 'Social')]
            for col, metric in enumerate(shown):
                means = [data[(data.intervention == i) & (data.access == a)][metric].mean() for i, a in groups]
                errors = [data[(data.intervention == i) & (data.access == a)][metric].sem() for i, a in groups]
                axes[row, col].bar(range(4), means, yerr=np.nan_to_num(errors), capsize=3,
                                   color=['#E69F00', '#009E73'] * 2)
                axes[row, col].axhline(0, color='black', linewidth=.7)
                axes[row, col].set_xticks(range(4), ['Reserve\nsolo', 'Reserve\nsocial', 'Search\nsolo', 'Search\nsocial'], rotation=25)
                axes[row, col].set_title(metric.removesuffix('_effect').replace('_', ' ').capitalize())
                if col == 0:
                    axes[row, col].set_ylabel(model)
        fig.tight_layout()
        fig.savefig(output / 'causal_intervention_effects.png', dpi=180)
        plt.close(fig)


def analyze(roots: list[Path], output: Path, target: float, window: int) -> None:
    rows = []
    directories = sorted({p.parent.resolve() for root in roots for p in root.rglob('config.json')
        if (p.parent / 'events.jsonl').exists() or (p.parent / 'rounds').is_dir()})
    for directory in directories:
        metadata, rounds = read_population(directory)
        rows.extend(dict(**metadata, **r) for r in rounds)
    if not rows:
        raise ValueError('no completed population rounds found')
    curves, targets = summarize(pd.DataFrame(rows), target, window)
    output.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output / 'population_curves.csv', index=False)
    targets.to_csv(output / 'target_outcomes.csv', index=False)
    write_causal_contrasts(targets, output)
    targets.groupby(['rung', 'model', 'condition', 'observed_rounds']).agg(
        seeds=('seed', 'nunique'), reached=('reached', 'sum'), censored=('censored', 'sum'),
        reach_probability=('reached', 'mean'), reach_se=('reached', 'sem')).to_csv(output / 'target_summary.csv')
    pairs = []
    for (key, seed), group in targets.groupby(['match_key', 'seed']):
        solos = group[~group.social]
        if len(solos) > 1:
            raise ValueError(f'ambiguous solo comparator for {key}, seed {seed}; select one run root')
        if len(solos) != 1:
            continue
        solo = solos.iloc[0]
        for _, social in group[group.social].iterrows():
            # Equal exposure is essential; partial runs do not create a matched endpoint.
            if social.observed_rounds != solo.observed_rounds:
                continue
            pairs.append(dict(match_key=key, seed=seed, rung=social.rung, model=social.model,
                social_condition=social.condition, solo_condition=solo.condition,
                reward_difference=social.mean_reward - solo.mean_reward,
                reach_difference=int(social.reached) - int(solo.reached),
                social_censored=social.censored, solo_censored=solo.censored,
                # No complete-case target-time contrast: would select successful seeds.
                observed_rounds=solo.observed_rounds))
    pd.DataFrame(pairs).to_csv(output / 'matched_seed_contrasts.csv', index=False)
    if pairs:
        p = pd.DataFrame(pairs)
        summary = p.groupby(['match_key', 'rung', 'model', 'social_condition', 'observed_rounds']).agg(
            seeds=('seed', 'nunique'), reward_difference=('reward_difference', 'mean'),
            reward_se=('reward_difference', 'sem'), reach_difference=('reach_difference', 'mean'),
            reach_se=('reach_difference', 'sem'))
        summary.to_csv(output / 'matched_summary.csv')
    else:
        pd.DataFrame(columns=['match_key', 'rung', 'model', 'social_condition', 'seeds',
                              'reward_difference', 'reward_se', 'reach_difference', 'reach_se']).to_csv(output / 'matched_summary.csv', index=False)
    for rung, data in curves.groupby('rung'):
        models = sorted(data.model.unique())
        fig, axes = plt.subplots(len(models), 4, figsize=(20, 4.2 * len(models)), squeeze=False)
        for row, model in enumerate(models):
            for color_index, (condition, values) in enumerate(data[data.model == model].groupby('condition')):
                # Plot each independent population; no agent-level uncertainty or interpolation beyond observations.
                for _, population in values.groupby('run'):
                    for col, (cost, label) in enumerate((('completion_tokens', 'Charged completion tokens'),
                            ('prompt_tokens', 'Prompt tokens'),
                            ('total_tokens', 'Prompt + completion tokens'), ('seconds', 'Measured wall seconds'))):
                        x = population[f'cumulative_{cost}']
                        if x.notna().all():
                            axes[row, col].plot(x, population.rolling_reward, alpha=.45, color=f'C{color_index % 10}', label=condition)
                for ax in axes[row]:
                    ax.set_title(f'{model} | Rung {rung}')
            for col, label in enumerate(('Charged completion tokens', 'Prompt tokens', 'Prompt + completion tokens', 'Measured wall seconds')):
                ax = axes[row, col]
                ax.axhline(target, color='black', linestyle=':', linewidth=1)
                ax.set(xlabel=f'Cumulative population {label.lower()}', ylabel=f'Mean reward over last {window} rounds')
                handles, labels = ax.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                if unique:
                    ax.legend(unique.values(), unique.keys(), fontsize=6, loc='upper left')
                else:
                    ax.text(.5, .5, 'No measured timing available', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        fig.savefig(output / f'rung{rung}_efficiency.png', dpi=160)
        plt.close(fig)
    (output / 'PLOT_GUIDE.md').write_text(f'''# Efficiency analysis

Each line is one independent population seed. Panels show the mean reward over the last {window} complete rounds against cumulative population completion tokens, prompt tokens, all inference tokens, and measured wall time. The dotted target is {target}; it is fixed by the analysis command, not estimated separately for each condition. Missing or invalid pulls earn the original logged zero reward. Prompt tokens include repeated context and are not equivalent in cost to completion tokens; their sum is an accounting measure, not a hardware FLOP estimate.

`population_curves.csv` retains per-seed evidence. `target_outcomes.csv` includes EVERY seed: `censored=true` means the target was not reached during observation. Its `observed_*` columns are censoring exposures, NOT time-to-target estimates. `to_target_*` is missing for nonreachers. Never average only the reached seeds. Paired summaries report reward and target-reaching probability differences, with standard errors across seeds; they do not manufacture target-time estimates from censored cases. Partial runs are visible via observed_rounds and only matched at equal horizons.

For rungs 1–3, timing is measured batched generation time, deduplicated by round, event, stage and wave. This excludes model loading and other environment overhead and is not per-agent latency. Old runs without measured timing remain missing. Rung 4 uses the sum of sequential agent-round wall times, including tools and learning calls; it excludes model-server startup and artifact serialization outside the timed round. These definitions differ: compare time only within the same rung, hardware, serving setup and population size. Hardware/batch-size changes remain timing confounds. No historical timing is inferred from token counts.

Matched contrasts use the same seed, model, environment, budget, intervention and policy family. Social information level may vary against the same solo reference, so these contrasts are correlated; do not pool them as independent replications. Repeated condition copies are rejected when they make solo matching ambiguous. Target thresholds must be appropriate to each task's reward units; run separate analyses for different rungs when necessary.

`causal_did_seeds.csv` compares each intervention's social-minus-solo reward gap against the untreated social-minus-solo gap within the same seed. Positive reward DID means the intervention improves the social advantage (or reduces its deficit). `causal_separate_effects_seeds.csv` shows intervention-minus-untreated effects separately for solo and each social information condition. Explicit p_independent and execution_reserve identify search-only, reserve-only and joint treatments. The base match key removes intervention settings but retains model, environment including shocks, budget, guidance, acquisition and policy family. All four DID endpoints must have equal observed horizons; summaries keep different horizons separate and compute uncertainty across population seeds. Joint-treatment DID is not a factorial interaction estimate. Unreached counts remain explicit for every endpoint; reach DID includes all matched seeds and no target-time contrast drops nonreachers. Missing or unequal-horizon counterparts are excluded from matched tables but remain in target_outcomes.csv. An execution-reserve effect can include intervention-induced fallback selection where the rung requires it; it is not automatically a pure execution mechanism.

`causal_intervention_effects.png` shows seed-level intervention-minus-control effects separately for solo and social populations. Independent-search assignment and completion rates are population fractions. Innovations are valid INNOVATE decisions in rung 2 and successful skill writes in rung 4; they are zero by construction in rungs 1 and 3. Unique selected options and effective selected diversity use completed payoff-earning choices within each round. The latter is the inverse Simpson concentration, so it differs from cumulative discoveries, acquired repertoire size, and lineage diversity. Forced search identifies the complete intervention—including opportunity and execution costs—not the effect of diversity while holding everything else fixed.
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=float, required=True)
    parser.add_argument('--window', type=int, default=10)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('window must be positive')
    analyze(args.roots, args.output, args.target, args.window)
