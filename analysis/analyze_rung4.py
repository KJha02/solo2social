"""Rung-4 learning, social provenance, and hidden artifact quality; use Slurm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_efficiency import read_population


def population(directory: Path) -> tuple[dict, list, list, list, list]:
    metadata, _ = read_population(directory)
    config = json.loads((directory / 'config.json').read_text())
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    learners = {x['agent_id']: x for x in checkpoint['learners']}
    horizon = int(checkpoint['next_round'])
    count = config['num_agents']
    artifacts = {p.stem: json.loads(p.read_text()) for p in (directory / 'artifacts').glob('*.json')}
    initial = [v for v, a in learners[0]['acquisitions'].items() if a['origin'] == 'initial']
    if len(initial) != 1:
        raise ValueError('Expected exactly one shared initial skill')
    initial = initial[0]

    def depth(version, seen=()):
        if version in seen or version not in artifacts:
            return np.nan
        parent = artifacts[version]['parent']
        return 0 if parent is None else 1 + depth(parent, (*seen, version))

    acquisitions = []
    for agent, learner in learners.items():
        for vid, acquisition in learner['acquisitions'].items():
            if acquisition['round'] < horizon:
                acquisitions.append(dict(**metadata, agent=agent, version=vid,
                    origin=acquisition['origin'], acquired_round=acquisition['round'],
                    source=acquisition.get('source'), parent=learner['versions'][vid]['parent'],
                    lineage_depth=depth(vid)))
    rounds, edges, selections = [], [], []
    prior_rewards = {}
    for r in range(horizon):
        values = json.loads((directory / 'rounds' / f'{r:04d}.json').read_text())
        results, events = values['results'], values['events']
        if len(results) != count or len({x['agent'] for x in results}) != count:
            raise ValueError('Incomplete or duplicate checkpointed population round')
        for x in results:
            if x['origin'] != learners[x['agent']]['acquisitions'][x['version']]['origin']:
                raise ValueError('Row provenance contradicts permanent first acquisition')
        call_sizes = {}
        for event in events:
            if 'call_id' in event:
                call_sizes[event['call_id']] = call_sizes.get(event['call_id'], 0) + 1
        observations = [e for e in events if e['tool'] == 'observe']
        development_results = []
        for event in events:
            if 'evaluation' in event:
                evaluation = event['evaluation']
                development_results.extend(evaluation.get('evaluations', [evaluation]))
        for event in observations:
            if 'target' not in event:
                continue
            fraction = 1 / call_sizes.get(event.get('call_id'), 1)
            eligible = [reward for agent, reward in prior_rewards.items() if agent != event['agent']]
            source_reward = prior_rewards.get(event['target'], np.nan)
            edges.append(dict(**metadata, round=r, agent=event['agent'], target=event['target'],
                acquired=('version' in event and not event.get('error', False)),
                version=event.get('version'),
                source_prior_reward=source_reward,
                source_advantage_over_available_peer_mean=source_reward - np.mean(eligible) if eligible else np.nan,
                source_tied_for_best_available=(source_reward == max(eligible)) if eligible else np.nan,
                decision_completion_tokens=event.get('decision_completion_tokens', np.nan) * fraction,
                decision_prompt_tokens=event.get('decision_prompt_tokens', np.nan) * fraction,
                decision_seconds=event.get('decision_inference_seconds', np.nan) * fraction))
        novel = [a for a in acquisitions if a['acquired_round'] == r and a['origin'] == 'independent']
        budget_keys = {'completion_tokens': 'completion', 'prompt_tokens': 'prompt',
                       'reasoning_tokens': 'reasoning_tokens', 'cost_usd': 'cost_usd',
                       'inference_seconds': 'inference_seconds', 'tool_seconds': 'tool_seconds'}
        row = dict(**metadata, round=r, reward=np.mean([x['reward'] for x in results]),
            submission_rate=np.mean([x['submitted'] for x in results]),
            reward_given_submission=np.mean([x['reward'] for x in results if x['submitted']])
                if any(x['submitted'] for x in results) else np.nan,
            parser_errors=sum(x['budget'].get('parser_errors', 0) for x in results),
            provider_generation_errors=sum(x['budget'].get('provider_generation_errors', 0) for x in results),
            unreported_completion_allowance=sum(x['budget'].get('unreported_completion_allowance', 0) for x in results),
            provider_cap_overruns=sum(x['budget'].get('provider_cap_overruns', 0) for x in results),
            phase_cap_violations=sum(x['budget'].get('phase_cap_violations', 0) for x in results),
            empty_completions=sum(x['budget'].get('empty_completions', 0) for x in results),
            calls_without_cost=sum(x['budget'].get('calls_without_cost', 0) for x in results),
            skill_write_attempts=sum(e['tool'] == 'write_skill' for e in events),
            skill_write_successes=sum(e['tool'] == 'write_skill' and not e.get('error', False) for e in events),
            social_first_selected_fraction=np.mean([x['origin'] == 'social' for x in results]),
            social_first_submitted_fraction=np.mean([x['origin'] == 'social' and x['submitted'] for x in results]),
            independent_acquisitions=len(novel), independent_unique_versions=len({a['version'] for a in novel}),
            observation_attempts=len(observations), observation_successes=sum('version' in e and not e.get('error', False) for e in observations),
            development_problem_evaluations=len(development_results),
            development_submissions=sum(x['submitted'] for x in development_results),
            development_candidates=sum('evaluation' in e for e in events),
            development_filtered_early=sum(e.get('evaluation', {}).get('filtered_early', False) for e in events),
            revision_attempts=sum(e['tool'] in {'revise', 'openevolve_revision'} for e in events),
            revision_successes=sum(e['tool'] == 'openevolve_revision' and e.get('changed', False) or
                                   e['tool'] == 'revise' and e.get('after') != e.get('before') for e in events),
            evolved_selected_fraction=np.mean([x['version'] != initial for x in results]),
            wall_seconds=sum(x['wall_seconds_round'] for x in results))
        row.update({name: sum(x['budget'].get(field, np.nan) for x in results)
                    for name, field in budget_keys.items()})
        row['completion_tokens_upper_bound'] = row['completion_tokens'] + row['unreported_completion_allowance']
        row['provider_usage_complete'] = row['provider_generation_errors'] == 0
        for phase in ('selection', 'revision', 'development', 'execution'):
            row[f'{phase}_completion_tokens'] = sum(x['budget']['phases'].get(phase, {}).get('completion_tokens', 0) for x in results)
            row[f'{phase}_provider_errors'] = sum(x['budget'].get('provider_errors_by_phase', {}).get(phase, 0) for x in results)
        rounds.append(row)
        prior_rewards = {x['agent']: x['reward'] for x in results if x.get('execution_attempted',
            x['budget'].get('phases', {}).get('execution', {}).get('calls', int(x['submitted']))) > 0}
        if r in config.get('holdout_rounds', [0, config['rounds'] - 1]):
            selections.extend(dict(x, round=r) for x in results)

    replay = {}
    reference_identity = None
    paired_seeds = {}
    replay_completion = replay_prompt = replay_wall = 0
    selection_path = directory / 'evaluation_selection.json'
    selected_tasks = (set(json.loads(selection_path.read_text())['task_ids'])
                      if selection_path.exists() else None)
    for path in (directory / 'holdout').glob('*.json'):
        value = json.loads(path.read_text())
        identity = value['identity']
        if selected_tasks is not None and identity['task'] not in selected_tasks:
            continue
        common = {k: identity[k] for k in ('model', 'budget', 'dataset')}
        common['source_sha256'] = identity.get('source_sha256')
        declared_model = {k: v for k, v in config['model'].items() if k != 'base_url'}
        if common['model'] != declared_model or common['budget'] != config['holdout_tokens']:
            raise ValueError('Hidden evaluator differs from the declared model or budget')
        if reference_identity is not None and common != reference_identity:
            raise ValueError('Cannot mix different hidden evaluators')
        reference_identity = common
        key = (identity['task'], identity['repeat'])
        if key in paired_seeds and paired_seeds[key] != identity.get('execution_seed'):
            raise ValueError('Hidden comparisons must share the task/repeat execution seed')
        paired_seeds[key] = identity.get('execution_seed')
        if key in replay.setdefault(identity['version'], {}):
            raise ValueError('Duplicate hidden version/task/repeat')
        replay[identity['version']][key] = float(value['result']['reward'])
        replay_completion += value['budget']['completion']
        replay_prompt += value['budget']['prompt']
        replay_wall += value['result']['wall_seconds']
    provenance = json.loads((directory / 'provenance.json').read_text())
    tasks = selected_tasks if selected_tasks is not None else provenance['holdout_ids']
    if not set(tasks) <= set(provenance['holdout_ids']):
        raise ValueError('Evaluation selection contains non-held-out tasks')
    expected = {(task, repeat) for task in tasks for repeat in range(config.get('holdout_repeats', 2))}
    def quality(version):
        scores = replay.get(version, {})
        return np.mean([scores[k] for k in sorted(expected)]) if expected and expected <= scores.keys() else np.nan

    comparisons = []
    for selection in selections:
        agent, r, vid = selection['agent'], selection['round'], selection['version']
        private = [a['version'] for a in acquisitions if a['agent'] == agent and
                   a['origin'] == 'independent' and a['acquired_round'] <= r]
        selected_quality, initial_quality = quality(vid), quality(initial)
        private_quality = [quality(v) for v in private]
        covered = bool(private) and np.isfinite(selected_quality) and all(np.isfinite(x) for x in private_quality)
        best = max(private_quality) if covered else np.nan
        comparisons.append(dict(**metadata, agent=agent, round=r, version=vid, origin=selection['origin'],
            selected_quality=selected_quality, initial_quality=initial_quality,
            selected_minus_initial=selected_quality - initial_quality,
            independent_candidate_count=len(private), comparator_covered=covered,
            best_independent_quality=best, selected_minus_best_independent=selected_quality - best,
            selected_better_than_best_independent=(selected_quality > best) if covered else np.nan))
    frame, edge_frame, comparison = pd.DataFrame(rounds), pd.DataFrame(edges), pd.DataFrame(comparisons)
    if frame.empty:
        raise ValueError('No checkpointed rounds')
    totals = edge_frame.groupby('target').decision_completion_tokens.sum(min_count=1) if len(edges) else pd.Series(dtype=float)
    counts = edge_frame.groupby('target').size() if len(edges) else pd.Series(dtype=float)
    independent_versions = {a['version'] for a in acquisitions if a['origin'] == 'independent'}
    summary = dict(**metadata, observed_rounds=horizon, reward=frame.reward.mean(),
        reward_given_submission=frame.reward.sum() / frame.submission_rate.sum()
            if frame.submission_rate.sum() else np.nan,
        parser_errors=int(frame.parser_errors.sum()),
        skill_write_attempts=int(frame.skill_write_attempts.sum()),
        skill_write_successes=int(frame.skill_write_successes.sum()),
        submission_rate=frame.submission_rate.mean(), social_first_selected_fraction=frame.social_first_selected_fraction.mean(),
        social_first_submitted_fraction=frame.social_first_submitted_fraction.mean(),
        independent_acquisitions=int(frame.independent_acquisitions.sum()), independent_unique_versions=len(independent_versions),
        social_acquisitions=sum(a['origin'] == 'social' for a in acquisitions),
        observation_attempts=int(frame.observation_attempts.sum()), observation_successes=int(frame.observation_successes.sum()),
        development_problem_evaluations=int(frame.development_problem_evaluations.sum()),
        development_submissions=int(frame.development_submissions.sum()),
        development_candidates=int(frame.development_candidates.sum()),
        development_filtered_early=int(frame.development_filtered_early.sum()),
        revision_attempts=int(frame.revision_attempts.sum()),
        revision_successes=int(frame.revision_successes.sum()),
        evolved_selected_fraction=frame.evolved_selected_fraction.mean(),
        observation_target_coverage=len(counts) / count,
        mean_personal_peer_coverage=(edge_frame.groupby('agent').target.nunique().sum() / (count * (count - 1))) if len(edges) and count > 1 else 0.,
        observation_top_target_share=counts.max() / counts.sum() if counts.sum() else np.nan,
        observation_target_hhi=((counts / counts.sum()) ** 2).sum() if counts.sum() else np.nan,
        observation_compute_top_target_share=totals.max() / totals.sum() if totals.sum() else np.nan,
        observation_completion_tokens=totals.sum(min_count=1) if len(edges) else 0.,
        observed_source_reward_advantage=edge_frame.source_advantage_over_available_peer_mean.mean() if len(edges) else np.nan,
        online_completion_tokens=frame.completion_tokens.sum(), online_prompt_tokens=frame.prompt_tokens.sum(),
        online_reasoning_tokens=frame.reasoning_tokens.sum(min_count=1),
        online_cost_usd=frame.cost_usd.sum(min_count=1),
        online_wall_seconds=frame.wall_seconds.sum(),
        replay_budget_per_evaluation=config.get('holdout_tokens'), replay_evaluation_count=sum(map(len, replay.values())),
        replay_completion_tokens=replay_completion, replay_prompt_tokens=replay_prompt, replay_wall_seconds=replay_wall)
    if len(comparison):
        final = comparison[comparison['round'] == comparison['round'].max()]
        social = final[final.origin == 'social']
        summary.update(final_selected_minus_initial=final.selected_minus_initial.mean(),
            final_initial_comparison_coverage=final.selected_minus_initial.notna().mean(),
            final_social_comparator_coverage=social.comparator_covered.mean() if len(social) else np.nan,
            final_social_minus_best_independent=social.selected_minus_best_independent.mean(),
            final_social_better_than_best_independent=social.selected_better_than_best_independent.mean())
    return summary, rounds, edges, comparisons, acquisitions


def analyze(roots, output):
    directories = sorted({p.parent.resolve() for root in roots for p in root.rglob('checkpoint.json')
        if (p.parent / 'rounds').is_dir() and (p.parent / 'provenance.json').exists()})
    tables = [[] for _ in range(5)]
    for directory in directories:
        result = population(directory)
        tables[0].append(result[0])
        for i in range(1, 5): tables[i].extend(result[i])
    if not tables[0]: raise ValueError('No rung4 populations found')
    output.mkdir(parents=True, exist_ok=True)
    names = ['population_summary', 'population_rounds', 'observation_edges', 'hidden_comparisons', 'acquisitions']
    frames = [pd.DataFrame(rows) for rows in tables]
    for name, frame in zip(names, frames): frame.to_csv(output / f'{name}.csv', index=False)
    seeds = frames[0]
    if seeds.duplicated(['condition', 'model', 'seed']).any():
        raise ValueError('Duplicate condition/model/seed populations; choose one authoritative output root')
    metrics = ['reward', 'submission_rate', 'social_first_submitted_fraction', 'independent_unique_versions',
               'mean_personal_peer_coverage', 'observation_top_target_share', 'observation_compute_top_target_share',
               'observed_source_reward_advantage',
               'final_selected_minus_initial', 'final_social_minus_best_independent', 'final_social_comparator_coverage']
    metrics = [m for m in metrics if m in seeds]
    aggregate = seeds.groupby(['model', 'condition', 'observed_rounds'])[metrics].agg(['mean', 'sem', 'count'])
    aggregate.to_csv(output / 'condition_summary.csv')

    # Interim causal snapshots compare each treated population with its exact
    # model/seed condition counterpart through their shared observed horizon.
    # This keeps partial checkpoint progress usable without pooling later rounds
    # from the population that happened to receive more cluster time.
    round_frame = frames[1]
    causal_rows = []
    snapshot_metrics = ['reward', 'submission_rate', 'social_first_submitted_fraction',
                        'independent_unique_versions', 'observation_attempts']
    indexed = {(row.model, row.seed, row.condition): row for _, row in seeds.iterrows()}
    for _, treated in seeds.iterrows():
        comparisons = []
        if treated.execution_reserve > 0 and '_r8192' in treated.condition:
            comparisons.append(('Execution reserve', treated.condition.replace('_r8192', '_r0')))
        if treated.p_independent > 0 and '_p15_' in treated.condition:
            comparisons.append(('Independent search', treated.condition.replace('_p15_', '_p0_')))
        for intervention, control_condition in comparisons:
            control = indexed.get((treated.model, treated.seed, control_condition))
            if control is None:
                continue
            horizon = min(int(treated.observed_rounds), int(control.observed_rounds))
            if horizon < 1:
                continue
            selected = []
            for population_row in (treated, control):
                selected.append(round_frame[(round_frame.run == population_row.run) &
                    (round_frame['round'] < horizon)])
            if any(x.empty for x in selected):
                continue
            values = []
            for data in selected:
                values.append(dict(reward=data.reward.mean(), submission_rate=data.submission_rate.mean(),
                    social_first_submitted_fraction=data.social_first_submitted_fraction.mean(),
                    independent_unique_versions=data.independent_unique_versions.sum(),
                    observation_attempts=data.observation_attempts.sum()))
            row = dict(model=treated.model, seed=treated.seed, intervention=intervention,
                treated_condition=treated.condition, control_condition=control_condition,
                common_horizon=horizon)
            row.update({f'{metric}_effect': values[0][metric] - values[1][metric]
                        for metric in snapshot_metrics})
            causal_rows.append(row)
    causal = pd.DataFrame(causal_rows)
    causal.to_csv(output / 'causal_snapshot_by_seed.csv', index=False)
    if not causal.empty:
        group = ['model', 'intervention', 'treated_condition', 'control_condition']
        aggregations = dict(seeds=('seed', 'nunique'), min_common_horizon=('common_horizon', 'min'),
                            max_common_horizon=('common_horizon', 'max'))
        for metric in snapshot_metrics:
            aggregations[f'{metric}_effect'] = (f'{metric}_effect', 'mean')
            aggregations[f'{metric}_effect_se'] = (f'{metric}_effect', 'sem')
        causal_summary = causal.groupby(group).agg(**aggregations).reset_index()
        causal_summary.to_csv(output / 'causal_snapshot_summary.csv', index=False)
        shown = ['reward_effect', 'submission_rate_effect', 'independent_unique_versions_effect']
        models = sorted(causal.model.unique())
        fig, axes = plt.subplots(len(models), len(shown), figsize=(15, 4 * len(models)), squeeze=False)
        for row_index, model in enumerate(models):
            data = causal[causal.model == model]
            labels = sorted(data.treated_condition.unique())
            for col, metric in enumerate(shown):
                means = [data[data.treated_condition == label][metric].mean() for label in labels]
                errors = [data[data.treated_condition == label][metric].sem() for label in labels]
                colors = ['#009E73' if '_p15_' in label else '#E69F00' for label in labels]
                axes[row_index, col].bar(range(len(labels)), means, yerr=np.nan_to_num(errors),
                                         color=colors, capsize=3)
                axes[row_index, col].axhline(0, color='black', linewidth=.7)
                axes[row_index, col].set_xticks(range(len(labels)),
                    [x.removeprefix('r4_full_') for x in labels], rotation=55, ha='right', fontsize=7)
                axes[row_index, col].set_title(metric.removesuffix('_effect').replace('_', ' ').capitalize())
                if col == 0:
                    axes[row_index, col].set_ylabel(model)
        fig.tight_layout()
        fig.savefig(output / 'causal_snapshot.png', dpi=180)
        plt.close(fig)

    rsi_rows = []
    for _, evolved in seeds[seeds.condition.isin(
            ['r4_full_openevolve_r0', 'r4_full_openevolve_r8192'])].iterrows():
        frozen = indexed.get((evolved.model, evolved.seed, 'r4_full_frozen_r0'))
        if frozen is None:
            continue
        horizon = min(int(evolved.observed_rounds), int(frozen.observed_rounds))
        if horizon < 1:
            continue
        pair = [round_frame[(round_frame.run == population_row.run) & (round_frame['round'] < horizon)]
                for population_row in (evolved, frozen)]
        rsi_rows.append(dict(model=evolved.model, seed=evolved.seed, condition=evolved.condition,
            common_horizon=horizon, reward_effect=pair[0].reward.mean()-pair[1].reward.mean(),
            submission_rate_effect=pair[0].submission_rate.mean()-pair[1].submission_rate.mean(),
            completion_tokens_effect=pair[0].completion_tokens.sum()-pair[1].completion_tokens.sum()))
    rsi = pd.DataFrame(rsi_rows)
    rsi.to_csv(output / 'rsi_vs_frozen_snapshot_by_seed.csv', index=False)
    if not rsi.empty:
        summary = rsi.groupby(['model', 'condition']).agg(seeds=('seed', 'nunique'),
            min_common_horizon=('common_horizon', 'min'), max_common_horizon=('common_horizon', 'max'),
            reward_effect=('reward_effect', 'mean'), reward_effect_se=('reward_effect', 'sem'),
            submission_rate_effect=('submission_rate_effect', 'mean'),
            submission_rate_effect_se=('submission_rate_effect', 'sem'),
            completion_tokens_effect=('completion_tokens_effect', 'mean'),
            completion_tokens_effect_se=('completion_tokens_effect', 'sem')).reset_index()
        summary.to_csv(output / 'rsi_vs_frozen_snapshot_summary.csv', index=False)
        models = sorted(rsi.model.unique())
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        width = .36
        for offset, condition in enumerate(['r4_full_openevolve_r0', 'r4_full_openevolve_r8192']):
            data = summary.set_index(['model', 'condition'])
            values, errors = [], []
            for model in models:
                row = data.loc[(model, condition)] if (model, condition) in data.index else None
                values.append(row.reward_effect if row is not None else np.nan)
                errors.append(row.reward_effect_se if row is not None else np.nan)
            axes[0].bar(np.arange(len(models)) + (offset - .5) * width, values, width,
                        yerr=np.nan_to_num(errors), capsize=3, label='No reserve' if offset == 0 else '8K reserve')
            values, errors = [], []
            for model in models:
                row = data.loc[(model, condition)] if (model, condition) in data.index else None
                values.append(row.submission_rate_effect if row is not None else np.nan)
                errors.append(row.submission_rate_effect_se if row is not None else np.nan)
            axes[1].bar(np.arange(len(models)) + (offset - .5) * width, values, width,
                        yerr=np.nan_to_num(errors), capsize=3)
            values, errors = [], []
            for model in models:
                row = data.loc[(model, condition)] if (model, condition) in data.index else None
                values.append(row.completion_tokens_effect if row is not None else np.nan)
                errors.append(row.completion_tokens_effect_se if row is not None else np.nan)
            axes[2].bar(np.arange(len(models)) + (offset - .5) * width, values, width,
                        yerr=np.nan_to_num(errors), capsize=3)
        for ax, title in zip(axes, ['Reward effect', 'Submission-rate effect', 'Extra completion tokens']):
            ax.axhline(0, color='black', linewidth=.7)
            ax.set_xticks(range(len(models)), models, rotation=20, ha='right')
            ax.set_title(title)
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / 'rsi_vs_frozen_snapshot.png', dpi=180)
        plt.close(fig)

    frontier_rows = []
    frontier = seeds[seeds.policy.isin(['frozen', 'openevolve'])]
    frozen_index = {(row.model, row.seed, row.initial_skill): row for _, row in
                    frontier[frontier.policy == 'frozen'].iterrows()}
    for _, evolved in frontier[frontier.policy == 'openevolve'].iterrows():
        frozen = frozen_index.get((evolved.model, evolved.seed, evolved.initial_skill))
        if frozen is None:
            continue
        horizon = min(int(evolved.observed_rounds), int(frozen.observed_rounds))
        pair = [round_frame[(round_frame.run == row.run) & (round_frame['round'] < horizon)]
                for row in (evolved, frozen)]
        frontier_rows.append(dict(model=evolved.model, seed=evolved.seed,
            initial_skill=evolved.initial_skill, common_horizon=horizon,
            online_reward_effect=pair[0].reward.mean() - pair[1].reward.mean(),
            submission_rate_effect=pair[0].submission_rate.mean() - pair[1].submission_rate.mean(),
            conditional_reward_effect=evolved.reward_given_submission - frozen.reward_given_submission,
            selected_minus_initial=evolved.get('final_selected_minus_initial', np.nan),
            evolved_selected_fraction=evolved.evolved_selected_fraction,
            revision_success_rate=evolved.revision_successes / evolved.revision_attempts
                if evolved.revision_attempts else np.nan,
            cascade_filter_rate=evolved.development_filtered_early / evolved.development_candidates
                if evolved.development_candidates else np.nan,
            evolved_online_cost_usd=evolved.online_cost_usd,
            frozen_online_cost_usd=frozen.online_cost_usd))
    pd.DataFrame(frontier_rows).to_csv(output / 'frontier_rsi_vs_frozen.csv', index=False)
    for name, chosen in [('learning_and_discovery', metrics[:4]), ('social_selection', metrics[4:8]),
                         ('hidden_artifact_quality', metrics[8:])]:
        if not chosen: continue
        fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 5), squeeze=False)
        labels = [f'{model}\n{condition}' for model, condition, _ in aggregate.index]
        for ax, metric in zip(axes[0], chosen):
            mean, error = aggregate[(metric, 'mean')], aggregate[(metric, 'sem')]
            ax.bar(range(len(mean)), mean, yerr=error.fillna(0), capsize=3)
            ax.set_xticks(range(len(mean)), labels, rotation=35, ha='right', fontsize=7)
            ax.set_title(metric.replace('_', ' ').capitalize())
            ax.axhline(0, color='black', linewidth=.5)
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=160)
        plt.close(fig)
    (output / 'PLOT_GUIDE.md').write_text('''# Rung 4: learning and social selection

Each observation in population_summary.csv is one independent seed population. Condition summaries average these populations and give seed-level standard errors; an absent error bar with one seed does not establish precision. Agent and checkpoint rows are evidence, not independent experimental replicates. Different observation horizons are not pooled.

**Learning and discovery:** Compare online reward and submission rates with how often submitted solutions used a social-first skill and how many distinct skill contents were independently created. Social-first means permanent first acquisition through observation; observing an already-known version does not change its origin. Submitted social-first use is divided by ALL agent-rounds, so failed submissions remain in the denominator. New independent content hashes count once per population in independent_unique_versions; independent_acquisitions separately counts each agent's first independent acquisition. Initial shared skills are neither new discoveries nor social copies. Descendants of socially acquired parents can be independent revisions; acquisition rows preserve parents and lineage depth for distinguishing these mechanisms.

**Social selection:** The first panel is mean personal peer coverage: the number of distinct peers queried by each agent divided by its available peers, averaged across all agents including non-observers. The separate population target coverage in the CSV is the fraction of population members queried at least once by anybody, NOT the fraction of peers each agent observed. Target shares and HHI describe concentration of valid-target observation attempts, including attempts before a peer has a prior result. observation_edges.csv retains observer identity, success and target for personal coverage. Per-target compute is the completion cost of the decision issuing observe, excluding later reasoning or reading. If one model response issues several tools, its decision cost is shared equally among those tools rather than counted repeatedly. Nested revision and development costs remain in phase budgets. Observed-source reward advantage compares the queried peer with the mean previous-round reward of other available peers, even in action-only conditions where that reward was hidden from the agent. These are private-task rewards, not clean estimates of skill quality. These are descriptive associations, not evidence that concentration caused discovery or reward changes.

**Hidden artifact quality:** At declared checkpoints, compare each selected artifact with the common initial artifact on identical hidden tasks, repeats, model and execution budget. The population averages agent-level paired differences; no held-out feedback enters learning. The social-versus-private comparison requires hidden evaluation of EVERY independently first-acquired candidate available to that recipient by the checkpoint, excluding the shared initial artifact. Missing evaluation of even one candidate makes the comparator unavailable. Coverage is reported, and a population with no social selection has undefined social comparator coverage. Selected-only replay often leaves this comparator incomplete; the analysis does not silently call the best evaluated subset the best independent skill. The maximum of noisy held-out candidate means is an optimistic retrospective comparator, not an independently validated oracle or a causal counterfactual.

Online inference and wall time, plus separately itemized hidden-replay cost, are in population_summary.csv. Hidden evaluation cost is research overhead, not learning budget. Pair this analysis with analyze_efficiency.py for token curves and censored target performance. No universal social advantage follows from a one-seed canary or a positive conditional comparison.

**Interim matched snapshots:** `causal_snapshot_by_seed.csv` compares reserve with no reserve and 15% with 0% independent search within the exact model, seed, policy and information condition. `rsi_vs_frozen_snapshot_by_seed.csv` compares each solo OpenEvolve population with its model-seed frozen reference. Every pair is truncated to its shared checkpoint horizon; the min/max horizons remain in summary tables. These estimates do not pool unmatched late rounds, but horizons can still differ across seeds and conditions. They are online diagnostics, not substitutes for complete 15-round hidden comparisons. Independent unique versions measure distinct new content, not quality.
''')
    with (output / 'PLOT_GUIDE.md').open('a') as handle:
        handle.write('\n## Values in this analysis\n\n')
        if any('fixture' in str(x).lower() for x in seeds.model):
            handle.write('These inputs include scripted fixtures: their rewards validate accounting, not model capability.\n\n')
        for (model, condition, horizon), group in seeds.groupby(['model', 'condition', 'observed_rounds']):
            handle.write(f'- {model}, {condition}: {len(group)} populations at {horizon} rounds; '
                f'mean reward {group.reward.mean():.3f}, submission {group.submission_rate.mean():.1%}, '
                f'social-first submitted use {group.social_first_submitted_fraction.mean():.1%}, '
                f'independent unique versions {group.independent_unique_versions.mean():.2f}.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.roots, args.output)
