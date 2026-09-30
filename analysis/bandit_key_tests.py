"""The ten finite-bandit contrasts, with one Bonferroni family."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/results'


def main():
    rows = []

    def add(model, contrast, metric, values):
        values = np.asarray(values, dtype=float)
        if len(values) != 8 or not np.isfinite(values).all():
            raise ValueError('Expected eight finite paired seed effects')
        rows.append(dict(model=model, contrast=contrast, metric=metric,
                         mean=values.mean(), se=values.std(ddof=1) / np.sqrt(8),
                         nominal_p=float(ttest_1samp(values, 0).pvalue), n=8))

    algorithms = pd.read_csv(OUT / 'rung1_matched_ucb_curves.csv')
    wide = algorithms[algorithms['round'] == 99].pivot(
        index='seed', columns='strategy', values='cumulative_reward')
    add('algorithmic', 'hierarchical minus solo UCB', 'cumulative_reward',
        wide['Hierarchical UCB'] - wide['Solo UCB'])
    frame = pd.read_csv(OUT / 'rung1_primary_diagnostics_by_seed.csv')
    percentage_rows = []
    for model, group in frame.groupby('model'):
        wide = group.pivot(index='seed', columns='strategy', values='reward')
        a, b = wide.solo_llm, wide.social_action_payoff
        add(model, 'social minus solo', 'cumulative_reward', 100 * (b - a))
        percentage_rows.append(dict(model=model, solo_mean_cumulative=100*a.mean(),
                                    social_mean_cumulative=100*b.mean(),
                                    change_percent=100*(b.mean()/a.mean()-1)))
        token = group.pivot(index='seed', columns='strategy', values='completion_tokens')
        add(model, 'social relative to solo', 'efficiency_percent',
            100 * ((b / token.social_action_payoff) / (a / token.solo_llm) - 1))
    causal = pd.read_csv(ROOT / 'runs/paper_analysis/analysis/causal_rung1/causal_separate_effects_seeds.csv')
    for model, p, reserve, field, scale in [
        ('qwen3_14b', 0, 2048, 'completion_rate_effect', 100),
        ('qwen3_14b', .15, 0, 'effective_selected_diversity_effect', 1),
        ('gpt_oss_20b', .15, 0, 'effective_selected_diversity_effect', 1),
    ]:
        values = causal[(causal.model == model) & causal.social &
                        (causal.p_independent == p) & (causal.execution_reserve == reserve)]
        add(model, 'PE' if reserve else 'IS', field, scale * values.sort_values('seed')[field])
    assert len(rows) == 10
    result = pd.DataFrame(rows)
    result['bonferroni_p'] = (10 * result.nominal_p).clip(upper=1)
    result.to_csv(OUT / 'rung1_table1_bonferroni.csv', index=False)
    pd.DataFrame(percentage_rows).to_csv(OUT / 'rung1_reward_percent_summary.csv', index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
