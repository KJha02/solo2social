"""Matched total learning-token audit from saved rounds. Run through Slurm; no API calls."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/population/full'
OUT = ROOT / 'outputs/results'
tests = pd.read_csv(OUT / 'r4_test_by_seed.csv')
ledger = pd.read_csv(OUT / 'r4_review_resource_by_seed.csv')
rows = []
audits = []
for (model, condition, seed), group in tests.groupby(['model', 'condition', 'seed']):
    mi = 0 if model == 'openai/gpt-oss-120b' else 1
    directory = RUN / f'model_{mi}' / condition / f'seed_{seed}'
    completion = prompt = 0
    for r in range(50):
        records = json.loads((directory / 'rounds' / f'{r:04d}.json').read_text())
        for rec in records:
            b = rec['budget']
            completion += b['completion']
            prompt += sum(p.get('prompt_tokens', 0) for p in b['phases'].values())
        if r in set(group['round']):
            row = group[group['round'] == r].iloc[0].to_dict()
            assert completion == row['cumulative_completion_tokens'], (model, condition, seed, r)
            row['cumulative_learning_total_tokens'] = completion + prompt
            rows.append(row)
    entry = ledger[(ledger.model == model) & (ledger.condition == condition) & (ledger.seed == seed)].iloc[0]
    audits.append(dict(model=model, condition=condition, seed=seed,
                       attributed_total_tokens=completion+prompt,
                       ledger_total_tokens=entry.learning_total_tokens,
                       unattributed_tokens=entry.learning_total_tokens-completion-prompt))
curves = pd.DataFrame(rows)
curves.to_csv(OUT / 'r4_total_token_checkpoints.csv', index=False)
pd.DataFrame(audits).to_csv(OUT / 'r4_total_token_accounting_audit.csv', index=False)
effects = []
for model in curves.model.unique():
    for treatment, control in [('llm_social_payoff','llm_solo'),('ucb_social','openevolve_solo'),('uniform_social','openevolve_solo'),('ucb_social','uniform_social')]:
        for seed in range(6):
            a = curves[(curves.model==model)&(curves.condition==control)&(curves.seed==seed)].sort_values('round')
            b = curves[(curves.model==model)&(curves.condition==treatment)&(curves.seed==seed)].sort_values('round')
            x = a.cumulative_learning_total_tokens.to_numpy()
            z = b.cumulative_learning_total_tokens.to_numpy()
            assert np.all(np.diff(x)>0) and np.all(np.diff(z)>0)
            target = min(x[-1],z[-1])
            assert max(x[0],z[0]) <= target
            for metric in ['prompt_accuracy','constraint_accuracy']:
                av = np.interp(target,x,a[metric]); bv = np.interp(target,z,b[metric])
                effects.append(dict(model=model,treatment=treatment,control=control,seed=seed,metric=metric,budget_tokens=target,control_accuracy=100*av,treatment_accuracy=100*bv,gain_pp=100*(bv-av)))
effects = pd.DataFrame(effects)
effects.to_csv(OUT / 'r4_matched_total_tokens_by_seed.csv',index=False)
summary=[]
for keys,g in effects.groupby(['model','treatment','control','metric']):
    t=abs(g.gain_pp.mean()/g.gain_pp.sem()); theta=math.atan(t/math.sqrt(5))
    p=1-(2*theta/math.pi+4*math.sin(2*theta)/(3*math.pi)+math.sin(4*theta)/(6*math.pi))
    row=dict(zip(['model','treatment','control','metric'],keys))
    row.update(gain_pp=g.gain_pp.mean(),se_pp=g.gain_pp.sem(),nominal_p=p,budget_million=g.budget_tokens.mean()/1e6,control_accuracy=g.control_accuracy.mean(),treatment_accuracy=g.treatment_accuracy.mean(),n=len(g))
    summary.append(row)
summary=pd.DataFrame(summary)
summary.to_csv(OUT / 'r4_matched_total_tokens_summary.csv',index=False)
print(summary[summary.metric=='prompt_accuracy'].to_string(index=False))
audit=pd.DataFrame(audits)
print('Ledger residuals:',audit.unattributed_tokens.describe().to_string())
print(audit.groupby(['model','condition'])[['attributed_total_tokens','ledger_total_tokens','unattributed_tokens']].mean().to_string())
assert (audit.unattributed_tokens == 0).all()
import plot_short_report as plot
plot.setup()
plot.paper_matched_token_results()
