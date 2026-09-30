"""Realized per-round reward at matched token costs; run on Slurm, no model calls."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from analyze_efficiency import read_population
import plot_short_report as plot

ROOT = Path(__file__).resolve().parents[1]
OUT = plot.PAPER_RESULTS
MODELS = ['qwen3_14b', 'ministral3_14b_reasoning', 'gpt_oss_20b']
NAMES = ['Qwen3-14B', 'Ministral-3-14B', 'GPT-OSS-20B']
existing = pd.read_csv(OUT / 'rung1_primary_diagnostics_by_seed.csv')
rows = []
for model in MODELS:
    for strategy in ['solo_llm', 'social_action_payoff']:
        for seed in range(8):
            directory = ROOT / 'runs/v3/rung1' / f'v3_{strategy}__use_it_or_lose_it__{model}' / f'seed_{seed}'
            _, values = read_population(directory)
            frame = pd.DataFrame(values).sort_values('round')
            assert list(frame['round']) == list(range(100))
            reference = existing[(existing.model == model) & (existing.strategy == strategy) & (existing.seed == seed)].iloc[0]
            assert np.isclose(frame.reward.mean(), reference.reward)
            assert frame.completion_tokens.sum() == reference.completion_tokens
            assert frame.prompt_tokens.notna().all()
            frame['total_tokens'] = frame.completion_tokens + frame.prompt_tokens
            for key in ['completion_tokens', 'total_tokens']:
                frame['cumulative_' + key] = frame[key].cumsum()
            frame['model'], frame['strategy'], frame['seed'] = model, strategy, seed
            rows.extend(frame.to_dict('records'))
curves = pd.DataFrame(rows)
curves.to_csv(OUT / 'rung1_matched_token_trajectories.csv', index=False)
effects = []
for model in MODELS:
    for seed in range(8):
        a = curves[(curves.model == model) & (curves.seed == seed) & (curves.strategy == 'solo_llm')]
        b = curves[(curves.model == model) & (curves.seed == seed) & (curves.strategy == 'social_action_payoff')]
        for resource in ['completion_tokens', 'total_tokens']:
            x, z = a['cumulative_'+resource].to_numpy(), b['cumulative_'+resource].to_numpy()
            assert np.all(np.diff(x)>0) and np.all(np.diff(z)>0)
            target = min(x[-1], z[-1])
            assert target >= max(x[0],z[0])
            for method in ['linear', 'last_completed_round']:
                ai, bi = np.searchsorted(x,target,side='right')-1, np.searchsorted(z,target,side='right')-1
                av, bv = a.reward.iloc[ai], b.reward.iloc[bi]
                if method == 'linear':
                    av, bv = np.interp(target,x,a.reward), np.interp(target,z,b.reward)
                effects.append(dict(model=model,seed=seed,resource=resource,method=method,budget_tokens=target,
                    solo_reward=av,social_reward=bv,delta_reward=bv-av,
                    solo_round=int(a['round'].iloc[ai])+1,social_round=int(b['round'].iloc[bi])+1))
effects = pd.DataFrame(effects)
effects.to_csv(OUT / 'rung1_matched_token_effects_by_seed.csv',index=False)
summaries=[]
for keys,g in effects.groupby(['model','resource','method']):
    mean,se=g.delta_reward.mean(),g.delta_reward.sem()
    theta=math.atan(abs(mean/se)/math.sqrt(7))
    p=1-(10*theta+7.5*math.sin(2*theta)+1.5*math.sin(4*theta)+math.sin(6*theta)/6)/(5*math.pi)
    summaries.append(dict(zip(['model','resource','method'],keys),mean=mean,se=se,n=len(g),nominal_p=p,
        bonferroni_three_p=min(1,3*p),mean_budget_tokens=g.budget_tokens.mean(),
        solo_reward=g.solo_reward.mean(),social_reward=g.social_reward.mean(),
        solo_round=g.solo_round.mean(),social_round=g.social_round.mean()))
summary=pd.DataFrame(summaries)
summary.to_csv(OUT / 'rung1_matched_token_summary.csv',index=False)
print(summary.to_string(index=False),flush=True)
plot.setup()
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(10.5,3.3),sharey=True)
for mi,model in enumerate(MODELS):
    ax=axes[mi]
    for j,(resource,method) in enumerate([('completion_tokens','linear'),('completion_tokens','last_completed_round'),('total_tokens','linear')]):
        v=effects[(effects.model==model)&(effects.resource==resource)&(effects.method==method)].delta_reward
        ax.scatter(j+np.linspace(-.12,.12,8),v,color=['#4C78A8','#777777','#008F7A'][j],alpha=.7,s=22)
        ax.errorbar(j,v.mean(),yerr=v.sem(),fmt='o',color='black',capsize=4)
    ax.axhline(0,color='.5',ls='--',lw=1)
    ax.set_xticks([0,1,2],['Completion\ninterp.','Completion\nlast round','All tokens\ninterp.'],fontsize=9)
    ax.set_title(f'({"abc"[mi]}) {NAMES[mi]}',fontsize=12)
axes[0].set_ylabel('Social − solo reward\nat matched tokens',fontsize=11)
fig.subplots_adjust(left=.09,right=.99,bottom=.25,top=.86,wspace=.12)
plot.save_paper_figure(fig,'rung1_matched_tokens');plt.close(fig)
