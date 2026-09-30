"""Reproduce the paper's reviewer-requested audits from saved runs; run via Slurm.

No model calls. Population seeds remain the independent experimental units.
"""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_short_report as plot

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/population'
OUT = plot.PAPER_RESULTS
MODELS = ['openai/gpt-oss-120b', 'z-ai/glm-5.3-flash']
NAMES = ['GPT-OSS-120B', 'GLM-5.3-Flash']
CONDITIONS = ['llm_solo', 'llm_social_payoff', 'openevolve_solo', 'ucb_social', 'uniform_social']
LABELS = ['LLM solo', 'LLM social', 'OpenEvolve solo', 'UCB social', 'Uniform social']
COLORS = ['#4C78A8', '#E69F00', '#008F7A', '#B65A9F', '#777777']


def save(name, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / ('r4_review_' + name + '.csv'), index=False)
    return frame


def summarize(frame, metrics, groups=('model', 'condition')):
    rows = []
    for keys, sub in frame.groupby(list(groups)):
        for metric in metrics:
            v = sub[metric].dropna()
            row = dict(zip(groups, keys if isinstance(keys, tuple) else (keys,)))
            row.update(mean=v.mean(), se=v.sem(), n=len(v))
            row['statistic' if 'metric' in groups else 'metric'] = metric
            rows.append(row)
    return pd.DataFrame(rows)


def resource_accounting():
    """Compare learning-only and full-run usage from persistent provider ledgers."""
    spending = pd.read_csv(OUT / 'r4_spending_by_seed.csv')
    required = {'completion_tokens', 'prompt_tokens', 'total_tokens', 'reported_cost'}
    assert required.issubset(spending.columns), spending.columns
    keys = ['model', 'condition', 'seed']
    all_usage = spending.groupby(keys, as_index=False)[list(required)].sum()
    learning = (spending[spending.phase != 'holdout']
                .groupby(keys, as_index=False)[list(required)].sum())
    frame = all_usage.merge(learning, on=keys, suffixes=('_all', '_learning'), validate='one_to_one')
    frame = frame.rename(columns={
        'completion_tokens_learning': 'learning_completion_tokens',
        'completion_tokens_all': 'all_completion_tokens',
        'total_tokens_learning': 'learning_total_tokens',
        'total_tokens_all': 'all_total_tokens',
        'reported_cost_learning': 'learning_cost_usd',
        'reported_cost_all': 'all_cost_usd',
        'prompt_tokens_learning': 'learning_prompt_tokens',
        'prompt_tokens_all': 'all_prompt_tokens',
    })
    # Some providers omit or inconsistently populate usage.total_tokens. Build
    # the auditable total directly from the two consistently logged fields.
    frame['learning_total_tokens'] = (frame['learning_prompt_tokens'] +
                                      frame['learning_completion_tokens'])
    frame['all_total_tokens'] = frame['all_prompt_tokens'] + frame['all_completion_tokens']
    metrics = ['learning_completion_tokens', 'all_completion_tokens',
               'learning_prompt_tokens', 'all_prompt_tokens',
               'learning_total_tokens', 'all_total_tokens',
               'learning_cost_usd', 'all_cost_usd']
    assert len(frame) == 60 and not frame.duplicated(keys).any()
    save('resource_by_seed', frame)
    save('resource_summary', summarize(frame, metrics))
    return frame


def matched_cost():
    tests = pd.read_csv(OUT / 'r4_test_by_seed.csv')
    rows = []
    for model in MODELS:
        for seed in range(6):
            solo = tests[(tests.model == model) & (tests.seed == seed) & (tests.condition == 'llm_solo')].sort_values('round')
            social = tests[(tests.model == model) & (tests.seed == seed) & (tests.condition == 'llm_social_payoff')].sort_values('round')
            assert len(solo) == len(social) == 4
            for resource in ['cumulative_attributed_learning_cost', 'cumulative_completion_tokens']:
                x, z = solo[resource].to_numpy(), social[resource].to_numpy()
                assert np.all(np.diff(x) > 0) and np.all(np.diff(z) > 0)
                for rule in ['social_endpoint', 'common_endpoint']:
                    target = z[-1] if rule == 'social_endpoint' else min(x[-1], z[-1])
                    covered = max(x[0], z[0]) <= target <= min(x[-1], z[-1])
                    for metric in ['prompt_accuracy', 'constraint_accuracy']:
                        row = dict(model=model, seed=seed, resource=resource, rule=rule, metric=metric,
                                   cost=target, covered=covered)
                        if covered:
                            y, w = solo[metric].to_numpy(), social[metric].to_numpy()
                            right = min(np.searchsorted(x, target, side='right'), len(x)-1)
                            left = right - 1
                            a, b = np.interp(target, x, y), np.interp(target, z, w)
                            row.update(gain_pp=100*(b-a), initial_adjusted_pp=100*((b-w[0])-(a-y[0])),
                                       solo_left_round=solo.iloc[left]['round'], solo_right_round=solo.iloc[right]['round'],
                                       left_checkpoint_gain_pp=100*(b-y[left]), right_checkpoint_gain_pp=100*(b-y[right]),
                                       solo_accuracy=100*a, social_accuracy=100*b)
                        rows.append(row)
    frame = save('matched_cost_by_seed', rows)
    stats = summarize(frame, ['gain_pp', 'initial_adjusted_pp', 'left_checkpoint_gain_pp', 'right_checkpoint_gain_pp'],
                      ('model', 'resource', 'rule', 'metric'))
    save('matched_cost_summary', stats)
    return frame


def raw_audit():
    allocation = json.loads((RUN / 'allocation.json').read_text())
    lineage, revisions, peers, budget, item_rows = [], [], [], [], []
    common_tasks = None
    for slot in allocation['slots']:
        cfg = slot['config']
        if cfg['smoke']:
            continue
        meta = dict(model=cfg['model']['name'], condition=cfg['condition'], seed=cfg['seed'])
        directory = RUN / slot['label']
        assert (directory / 'completed.json').exists(), directory
        states = [json.loads((directory/'agents'/str(i)/'checkpoint.json').read_text())['state'] for i in range(5)]
        acquisitions = [x['learner']['acquisitions'] for x in states]
        known = [{} for _ in range(5)]
        scores = [{} for _ in range(5)]
        prior = None
        for r in range(50):
            rr = json.loads((directory/'rounds'/f'{r:04d}.json').read_text())
            assert len(rr) == 5 and {x['agent'] for x in rr} == set(range(5))
            previous_known = copy.deepcopy(known)
            deployed = []
            for row in rr:
                i, vid = row['agent'], row['version']
                if r == 0:
                    known[i][vid] = dict(depth=0, copied=False, postcopy=False, transfers=0, root='initial')
                for event in row['events']:
                    if event['kind'] == 'observe':
                        source, imported = event['target'], event['version']
                        assert prior[source]['version'] == imported
                        ancestor = previous_known[source][imported]
                        if imported not in known[i]:
                            value = dict(ancestor)
                            value['copied'] = ancestor['copied'] or ancestor['depth'] > 0
                            value['transfers'] += int(ancestor['depth'] > 0)
                            known[i][imported] = value
                        scores[i].setdefault(imported, event['training_score'])
                        available = [x for x in prior.values() if x['agent'] != i]
                        first = min(x['agent'] for x in available)
                        first_score = prior[first]['training_score']
                        others = [x['training_score'] for x in available if x['agent'] != first]
                        all_scores = [x['training_score'] for x in available]
                        peers.append(dict(**meta, agent=i, round=r, chose_first=source == first,
                            first_minus_other_score=first_score-np.mean(others),
                            chosen_minus_available_score=event['training_score']-np.mean(all_scores),
                            available_score_sd=np.std(all_scores), first_best=first_score >= max(all_scores),
                            chosen_best=event['training_score'] >= max(all_scores)))
                    if event['kind'] == 'revision' and event['changed']:
                        parent, child = event['parent'], event['candidate']
                        assert parent in known[i], (meta, r, i, parent)
                        ancestor = known[i][parent]
                        if child not in known[i]:
                            known[i][child] = dict(depth=ancestor['depth']+1, copied=ancestor['copied'],
                                postcopy=ancestor['postcopy'] or ancestor['copied'], transfers=ancestor['transfers'],
                                root=child if ancestor['depth'] == 0 else ancestor['root'])
                        before = row['parent_reward'] if row['parent_reward'] is not None else scores[i].get(parent)
                        after = row['candidate_reward']
                        revisions.append(dict(**meta, agent=i, round=r, copied_parent=ancestor['copied'],
                            direct_copied_parent=acquisitions[i][parent]['origin'] == 'social',
                            comparable=before is not None and after is not None,
                            training_gain_pp=100*(after-before) if before is not None and after is not None else None,
                            fresh_parent_evaluation=row['parent_reward'] is not None))
                        if after is not None:
                            scores[i][child] = after
                if row['parent_reward'] is not None:
                    scores[i][row['parent_version']] = row['parent_reward']
                assert vid in known[i], (meta, r, i, vid)
                node = known[i][vid]
                deployed.append(node)
                scores[i][vid] = row['training_score']
                b = row['budget']
                budget.append(dict(**meta, agent=i, round=r, allowance=b['limit'], completion=b['completion'],
                    fraction_used=b['completion']/b['limit'], hit_limit=b['completion'] >= b['limit'],
                    phase_cap_violations=b.get('phase_cap_violations', 0), errors=b.get('provider_generation_errors', 0)))
            lineage.append(dict(**meta, round=r, mean_depth=np.mean([x['depth'] for x in deployed]),
                copied_ancestry_pct=100*np.mean([x['copied'] for x in deployed]),
                copy_then_revision_pct=100*np.mean([x['postcopy'] for x in deployed]),
                distinct_deployed_lineages=len({x['root'] for x in deployed if x['root'] != 'initial'}),
                distinct_deployed_versions=len({x['version'] for x in rr}),
                max_depth=max(x['depth'] for x in deployed)))
            prior = {x['agent']: x for x in rr}
        for i in range(5):
            assert set(known[i]) == set(acquisitions[i]), (meta, i, 'acquisition coverage')
            assert set(scores[i]) == set(states[i]['scores']), (meta, i, 'score coverage')
            assert all(np.isclose(value, states[i]['scores'][vid]) for vid, value in scores[i].items()), (meta, i, 'score reconstruction')
        if cfg['condition'] in ['llm_solo', 'llm_social_payoff']:
            for r in [0, 10, 25, 49]:
                arrays = []
                for i in range(5):
                    summary = json.loads((directory/'test_summary'/f'{r:04d}-{i}.json').read_text())
                    records = json.loads((directory/'agents'/str(i)/'holdout'/summary['version']/'complete.json').read_text())
                    by_task = {x['identity']['task']: x['result'] for x in records}
                    tasks = sorted(by_task)
                    assert len(tasks) == len(records) == 200
                    if common_tasks is None:
                        common_tasks = tasks
                    assert tasks == common_tasks
                    arr = np.array([[by_task[t]['prompt_level_reward'], by_task[t]['constraint_accuracy']] for t in tasks])
                    assert np.allclose(arr.mean(axis=0), [summary['prompt_accuracy'], summary['constraint_accuracy']])
                    arrays.append(arr)
                for task, values in zip(common_tasks, np.mean(arrays, axis=0)):
                    item_rows.append(dict(**meta, round=r, task=task, prompt_accuracy=values[0], constraint_accuracy=values[1]))
        print('Audited', slot['label'], flush=True)
    lf = save('lineage_by_seed', lineage)
    save('lineage_summary', summarize(lf[lf['round'] == 49], ['mean_depth','copy_then_revision_pct','copied_ancestry_pct','distinct_deployed_lineages']))
    rv = save('revision_events', revisions)
    rvseed = rv.groupby(['model','condition','seed','copied_parent'], as_index=False).agg(
        training_gain_pp=('training_gain_pp','mean'), comparisons=('training_gain_pp','count'), proposals=('round','count'))
    save('revision_gains_by_seed', rvseed)
    pf = save('peer_quality_events', peers)
    pseed = pf.groupby(['model','condition','seed'], as_index=False).mean(numeric_only=True)
    save('peer_quality_by_seed', pseed)
    save('peer_quality_summary', summarize(pseed, ['chose_first','first_minus_other_score','chosen_minus_available_score','available_score_sd','first_best','chosen_best']))
    bf = save('budget_by_round', budget)
    bs = bf.groupby(['model','condition','seed'], as_index=False).agg(max_fraction=('fraction_used','max'),
        mean_fraction=('fraction_used','mean'), hit_limits=('hit_limit','sum'), phase_violations=('phase_cap_violations','sum'), errors=('errors','sum'))
    save('budget_by_seed', bs)
    items = save('heldout_items', item_rows)
    return lf, pseed, bs, items


def bootstrap(items):
    rng = np.random.default_rng(20260919)
    rows = []
    weights = np.array([5, 12.5, 19.5, 12.0])/49
    for model in MODELS:
        for metric in ['prompt_accuracy', 'constraint_accuracy']:
            arrays = []
            for condition in ['llm_solo','llm_social_payoff']:
                z = items[(items.model == model) & (items.condition == condition)]
                arrays.append(z.sort_values(['seed','round','task'])[metric].to_numpy().reshape(6,4,200))
            delta = arrays[1]-arrays[0]
            estimates = {'final':delta[:,-1,:], 'trajectory':np.einsum('srt,r->st',delta,weights),
                         'initial_adjusted_trajectory':np.einsum('srt,r->st',delta,weights)-delta[:,0,:]}
            for name, values in estimates.items():
                draws = {'item_only':[], 'seed_and_item':[]}
                for _ in range(5000):
                    tasks = rng.integers(0,200,200)
                    seeds = rng.integers(0,6,6)
                    draws['item_only'].append(values[:,tasks].mean())
                    draws['seed_and_item'].append(values[seeds][:,tasks].mean())
                rows.append(dict(model=model, metric=metric, contrast=name, mean_pp=100*values.mean(),
                    seed_se_pp=100*values.mean(axis=1).std(ddof=1)/np.sqrt(6),
                    item_bootstrap_se_pp=100*np.std(draws['item_only'],ddof=1),
                    joint_bootstrap_se_pp=100*np.std(draws['seed_and_item'],ddof=1),
                    draws=5000, tasks=200, seeds=6))
    return save('bootstrap_summary', rows)


def figures(cost, lineage, peers, uncertainty):
    timing = pd.read_csv((ROOT / 'runs/timing/analysis/test_by_seed.csv'))
    fig, axes = plt.subplots(1,2,figsize=(8.6,3.0))
    for mi, model in enumerate(MODELS):
        for condition,label,color in [('llm_solo','Solo','#4C78A8'),('forced_early','Four early observations','#E69F00'),('forced_distributed','Four distributed observations','#8E63B0')]:
            z=timing[(timing.model==model)&(timing.condition==condition)&timing['round'].isin([0,10,25,49])].groupby('round').prompt_accuracy.agg(['mean','sem'])
            axes[mi].plot(z.index,100*z['mean'],color=color,label=label)
            axes[mi].fill_between(z.index,100*(z['mean']-z['sem']),100*(z['mean']+z['sem']),color=color,alpha=.16)
        axes[mi].set_title(f'({"ab"[mi]}) {NAMES[mi]}',fontsize=12)
        axes[mi].set_xlabel('Learning round');axes[mi].set_xticks([0,10,25,49])
    axes[0].set_ylabel('Test accuracy (%)',fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.11,right=.98,bottom=.28,wspace=.26)
    plot.save_paper_figure(fig,'r4_assigned_timing_main');plt.close(fig)
    plot.paper_matched_token_results()
    fig, axes=plt.subplots(3,2,figsize=(9,7),sharex=True)
    for mi,model in enumerate(MODELS):
        for ri,(metric,ylabel) in enumerate([('mean_depth','Revision depth'),('copy_then_revision_pct','Copy then revise\ndeployments (%)'),('distinct_deployed_lineages','Deployed innovation\nlineages')]):
            ax=axes[ri,mi]
            for c,label,color in zip(CONDITIONS,LABELS,COLORS):
                z=lineage[(lineage.model==model)&(lineage.condition==c)].groupby('round')[metric].agg(['mean','sem'])
                ax.plot(z.index,z['mean'],label=label,color=color);ax.fill_between(z.index,z['mean']-z['sem'],z['mean']+z['sem'],color=color,alpha=.14)
            ax.set_title(f'({"abcdef"[2*ri+mi]}) {NAMES[mi]}',fontsize=12)
            if mi==0:ax.set_ylabel(ylabel,fontsize=11)
            if ri==2:ax.set_xlabel('Learning round')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.12,right=.98,bottom=.15,hspace=.42,wspace=.26)
    plot.save_paper_figure(fig,'r4_skill_lineages');plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(8.6,3.0))
    for mi,model in enumerate(MODELS):
        z=uncertainty[(uncertainty.model==model)&(uncertainty.metric=='prompt_accuracy')].set_index('contrast').loc[['trajectory','initial_adjusted_trajectory','final']]
        for j,(col,label,color) in enumerate([('seed_se_pp','Seed SE',COLORS[0]),('item_bootstrap_se_pp','Item bootstrap SE',COLORS[1]),('joint_bootstrap_se_pp','Seed + item SE',COLORS[2])]):
            axes[mi].errorbar(np.arange(3)+(j-1)*.17,z.mean_pp,yerr=z[col],fmt='o',capsize=3,color=color,label=label)
        axes[mi].axhline(0,color='.55',ls=':');axes[mi].set_xticks(range(3),['Trajectory','Initial-adjusted','Final'],fontsize=9)
        axes[mi].set_title(f'({"ab"[mi]}) {NAMES[mi]}',fontsize=12)
    axes[0].set_ylabel('Social minus solo\naccuracy (pp)',fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.12,right=.98,bottom=.28,wspace=.28)
    plot.save_paper_figure(fig,'r4_sampling_uncertainty');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.6,3))
    for mi,model in enumerate(MODELS):
        z=peers[(peers.model==model)&(peers.condition=='llm_social_payoff')]
        cols=['first_minus_other_score','chosen_minus_available_score']
        for j,col in enumerate(cols):
            axes[mi].scatter(np.full(len(z),j),100*z[col],s=17,color=COLORS[mi],alpha=.6)
            axes[mi].errorbar(j,100*z[col].mean(),yerr=100*z[col].sem(),color='black',fmt='o',capsize=3)
        axes[mi].axhline(0,color='.55',ls=':');axes[mi].set_xticks([0,1],['First peer\nminus others','Chosen peer\nminus available'],fontsize=9)
        axes[mi].set_xlim(-.4,1.4)
        axes[mi].set_title(f'({"ab"[mi]}) {NAMES[mi]}',fontsize=12)
    axes[0].set_ylabel('Training score\ndifference (pp)',fontsize=11)
    fig.subplots_adjust(left=.12,right=.98,bottom=.24,wspace=.28)
    plot.save_paper_figure(fig,'r4_peer_quality');plt.close(fig)


def paired_significance():
    """Two-sided paired t-tests, using one matched contrast per population seed."""
    from mpmath import betainc
    def two_sided_p(t, df):
        # Exact Student-t tail: I_{df/(df+t^2)}(df/2, 1/2).
        return float(betainc(df/2, .5, 0, df/(df+t*t), regularized=True))
    assert np.isclose(two_sided_p(1, 1), .5, atol=1e-12)
    assert np.isclose(two_sided_p(2.570581835636305, 5), .05, atol=1e-10)
    assert np.isclose(two_sided_p(3.4994832973505026, 7), .01, atol=1e-10)
    rows=[]
    def record(study, model, contrast, metric, values):
        values=np.asarray(values,dtype=float)
        assert len(values)>=6 and np.isfinite(values).all()
        se=values.std(ddof=1)/np.sqrt(len(values))
        t=float(values.mean()/se) if se>0 else float('nan')
        p=two_sided_p(t,len(values)-1) if se>0 else float('nan')
        threshold=(next((f'p<{x:g}' for x in [.001,.01,.05] if p<x),'p>=0.05')
                   if np.isfinite(p) else 'undefined: zero variance')
        rows.append(dict(study=study,model=model,contrast=contrast,metric=metric,n=len(values),
                         mean=values.mean(),se=se,
                         t=t,df=len(values)-1,p_two_sided=p,threshold=threshold,
                         multiplicity='unadjusted'))
    frame=pd.read_csv(OUT/'r4_paired_effects_by_seed.csv')
    assert not frame.duplicated(['model','contrast','metric','seed']).any()
    for (model,contrast,metric),z in frame.groupby(['model','contrast','metric']):
        record('skills',model,contrast,metric,z.effect)
    resources=pd.read_csv(OUT/'r4_review_resource_by_seed.csv')
    resource_metrics=['learning_completion_tokens','all_completion_tokens',
                      'learning_prompt_tokens','all_prompt_tokens',
                      'learning_total_tokens','all_total_tokens',
                      'learning_cost_usd','all_cost_usd']
    for model,z in resources.groupby('model'):
        for metric in resource_metrics:
            wide=z.pivot(index='seed',columns='condition',values=metric)
            for a,b in [('llm_social_payoff','llm_solo'),('ucb_social','openevolve_solo'),('uniform_social','openevolve_solo'),('ucb_social','uniform_social')]:
                record('skills',model,a+' minus '+b,metric,wide[a]-wide[b])
    ucb=pd.read_csv(OUT/'rung1_matched_ucb_curves.csv')
    for metric in ['cumulative_reward','cumulative_regret']:
        wide=ucb[ucb['round']==99].pivot(index='seed',columns='strategy',values=metric)
        record('bandit','algorithmic','Hierarchical minus solo UCB',metric,wide['Hierarchical UCB']-wide['Solo UCB'])
    causal=pd.read_csv(ROOT/'runs/paper_analysis/analysis/causal_rung1/causal_separate_effects_seeds.csv')
    for (model,p,reserve,social),z in causal.groupby(['model','p_independent','execution_reserve','social']):
        for metric in ['reward_effect','completion_rate_effect','effective_selected_diversity_effect']:
            assert not z.seed.duplicated().any()
            record('bandit',model,f'IS={p};PE={reserve};social={social}',metric,z[metric])
    timing=pd.read_csv((ROOT / 'runs/timing/analysis/paired_by_seed.csv'))
    for (model,contrast,metric),z in timing.groupby(['model','contrast','metric']):
        for value in ['final_gain','learning_curve_gain','baseline_adjusted_learning_curve_gain']:
            assert not z.seed.duplicated().any()
            record('timing',model,contrast,metric+':'+value,100*z[value])
    return save('paired_ttests',rows)


def primary_bandit_diagnostics():
    frame=pd.read_csv(ROOT/'runs/v3/rung1/analysis/report_metrics_by_seed.csv')
    frame=frame[(frame.model.isin(plot.MODELS))&(frame.strategy.isin(['solo_llm','social_action_payoff']))&
                (frame.guidance=='neutral')&(frame.budget=='use_it_or_lose_it')&(frame.period==0)&(frame['shape']==1)]
    assert len(frame)==48 and not frame.duplicated(['model','strategy','seed']).any()
    frame.to_csv(OUT/'rung1_primary_diagnostics_by_seed.csv',index=False)
    fig,axes=plt.subplots(3,3,figsize=(9,7),sharey='row')
    for mi,model in enumerate(plot.MODELS):
        for ri,(metric,label,scale) in enumerate([('reward','Population reward',1),('conditional_reward','Reward given\nvalid pull',1),('completion','Completed pulls (%)',100)]):
            for j,c in enumerate(['solo_llm','social_action_payoff']):
                v=frame[(frame.model==model)&(frame.strategy==c)][metric]*scale
                axes[ri,mi].bar(j,v.mean(),yerr=v.sem(),color=COLORS[j],capsize=3)
            axes[ri,mi].set_xticks([0,1],['Solo','Social'])
            axes[ri,mi].set_title(f'({"abcdefghi"[ri*3+mi]}) {plot.NAMES[model]}',fontsize=12)
            if mi==0:axes[ri,mi].set_ylabel(label,fontsize=11)
    fig.subplots_adjust(left=.13,right=.98,hspace=.55,wspace=.15)
    plot.save_paper_figure(fig,'rung1_completion_primary');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(9,3),sharey=True)
    for mi,model in enumerate(plot.MODELS):
        for j,c in enumerate(['solo_llm','social_action_payoff']):
            z=frame[(frame.model==model)&(frame.strategy==c)]
            axes[mi].scatter(z.completion_tokens/1e6,z.reward,color=COLORS[j],s=24,label=['Solo','Social'][j])
        axes[mi].set_xlabel('Tokens (millions)',fontsize=10)
        axes[mi].set_title(f'({"abc"[mi]}) {plot.NAMES[model]}',fontsize=12)
    axes[0].set_ylabel('Mean reward per round',fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.11,right=.98,bottom=.29,wspace=.15)
    plot.save_paper_figure(fig,'rung1_efficiency_primary');plt.close(fig)


def timing_contrasts():
    root=(ROOT / 'runs/timing/analysis')
    pairs=pd.read_csv(root/'paired_by_seed.csv')
    tests=pd.read_csv(root/'test_by_seed.csv')
    contrasts=[('forced_early','forced_distributed','Early − distributed'),('forced_distributed','llm_solo','Distributed − solo'),
               ('forced_early','llm_solo','Early − solo'),('optional_early','llm_social_payoff','Early access − unrestricted')]
    rows=[]
    fig,axes=plt.subplots(3,2,figsize=(10,8))
    for mi,model in enumerate(MODELS):
        for ci,(a,b,label) in enumerate(contrasts):
            p=pairs[(pairs.model==model)&(pairs.metric=='prompt_accuracy')&(pairs.contrast==a+' minus '+b)]
            z=tests[(tests.model==model)&(tests['round']==49)].pivot(index='seed',columns='condition',values='cumulative_attributed_learning_cost')
            for ri,(metric,values) in enumerate([('Final accuracy (pp)',100*p.final_gain),('Trajectory accuracy (pp)',100*p.learning_curve_gain),('Learning expense (USD)',z[a]-z[b])]):
                assert len(values)==6 and values.notna().all()
                axes[ri,mi].errorbar(values.mean(),ci,xerr=values.sem(),fmt='o',capsize=3,color=COLORS[mi])
                rows.append(dict(model=model,contrast=a+' minus '+b,metric=metric,mean=values.mean(),se=values.sem(),n=6))
        for ri,metric in enumerate(['Final accuracy (pp)','Trajectory accuracy (pp)','Learning expense (USD)']):
            axes[ri,mi].axvline(0,color='.5',ls=':',lw=1)
            axes[ri,mi].set_yticks(range(4),[c[2] for c in contrasts],fontsize=8)
            axes[ri,mi].invert_yaxis();axes[ri,mi].set_xlabel(metric,fontsize=10)
            axes[ri,mi].set_title(f'({"abcdef"[2*ri+mi]}) {NAMES[mi]}',fontsize=12)
    fig.subplots_adjust(left=.23,right=.98,bottom=.09,wspace=1.1,hspace=.6)
    plot.save_paper_figure(fig,'r4_timing_paired');plt.close(fig)
    save('timing_contrasts',rows)


def supplemental_audits():
    """Summarize mechanism checks from cached event/seed exports."""
    budgets = pd.read_csv(OUT / 'r4_review_budget_by_seed.csv')
    save('budget_summary', summarize(budgets, ['mean_fraction', 'max_fraction', 'hit_limits', 'phase_violations', 'errors']))
    events = pd.read_csv(OUT / 'r4_review_revision_events.csv')
    frames = []
    # Revisions use recorded parent fitness; reevaluation is a separate action.
    assert not events.fresh_parent_evaluation.any()
    for name, subset in [('Recorded parent score', events)]:
        seed = subset.groupby(['model','condition','seed','copied_parent'],as_index=False).agg(
            training_gain_pp=('training_gain_pp','mean'), proposals=('training_gain_pp','count'))
        seed['comparison'] = name
        frames.append(seed)
    gains = pd.concat(frames,ignore_index=True)
    save('revision_gain_checks_by_seed', gains)
    save('revision_gain_checks_summary', summarize(gains,['training_gain_pp','proposals'],('model','condition','comparison','copied_parent')))
    fig, axes = plt.subplots(1,2,figsize=(9,3.2))
    for mi,model in enumerate(MODELS):
        for j,(name,color) in enumerate([('Recorded parent score',COLORS[0])]):
            z=gains[(gains.model==model)&(gains.condition=='llm_social_payoff')&(gains.comparison==name)]
            for copied in [False,True]:
                v=z[z.copied_parent==copied].training_gain_pp.dropna()
                axes[mi].errorbar(int(copied)+(j-.5)*.18,v.mean(),yerr=v.sem(),fmt='o',color=color,capsize=3,label=name if not copied else None)
        axes[mi].axhline(0,color='.55',ls=':')
        axes[mi].set_xticks([0,1],['Private ancestry','Copied ancestry'],fontsize=10)
        axes[mi].set_title(f'({"ab"[mi]}) {NAMES[mi]}',fontsize=12)
    axes[0].set_ylabel('Candidate minus parent\ntraining score (pp)',fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.13,right=.98,bottom=.27,wspace=.3)
    plot.save_paper_figure(fig,'r4_revision_ancestry_gains');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.2))
    for mi,model in enumerate(MODELS):
        for j,c in enumerate(CONDITIONS):
            v=100*budgets[(budgets.model==model)&(budgets.condition==c)].mean_fraction
            axes[mi].bar(j,v.mean(),yerr=v.sem(),color=COLORS[j],capsize=3)
            maxima=100*budgets[(budgets.model==model)&(budgets.condition==c)].max_fraction
            axes[mi].scatter(np.full(len(maxima),j),maxima,color='.35',marker='x',s=16,label='Seed maximum' if j==0 else None)
        axes[mi].axhline(100,color='.5',ls=':',lw=1)
        axes[mi].set_xticks(range(5),['LLM\nsolo','LLM\nsocial','OE\nsolo','UCB','Uniform'],fontsize=9)
        axes[mi].set_title(f'({"ab"[mi]}) {NAMES[mi]}',fontsize=12)
    axes[0].set_ylabel('Round allowance\nused (%)',fontsize=11)
    axes[1].legend(fontsize=8,frameon=False)
    fig.subplots_adjust(left=.13,right=.98,bottom=.24,wspace=.28)
    plot.save_paper_figure(fig,'r4_budget_use');plt.close(fig)
    peers=pd.read_csv(OUT/'r4_review_peer_quality_by_seed.csv')
    fig, axes=plt.subplots(1,2,figsize=(9,3.1))
    for mi,model in enumerate(MODELS):
        z=peers[peers.model==model]
        for j,c in enumerate(['llm_social_payoff','ucb_social','uniform_social']):
            v=100*z[z.condition==c].available_score_sd
            axes[mi].bar(j,v.mean(),yerr=v.sem(),color=[COLORS[1],COLORS[3],COLORS[4]][j],capsize=3)
        axes[mi].set_xticks(range(3),['LLM social','UCB','Uniform'],fontsize=10)
        axes[mi].set_title(f'({"ab"[mi]}) {NAMES[mi]}',fontsize=12)
    axes[0].set_ylabel('Within-choice peer\nscore SD (pp)',fontsize=11)
    fig.subplots_adjust(left=.13,right=.98,bottom=.22,wspace=.3)
    plot.save_paper_figure(fig,'r4_peer_score_spread');plt.close(fig)
    causal=ROOT/'runs/paper_analysis/analysis/causal_rung1'
    sep=pd.read_csv(causal/'causal_separate_effects_summary.csv')
    did=pd.read_csv(causal/'causal_did_summary.csv')
    sep=sep[(sep.model.str.contains('ministral'))&(sep.p_independent==.15)&(sep.execution_reserve==0)].set_index('social')
    did=did[(did.model.str.contains('ministral'))&(did.p_independent==.15)&(did.execution_reserve==0)].iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(9,3.1))
    for j,(metric,title,scale) in enumerate([('reward','(a) Reward effect',1),('completion_rate','(b) Completion effect',100)]):
        means=[sep.loc[s,metric+'_effect'] for s in [False,True]]+[did[metric+'_did']]
        ses=[sep.loc[s,metric+'_effect_se'] for s in [False,True]]+[did[metric+'_did_se']]
        axes[j].bar(range(3),np.array(means)*scale,yerr=np.array(ses)*scale,color=[COLORS[0],COLORS[1],'#777777'],capsize=3)
        axes[j].axhline(0,color='.5',lw=1)
        axes[j].set_xticks(range(3),['Solo','Social','Interaction'],fontsize=10)
        axes[j].set_title(title,fontsize=12)
        axes[j].set_ylabel('Reward per round' if j==0 else 'Percentage points',fontsize=11)
    fig.subplots_adjust(left=.1,right=.98,bottom=.22,wspace=.36)
    plot.save_paper_figure(fig,'rung1_ministral_search_effects');plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    plot.setup()
    resources=resource_accounting()
    cost=matched_cost()
    lineage,peers,budgets,items=raw_audit()
    uncertainty=bootstrap(items)
    figures(cost,lineage,peers,uncertainty)
    supplemental_audits()
    primary_bandit_diagnostics()
    timing_contrasts()
    paired_significance()
    print(uncertainty.to_string(index=False),flush=True)
