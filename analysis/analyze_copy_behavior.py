"""Recorded observation versus deployment; execute via Slurm, no model calls."""
import json,statistics,math
from pathlib import Path
root=Path(__file__).resolve().parents[1] / 'runs/population'
seeds=[]
for slot in json.loads((root/'allocation.json').read_text())['slots']:
    c=slot['config']
    if c['smoke']: continue
    p=root/slot['label']; n=c['num_agents']; rounds=c['rounds']-1
    learners=[json.loads((p/'agents'/str(i)/'checkpoint.json').read_text())['state']['learner'] for i in range(n)]
    counts=dict(observations=0,new_social_acquisitions=0,immediate_new_copy=0,social_deployments=0,revise_actions=0,changed_revisions=0,revise_social_parent=0,early_observations=0,late_observations=0)
    observing=set(); copying=set(); prev={}
    for r in range(c['rounds']):
        rows=json.loads((p/'rounds'/f'{r:04d}.json').read_text())
        for row in rows:
            i=row['agent']; vid=row['version']; acq=learners[i]['acquisitions']
            if r==0: prev[i]=vid; continue
            social=lambda v: acq[v]['origin']=='social' and acq[v]['round']<=r
            if social(vid): counts['social_deployments']+=1; copying.add(i)
            if row['action']['action']=='revise':
                counts['revise_actions']+=1
                if social(row['parent_version']): counts['revise_social_parent']+=1
            for e in row['events']:
                if e['kind']=='revision': counts['changed_revisions']+=bool(e['changed'])
                if e['kind']!='observe': continue
                counts['observations']+=1; observing.add(i)
                counts['early_observations' if r<=10 else 'late_observations']+=1
                new=social(e['version']) and acq[e['version']]['round']==r
                counts['new_social_acquisitions']+=new
                counts['immediate_new_copy']+=bool(new and vid==e['version'] and vid!=prev[i])
            prev[i]=vid
    record=dict(model=c['model']['name'],condition=c['condition'],seed=c['seed'],raw=counts)
    record.update({k+'_per_agent':v/n for k,v in counts.items()})
    record.update(observe_pct=100*counts['observations']/(n*rounds),social_deploy_pct=100*counts['social_deployments']/(n*rounds),agents_ever_observe_pct=100*len(observing)/n,agents_ever_copy_pct=100*len(copying)/n,early_observe_pct=100*counts['early_observations']/(n*10),late_observe_pct=100*counts['late_observations']/(n*(rounds-10)))
    seeds.append(record)
groups=[]
for model,condition in sorted({(x['model'],x['condition']) for x in seeds}):
    rows=[x for x in seeds if (x['model'],x['condition'])==(model,condition)]
    metrics={k:dict(mean=statistics.mean(x[k] for x in rows),se=statistics.stdev(x[k] for x in rows)/math.sqrt(len(rows))) for k in rows[0] if k not in ['model','condition','seed','raw']}
    groups.append(dict(model=model,condition=condition,n=len(rows),metrics=metrics))
(root/'analysis/copy_behavior.json').write_text(json.dumps(dict(seed_level=seeds,groups=groups),indent=2)+'\n')
print(json.dumps(groups),flush=True)
