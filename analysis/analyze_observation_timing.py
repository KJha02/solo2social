"""Timing of recorded observations; run through Slurm without model calls."""
import json,statistics,math
from pathlib import Path
root=Path(__file__).resolve().parents[1] / 'runs/population'
records=[]
for slot in json.loads((root/'allocation.json').read_text())['slots']:
    c=slot['config']
    if c['smoke'] or c['condition'] not in {'llm_social_payoff','ucb_social','uniform_social'}: continue
    agents=[]
    for i in range(c['num_agents']):
        state=json.loads((root/slot['label']/'agents'/str(i)/'checkpoint.json').read_text())['state']['learner']
        times=[x['acquired_round'] for x in state['observations']]
        assert all(1<=t<=49 for t in times)
        agents.append(times)
    times=sorted(t for a in agents for t in a)
    assert times
    records.append(dict(model=c['model']['name'],condition=c['condition'],seed=c['seed'],observations=len(times),mean_round=statistics.mean(times),median_round=statistics.median(times),fraction_first10=100*sum(t<=10 for t in times)/len(times),agent_weighted_mean_round=statistics.mean(statistics.mean(a) for a in agents if a),observers=sum(bool(a) for a in agents),round_counts={str(r):times.count(r) for r in range(1,50)}))
def ms(xs): return dict(mean=statistics.mean(xs),se=statistics.stdev(xs)/math.sqrt(len(xs)),n=len(xs))
groups=[]
for model,condition in sorted({(x['model'],x['condition']) for x in records}):
    rows=[x for x in records if (x['model'],x['condition'])==(model,condition)]
    assert len(rows)==6
    groups.append(dict(model=model,condition=condition,metrics={k:ms([x[k] for x in rows]) for k in ['mean_round','median_round','fraction_first10','agent_weighted_mean_round','observers','observations']}))
contrasts=[]
index={(x['model'],x['condition'],x['seed']):x for x in records}
glm='z-ai/glm-5.3-flash'; gpt='openai/gpt-oss-120b'
for name,a,b in [('GLM minus GPT LLM',(glm,'llm_social_payoff'),(gpt,'llm_social_payoff')),('GLM LLM minus UCB',(glm,'llm_social_payoff'),(glm,'ucb_social')),('GPT LLM minus UCB',(gpt,'llm_social_payoff'),(gpt,'ucb_social'))]:
    contrasts.append(dict(contrast=name,mean_round_difference=ms([index[(*a,s)]['mean_round']-index[(*b,s)]['mean_round'] for s in range(6)])))
out=dict(seed_level=records,groups=groups,contrasts=contrasts)
(root/'analysis/observation_timing.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(groups=groups,contrasts=contrasts)),flush=True)
