"""Seed-level static social RSI curves and paired effects. Run through Slurm."""
import argparse
import hashlib
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ('constraint_accuracy', 'prompt_accuracy')


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_se(values):
    return dict(mean=statistics.mean(values),
                se=statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
                seeds=len(values))


def analyze(root, output, smoke=False):
    allocation = json.loads((root / 'allocation.json').read_text())
    learning, tests, expenses, summaries = [], [], [], []
    for slot in allocation['slots'] + allocation.get('reused_controls', []):
        config = slot['config']
        if config['smoke'] != smoke:
            continue
        directory = Path(slot['directory']) if slot.get('reused') else root / slot['label']
        if slot.get('reused') and hashlib.sha256((directory / 'provenance.json').read_bytes()).hexdigest() != slot['provenance_sha256']:
            raise ValueError('Reused control provenance changed')
        metadata = dict(model=config['model']['name'], condition=config['condition'], seed=config['seed'],
                        reused_control=bool(slot.get('reused')))
        # These are complete population rounds only, never partial-agent averages.
        costs = tokens = reward_sum = revisions = observations = 0
        per_round = {}
        for path in sorted((directory / 'rounds').glob('*.json')):
            rows = json.loads(path.read_text())
            assert len(rows) == config['num_agents']
            r = rows[0]['round']
            costs += sum(x['budget']['cost_usd'] for x in rows)
            tokens += sum(x['budget']['completion'] for x in rows)
            events = [e for row in rows for e in row['events']]
            revisions += sum(e['kind'] == 'revision' and e['changed'] for e in events)
            observations += sum(e['kind'] == 'observe' for e in events)
            fresh = statistics.mean(x['execution_constraint'] for x in rows)
            reward_sum += fresh if r else 0
            values = [x['training_score'] for x in rows]
            record = dict(**metadata, round=r, cumulative_attributed_learning_cost=costs,
                cumulative_completion_tokens=tokens,
                deployed_training_fitness=statistics.mean(values) if all(x is not None for x in values) else None,
                fresh_execution_constraint=fresh,
                fresh_execution_prompt=statistics.mean(x['execution_prompt'] for x in rows),
                cumulative_execution_constraint_reward=reward_sum,
                cumulative_valid_revisions_per_agent=revisions / len(rows),
                cumulative_observations_per_agent=observations / len(rows),
                execution_completion=statistics.mean(x['execution_submitted_rate'] for x in rows))
            learning.append(record)
            per_round[r] = record
        trajectory = []
        for r in config['snapshots']:
            paths = [directory / 'test_summary' / f'{r:04d}-{i}.json' for i in range(config['num_agents'])]
            if not all(p.exists() for p in paths) or r not in per_round:
                continue
            rows = [json.loads(p.read_text()) for p in paths]
            assert all(x['tasks'] == 200 and x['round'] == r for x in rows)
            record = dict(**metadata, round=r, **{m: statistics.mean(x[m] for x in rows) for m in METRICS},
                cumulative_attributed_learning_cost=per_round[r]['cumulative_attributed_learning_cost'],
                cumulative_completion_tokens=per_round[r]['cumulative_completion_tokens'])
            tests.append(record)
            trajectory.append(record)
        if len(trajectory) == len(config['snapshots']):
            summary = dict(**metadata)
            for metric in METRICS:
                summary[metric] = trajectory[-1][metric]
                summary[metric + '_initial'] = trajectory[0][metric]
                summary[metric + '_gain'] = trajectory[-1][metric] - trajectory[0][metric]
                summary[metric + '_auc_per_round'] = sum((b['round'] - a['round']) *
                    (a[metric] + b[metric]) / 2 for a, b in zip(trajectory, trajectory[1:])) / (trajectory[-1]['round'] - trajectory[0]['round'])
                reached = next((x for x in trajectory if x[metric] >= trajectory[0][metric] + .05), None)
                summary[metric + '_reached_5pp'] = reached is not None
                summary[metric + '_first_checkpoint_5pp'] = reached['round'] if reached else None
                summary[metric + '_cost_at_5pp'] = reached['cumulative_attributed_learning_cost'] if reached else None
                common = [x for x in trajectory if x['round'] in allocation.get('comparison_snapshots', config['snapshots'])]
                summary[metric + '_common_auc'] = sum((b['round'] - a['round']) *
                    (a[metric] + b[metric]) / 2 for a, b in zip(common, common[1:])) / (common[-1]['round'] - common[0]['round'])
            summaries.append(summary)
        for i in range(config['num_agents']):
            ledger = directory / 'agents' / str(i) / 'usage.json'
            if not ledger.exists():
                continue
            records = list(json.loads(ledger.read_text()).values())
            for phase in sorted({x.get('phase', 'unknown') for x in records}):
                chosen = [x for x in records if x.get('phase', 'unknown') == phase]
                expenses.append(dict(**metadata, agent=i, phase=phase,
                    reported_cost=sum(x.get('cost_usd') or 0 for x in chosen),
                    completion_tokens=sum((x.get('usage') or {}).get('completion_tokens', 0) or 0 for x in chosen),
                    prompt_tokens=sum((x.get('usage') or {}).get('prompt_tokens', 0) or 0 for x in chosen),
                    total_tokens=sum((x.get('usage') or {}).get('total_tokens', 0) or 0 for x in chosen),
                    unresolved_requests=sum(x.get('cost_usd') is None for x in chosen),
                    conservative_exposure=sum(x['cost_usd'] if x.get('cost_usd') is not None
                                              else x.get('reserved_usd', 0) for x in chosen)))
    paired = []
    index = {(x['model'], x['condition'], x['seed']): x for x in summaries}
    contrasts = [('llm_social_payoff', 'llm_solo'), ('ucb_social', 'openevolve_solo'),
                 ('uniform_social', 'openevolve_solo'), ('ucb_social', 'uniform_social')]
    if allocation.get('study') == 'observation_timing_v1':
        contrasts += [('forced_early', 'forced_distributed'), ('forced_early', 'forced_middle'),
                      ('forced_early', 'forced_late'), ('optional_early', 'llm_social_payoff'),
                      ('optional_middle', 'llm_social_payoff'), ('optional_late', 'llm_social_payoff')]
        contrasts += [(c, 'llm_solo') for c in allocation['planned_conditions']]
    for row in summaries:
        for treatment, control in contrasts:
            other = index.get((row['model'], control, row['seed']))
            if row['condition'] != treatment or other is None:
                continue
            for metric in METRICS:
                paired.append(dict(model=row['model'], seed=row['seed'], contrast=treatment+' minus '+control,
                    metric=metric, final_gain=row[metric]-other[metric],
                    learning_curve_gain=row[metric+'_common_auc']-other[metric+'_common_auc'],
                    baseline_adjusted_learning_curve_gain=(row[metric+'_common_auc']-row[metric+'_initial'])
                        -(other[metric+'_common_auc']-other[metric+'_initial'])))
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [('learning_by_seed', learning), ('test_by_seed', tests), ('final_by_seed', summaries),
                         ('paired_by_seed', paired), ('spending_by_phase', expenses)]:
        write_csv(output / (name + '.csv'), values)
    aggregate = []
    for model, condition, r in sorted({(x['model'], x['condition'], x['round']) for x in tests}):
        values = [x for x in tests if (x['model'], x['condition'], x['round']) == (model, condition, r)]
        for metric in METRICS:
            aggregate.append(dict(model=model, condition=condition, round=r, metric=metric,
                                  **mean_se([x[metric] for x in values])))
    write_csv(output / 'test_summary.csv', aggregate)
    (output / 'PLOT_GUIDE.md').write_text('''# Static social RSI analysis

Each row in the by-seed learning/test files is a five-agent population mean.
Standard errors in test_summary.csv are across independent population seeds.
The initial snapshot supplies the matched frozen-skill reference. Never pool
partially evaluated populations with complete ones. Missing seeds are not zeros.

Plot both test metrics against rounds and cumulative learning tokens/dollars,
with matching model panels, shared axes, and seed-SE bands. The sparse hidden
curve connects the declared test checkpoints; it is not a measured
per-round test trajectory. Area is trapezoidal between these checkpoints.
Five-percentage-point threshold crossings are first observed checkpoints,
not exact crossing times; unreached populations are censored, not discarded.

Training fitness can reuse earlier evaluations and is subject to search
selection. Fresh execution is on training tasks and is not hidden generalization.
The cumulative execution metric excludes initialization and includes failed
answers as zero. Final paired files compare matched seeds, not agent replicates.

Attributed learning cost uses completed-round accounting; spending_by_phase.csv
uses persistent ledgers and also includes retry overhead and unresolved charge
reservations. Hidden-test cost is research overhead, never a learning expense.
Repeated artifact tests can share cached answers: do not sum their copied budget
fields as new charges. No test metric feeds back into learning or skill selection.

For timing interventions, reused_controls are read-only completed predecessor
populations, not new jobs. Their ledger costs are historical, not new spending.
Paired learning_curve_gain uses only the common 0/10/25/49 checkpoints when
declared by the allocation, so denser new curves do not bias baseline comparisons.
The by-seed auc_per_round also retains the full available checkpoint curve.
''')
    print(json.dumps(dict(completed_populations=len(summaries), population_test_points=len(tests))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    analyze(args.root, args.output, args.smoke)
