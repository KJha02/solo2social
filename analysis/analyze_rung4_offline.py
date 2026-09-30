"""Summarize both saved IFBench test metrics for matched offline seeds; run in Slurm."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(values):
    return dict(mean=statistics.mean(values),
                se=statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None)


def analyze(roots):
    records, summaries, pending = [], [], []
    for root in roots:
        pairs = []
        for frozen in sorted((root / 'minimal_frozen').glob('seed_*')):
            evolved = root / 'minimal_openevolve_full' / frozen.name
            config = json.loads((evolved / 'config.json').read_text())
            final = evolved / 'rounds' / f"{config['rounds'] - 1:04d}.json"
            if not final.exists():
                pending.append(str(evolved))
                continue
            versions = [json.loads((frozen / 'rounds/0000.json').read_text())['results'][0]['version'],
                        json.loads(final.read_text())['results'][0]['version']]
            selections = [json.loads((p / 'evaluation_selection.json').read_text())['task_ids']
                          for p in (frozen, evolved)]
            assert selections[0] == selections[1]
            expected = {(task, repeat) for task in selections[0]
                        for repeat in range(config.get('holdout_repeats', 2))}
            scores = []
            identities = []
            for directory, version in zip((frozen, evolved), versions):
                rows = [json.loads(p.read_text()) for p in (directory / 'holdout').glob(f'{version}-*.json')]
                keyed = {(r['identity']['task'], r['identity']['repeat']): r for r in rows}
                assert len(keyed) == len(rows), 'Duplicate test results'
                if not expected <= keyed.keys():
                    break
                rows = [keyed[k] for k in sorted(expected)]
                identities.append([{k: r['identity'][k] for k in
                    ('task', 'repeat', 'execution_seed', 'model', 'budget', 'dataset')} for r in rows])
                # Whole-run spending ceilings differ between frozen and search;
                # neither changes the matched per-answer execution budget.
                for identity in identities[-1]:
                    identity['model'] = {k: v for k, v in identity['model'].items()
                                         if k not in ('run_cost_limit_usd', 'total_cost_limit_usd')}
                scores.append({metric: statistics.mean(r['result'][metric] for r in rows)
                               for metric in ('prompt_level_loose', 'constraint_accuracy')})
            if len(scores) != 2:
                pending.append(str(evolved))
                continue
            assert identities[0] == identities[1], 'Unmatched evaluator or test execution seeds'
            row = dict(model=config['model']['name'], seed=config['seed'], tasks=len(selections[0]),
                       frozen=scores[0], evolved=scores[1],
                       gain={m: scores[1][m] - scores[0][m] for m in scores[0]})
            records.append(row)
            pairs.append(row)
        if pairs:
            summary = dict(model=pairs[0]['model'], seeds=len(pairs), metrics={
                metric: {group: summarize([p[group][metric] for p in pairs])
                         for group in ('frozen', 'evolved', 'gain')}
                for metric in ('prompt_level_loose', 'constraint_accuracy')})
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    return dict(summary=summaries, by_seed=records, incomplete=pending,
                constraint_definition='Mean per-task fraction of loose constraints satisfied; tasks weighted equally.',
                prompt_definition='Fraction of tasks satisfying every loose constraint.',
                uncertainty='Standard error across independent seeds; gains paired within seed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path, help='IFBench directories containing both policies')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
