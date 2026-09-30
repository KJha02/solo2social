"""Cheap release checks: syntax, paths, split provenance, and safe preparation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_parse():
    for directory in ('code', 'analysis', 'runners'):
        for path in (ROOT / directory).glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))


def test_prompt_paths_exist():
    for path in (ROOT / 'configs').glob('*.json'):
        config = json.loads(path.read_text())
        if config.get('prompt'):
            assert (ROOT / config['prompt']).is_file(), path


def test_pinned_ifbench_source():
    import ifbench
    assert hashlib.sha256(ifbench.data_path().read_bytes()).hexdigest() == (
        'a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85')


def test_public_spending_allocation(tmp_path):
    output = tmp_path / 'population'
    args = [sys.executable, str(ROOT / 'runners/prepare_population.py'),
            '--root', str(output), '--cap-usd', '40']
    subprocess.run(args, check=True, capture_output=True)
    allocation = json.loads((output / 'allocation.json').read_text())
    assert len(allocation['slots']) == 70
    assert abs(sum(s['config']['num_agents'] * s['config']['model']['total_cost_limit_usd']
                   for s in allocation['slots']) - 40) < 1e-8
    assert all(s['config']['smoke_train_tasks'] == 10 and s['config']['smoke_holdout_tasks'] == 20
               for s in allocation['slots'] if s['config']['smoke'])
    subprocess.run(args, check=True, capture_output=True)
    assert not list(output.rglob('usage.json'))  # Preparation cannot spend money.
