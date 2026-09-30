"""Matched frozen versus full OpenEvolve on the static 100/200 IFBench split."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import rung4
import rung4_instruction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gpt_oss", "glm"), required=True)
    parser.add_argument("--policy", choices=("frozen", "openevolve_full"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-skill", choices=("minimal", "strong"), default="minimal")
    parser.add_argument("--budget-usd", type=float, required=True,
                        help="Explicit ceiling for this run, including hidden evaluation")
    args = parser.parse_args()
    if args.budget_usd <= 0:
        parser.error("--budget-usd must be positive")
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/offline_ifbench.json").read_text())
    config.update(seed=args.seed, initial_skill=args.initial_skill, policy=args.policy)
    if args.policy == "frozen":
        config.update(rounds=1, holdout_rounds=[0])
    if args.model == "glm":
        config["model"]["name"] = "z-ai/glm-5.3-flash"
        config["model"]["request_options"]["provider"]["max_price"] = {"prompt": .15, "completion": .5}
    config["model"].update(run_cost_limit_usd=args.budget_usd, total_cost_limit_usd=args.budget_usd)
    output = root / "runs/offline" / args.model / "ifbench" / f"{args.initial_skill}_{args.policy}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    # Keep every seed/condition's persistent spend ledger separate.
    os.environ['AGENT_MARKET_USAGE_LEDGER'] = str(output / 'usage.json')
    tasks = json.loads((root / config["dataset"]).read_text())["tasks"]
    selection = {"task_ids": [t["id"] for t in tasks if t["split"] == "holdout"],
                 "training_examples": 100, "rule": "all 200 fixed held-out tasks"}
    if len(selection["task_ids"]) != 200:
        raise ValueError("Expected the paper's 100/200 split")
    rung4.atomic_json(output / "evaluation_selection.json", selection)
    prepared = output / 'requested_config.json'
    if prepared.exists() and json.loads(prepared.read_text()) != config:
        raise ValueError('Refusing changed configuration on resume')
    rung4.atomic_json(prepared, config)
    command = [sys.executable, str(root / 'code/rung4_instruction.py'),
               '--config', str(prepared), '--output', str(output)]
    subprocess.run(command, check=True)
    subprocess.run(command + ['--replay'], check=True)


if __name__ == "__main__":
    main()
