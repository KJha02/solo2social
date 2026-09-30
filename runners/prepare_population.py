"""Create an immutable, explicitly budgeted paper sweep. Does not submit jobs."""
import argparse
import json
import math
import tempfile
from pathlib import Path
from rung4_population import prepare
from rung4 import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/population"))
    parser.add_argument("--cap-usd", type=float, required=True,
                        help="Total ceiling across both models, all seeds, smoke, and hidden tests")
    args = parser.parse_args()
    if not math.isfinite(args.cap_usd) or args.cap_usd <= 0:
        parser.error("Provide a finite positive spending ceiling")
    with tempfile.TemporaryDirectory() as temporary:
        prepare(Path(temporary))
        allocation = json.loads((Path(temporary) / "allocation.json").read_text())
    scale = args.cap_usd / allocation["cap_usd"]
    for slot in allocation["slots"]:
        for field in ("run_cost_limit_usd", "total_cost_limit_usd"):
            slot["config"]["model"][field] *= scale
    allocation["cap_usd"] = args.cap_usd
    path = args.root / "allocation.json"
    if path.exists():
        if json.loads(path.read_text()) != allocation:
            raise ValueError("Existing allocation differs; never overwrite a started experiment")
    else:
        atomic_json(path, allocation)
    for name, smoke in (("smoke", True), ("production", False)):
        indices = [str(i) for i, s in enumerate(allocation["slots"]) if s["config"]["smoke"] == smoke]
        print(f"{name} array: {','.join(indices)}")
    print(f"Prepared {path}; total ceiling ${args.cap_usd:.2f}. No jobs submitted.")


if __name__ == "__main__":
    main()
