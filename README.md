# solo2social

Can agents that improve on their own tasks also learn effectively from one
another? This repository contains the experiments for **From Solo to Social
Learning: Characterizing Recursive Social Improvement in LLMs**.

Each agent pursues its own reward. It can search independently, learn from a
peer, or use what it already knows. We study these choices first in bandit
environments and then in populations that revise and exchange Markdown skill
files. Model weights stay fixed. We measure reward, learning speed, copying,
and computation—not just final accuracy.

## What's here

| Directory | Contents |
|---|---|
| `code/` | Environments, agents, evolutionary search adapter, API accounting |
| `prompts/` | Fixed bandit and scheduling prompts |
| `skills/` | The fixed scheduling skill library |
| `configs/` | Experiment templates and the offline IFBench configuration |
| `runners/` | Commands for individual runs, paper sweeps, and Slurm |
| `analysis/` | Seed-level statistics, matched-token comparisons, paper figures |
| `tests/` | Offline checks of budgets, copying, evaluation, and checkpoint resume |
| `docs/` | Experiment recipes and implementation details |

Historical file names `rung1`–`rung4` identify finite bandits, infinite bandits,
fixed scheduling skills, and evolving skills, respectively. They are retained
so the code and saved result schemas remain easy to trace. The principal
skill-evolution experiment uses `code/rung4_population.py`.

No logs, runs, model weights, benchmark data, API responses, or paper drafts are
included. Generated data go in ignored `data/`, `runs/`, `logs/`, and `outputs/`
directories. Start from new output directories: this reorganized release is not
a drop-in resume environment for checkpoints made with the private source tree.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Shell launchers require
Bash and `jq`; population jobs also use `flock`.

```bash
uv sync --frozen --extra evolution --extra test
bash runners/python.sh -m pytest -q
```

Add `--extra vllm` when installing for local GPU inference. That optional stack
is pinned to the versions used in the experiments and requires suitable NVIDIA
GPUs and drivers. API-based IFBench experiments need no GPU. OpenEvolve and
IFBench are pinned to exact upstream commits; there is no separate OpenEvolve
fork to install. See [dependencies and adaptations](docs/dependencies.md).

**On a cluster, install, test, analyze, and run experiments inside an allocated
job—not on the login node.** The provided Slurm files contain no account,
partition, private filesystem path, or credentials. Supply your site's values:

```bash
mkdir -p logs
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION runners/setup.slurm
```

Submit from the repository root. `runners/python.sh` sets the import paths and
the fixed Python hash seed used for reproducible OpenEvolve resume. Set
`SOLO2SOCIAL_PYTHON` to use an interpreter outside `.venv`, or
`UV_PROJECT_ENVIRONMENT` to install somewhere with sufficient storage.

## Start with a no-API example

This runs solo UCB on the paper's finite bandit. It does not load an LLM or spend
API credits.

```bash
bash runners/python.sh code/main.py \
  --config configs/rung1_ucb.json --seed 0 \
  --num-agents 10 --num-arms 25 --output-root runs/example
```

On Slurm, replace `bash runners/python.sh` with
`sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION runners/job.slurm`.

## Evolving skills on IFBench

Prepare the fixed split: 100 training examples and 200 held-out examples.
This downloads the pinned IFBench source data and NLTK resources, but
does not call a model.

```bash
bash runners/python.sh runners/prepare_data.py
```

Export `OPENROUTER_KEY` in your shell; never put it in a configuration or commit
it. The paper uses `openai/gpt-oss-120b` and `z-ai/glm-5.3-flash`. Availability,
provider behavior, and pricing can change; the original price ceilings are
recorded in the configurations and requests fail if no eligible provider exists.

The standalone comparison is frozen versus full solo OpenEvolve, with one agent
and a weak initial skill. Specify a spending ceiling for **each** run:

```bash
bash runners/python.sh runners/offline.py \
  --model gpt_oss --policy frozen --seed 0 --budget-usd 2
bash runners/python.sh runners/offline.py \
  --model gpt_oss --policy openevolve_full --seed 0 --budget-usd 6
```

These are example ceilings, not cost estimates or guarantees of completion.
The evolved condition runs 50 rounds. Both conditions evaluate on the same 200
held-out tasks; training optimizes individual-constraint accuracy, and testing
also reports whole-answer accuracy. The same command resumes interrupted work
with unchanged code, configuration, and data.

For the population study, prepare an allocation with a **total** spending cap:

```bash
bash runners/python.sh runners/prepare_population.py \
  --root runs/population --cap-usd 1000
```

Preparation makes no model calls. It defines two models × five conditions × six
seeds, plus ten small smoke populations. Each population has five agents. The
conditions are LLM solo, LLM social, full solo OpenEvolve, UCB social, and uniform
social. Smoke runs use 10 training and 20 test examples. Complete and audit them
before launching the production slots:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --array=0-9 runners/population.slurm runs/population
# After those ten jobs finish:
bash runners/python.sh code/rung4_population.py \
  --root runs/population --audit-smoke
# Only after the audit passes:
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  --array=10-69 runners/population.slurm runs/population
```

Use a cap you are willing to spend. It is divided among runs; an individual
run may stop even when another has unused allowance. Unknown charges retain
conservative reservations. No launcher automatically raises the cap. Keep the
persistent ledgers and caches when resuming. Slurm requeue behavior is
site-dependent; resubmitting the same slot resumes it, and a lock prevents
duplicate writers.

## Reproduce the paper

[Experiment recipes](docs/reproduction.md) list the local-model sweeps,
mechanism interventions, algorithmic controls, copying-time experiments, and
analysis order. [The figure guide](analysis/PLOT_GUIDE.md) explains what the
plots measure.

The unit of uncertainty is an independent population seed, not an individual
agent. Matched-cost comparisons interpolate each pair's observed performance
against total population expenditure; reward divided by tokens is a different
metric. Tests never feed back into skill evolution. Peer lists retain the fixed
ordering used in the paper—this release does not silently change that design.
