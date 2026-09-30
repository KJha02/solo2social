# Reproducing the experiments

Run commands from the repository root with `bash runners/python.sh`, or submit
them with `runners/job.slurm` on a CPU/GPU allocation. None of the preparation
commands submits experiments implicitly. Slurm examples deliberately omit your
account and partition; supply them with `sbatch`.

## Finite bandits, infinite bandits, and fixed scheduling skills

The paper uses Qwen3-14B, Ministral-3-14B-Reasoning-2512, and GPT-OSS-20B.
The launcher profiles are `q14`, `ministral`, and `oss`.
All use eight population seeds (0–7), ten agents, and 100 rounds.
Finite bandits have 25 arms. Main comparisons are stationary, neutral-prompt,
expiring-budget solo versus action-plus-payoff social access.

Install the vLLM extra and make the model weights accessible to the compute
nodes. Choose GPU resources appropriate to the model; the paper used one
supported GPU per run. These arrays reproduce the final V3 experiment design,
including the supporting conditions—not an older V1/V2 sweep:

```bash
# Repeat these for q14, ministral, and oss.
sbatch --gpus=1 --array=0-111 runners/rung1_array.slurm q14 v3
sbatch --gpus=1 --array=0-207 runners/rung2_array.slurm q14 v3
sbatch --gpus=1 --array=0-39 runners/rung3_array.slurm q14 v3

# Algorithmic finite-bandit solo UCB and oracle: no GPU or model calls.
sbatch --array=0-47 runners/rung1_baseline_array.slurm v3
```

Supply your account, partition, GPU constraint, memory, and wall time at submission.
The first three arrays contain 14, 26, and 5 conditions per model, respectively.
Array index = 8 × condition index + seed. The arrays preserve the paper's
model-specific sampling settings. The configuration templates alone do not
encode the full paper sweep: the launcher overrides population size, prompt,
budget behavior, and experimental condition.

Protected execution (PE) and independent search (IS) form a 2×2 intervention
design applied to both solo and social populations. In this array, indices
0–15 are default, 16–31 IS, 32–47 PE, and 48–63 both; each block contains solo
seeds 0–7 followed by social seeds 0–7.

```bash
# Rung number, model profile, payoff-change period (0 = stationary).
sbatch --gpus=1 --array=0-63 runners/causal_array.slurm 1 q14 0
sbatch --gpus=1 --array=0-63 runners/causal_array.slurm 2 q14 0
sbatch --gpus=1 --array=0-63 runners/causal_array.slurm 3 q14 0
```

Repeat for the three profiles. For finite/infinite bandits, periods 10 and 25
give the volatility controls. Fixed-skill acquisition and fixed-selection replay:

```bash
sbatch --gpus=1 --array=0-31 runners/rung3_array.slurm q14 v3_acquisition
# Run after the matched acquisition populations finish.
sbatch --gpus=1 --array=0-31 runners/rung3_array.slurm q14 v3_replay
sbatch --gpus=1 --array=0-15 runners/rung3_deoe.slurm q14
```

Repeat each for the three profiles. Replay fixes the selection trajectory from
the acquisition run and reevaluates execution; it is not a new independent seed.
The DEOE hybrid selects skills algorithmically but uses the same LLM executor.

Analyze saved populations, then run the algorithmic social baselines on those
same environments:

```bash
bash runners/python.sh analysis/analysis.py --rung 1 --runs-root runs/v3/rung1
bash runners/python.sh analysis/analysis.py --rung 2 --runs-root runs/v3/rung2
bash runners/python.sh analysis/analysis.py --rung 3 --runs-root runs/v3/rung3
bash runners/python.sh analysis/analysis.py --rung 3 --runs-root runs/v3_acquisition/rung3
bash runners/python.sh analysis/analysis.py --rung 3 --runs-root runs/v3_replay/rung3
bash runners/python.sh code/social_baseline.py
bash runners/python.sh analysis/analyze_social_baseline.py
bash runners/python.sh analysis/analyze_efficiency.py runs/causal/rung1 \
  --output runs/paper_analysis/analysis/causal_rung1 --target 4
bash runners/python.sh analysis/analyze_efficiency.py runs/causal/rung2 \
  --output runs/paper_analysis/analysis/causal_rung2 --target 4
bash runners/python.sh analysis/analyze_efficiency.py runs/causal/rung3 \
  --output runs/paper_analysis/analysis/causal_rung3 --target .9
```

`--target` sets a threshold for the optional time-to-target diagnostic; it does
not change the reward or intervention contrasts. DEOE is inspired by the
discountmachine strategy, not an exact reproduction or an optimality claim.
See [algorithmic baselines](algorithmic_baselines.md).
The paper plotting pipeline reads the DEOE+LLM scheduling summaries directly.
The general analysis commands above expect complete rung-level sweeps; for a
single example run, inspect its `summary.json` rather than invoking the full
paper analysis.

## Solo skill evolution

The README gives single-run commands. Repeat weak-skill frozen and
`openevolve_full` for seeds 0–7 and models `gpt_oss` and `glm`.
Each run has one agent. Training uses all 100 fixed examples in parallel;
the initial and evolved skills are tested on the same 200 held-out examples.
Use the same code, model configuration, split, and per-answer caps for each
paired seed. `--initial-skill strong` retains the stronger-skill diagnostic.

```bash
bash runners/python.sh analysis/analyze_rung4_offline.py \
  runs/offline/gpt_oss/ifbench runs/offline/glm/ifbench \
  --output outputs/offline_summary.json
bash runners/python.sh analysis/plot_short_report.py --paper-offline
```

## Social skill evolution and copying time

The README prepares the corrected 60-population experiment, with 50 rounds
including initialization and held-out checkpoints at 0, 10, 25, and 49.
Every condition uses a weak initial skill. Train feedback scores individual
constraints. Test reports both that score and the stricter whole-answer score.

The source UCB and uniform controls choose between self (private revision)
and a peer (acquire the peer's previous skill and training score). Both use the
OpenEvolve archive for lower-level skill selection. The uniform control changes
both self-versus-social frequency and peer identity; it is not a peer-only
randomization with a fixed copying schedule.

After all baseline populations have completed, prepare the paper's timing study:

```bash
bash runners/python.sh code/rung4_population.py --root runs/timing \
  --prepare-timing --prior-root runs/population --timing-cap 400
```

This prepares six small smoke slots followed by 36 production slots:
two models × three timing conditions × six seeds. Conditions are four forced
early observations, four forced distributed observations, and optional early
access. Matched completed solo and unrestricted-social controls are reused
read-only, not paid for again. This cap is for the new timing jobs.

```bash
sbatch --array=0-5 runners/population.slurm runs/timing
# After smoke finishes:
bash runners/python.sh code/rung4_population.py --root runs/timing --audit-smoke
sbatch --array=6-41 runners/population.slurm runs/timing
```

Add your cluster options. The runner requires completed controls with the same
protocol and data. `--full-timing` prepares additional middle/late-window
conditions; those are optional extensions, not required to reproduce the paper.

## Analysis and figures

The analysis keeps original schemas and run identifiers so existing readers can
be traced to the experimental code. All paths are repository-relative.

```bash
bash runners/python.sh analysis/analyze_rung4_population.py \
  --root runs/population --output runs/population/analysis
bash runners/python.sh analysis/analyze_copy_behavior.py
bash runners/python.sh analysis/analyze_observation_timing.py
bash runners/python.sh analysis/analyze_rung4_population.py \
  --root runs/timing --output runs/timing/analysis
bash runners/python.sh runners/analyze_paper.py
```

The final command expects the completed paper sweeps above and builds the
bandit and skill figures, bootstrap audits, lineages, intervention summaries,
and matched-token contrasts. It fails if required inputs are missing rather
than silently substituting published numbers. Full analysis is CPU-only but
reads many event files; allocate enough memory and time. Individual analysis
scripts can be used without running the entire paper pipeline.

A historical original-versus-corrected-loop plot is generated only if its
separate original-loop analysis is supplied under `runs/original_population/analysis`.
Deprecated training code and original runs are not bundled. Corrected results
never depend on those historical data.

## Earlier coding-task diagnostics

`code/rung4.py` and `code/rung4_benchmark.py` retain the coding executor
and frozen/solo/social skill-file interfaces used during the study. They are
not necessary for the main IFBench experiment. Running generated code requires
Apptainer with network isolation and the official LiveCodeBench checkout at
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.

Set `SOLO2SOCIAL_DEPS` to a directory containing `LiveCodeBench/` and
`lcb_runtime/`. The latter must contain `python311.sif`, a Python-3.11-compatible
`python/` dependency directory with NumPy 2.2.6, and `test6.jsonl` from
`livecodebench/code_generation_lite` revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.
The scorer uses official public and private tests; never expose the private
tests in agent-visible prompts. Run the sandbox diagnostic before inference:

```bash
bash runners/python.sh code/rung4_benchmark.py --diagnostic
```

No SIF image, model weights, or downloaded benchmark data are distributed here.
The container protects the host; the upstream in-process checker is not an
adversarially secure evaluator against code deliberately inspecting its grader.
