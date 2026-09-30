#!/usr/bin/env bash
#SBATCH --job-name=solo2social-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Pass --extra vllm as additional arguments for local GPU inference.
uv sync --extra evolution --extra test "$@"
bash runners/python.sh -m pytest -q
