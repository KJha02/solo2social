#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

if [[ -z ${SLURM_ARRAY_TASK_ID:-} ]]; then
    echo "This script must run as a Slurm array." >&2
    exit 2
fi
configs=(configs/rung1_ucb.json configs/rung1_oracle.json)
task=$SLURM_ARRAY_TASK_ID
extra=()
if [[ ${1:-} == v3 ]]; then
    v3_configs=(
        configs/rung1_ucb.json configs/rung1_oracle.json
        configs/rung1_ucb.json configs/rung1_oracle.json
        configs/rung1_ucb.json configs/rung1_oracle.json
    )
    v3_regimes=(0 0 25 25 10 10)
    condition_index=$((task / 8))
    seed=$((task % 8))
    config=${v3_configs[$condition_index]}
    extra=(
        --output-root runs/v3/rung1 --condition-prefix v3_
        --num-agents 10 --num-arms 25
    )
    regime=${v3_regimes[$condition_index]}
    if (( regime > 0 )); then extra+=(--regime-period "$regime"); fi
else
    echo 'Use the paper configuration: v3' >&2
    exit 2
fi
(( task >= 0 && task < 48 )) || exit 2
repo_dir=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}
test -f "$repo_dir/code/main.py"
exec "${SOLO2SOCIAL_PYTHON:-$repo_dir/.venv/bin/python}" "$repo_dir/code/main.py" \
    --config "$repo_dir/$config" --seed "$seed" "${extra[@]}"
