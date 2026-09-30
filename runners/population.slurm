#!/usr/bin/env bash
#SBATCH --job-name=solo2social-population
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${OPENROUTER_KEY:?Export OPENROUTER_KEY before submission}"
root=${1:?Allocation directory required}
slot=${SLURM_ARRAY_TASK_ID:?Submit a job array}
label=$(jq -er --argjson slot "$slot" '.slots[$slot].label' "$root/allocation.json")
if [[ $(jq -r --argjson slot "$slot" '.slots[$slot].config.smoke' "$root/allocation.json") == false ]]; then
    expected=$(jq -r '.expected_smoke_conditions // 10' "$root/allocation.json")
    jq -e --argjson expected "$expected" '.passed == true and (.conditions | length == $expected)' "$root/smoke_audit.json" >/dev/null
fi
mkdir -p "$root/$label"
exec 9>"$root/$label/run.lock"
flock -n 9 || { echo 'This population already has a writer.' >&2; exit 1; }
# Re-running the identical command resumes saved rounds and paid-task caches.
# Budgets are never raised automatically.
exec bash runners/python.sh code/rung4_population.py --root "$root" --slot "$slot"
