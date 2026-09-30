#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
case ${1:-} in
    q14) model=Qwen/Qwen3-14B; label=qwen3_14b ;;
    oss) model=openai/gpt-oss-20b; label=gpt_oss_20b ;;
    ministral) model=mistralai/Ministral-3-14B-Reasoning-2512; label=ministral3_14b_reasoning ;;
    *) echo 'Specify q14, oss, or ministral' >&2; exit 2 ;;
esac
task=${SLURM_ARRAY_TASK_ID:?Submit array 0-15}
(( task >= 0 && task < 16 )) || exit 2
mode=solo
if (( task >= 8 )); then mode=social; fi
root=runs/v3_deoe/rung3
mkdir -p "$root/configs"
config="$root/configs/${label}_${mode}_${task}.json"
temporary=$(mktemp "$root/configs/prepare-XXXXXXXX.json")
trap 'rm -f -- "$temporary"' EXIT
jq --arg mode "$mode" --arg root "$root" '
  .condition=("rung3_" + $mode) | .strategy=("deoe_" + $mode) |
  .policy=($mode + "_deoe") | .social_info=(if $mode == "solo" then "none" else "full" end) |
  .output_root=$root | .independent_skill_acquisition=true |
  .environment.num_agents=10 | .budget={tokens_per_round:8192,carry_over:false} |
  .prompt="prompts/v2_rung3_policy.txt" | .guidance="neutral"
' configs/rung3_solo.json > "$temporary"
if [[ -f $config ]]; then cmp --silent "$temporary" "$config"; else cp "$temporary" "$config"; fi
bash runners/run_condition.sh "$config" "$((task % 8))" "$model" "$label" 1 --condition-prefix v3_deoe_
