#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

if [[ ($# -ne 1 && $# -ne 2) || -z ${SLURM_ARRAY_TASK_ID:-} ]]; then
    echo "usage: sbatch --array=0-111 runners/rung1_array.slurm PROFILE v3" >&2
    exit 2
fi

configs=(
    configs/rung1_social_action.json
    configs/rung1_social_action_payoff.json
    configs/rung1_solo_llm.json
)
task=$SLURM_ARRAY_TASK_ID
(( task >= 0 && task < 112 )) || { echo 'Array index must be 0–111' >&2; exit 2; }
extra=()
if [[ ${2:-} == v3 ]]; then
    v3_configs=(
        configs/rung1_solo_llm.json configs/rung1_solo_llm.json
        configs/rung1_social_action.json configs/rung1_social_action.json
        configs/rung1_social_action_payoff.json configs/rung1_social_action_payoff.json
        configs/rung1_social_action_payoff.json configs/rung1_social_action_payoff.json
        configs/rung1_solo_llm.json configs/rung1_social_action_payoff.json
        configs/rung1_social_action_payoff.json configs/rung1_solo_llm.json
        configs/rung1_social_action_payoff.json configs/rung1_social_action_payoff.json
    )
    v3_prompts=(
        prompts/v2_solo.txt prompts/v2_solo.txt
        prompts/v2_social_action_only.txt prompts/v2_social_action_only.txt
        prompts/v2_social_action_payoff.txt prompts/v2_social_action_payoff.txt
        prompts/v2_social_action_payoff.txt prompts/v2_social_action_payoff.txt
        prompts/v2_solo.txt prompts/v2_social_action_payoff.txt
        prompts/v2_social_action_payoff.txt prompts/v2_solo.txt
        prompts/v2_social_action_payoff.txt prompts/v2_social_action_payoff.txt
    )
    v3_budgets=(carry use_it_or_lose_it carry use_it_or_lose_it carry use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it)
    v3_guidance=(neutral neutral neutral neutral neutral neutral deliberate strategic neutral neutral strategic neutral neutral strategic)
    v3_regimes=(0 0 0 0 0 0 0 0 25 25 25 10 10 10)
    condition_index=$((task / 8))
    seed=$((task % 8))
    config=${v3_configs[$condition_index]}
    prompt=${v3_prompts[$condition_index]}
    budget_mode=${v3_budgets[$condition_index]}
    extra=(
        --output-root runs/v3/rung1 --prompt "$prompt"
        --condition-prefix v3_ --num-agents 10 --num-arms 25
        --budget-mode "$budget_mode" --guidance "${v3_guidance[$condition_index]}"
    )
    regime=${v3_regimes[$condition_index]}
    if (( regime > 0 )); then extra+=(--regime-period "$regime"); fi
else
    echo "Use the paper configuration: v3" >&2; exit 2
fi

case $1 in
    q14) model=Qwen/Qwen3-14B; label=qwen3_14b; tp=1 ;;
    oss) model=openai/gpt-oss-20b; label=gpt_oss_20b; tp=1 ;;
    ministral) model=mistralai/Ministral-3-14B-Reasoning-2512; label=ministral3_14b_reasoning; tp=1 ;;
    *) echo "unknown model profile: $1" >&2; exit 2 ;;
esac

repo_dir=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}
test -f "$repo_dir/code/main.py"
exec "$repo_dir/runners/run_condition.sh" \
    "$config" "$seed" "$model" "$label" "$tp" "${extra[@]}"
