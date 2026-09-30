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
    echo "usage: sbatch --array=0-207 runners/rung2_array.slurm PROFILE v3" >&2
    exit 2
fi
configs=(
    configs/rung2_unstructured_social_action.json
    configs/rung2_unstructured_social_action_payoff.json
    configs/rung2_unstructured_solo_llm.json
    configs/rung2_structured_social_action.json
    configs/rung2_structured_social_action_payoff.json
    configs/rung2_structured_solo_llm.json
)
task=$SLURM_ARRAY_TASK_ID
extra=()
if [[ ${2:-} == v3 ]]; then
    u_solo=configs/rung2_unstructured_solo_llm.json
    u_social=configs/rung2_unstructured_social_action_payoff.json
    s_solo=configs/rung2_structured_solo_llm.json
    s_social=configs/rung2_structured_social_action_payoff.json
    s_action=configs/rung2_structured_social_action.json
    v3_configs=(
        "$u_solo" "$u_social" "$u_social" "$s_solo" "$s_social" "$s_social"
        "$s_social" "$s_action"
        "$u_solo" "$u_social" "$u_solo" "$u_social"
        "$s_solo" "$s_social" "$s_solo" "$s_social"
        "$s_solo" "$s_social" "$s_social" "$s_solo" "$s_social" "$s_social"
        "$u_solo" "$u_social" "$s_solo" "$s_social"
    )
    v3_prompts=(
        prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt prompts/v3_rung2_social.txt
        prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt prompts/v3_rung2_social.txt
        prompts/v3_rung2_social.txt prompts/v3_rung2_social.txt
        prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt
        prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt
        prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt prompts/v3_rung2_social.txt
        prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt prompts/v3_rung2_social.txt
        prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt prompts/v3_rung2_solo.txt prompts/v3_rung2_social.txt
    )
    v3_budgets=(use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it use_it_or_lose_it carry carry carry carry)
    v3_guidance=(neutral neutral strategic neutral neutral strategic deliberate neutral neutral neutral neutral neutral neutral neutral neutral neutral neutral neutral strategic neutral neutral strategic neutral neutral neutral neutral)
    v3_shapes=(1 1 1 1 1 1 1 1 0.6 0.6 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
    v3_correlations=(0 0 0 0.15 0.15 0.15 0.15 0.15 0 0 0 0 0.05 0.05 0.30 0.30 0.15 0.15 0.15 0.15 0.15 0.15 0 0 0.15 0.15)
    v3_regimes=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 25 25 25 10 10 10 0 0 0 0)
    condition_index=$((task / 8))
    seed=$((task % 8))
    config=${v3_configs[$condition_index]}
    prompt=${v3_prompts[$condition_index]}
    extra=(
        --output-root runs/v3/rung2 --prompt "$prompt"
        --condition-prefix v3_ --num-agents 10 --coordinate-innovation
        --budget-mode "${v3_budgets[$condition_index]}"
        --guidance "${v3_guidance[$condition_index]}"
        --reward-shape "${v3_shapes[$condition_index]}"
    )
    correlation=${v3_correlations[$condition_index]}
    if [[ $correlation != 0 ]]; then extra+=(--spatial-length-scale "$correlation"); fi
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
