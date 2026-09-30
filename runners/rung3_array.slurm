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

if [[ ($# -ne 1 && $# -ne 2) || -z ${SLURM_ARRAY_TASK_ID:-} ]]; then
    echo "usage: sbatch --array=... runners/rung3_array.slurm PROFILE v3|v3_acquisition|v3_replay" >&2
    exit 2
fi
configs=(
    configs/rung3_solo.json
    configs/rung3_social_id.json
    configs/rung3_social_payoff.json
    configs/rung3_social_reputation.json
    configs/rung3_social_full.json
    configs/rung3_skill_oracle.json
)
task=$SLURM_ARRAY_TASK_ID
extra=()
if [[ ${2:-} == v3_replay || ${2:-} == v3_replay_canary ]]; then
    if (( task < 0 || task >= 32 )); then echo "replay task must be 0-31" >&2; exit 2; fi
    if [[ $2 == v3_replay_canary && $task -ne 0 ]]; then echo "replay canary must use task 0" >&2; exit 2; fi
    condition_index=$((task / 8))
    seed=$((task % 8))
    replay_conditions=(
        v3_acquisition_rung3_solo__independent_acquisition__use_it_or_lose_it
        v3_acquisition_rung3_social_id__independent_acquisition__use_it_or_lose_it
        v3_acquisition_rung3_social_full__independent_acquisition__use_it_or_lose_it
        v3_acquisition_rung3_social_full__independent_acquisition__use_it_or_lose_it__strategic
    )
    replay_source_condition=${replay_conditions[$condition_index]}
elif [[ ${2:-} == v3 || ${2:-} == v3_acquisition ]]; then
    v3_configs=(
        configs/rung3_solo.json
        configs/rung3_social_id.json
        configs/rung3_social_full.json
        configs/rung3_social_full.json
        configs/rung3_skill_oracle.json
    )
    v3_guidance=(neutral neutral neutral strategic neutral)
    condition_index=$((task / 8))
    config=${v3_configs[$condition_index]}
    seed=$((task % 8))
    extra=(
        --output-root runs/v3/rung3 --prompt prompts/v2_rung3_policy.txt
        --condition-prefix v3_ --budget-mode use_it_or_lose_it
        --num-agents 10 --guidance "${v3_guidance[$condition_index]}"
    )
    if [[ $2 == v3_acquisition ]]; then
        if (( task < 0 || task >= 32 )); then echo "acquisition task must be 0-31" >&2; exit 2; fi
        extra+=(--output-root runs/v3_acquisition/rung3 --condition-prefix v3_acquisition_ --independent-skill-acquisition)
    fi
else
    echo "Use v3, v3_acquisition, or v3_replay" >&2; exit 2
fi

case $1 in
    q14) model=Qwen/Qwen3-14B; label=qwen3_14b; tp=1 ;;
          oss) model=openai/gpt-oss-20b; label=gpt_oss_20b; tp=1 ;;
    ministral) model=mistralai/Ministral-3-14B-Reasoning-2512; label=ministral3_14b_reasoning; tp=1 ;;
    *) echo "unknown model profile: $1" >&2; exit 2 ;;
esac

repo_dir=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}
test -f "$repo_dir/code/main.py"
if [[ ${2:-} == v3_replay || ${2:-} == v3_replay_canary ]]; then
    cd "$repo_dir"
    guard_root=runs/v3_replay/rung3
    root=$guard_root
    rounds=100
    condition_prefix=rung3_replay__
    if [[ $2 == v3_replay_canary ]]; then
        root=runs/canary/rung3_replay_sweep
        rounds=3
        condition_prefix=rung3_replay_canary__
    fi
    source_run="$repo_dir/runs/v3_acquisition/rung3/${replay_source_condition}__${label}/seed_${seed}"
    test "$(jq -r '.completed' "$source_run/summary.json")" = true
    test "$(jq -r '.quality_gate_passed' "$source_run/status.json")" = true
    if [[ -f "$guard_root/SOURCE.sha256" ]]; then
        sha256sum --check --quiet "$guard_root/SOURCE.sha256"
    fi
    mkdir -p "$root/configs"
    config="$root/configs/${label}_${task}.json"
    temporary=$(mktemp "$root/configs/${label}_${task}.json.XXXXXX")
    jq --arg source "$source_run" --arg output "$root" \
       --arg prefix "$condition_prefix" --argjson rounds "$rounds" '
      .condition = ($prefix + .condition) |
      .strategy = "skill_replay" | .policy = "skill_replay" |
      .social_info = "none" | .source_run = $source | .output_root = $output |
      .environment.rounds = $rounds |
      .budget = {tokens_per_round: 8192, carry_over: false} |
      .execution_replicate = 0
    ' "$source_run/config.json" > "$temporary"
    if [[ -f "$config" ]] && ! cmp --silent "$temporary" "$config"; then
        echo "prepared replay configuration changed: $config" >&2
        exit 1
    fi
    mv "$temporary" "$config"
    exec "$repo_dir/runners/run_condition.sh" "$config" "$seed"
fi
exec "$repo_dir/runners/run_condition.sh" \
    "$config" "$seed" "$model" "$label" "$tp" "${extra[@]}"
