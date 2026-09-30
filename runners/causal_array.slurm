#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=03:30:00
#SBATCH --time-min=00:45:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Preparation only: this file does not submit any jobs itself.
# Each rung/model/regime has 64 tasks: 4 treatments x 2 social conditions x 8 seeds.
# Example: sbatch --array=0-63 --gpus=1 runners/causal_array.slurm 1 q14 0
set -euo pipefail
rung=${1:?rung 1/2/3 required}
profile=${2:?model profile q14/oss/ministral required}
regime=${3:-0}
mode=full
if [[ $mode != full && $mode != --canary ]]; then echo 'fourth argument must be --canary' >&2; exit 2; fi
task=${SLURM_ARRAY_TASK_ID:?array task required}
if (( task < 0 || task >= 64 )); then echo 'task must be 0-63' >&2; exit 2; fi
case $profile in
 q14) model=Qwen/Qwen3-14B; label=qwen3_14b ;;
 oss) model=openai/gpt-oss-20b; label=gpt_oss_20b ;;
 ministral) model=mistralai/Ministral-3-14B-Reasoning-2512; label=ministral3_14b_reasoning ;;
 *) echo 'unknown model profile' >&2; exit 2 ;;
esac
seed=$((task % 8))
social=$(((task / 8) % 2))
treatment=$((task / 16))
probability=0
if (( treatment % 2 )); then probability=0.15; fi
extra=(--num-agents 10 --budget-mode use_it_or_lose_it --guidance neutral --condition-prefix causal_)
case $rung in
 1)
   config=configs/rung1_solo_llm.json; prompt=prompts/v2_solo.txt
   if (( social )); then config=configs/rung1_social_action_payoff.json; prompt=prompts/v2_social_action_payoff.txt; fi
   reserve=2048; extra+=(--num-arms 25) ;;
 2)
   config=configs/rung2_structured_solo_llm.json; prompt=prompts/v3_rung2_solo.txt
   if (( social )); then config=configs/rung2_structured_social_action_payoff.json; prompt=prompts/v3_rung2_social.txt; fi
   reserve=2048; extra+=(--coordinate-innovation --spatial-length-scale 0.15 --reward-shape 1.0) ;;
 3)
   config=configs/rung3_solo.json; prompt=prompts/v2_rung3_policy.txt
   if (( social )); then config=configs/rung3_social_full.json; fi
   reserve=4096; extra+=(--independent-skill-acquisition)
   if (( regime )); then echo 'rung3 has no payoff shock treatment' >&2; exit 2; fi ;;
 *) echo 'rung must be 1, 2 or 3' >&2; exit 2 ;;
esac
if (( treatment < 2 )); then reserve=0; fi
if (( regime )); then extra+=(--regime-period "$regime"); fi
output_root="runs/causal/rung$rung"
repo_dir=${SLURM_SUBMIT_DIR:?}
cd "$repo_dir"

extra+=(--prompt "$prompt" --output-root "$output_root" --independent-search-probability "$probability" --execution-reserve-tokens "$reserve")
exec bash runners/run_condition.sh "$config" "$seed" "$model" "$label" 1 "${extra[@]}"
