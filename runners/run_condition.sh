#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=${1:?CONFIG required}
seed=${2:?SEED required}
shift 2
args=(--config "$config" --seed "$seed")
if (( $# >= 3 )); then
    args+=(--model "$1" --model-label "$2" --tensor-parallel-size "$3")
    shift 3
fi
exec bash "$repo_dir/runners/python.sh" code/main.py "${args[@]}" "$@"
