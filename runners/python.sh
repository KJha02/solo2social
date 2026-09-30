#!/usr/bin/env bash
# Run from any directory while keeping data and configuration paths consistent.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
export PYTHONPATH="$repo_dir/code:$repo_dir/analysis${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export NLTK_DATA="${NLTK_DATA:-$repo_dir/data/nltk_data}"
export MPLBACKEND=Agg
exec "${SOLO2SOCIAL_PYTHON:-$repo_dir/.venv/bin/python}" "$@"
