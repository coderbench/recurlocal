#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-./build/recur_local_cuda_bench}"
python3 eval/run_eval.py --binary "${BIN}" --output eval-result.json
