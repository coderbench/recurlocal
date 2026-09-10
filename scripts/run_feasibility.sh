#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-./build/tensortransit_bench}"
python3 eval/run_eval.py --binary "${BIN}" --output eval-result.json
