#!/usr/bin/env bash
# Everything a contributor should run before opening a PR, in one command.
#
#   scripts/check.sh            CPU only -- the planners, the graph, the scorer, the schemas
#   scripts/check.sh --cuda     also builds the CUDA half and runs the device tests
#   scripts/check.sh --cuda --sanitize   ...and compute-sanitizer over both engines
#
# It exists because the pre-PR list was four commands in three documents, and a list like that
# is one people run three quarters of.
set -euo pipefail

CUDA=0
SANITIZE=0
for arg in "$@"; do
    case "$arg" in
        --cuda)     CUDA=1 ;;
        --sanitize) CUDA=1; SANITIZE=1 ;;
        -h|--help)  sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "!! unknown option $arg"; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BUILD="build-check"
JOBS="$(nproc 2>/dev/null || echo 4)"

step() { printf '\n\033[1m>> %s\033[0m\n' "$*"; }

step "build (CUDA=$CUDA)"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DTENSORTRANSIT_BUILD_CUDA="$([ "$CUDA" = 1 ] && echo ON || echo OFF)" \
      ${CMAKE_CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES"} \
      > /dev/null
cmake --build "$BUILD" -j "$JOBS" > /dev/null

step "tests"
# Planners, the graph, the plan schema, the compat shim, the frontier scorer, the frontier CLI,
# the overhead budget -- and, with --cuda, the controller and executor contract tests.
ctest --test-dir "$BUILD" --output-on-failure

step "the cost model still fits the measurements it claims to"
python3 eval/cost_model_fit.py > /dev/null && echo "   residency beats linear on the arms in results/"

step "every frozen generation still loads and still hashes"
for gen in eval/generations/*/; do
    name="$(basename "$gen")"
    python3 tools/tt-frontier generation show "$name" | sed -n '1,2p'
done

step "the manifest still matches the repository"
python3 scripts/manifest.py --check

step "the ledger audits clean"
for gen in frontier/*/; do
    name="$(basename "$gen")"
    [ "$name" = "README.md" ] && continue
    python3 tools/tt-frontier ledger audit "$name" > /dev/null && echo "   $name ok"
done

if [ "$SANITIZE" = 1 ]; then
    step "compute-sanitizer, both engines"
    bash scripts/sanitize.sh "$BUILD"
fi

cat <<'DONE'

>> all checks passed.

Before opening a PR, the two things this script cannot do for you:

  1. Say what you measured, and with what. A performance PR carries the baseline commit, the
     GPU/driver/CUDA versions, the exact command, and the raw result. Do not type a benchmark
     number into a description -- `tools/tt-frontier report <receipt> --format comment`
     generates it from the receipt, and the receipt from the raw measurements.

  2. Say which model produced any predicted figure. `tensortransit plan` and `compare` output
     is evidence about a PLANNER. Only `eval/decide.py --real` and `tools/tt-frontier` are
     evidence about a speedup.
DONE
