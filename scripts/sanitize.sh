#!/usr/bin/env bash
# compute-sanitizer over the CUDA code. CONTRIBUTING.md requires contributors' kernels to be
# clean; before this script there was nothing that actually ran it, so "must be clean" was an
# instruction with no way to check compliance.
#
#   scripts/sanitize.sh [build-dir]
#
# memcheck found a real defect the first time it was run: destroying a controller while its
# compute stream was still capturing issued 14 illegal CUDA calls into the caller's context.
set -euo pipefail
BUILD="${1:-build}"
BIN="$BUILD/test_cuda_controller"
BENCH="$BUILD/tensortransit_bench"
[ -x "$BIN" ] || { echo "!! $BIN not built; cmake -B $BUILD -DTENSORTRANSIT_BUILD_CUDA=ON first"; exit 2; }

fail=0
# One controller test deliberately provokes a failing cudaMalloc to reproduce a stale-error
# bug. compute-sanitizer reports every error-returning API call, so that test is skipped here
# and the skip is printed by the binary itself. The alternative - a "2 expected errors"
# baseline in this script - would hide a third, real error behind the same number.
export RECURLOCAL_TEST_NO_DELIBERATE_ERRORS=1
for tool in memcheck initcheck synccheck; do
    echo ">> $tool (controller contract tests)"
    compute-sanitizer --tool "$tool" --error-exitcode 9 "$BIN" > /tmp/san.$tool.log 2>&1 || fail=1
    grep -E "^\[skip\]" /tmp/san.$tool.log || true
    tail -1 /tmp/san.$tool.log
done
if [ -x "$BENCH" ]; then
    # racecheck needs kernels doing real concurrent work; the benchmark's pre-touch against a
    # live update kernel is the only place in the repo where two streams touch one buffer.
    echo ">> racecheck (pre-touch against the update kernel)"
    compute-sanitizer --tool racecheck --error-exitcode 9 "$BENCH" combined \
        --layers 4 --tokens 2 --warmup-tokens 1 --state-bytes 262144 > /tmp/san.race.log 2>&1 || fail=1
    grep -E "RACECHECK SUMMARY" /tmp/san.race.log || true
fi
[ "$fail" = 0 ] && echo "sanitizers clean" || { echo "!! sanitizer findings; see /tmp/san.*.log"; exit 1; }
