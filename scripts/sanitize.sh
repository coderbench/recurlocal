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
EXEC="$BUILD/test_cuda_executor"
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

# The TRANSIT executor, which is the half of this library that is now in the measured path.
# Leaving it out would have meant the sanitizers covered the code a submission cannot change
# and not the code it can -- and it is the side that manipulates a live graph capture, drops
# borrowed streams and hands a device-wide L2 partition back.
if [ -x "$EXEC" ]; then
    for tool in memcheck initcheck synccheck; do
        echo ">> $tool (transit executor contract tests)"
        compute-sanitizer --tool "$tool" --error-exitcode 9 "$EXEC" \
            > /tmp/san.exec.$tool.log 2>&1 || fail=1
        tail -1 /tmp/san.exec.$tool.log
    done
else
    echo "!! $EXEC not built; the transit executor is NOT covered by this run"
    fail=1
fi
if [ -x "$BENCH" ]; then
    # racecheck needs kernels doing real concurrent work; the benchmark's pre-touch against a
    # live update kernel is the only place in the repo where two streams touch one buffer.
    # BOTH engines, because they issue that pre-touch from different code: `v0` from the 0.1
    # controller, `transit` from the plan an ITransitPlanner produced.
    for engine in v0 transit; do
        echo ">> racecheck (pre-touch against the update kernel, engine=$engine)"
        compute-sanitizer --tool racecheck --error-exitcode 9 "$BENCH" combined \
            --engine "$engine" --layers 4 --tokens 2 --warmup-tokens 1 --state-bytes 262144 \
            > "/tmp/san.race.$engine.log" 2>&1 || fail=1
        grep -E "RACECHECK SUMMARY" "/tmp/san.race.$engine.log" || true
    done
    # And memcheck over the transit engine's whole path with a streaming buffer present, which
    # is what exercises the Stream action and the plan's region arithmetic against real
    # allocations.
    echo ">> memcheck (transit engine, with streaming interference)"
    compute-sanitizer --tool memcheck --error-exitcode 9 "$BENCH" combined \
        --engine transit --planner budgeted --admission survival --layers 6 --tokens 2 \
        --warmup-tokens 1 --state-bytes 262144 --stream-bytes 4194304 --stream-mode distinct \
        > /tmp/san.transit.log 2>&1 || fail=1
    tail -1 /tmp/san.transit.log
fi
[ "$fail" = 0 ] && echo "sanitizers clean" || { echo "!! sanitizer findings; see /tmp/san.*.log"; exit 1; }
