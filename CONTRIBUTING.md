# Contributing

RecurLocal is a frontier-optimization project. You do not need a maintainer-created issue to contribute.

A performance PR should include the exact baseline commit, GPU/CUDA versions, benchmark command, raw result, real-runtime result when relevant, correctness evidence, and why the change should generalize. State them in the PR description.

Before opening a PR:

```bash
cmake -S . -B build -DRECURLOCAL_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure          # planner + label bands must pass
python3 eval/run_eval.py --binary ./build/recur_local_cuda_bench --repeats 5
```

Your kernels must be sanitizer-clean: `scripts/sanitize.sh build` runs memcheck, initcheck,
synccheck and racecheck. It is not optional decoration — memcheck found a real defect (a
controller destroyed mid-capture issuing 14 illegal CUDA calls) the first time it was run.

Do not change model math in the exact-locality track.

## The synthetic benchmark is not the score

`bench/cuda_bench.cu` explains mechanism. It has now disagreed with the real model on three
separate axes — prefetch distance, hot-set policy, and whether a pre-touch helps at all — for
one structural reason: it does not capture a CUDA graph, and production decode does. If your
change moves a synthetic number, say so, and then measure it where it counts:

```bash
integrations/sparkinfer/build.sh $WORK          # pinned commit, patched, one binary
eval/real_sweep.py --binary $WORK/sparkinfer/build/runtime/qwen3_gguf_bench \
                   --model $MODEL --axis <your axis> --repeats 3
eval/real_eval.py  ... --candidate RECURLOCAL=<your mode> --output real-result.json
eval/decide.py --real real-result.json
```

Before optimizing for a model, check what the best possible result would be worth:

```bash
eval/traffic_budget.py --ms-per-token <measured> --bandwidth-gbs <device> --sequences <N>
eval/traffic_budget.py --matrix configs/rtx5090-section44-ceiling.json --bandwidth-gbs 1792
```

The first gives one workload's ceiling; the second gives the highest weighted score the whole
section 44 matrix can physically return, so you can see which bands are reachable at all
before choosing what to work on.

## Adding a mechanism

A mechanism is an enumerator plus an implementation, never a file replacement — that is what
lets two of them be A/B'd in one process against identical state. In practice:

1. a value on the relevant `enum class` in `include/recurlocal/planner.h`, with its
   `to_string`/`parse_` pair and a round-trip test;
2. the implementation, behind that value only;
3. a row in the axis table of `eval/sweep.py` (synthetic) or `eval/real_sweep.py` (real);
4. a measurement, with the noise floor of the thing you measured it against.

If your axis will not resolve outside its own noise floor, say that. `sweep.py` and
`real_sweep.py` both exit non-zero rather than name a winner, and an unresolved axis is a
useful contribution — it is open, not solved.
