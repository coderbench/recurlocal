# Contributing

TensorTransit is a frontier-optimization project. You do not need a maintainer-created issue
to contribute.

A performance PR should include the exact baseline commit, GPU/CUDA versions, benchmark command, raw result, real-runtime result when relevant, correctness evidence, and why the change should generalize. State them in the PR description.

Before opening a PR:

```bash
cmake -S . -B build -DTENSORTRANSIT_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure   # planners, golden plans, schemas, compat shim
python3 eval/run_eval.py --binary ./build/tensortransit_bench --repeats 5
```

Your kernels must be sanitizer-clean: `scripts/sanitize.sh build` runs memcheck, initcheck,
synccheck and racecheck. It is not optional decoration — memcheck found a real defect (a
controller destroyed mid-capture issuing 14 illegal CUDA calls) the first time it was run.

Do not change model math in the exact-locality track.

A CPU-only build is enough for most of the frontier — the planners, the graph, the plan
schema and the golden tests all run without a GPU:

```bash
cmake -S . -B build -DTENSORTRANSIT_BUILD_CUDA=OFF
cmake --build build -j && ctest --test-dir build --output-on-failure
```


## You do not need an issue

There are deliberately no bounty-style optimization issues to claim, and no maintainer has to
create work for you.

> Profile the current frontier, find a measurable cross-kernel data-movement bottleneck, and
> submit a reproducible improvement.

Start with [`docs/MINING.md`](docs/MINING.md) — it is written to talk you out of the two
obvious mistakes before you spend a week on them — and with one command:

```bash
tensortransit inspect <trace.json> --device rtx5090
```

If the device-bounded ceiling for the roles your policy is allowed to touch is under the 2%
significance floor, nothing in this repository can help you, and you have found that out for
free. Several of the surfaces worth taking need **no GPU at all**: the cost model, a new
admission rule, a reuse metric, trace fidelity. `docs/MINING.md` says which.

Impact is applied by `eval/decide.py`, mechanically, from the bands in that document — not by
a reviewer's judgement. Most of this repository's own results land in the `none` band, and
saying so is the point rather than an embarrassment.

## The synthetic benchmark is not the score

`workloads/recurrent/synthetic/cuda_bench.cu` explains mechanism. It has now disagreed with the real model on three
separate axes — prefetch distance, hot-set policy, and whether a pre-touch helps at all — for
one structural reason: it does not capture a CUDA graph, and production decode does. If your
change moves a synthetic number, say so, and then measure it where it counts:

```bash
adapters/sparkinfer/build.sh $WORK          # pinned commit, patched, one binary
eval/real_sweep.py --binary $WORK/sparkinfer/build/runtime/qwen3_gguf_bench \
                   --model $MODEL --axis <your axis> --repeats 3
eval/real_eval.py  ... --candidate RECURLOCAL=<your mode> --output real-result.json
eval/decide.py --real real-result.json
```

Before optimizing for a model, check what the best possible result would be worth:

```bash
eval/traffic_budget.py --ms-per-token <measured> --bandwidth-gbs <device> --sequences <N>
eval/traffic_budget.py --matrix configs/rtx5090-section44-ceiling.json --bandwidth-gbs 1792
# and the tighter bound the persist family is actually held to:
eval/traffic_budget.py --matrix configs/qwen3.6-35b-a3b-moe-ceiling.json                        --bandwidth-gbs 1792 --persisting-l2-bytes 62914560
```

The first gives one workload's ceiling; the second gives the highest weighted score the whole
section 44 matrix can physically return, so you can see which bands are reachable at all
before choosing what to work on.

**Which model you pick decides more than which policy you pick.** The persist-family bound is
`2 × min(persisting_capacity, footprint) / step_traffic`, and the capacity is the device's, so
the only lever is the step. `--persisting-l2-bytes` prints `break_even_step_traffic_bytes` —
the step traffic a model has to come in under before a persisting window is worth anything at
all. On an RTX 5090 that is 6.42 GB: the dense Qwen3.8-27B moves 18.5 GB and cannot clear it,
the sparse-MoE Qwen3.6-35B-A3B moves 3.56 GB and does. Two configs ship so the difference can
be read side by side, and a matrix spec may carry its own `model` geometry so a second model's
decode rates cannot be scored against the first model's state shape.

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
