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

If the device-bounded ceiling for the roles your policy is allowed to touch is under the
run-to-run spread of the cells it would be measured in, nothing in this repository can help
you, and you have found that out for free. `frontier/TTF-1/reference.json` publishes both
numbers per cell — the measured control and its spread — so the size of the prize and the noise
you have to beat are on the page before you start.

Several of the surfaces worth taking need **no GPU at all**: the cost model, a new admission
rule, a reuse metric, trace fidelity, plan replay. `docs/MINING.md` says which.

## How a submission is scored

One continuous number, computed and never assigned:

```text
Frontier Gain: dF = F(candidate) / F(main) - 1
```

`F` is the normalized Pareto hypervolume of the serving frontier — goodput against p99
inter-token latency — over a frozen generation's workload cells. **There are no XS/S/M/L/XL
bands**; the reason they went away is in [`frontier/README.md`](frontier/README.md) and it is
not stylistic. Every figure in a PR comment is generated from the receipt, and the receipt
from the raw measurements: do not type a benchmark number into a PR description.

Most of this repository's own results are `NO_FRONTIER_GAIN` or `INCONCLUSIVE`, and saying so
is the point rather than an embarrassment.

Before asking for hardware, do all of this locally:

```bash
tensortransit compare <trace.json>                       # what your plan does, offline
tensortransit plan <trace.json> --cost-model linear      # and whether it survives the control model
tensortransit replay <plan.json> --trace <trace.json>    # and that an executor would fire it
ctest --test-dir build --output-on-failure               # including the frontier scorer
```

The evaluator runs the instrument from the **base** commit, not from your tree: `eval/`,
`tools/tt-frontier`, `schemas/`, `configs/`, `tests/golden/`, `workloads/` and
`adapters/sparkinfer/pin.json` are overlaid from the baseline before anything is measured, and
anything you changed in them is *named* in the output rather than silently dropped. Propose a
change to what is measured separately from the optimization it would score.

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
