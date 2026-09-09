# RecurLocal

**RecurLocal is a feasibility-first CUDA library for software-directed locality of mutable recurrent neural state in hybrid LLM inference.**

> Keep recurrent AI state close to compute.

RecurLocal is intentionally **not** another inference engine, KV-cache manager, or model quantizer. Its first technical question is narrower:

> Can explicit L2 residency hints plus layer-ahead state prefetch reduce recurrent-state HBM traffic enough to improve real hybrid-LLM decode throughput?

The project starts with a small, measurable primitive and is designed to integrate with runtimes such as SparkInfer.

## Why this exists

Hybrid LLMs increasingly combine full attention with recurrent / linear-attention layers. Qwen3.8-27B, for example, has 64 language layers with a repeating pattern of three linear-attention layers followed by one full-attention layer. Its recurrent state uses FP32 and has 48 value heads of dimension 128×128, which is 3 MiB of matrix state per recurrent layer.

That state is mutable and repeatedly read/written during decode. RecurLocal experiments with the CUDA memory hierarchy rather than changing model math:

- reserve a bounded persisting-L2 set-aside when supported;
- mark hot recurrent-state windows as persisting;
- pre-touch the next recurrent layer's state asynchronously;
- rotate the hot window according to known layer execution order;
- measure whether this improves **end-to-end** decode, not just a microbenchmark.

CUDA's access-policy windows are hints, not placement guarantees. RecurLocal therefore treats every policy as an experimentally measured optimization, never as an assumed win.

## Non-goals

RecurLocal v0 does **not**:

- implement Gated DeltaNet / KDA / Mamba math;
- replace SparkInfer, vLLM, or SGLang;
- page recurrent state to CPU/NVMe;
- quantize recurrent state;
- change model outputs;
- claim any speedup before hardware measurements exist.

## Architecture

```text
Inference runtime
      |
      | recurrent layer N is about to run
      v
+---------------------------+
|         RecurLocal        |
|                           |
| locality planner          |
| persisting-L2 window      |
| next-layer pre-touch      |
| cache-window rotation     |
+-------------+-------------+
              |
              v
      CUDA memory hierarchy
        L2 <----> HBM
              |
              v
        recurrent kernel
```

## First real result

One RTX 5090, CUDA 13.3. **Qwen3.8-27B** (NVFP4, 64 layers, 48 of them recurrent) on a
**pinned SparkInfer commit** (`5347b27c`), through the adapter in
[`integrations/sparkinfer/`](integrations/sparkinfer/). Control and candidate are the same
binary — the hook is inert unless `RECURLOCAL` names a mode — run interleaved on the same box.
Control: **96.08 tok/s** at ctx 128, batch 1. Noise floor **0.023%**.

| mode | real end-to-end decode |
|---|--:|
| `baseline` — hook on, no policy | -0.01% |
| `persist` | **+0.13%** |
| `prefetch` | -1.29% |
| `combined` | -1.17% |

Output is token-exact against the unhooked runtime under greedy replay.

**All of the persist gain is an artefact of how the window is delivered, not of the policy.**
Production decode is a captured CUDA graph, and a stream access-policy window is host-side
state a graph never records:

| window delivery | gain |
|---|--:|
| `stream` (default; hand the window back for the runtime to attach) | **-0.019%** |
| `capture_node` (set the attribute on the node the capture just recorded) | **+0.129%** |

Under the safe path the policy never reaches the replayed graph, so there is nothing to
measure. `capture_node` reaches it by mutating a graph mid-capture, which CUDA does not
document as supported — a documentation gap rather than an observed defect: it has not failed
in any run taken, including twelve isolated runs at 32 sequences. It is opt-in, counted, and
self-disabling after a first detected invalidation.

(An earlier version of this README blamed a 32-sequence throughput collapse on that mutation.
**That was wrong**: the arms that collapsed were `baseline` and `prefetch`, which never arm the
mechanism at all. It is the runtime declining to batch. One instance of that family has since
been identified and proven — a row cap in a multi-row GEMV, worth 5.4× — but *this* collapse,
on this checkpoint, is intermittent and still unexplained. See below and
[`docs/OPTIMIZATION-SURFACES.md`](docs/OPTIMIZATION-SURFACES.md).)

The settled part is the boundary: **a locality library can compute a persisting window, but
under graph decode it cannot deliver one without the runtime attaching it at its own launch
site.** Pre-touch has no such problem — its kernels and events are recorded into the graph
like any other work.

### The scored result

The candidate is `persist` with `window_attach=capture_node` — the only mode that measures
positive anywhere, and the only way the persist family delivers a window at all under graph
decode. The **complete** section 44 matrix: batch 1 at three contexts plus concurrency 4, 16
and 32, three interleaved pairs each, one box, token-exact output with the hook proven active
during the gate.

```
$ python3 eval/decide.py --real results/rtx5090-real-complete.json   # JSON on stdout, this on stderr

verdict: reject   weighted gain +0.059%   impact none   significant false
  batch1          +0.184%  (w=0.40)
  concurrency16   +0.000%  (w=0.20)
  concurrency32   -0.163%  (w=0.20)
  concurrency4    +0.090%  (w=0.20)
  unresolved: ['concurrency16', 'concurrency32', 'concurrency4']
```

Batch 1 is the only arm that resolves, at +0.184% against a 0.01% noise floor — a real gain,
and roughly a quarter of the 0.68% a persisting cache can reach there. Every concurrency arm
sits inside its own run-to-run spread, and the verdict is **reject**: +0.059% weighted, far
below the 2% floor.

Two earlier runs are kept in `results/` because each is the reason a guard exists.
`rtx5090-real.json` has **no concurrency-32 arm** — and a missing workload is renormalised
away rather than averaged in as a zero, so omitting the arm with the most room *improves* a
score; `decide.py` now names what is absent and refuses to call such a matrix significant.
Before that, a scored run reported +0.053% for `persist` with safe window delivery, which was
a **null candidate** that applied no policy at all — 192 windows computed, none attached —
and `real_eval.py` now refuses that by name too.

**That is a rejection under the project's own gate**, and the reason is arithmetic rather
than implementation:

```
recurrent state per token   48 x (3 MiB fp32 + 60 KiB bf16) x 2  =  294 MiB
decode step                 10.34 ms x 1792 GB/s                 =   18.5 GB
recurrent share                                                      1.66%
```

Qwen3.8-27B is a **dense** hybrid: every weight is read every token, so at batch 1 the
recurrent state is 1.66% of the memory traffic. Making it *free* would be worth 1.69% in
throughput — a step carrying *f* less traffic runs in *(1−f)* of the time, so tok/s rise by
*f/(1−f)* — still below the 2% floor the go/no-go table rejects at, before any policy is
chosen.
`eval/traffic_budget.py` computes this from the pinned geometry, and it is worth running
before optimizing for any new model or concurrency.

The one thing that did move is not locality at all. Production decode is a captured CUDA graph
replayed per token, so every fork and join the pre-touch needs is a permanent graph node:

| pre-touch ordering | gain |
|---|--:|
| join per layer (v0.1 shape) | -1.29% |
| one join per token | **-0.09%** |

1.20 points of the 1.29 were the ordering, not the memory. The synthetic benchmark cannot see
this, because it does not capture a graph — and that is one of four axes on which it has now
disagreed with the real model.

### Concurrency is where the room is, and a persisting cache cannot reach it

Weights are read once per decode step whatever the batch; recurrent state once per sequence.
So the share of traffic RecurLocal can address grows with concurrency. Measured on one RTX
5090, three interleaved pairs per arm ([`results/rtx5090-baseline-matrix.json`](results/rtx5090-baseline-matrix.json)):

| | control | floor | traffic ceiling | `persist` | `prefetch` | persist ceiling |
|---|--:|--:|--:|--:|--:|--:|
| batch 1 | 96.67 tok/s | 0.04% | 1.69% | **+0.10%** | -1.27% | 0.68% |
| concurrency 4 | 334.17 tok/s | 0.09% | 3.01% | +0.15% | -2.07% | 0.59% |
| concurrency 16 | 790.60 tok/s | 0.13% | 7.44% | +0.06% | -5.68% | 0.35% |
| concurrency 32 | 1287.57 tok/s | 0.34% | **12.71%** | -0.59% | -7.51% | 0.28% |

The traffic ceiling rises with concurrency. The last column *falls*, and it is the one that
binds. A persisting window cannot save traffic it cannot hold, and state written at layer *i*
is read again at layer *i* of the **next token** — so the footprint that has to stay resident
is every recurrent layer for every sequence at once. Against this device's 60 MiB persisting-L2
capacity that is 2.4x oversubscribed at batch 1 and **39.9x at 32 sequences**. The room grows;
the fraction a cache can address shrinks faster.

Weighted across the section 44 matrix: the most any submission could score is **5.22%**, and
the persist family specifically tops out at **0.52%** — below the 2% floor, at every
concurrency. Both are computed, not asserted:

```bash
eval/traffic_budget.py --matrix configs/rtx5090-section44-ceiling.json --bandwidth-gbs 1792
```

So the open problem this work handed over was not "why does persist not capture the concurrency
room". That is answered. It was whether a *different* reuse distance, or a model with less
weight traffic per token, moves the terms that this bound is made of. Both have now been tested.

### A different reuse distance: bounded, and there is nothing there

Reuse *within* a recurrent layer is a distance L2 serves for free, so the only question was
whether any bytes sit at it. On the pinned runtime's kernels, almost none do. The Gated-DeltaNet
kernel holds each state column in registers across both of its passes — one global read, one
global write — so 98% of the recurrent bytes are touched exactly twice and there is no second
touch to catch. What remains is the convolution window's shift re-read: `conv × (K−2)/(K−1)` per
layer, **1.97 MB per token against an 18.5 GB step**. `eval/traffic_budget.py` reports it as
`within_layer_family` — a **0.011%** ceiling, two orders of magnitude under the floor. That
surface is closed by arithmetic rather than by effort. It is a property of this runtime, not of
Gated DeltaNet: the naive kernel SparkInfer replaced touched the state twice each way, which
would have put 288 MiB per token at that distance — 153× as much, and a 1.66% ceiling. The reuse
was real and large; somebody else already took it, in registers.

### A model with less weight traffic: this one moves, and it moves a lot

The persist bound is `2 × min(capacity, footprint) / step_traffic`. The capacity is the device's
and cannot be raised, so the only lever is the denominator — and the tool now prints the
threshold outright: **a decode step must move at most 6.42 GB** before a persisting window over
this footprint can reach 2% at all. Qwen3.8-27B moves 18.5 GB.

**Qwen3.6-35B-A3B** is the same architecture family on the same pinned commit, the same hook and
the same box — a sparse MoE reading 8 of 256 experts per token. The adapter needed no change: it
reads the state geometry from the runtime's config and brackets a 30-layer 2 MiB state as readily
as a 48-layer 3 MiB one. Two terms move at once:

| batch 1 | Qwen3.8-27B (dense) | Qwen3.6-35B-A3B (sparse MoE) |
|---|--:|--:|
| recurrent footprint | 146.8 MiB | **61.4 MiB** |
| vs 60 MiB persisting capacity | 2.4× | **1.02×** |
| resident fraction of the state | 41% | **98%** |
| decode step traffic | 18.5 GB | **3.56 GB** |
| traffic ceiling | 1.69% | **3.75%** |
| persist-family ceiling | 0.68% | **3.67%** |
| measured `persist`, default settings | +0.10% | **+1.26%** |
| measured `persist`, `budget_fraction=1.00` | — | **+1.53%** |

Three interleaved pairs, control 503.2 tok/s, noise floor 0.078%, paired ratios
1.0137 / 1.0120 / 1.0126 at the default. That is the largest real-model gain this repository has
measured — and the L2 set-aside turns out to be the dial that matters, resolved and monotonic:

| `budget_fraction` | set-aside | gain |
|---|--:|--:|
| 0.25 | 15 MiB | +0.68% |
| 0.50 | 30 MiB | +0.96% |
| 0.75 (the shipped default) | 45 MiB | +1.28% |
| **1.00** | 60 MiB | **+1.53%** |

Span 0.85% against a 0.10% noise floor. `--axis hit-ratio` says the same from the other side —
+0.76% / +1.13% / +1.41% / **+1.53%** over 0.25 → 1.00, span 0.74% against a 0.063% floor. Both
dials are monotonic to their maximum with no interior optimum, which is what a footprint that
nearly fits predicts: nowhere on this model is asking for less persistence better than asking
for more. On the dense model the same axes span 0.02%, because no fraction of a 2.4×
oversubscribed footprint can be held.

The two dials compound: together they give **+1.63%**, the best measured configuration on either
model. The shipped defaults (0.75, 0.70) were leaving a third of a point on the floor, and even
at the far end there are **2.0 points of headroom** to the ceiling — which is what makes batch-1
decode a surface here rather than the dead end it is on the dense model.

The new `quota` hot-set policy — admit whole layers at the full hit ratio rather than shave
every layer's — ties the best figure there (+1.631%) but the axis spans 0.075% inside a 0.190%
floor, so it is **open, not solved**. That fits: at 1.02× oversubscription there is almost
nothing for a policy to ration. The regime where it should differ is concurrency, where the
footprint is 8–16× the set-aside, and that is unmeasured.

### And a 5.4× cliff in the runtime, found on the way

Concurrency on this checkpoint could not be measured at first: above 8 rows the runtime stops
batching, packs 127 of 4205 tokens at 32 sequences and decodes the rest one row at a time, so
aggregate throughput falls *below* the single-sequence rate. `launch_mmvq_q4k_rows` refuses
`M > 8` and the bf16 `launch_mmvq_rows` dispatcher has no chunking loop, where its own `_f32`
sibling does. The runtime says so itself — `mmvq_rows refused type=12 N=16 n_out=8192 K=2048`,
then `declined at layer=0`.

Proven without a patch, because the engine already chunks a wide batch by its packed-row cap and
the cap is an environment variable:

| | default cap | `SPARKINFER_PACKED_MAX_ROWS=8` |
|---|--:|--:|
| c=16 | 5.8% packed, 452.8 tok/s | **94.4% packed, 1204.4 tok/s** |
| c=32 | 3.0% packed, 455.6 tok/s | **97.3% packed, 1226.8 tok/s** |

The cliff is **5.4×** (2456 tok/s at 8 rows against 452 at 16); the cap recovers **2.7×** of it,
because chunking re-reads the weights once per chunk. That is two orders of magnitude more than
anything this library does to a decode step, and it is SparkInfer's to fix.
`eval/real_eval.py` now refuses a concurrency arm that fell off the batched path rather than
scoring it. Details in [`docs/MINING.md`](docs/MINING.md).

Raw data: [`results/rtx5090-moe-matrix.json`](results/rtx5090-moe-matrix.json).
[`docs/OPTIMIZATION-SURFACES.md`](docs/OPTIMIZATION-SURFACES.md) has the full matrix.

## First measured result (synthetic)

One RTX 5090, CUDA 13.3, synthetic recurrent-state benchmark — 48 layers x 3 MiB, 32 timed
tokens after 4 warm-up, 15 interleaved repeats. Full data in
[`results/rtx5090-synthetic.json`](results/rtx5090-synthetic.json).

| mode | gain vs baseline | spread |
|---|--:|--:|
| persist | +10.1% | 0.9% |
| prefetch | **+15.9%** | 4.7% |
| combined | +10.4% | 4.6% |

Tuning the prefetch distance, which v0.1 had hard-coded to one layer, reaches **+21.0% at
distance 6**. Correctness held across all 39 strategy x distance configurations: every one
produced a bit-identical final state.

**This is a synthetic locality benchmark, not a model speedup.** No model and no runtime were
involved. It was enough to justify building the real integration, and that integration has
since contradicted it on four separate axes — including this headline. Read the real section
above first; this one explains mechanism and decides nothing.

### The batch-1 policy does not survive batch N

Every number above is one sequence. Under concurrency the picture inverts
([`results/rtx5090-surfaces.json`](results/rtx5090-surfaces.json)):

| sequences | 1 | 4 | 16 | 32 |
|---|--:|--:|--:|--:|
| persist | +9.9% | **-11.1%** | -1.8% | -0.6% |
| prefetch | +13.9% | +7.3% | **+22.9%** | -4.8% |
| combined | +9.9% | -0.4% | -17.6% | **-21.0%** |

`persist` turns harmful at four sequences — and the v0.1 planner did not consider that
oversubscribed at all, because its hot-set model counted one layer's state and ignored that
the other 47 layers' are equally live. **That accounting is now fixed**
(`--hot-set-model token_footprint`, with the old model kept as the control); what the
corrected number should make the policy *do* at each concurrency is still open. Separately,
64 MiB of streaming weight traffic against a 48 MiB set-aside costs **two thirds of
throughput**.

Three more that overturn stated assumptions: **kernel-integrated prefetch is 30-33 points
worse** than a separate stream (section 34.4 and the v0.5 roadmap assume the opposite);
**state layout alone is a 2.3x effect** and flips which policy wins; and the `cliff` hot-set
policy scores **+22.8% where every other policy scores -17.4%** at 16 sequences.

These are the open problems, not the settled results — and the real integration has since
shown that several of them are artefacts of a benchmark that does not capture a CUDA graph.
[`docs/OPTIMIZATION-SURFACES.md`](docs/OPTIMIZATION-SURFACES.md) maps every surface on both
benchmarks, the flag that isolates each, and where the two disagree.

## Go / no-go gate

| Real Qwen3.8 / SparkInfer result | Decision |
|---|---|
| <2% end-to-end gain | reject the project |
| 2–4% | probably reject |
| 4–7% | promising |
| 7–10% | strong candidate |
| >10% | expand immediately |

**The gate has been run and the result is in the first band.** The scored measurement lives
in `results/rtx5090-real.json` and the verdict is derived from it by `eval/decide.py --real`,
not asserted — the numbers are quoted once, above, from that command's actual output.

Section 21 says the project should be willing to fail this test. On Qwen3.8-27B, on this
hardware, with these policies, it fails.

What that does and does not mean:

- It does **not** mean the mechanism is broken. The pre-touch demonstrably runs, and the
  persisting window is verifiably present in a replayed graph when it is attached.
- It does mean the mechanism is aimed at 1.65% of the problem at batch 1 on a dense hybrid,
  and that the policies do not convert the 7.4% that 16 sequences puts on the table.
- It is now clear *why* they do not, and it is not tuning: the reuse distance is a full model
  pass, so the resident footprint a persisting window would need is up to 40x the device's
  persisting-L2 capacity. That bound tightens as concurrency grows.
- The next honest experiment is therefore not more tuning of these policies. It is a model
  whose weight traffic per token is smaller — a sparse MoE, where the same recurrent state is a
  much larger share and the residency requirement is unchanged — or a policy aimed at reuse
  *within* a layer, which is a distance the cache can actually serve.

No synthetic result should be marketed as a model speedup.

This table is executable rather than advisory. `eval/decide.py` applies it, along with the
impact tiers and the weighted workload score, as a deterministic function of measurements:

```bash
python3 eval/decide.py --synthetic eval-result.json   # reports, never tiers
python3 eval/decide.py --real real-result.json        # scores the real workload matrix
```

The synthetic path exists to say no: a locality microbenchmark is reported but never
labelled, because the only number that decides this project is real end-to-end decode. The
real path gates on bit-identical output, blocks any workload regressing more than 2%, and
combines the matrix with a weighted geometric mean so one cherry-picked win cannot carry a
PR that loses elsewhere.

## Using it from a runtime

RecurLocal is a component, not an application. It installs a CMake package:

```cmake
find_package(RecurLocal 0.1 REQUIRED)
target_link_libraries(your_runtime PRIVATE RecurLocal::recurlocal_cuda)
```

Construction and every entry point are `noexcept` and report `cudaError_t`; the controller
detects CUDA Graph capture and hands the access-policy window back for the caller to attach to
its kernel node, because a stream attribute is not recorded into a graph. See
`integrations/sparkinfer/README.md` for the full contract.

## Build: CPU-only

```bash
cmake -S . -B build -DRECURLOCAL_BUILD_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/recur_local_info
```

## Build: CUDA

```bash
cmake -S . -B build \
  -DRECURLOCAL_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Synthetic CUDA benchmark

```bash
./build/recur_local_cuda_bench baseline
./build/recur_local_cuda_bench persist
./build/recur_local_cuda_bench prefetch
./build/recur_local_cuda_bench combined

python3 eval/run_eval.py --binary ./build/recur_local_cuda_bench
```

The default benchmark emulates 48 recurrent layers with 3 MiB of state per layer, over 32 timed tokens preceded by 4 untimed warm-up tokens. It is a memory-locality experiment, **not** a faithful GDN model benchmark.

Benchmark options: `--layers`, `--tokens`, `--warmup-tokens`, `--inner-iters`, `--state-bytes`, `--device`.

The evaluator runs every mode `--repeats` times (default 3), interleaved so drift does not land on one mode, and compares median timings. It refuses to report a result unless every run of every mode produced the same final-state checksum, and marks the run unstable when the run-to-run spread is wider than the difference being measured — the bands below start at 2%, which a single run per mode cannot resolve. Pre-touch work is joined into the timed region, so prefetch overhead counts against prefetch modes.

Every result records what produced it — GPU, compute capability, driver and runtime version,
commit and whether the tree was dirty, and the observed graphics clock. Pass
`--pin-clock-mhz auto` (needs privileges) to lock the clock so an absolute time is
reproducible off the box, not merely same-box comparable.

## Integration contract

A runtime needs only the lifecycle hook shown in `integrations/sparkinfer/README.md`. The
SparkInfer integration is 88 lines of insertions against a pinned commit, and CI asserts it
deletes nothing.

The CUDA code is tested: `tests/test_cuda_controller.cu` runs 120 device-side checks over the
controller, the graph-capture state machine and every pre-touch strategy, and
`scripts/sanitize.sh` keeps it memcheck/initcheck/synccheck/racecheck clean. Writing those,
and the audit that went with them, found eleven real defects in the library, including a
persisting-L2 set-aside that was never given back and a failing pre-touch that could strand a
host runtime's graph capture — see the CHANGELOG.

`stats()` reports windows applied, windows *deferred* under graph capture, windows actually
attached to a captured graph node, layers where the hot set was oversubscribed, and pre-touch
volume — so a null end-to-end result can be explained rather than guessed at. That mattered
here: without the node attachment the persisting policy would have been absent from every
graph replay, `windows_attached_to_node` would have read 0, and `persist` would have measured
its cost with none of its effect.

## Competing on this repository

[`docs/MINING.md`](docs/MINING.md) is the competition brief, and it is written to talk people
out of the two obvious mistakes. **Batch-1 decode is a dead surface** — a 1.69% ceiling, below
the floor the gate rejects at. **Concurrent decode has the room** — 12.71% at 32 sequences —
but **a persisting L2 window is not the instrument that reaches it**: the resident footprint
required is 40x the cache, so that policy family tops out at 0.52% weighted however well it is
delivered. The brief states the highest score physically available on this model and device
(**5.22%**, impact `S`) so that nobody spends a week chasing a band that does not exist here.

## Contribution model

There are deliberately no bounty-style optimization issues required. Profile `main`, find a bottleneck, and move the frontier.

[`docs/OPTIMIZATION-SURFACES.md`](docs/OPTIMIZATION-SURFACES.md) maps every surface, the flag
that isolates it, and the current frontier on each — including the two that are measured and
open, and the two that cannot be competed on yet because nothing measures them.

## References

- Qwen3.8-27B config: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- CUDA L2 cache control: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
- SparkInfer: https://github.com/gittensor-ai-lab/sparkinfer

## License

MIT
