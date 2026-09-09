# Optimization surfaces

Where the work is, what isolates it, and what is already known.

Every surface is a named value on one enumeration, selected at run time and reported in the
benchmark's JSON. Two people can work on different mechanisms without conflicting, and any
two can be A/B'd in one process against identical state. A change that replaces a file
instead of adding an enumerator cannot be compared against what it replaced.

Sweep one axis with everything else fixed:

```bash
python3 eval/sweep.py --binary ./build/recur_local_cuda_bench --axis prefetch-distance
python3 eval/sweep.py --binary ./build/recur_local_cuda_bench --axis hot-set-policy -- --sequences 32
```

`sweep.py` compares the spread across an axis to that axis's own noise floor and **refuses to
name a winner inside it**, exiting non-zero. An axis that will not resolve is open, not solved.

Two benchmarks appear below and they are not interchangeable. The **real** section is
Qwen3.8-27B on a pinned SparkInfer commit; it is the one the go/no-go gate reads. The
**synthetic** section is `bench/cuda_bench.cu`, which explains mechanism and decides nothing
(sections 17 and 28).

---

# Real model: Qwen3.8-27B, pinned SparkInfer, RTX 5090

`integrations/sparkinfer/` · NVFP4 checkpoint, 64 layers (48 recurrent), CUDA 13.3, sm_120 ·
control 96.08 tok/s at ctx 128 · every number is a control/candidate pair measured minutes
apart on the same box.

## Read this first: the ceiling is 1.65%

```
recurrent state per token   48 layers x (3 MiB fp32 + 60 KiB bf16) x 2 (read+write) = 294 MiB
decode step                 10.41 ms at 1792 GB/s                                   =  18.6 GB
recurrent share                                                                       1.65%
```

`eval/traffic_budget.py` computes it. Qwen3.8-27B is a **dense** hybrid: every weight is read
every token, and at batch 1 that is 98.35% of the traffic. Making recurrent state *free* would
be worth 1.65%, which is below the 2% floor section 21 rejects at — **before** any policy is
chosen, and no implementation can move it.

The traffic model is checkable rather than assumed, and checking it produces a second result.
A pre-touch adds exactly one extra read of the recurrent state per token — +0.83% of the
traffic budget — so a saturated memory system should charge about 0.83% for it. Measured with
the per-layer ordering removed (`prefetch_join=token_end`), the whole pre-touch costs
**0.09%**. Either the step has that much spare bandwidth, or the pre-touched lines are served
from L2 when the layer actually reads them and DRAM traffic never doubles. Both say the
locality mechanism is working. Neither produces a speedup, because the recurrent kernel was
not waiting on that memory in the first place — 98.35% of the step is weights.

## Modes

3 interleaved pairs, noise floor **0.023%** — two orders of magnitude tighter than the
synthetic benchmark's, because the timed region here is a pure `cudaGraphLaunch` loop.

| mode | gain | what it says |
|---|--:|---|
| `baseline` (hook on, no policy) | **-0.01%** | the integration costs nothing |
| `persist` | **+0.13%** | resolved, real, 12x smaller than the ceiling — and see below |
| `prefetch` | **-1.29%** | |
| `combined` | **-1.17%** | |

## The persist gain is not free, and it is not the library's to take

Production decode is a captured CUDA graph. A stream access-policy window is host-side state
that a graph never records, so under capture the window has to reach the kernel some other
way. There are exactly two, and this axis separates them:

| `--axis window-attach` | gain | what it does |
|---|--:|---|
| `stream` (default) | **-0.019%** | applies on the stream when not capturing; under capture, hands the window back for the runtime to attach |
| `capture_node` | **+0.129%** | sets the attribute on the kernel node the capture just recorded |

**Every point of the persist result comes from `capture_node`.** Under the safe path the
window is never delivered and `persist` measures nothing, because there is nothing to measure:
the policy does not exist in the replayed graph.

And `capture_node` is not a free lunch. Setting an attribute on a node of a graph that is
still being captured is not an operation CUDA documents as supported. On batch-1 decode it
works — 48 of 48 nodes, no failure in any run taken. At 32 concurrent sequences, during the
scored evaluation, something else happened:

```
[prefill] graph capture failed -> fallback
tokens 270, tokens_packed 128       # half the decode steps fell back to per-row
agg_tok_s 1260.5 -> 901.8           # -28%, with no cache policy anywhere in it
```

Two of three repeats of that arm collapsed that way, and an immediate targeted rerun of the
same configuration reproduced it. **A later 12-run probe did not.** Four unhooked control
runs, four `stream` runs and four `capture_node` runs, each in isolation at 32 sequences, all
landed between 1242 and 1272 tok/s with zero capture failures.

So what is established is narrower than it first looked:

- the collapse is real, is a *runtime fallback* and not a cache effect, and costs 28%;
- it was seen only in the eval's run sequence — after the c=4 and c=16 arms — and never in
  isolation, so accumulated device state is a co-factor and node mutation alone is **not**
  a demonstrated cause;
- it has never been seen with `stream` or with the unhooked control, but 8 clean runs are
  not enough to establish a rate for something this rare.

That is why `Stream` is the default rather than why `CaptureNode` was deleted. The controller
checks `cudaStreamIsCapturing` for `Invalidated` after every attach, counts it in
`stats().capture_invalidations`, and latches itself off after the first — best-effort, since
an invalidation that only surfaces at the runtime's `cudaStreamEndCapture` will not be seen by
it. The runtime's own message remains the ground truth, and a bimodal concurrency arm should
be read as a fallback until proven otherwise.

**The part that *is* settled is about the boundary, not the policy.** A locality library can
compute a persisting window; under graph decode it cannot deliver one without the runtime
attaching it at its own launch site, and the one shortcut that avoids that is undocumented.
Pre-touch has no such problem — kernels and events are recorded into the graph like any other
work — so two mechanisms that look symmetrical in the API have very different integration
costs.

## The finding: graph nodes, not memory traffic

Production decode is one captured CUDA graph replayed per token, so every fork and join the
pre-touch needs is a permanent graph node — 47 of each for this model.

| `--axis prefetch-join` | gain |
|---|--:|
| `per_layer` (v0.1 shape) | -1.29% |
| `token_end` | **-0.09%** |

**1.20 points**, for removing 47 joins and changing nothing about what is read. Every other
prefetch axis is the same effect seen from a different side — each one that does *less*
ordering costs less, in proportion:

| `--axis prefetch-schedule` | pre-touched layers | gain |
|---|--:|--:|
| `uniform` | 47 | -1.28% |
| `ramp` | 47 | -1.26% |
| `alternating` (every 2nd) | 24 | -0.65% |
| `sparse` (every 4th, twice as far) | 12 | -0.30% |

Cost is straight-line in the number of pre-touched layers across a 4x range — about 0.027% of
a decode step each — and extrapolates to zero at zero layers. The synthetic benchmark measured
this axis as a 16-point *penalty* for doing less; the real model measures it as a saving,
because the synthetic benchmark does not capture a graph and so never pays for the nodes.

| `--axis pre-touch-coverage` | gain |
|---|--:|
| `conv` (60 KiB/layer) | -0.37% |
| `matrix` (3 MiB/layer) | -0.98% |
| `both` | -1.29% |

Touching the small state costs a third of touching the large one; the difference between them
is bandwidth. Both are costs; neither is a gain. Note that this axis could not be *expressed*
before this release — the pre-touch took a `const float*` and the convolution state is bf16.

| `--axis pre-touch` | gain |
|---|--:|
| `ptx_l2` | **-0.74%** |
| `partial` | -1.02% |
| `scalar` | -1.26% |
| `vec4` | -1.30% |
| `vec4_ldcg` | -1.31% |
| `warp_tile` | -1.44% |

On the synthetic benchmark this axis spans 1.6 points inside a 4-5% noise floor and is
formally unresolved. On the real model it spans 0.70 points against a floor two orders of
magnitude tighter, and `ptx_l2` is clearly ahead: `prefetch.global.L2` moves a line into L2
without it ever entering a register, so it neither occupies L1 nor competes for issue slots
with the kernel it is trying to help. `warp_tile` is the worst, which is the opposite of what
a coalescing argument predicts. The synthetic benchmark ranked them in a different order and
could not tell any of them apart.

`--axis prefetch-distance` moves monotonically from -1.29% at d=1 to -1.15% at d=8 — reaching
further is slightly cheaper and never positive. The synthetic benchmark's +21.0% at distance 6
does not survive contact with a real model.

## Concurrency: the ceiling rises and the policies still do not catch it

Aggregate continuous-batching throughput (`qwen3_gguf_cb_bench`, a different code path —
`decode_packed` — bracketed by the same adapter; `tokens_packed` in the telemetry proves it
was used). 2 interleaved pairs each.

| | ceiling | control | `baseline` | `persist` | `prefetch` | `combined` |
|---|--:|--:|--:|--:|--:|--:|
| batch 1 | 1.65% | 96.1 tok/s | -0.01% | **+0.13%** | -1.29% | -1.17% |
| concurrency 4 | 2.88% | 329 tok/s | -0.09% | +0.06% | -2.19% | -2.34% |
| concurrency 16 | 6.73% | 769 tok/s | +0.36% | -0.21% | **-5.88%** | -5.40% |
| concurrency 32 | — | 1262 tok/s | -1.13% | -1.13% | *unresolved* | *unresolved* |

"Ceiling" is `eval/traffic_budget.py` at that step time and sequence count, with the state
bf16-compacted as the runtime does for batched decode. Weights are read once per step
whatever the batch; recurrent state once per sequence — so the share of decode traffic that
recurrent state accounts for **quadruples** from batch 1 to concurrency 16, and by 16 there is
genuinely 6.7% on the table.

**None of it is captured.** `persist` goes from +0.13% to -0.21% as the room to win grows, and
`prefetch` gets worse in near-exact proportion to the bytes it moves — -1.29%, -2.19%, -5.88%
against 1x, 4x and 16x the pre-touch traffic. The pre-touch is not competing with the
recurrent read it is meant to hide; it is competing with the weight stream that is the other
93% of the step.

### Concurrency 32: the screen did not resolve, and a dedicated probe did

The 2-repeat screen produced 30%+ spreads on two of its four arms — `baseline` measured
0.7129 and 0.9906 against the same control, `prefetch` 0.9221 and 0.6761. A median of two
turns those into "-14.8%" and "-20.1%", which read like results. They are not: `baseline`
installs no window and issues no pre-touch, so it *cannot* cost 14.8%. A number that a
configuration doing nothing can produce is a number that means nothing.

The collapses are the runtime falling back off its batched decode path (see above), not a
cache effect. A dedicated 12-run probe — 4 unhooked control, 4 `stream`, 4 `capture_node`,
each run in isolation — has no collapses in it at all and gives a usable figure:

| c=32, isolated runs | aggregate tok/s | vs control |
|---|--:|--:|
| control (unhooked) | 1261.1, 1261.1, 1271.7, 1262.5 | — |
| hooked, `stream` | 1246.4, 1248.6, 1241.9, 1251.1 | **-1.13%** |
| hooked, `capture_node` | 1246.2, 1248.1, 1248.7, 1248.3 | **-1.08%** |

So at 32 sequences the hook costs about 1.1% and the attach mode does not matter, which is
consistent with 4 and 16 and inconsistent with the screen's -14.8%. `real_sweep.py` refused
to name a winner on the screen, and `real_eval.py` now reports a `resolution` block beside
every workload's gain rather than letting a median hide the repeats behind it. That is why
concurrency 32 is reported here and left out of the scored matrix.

That is the open problem this release hands over, and it is a much better-posed one than the
project started with: the accounting is now correct, the concurrency path is instrumented,
the ceiling at each batch size is computable, and the mechanism that is supposed to capture it
demonstrably does not. A policy that does capture it has 6.7% to aim at, at 16 sequences, on
this model.

## Axes that do not resolve

| axis | span | verdict |
|---|--:|---|
| `window-target` (`matrix`/`conv`/`widest`/`narrowest`) | 0.02% | **open** — inside its noise floor |
| `hot-set-policy` (all four) | 0.01% | flat; there is nothing for a policy to protect |
| `window-scope` | 0.15% | `layer` and `ahead` tie at +0.14%, `allocation` at 0.00% |

The hot-set policy surface is the synthetic benchmark's most dramatic result — a 40-point
swing at 16 sequences — and on the real model at batch 1 the four policies span 0.01%. Both
are true; they are measuring different regimes.

Every 2-repeat sweep above is reported with its winner named only where the span is large
compared to the 3-repeat floor measured on the mode axis. `real_sweep.py` now refuses to
estimate a floor from two repeats at all.

## What is fixed and what it changed

The hot-set accounting was wrong and is now right: on this model `TokenFootprint` reports
**48/48 layers oversubscribed** (147 MiB of live recurrent state against a 48 MiB set-aside)
where `CurrentLayer` reported **0**. It changed no batch-1 number, because at batch 1 there
is 1.65% to fight over. It is the input every concurrency policy depends on.

`windows_attached_to_node` is 48/48 under `--window-attach capture_node`, so the persisting
policy really is present in the replayed graph there — and 0 under the safe default, which is
the whole point of the section above. Telemetry is what distinguishes "the policy did not
help" from "the policy was never applied", and those are different results with the same
number.

---

# Synthetic benchmark

All numbers below: one RTX 5090, CUDA 13.3, 48 layers x 3 MiB, 32 timed tokens after 4
warm-up. **Synthetic locality benchmark, not a model speedup** — read `docs/FRONTIER.md`
first, and note above where it disagrees with the real model.

---

## Read this before believing any number

| mode | spread across repeats |
|---|--:|
| baseline, persist | < 1% |
| prefetch, combined | ~4-5% |

The pre-touch stream contends for SMs, so modes that use it are intrinsically noisier. Graphics
clocks could not be pinned on the measurement box, so absolute times are not reproducible off
it; same-box deltas are.

---

## The headline finding: the batch-1 policy is wrong

Everything the project measured before concurrency existed was measured at one sequence.

| sequences | hot set | persist | prefetch | combined |
|--:|--:|--:|--:|--:|
| 1 | 3 MiB | +9.9% | +13.9% | +9.9% |
| 2 | 6 MiB | +8.3% | +14.9% | +12.6% |
| 4 | 12 MiB | **-11.1%** | +7.3% | -0.4% |
| 8 | 24 MiB | -4.4% | +20.2% | -9.0% |
| 16 | 48 MiB | -1.8% | **+22.9%** | -17.6% |
| 32 | 96 MiB | -0.6% | **-4.8%** | **-21.0%** |

`persist` turns **harmful at four concurrent sequences** and `combined` loses 21% at 32.
Overview section 32.5 predicted this in prose; it is now a number.

Note where it turns: four sequences is 12 MiB against a 48 MiB set-aside, so the planner did
not consider it oversubscribed at all (`hot_set_oversubscribed` was 0 until 16 sequences),
because the v0.1 accounting counted one layer's state and ignored that the other 47 layers'
state is equally live.

**That is fixed.** `--hot-set-model token_footprint` counts every recurrent layer for every
sequence — 144 MiB at one sequence, oversubscribed from the start — and `reuse_window` adds
the streaming traffic in between. `current_layer` remains as the control, so the correction is
measurable rather than asserted, and re-running this table under the corrected model is now a
one-flag experiment (`eval/sweep.py --axis hot-set-model -- --sequences 4`). What the
corrected number should make the *policy* do at each concurrency is still open.

## Surface: hot-set policy

**`src/planner.cpp` · `--hot-set-policy` · 4 policies · frontier: `cliff`**

At 32 sequences (96 MiB hot against a 48 MiB set-aside):

`combined` mode, by sequence count:

| policy | 8 seqs | 16 seqs | 32 seqs |
|---|--:|--:|--:|
| `fixed` (naive control) | -9.0% | -17.4% | -20.8% |
| `proportional` (the v0.1 heuristic) | -9.1% | -17.4% | -21.0% |
| `sqrt` | -9.0% | -17.5% | -21.4% |
| `cliff` | -9.0% | **+22.8%** | **-4.8%** |

At 16 sequences `cliff` scores +22.8% where every other policy scores -17.4% — a **40-point
swing**, and it exactly matches pure `prefetch` at that concurrency, because declining the
window is what lets prefetch work unimpeded.

The shipped heuristic is the *worst* of the backing-off policies, and the policy that simply
refuses the window when oversubscribed loses 16 points less than it. Backing off the hit ratio
does not work; declining to compete does. Nobody has explained why — that explanation, with
hardware counters behind it, is a real contribution.

## Surface: cache interference and QoS

**`bench/cuda_bench.cu` · `--stream-bytes`, `--qos` · frontier: unexplained**

Attention and MoE weights stream through the same L2 the recurrent state wants to sit in.

| streaming | persist | prefetch | combined |
|--:|--:|--:|--:|
| 0 | +10.2% | +14.4% | +9.9% |
| 16 MiB | -6.9% | +9.2% | +9.0% |
| 64 MiB | **-66.7%** | +3.0% | **-67.4%** |
| 256 MiB | +0.4% | -0.6% | -0.3% |

A 48 MiB persisting set-aside against 64 MiB of streaming traffic costs **two thirds of
throughput**. This is the interference section 32.4 warns about, and it is far worse than the
warning implies. At 256 MiB everything is bandwidth-bound and the policy stops mattering.

### QoS: the answer depends on what the traffic is

`--stream-mode` picks the shape of the competing traffic, and it decides the policy:

| 64 MiB competing | baseline | base+qos | persist | persist+qos |
|---|--:|--:|--:|--:|
| `reuse` (one buffer, re-read every layer) | **34.0 ms** | 102.4 ms | 101.7 ms | 101.4 ms |
| `distinct` (each layer its own slice) | 10.9 ms | 10.9 ms | 10.5 ms | **10.0 ms** |

With genuinely streamed per-layer weights, `persist` + `qos` is the best configuration
(**+8.4%** over control at 64 MiB, +2.8% at 256 MiB, +1.7% at 512 MiB) — section 34.6 works.

With a re-read buffer, the same hints cost **3x**. That buffer fits in L2 and is the hottest
data on the device; reserving 48 MiB for recurrent state evicts it, and marking it streaming
evicts it deliberately. Both models are realistic — a shared expert is reused, attention
weights are not — and they give opposite answers. Deciding which regime a runtime is in, at
run time, is an open problem and nothing in the planner currently even asks.

## Surface: prefetch distance

**`src/planner.cpp` · `--prefetch-distance 0..8` · frontier: +21.0% at d=6**

| distance | 1 | 2 | 3 | 4 | 6 | 8 |
|---|--:|--:|--:|--:|--:|--:|
| gain | +14.7% | +15.6% | +17.0% | +18.2% | **+21.0%** | +20.5% |

Hard-coded to 0-or-1 until v0.2, leaving six points on the table. The curve has an interior
optimum: too far ahead and the state is evicted before its layer runs. That optimum depends on
state size, layer compute time and L2 pressure, so it is almost certainly not 6 on a real
model. A planner that derives it instead of taking a constant is an open problem.

## Surface: prefetch schedule

**`src/planner.cpp` · `--prefetch-schedule` · 4 schedules · frontier: `uniform`, at batch 1**

| schedule | uniform | ramp | alternating | sparse |
|---|--:|--:|--:|--:|
| gain (d=6) | **+21.7%** | +21.5% | +14.6% | +5.9% |

Cutting pre-touch traffic costs proportionally: skipping every other layer loses 7 points,
every fourth loses 16. At batch 1 the pre-touch is not the bottleneck, so there is nothing to
save by doing less of it. Where these schedules should earn their keep is exactly where
prefetch collapsed — 32 concurrent sequences — and that has not been measured.

## Surface: prefetch implementation

**`bench/cuda_bench.cu` · `--prefetch-impl` · frontier: `stream`, decisively**

| distance | 1 | 2 | 4 | 6 |
|---|--:|--:|--:|--:|
| `stream` | +15.2% | +15.9% | +19.4% | **+21.0%** |
| `fused` | -15.3% | -14.2% | -13.1% | -12.1% |

Kernel-integrated prefetch is **30 to 33 points worse at every distance**, and worse than
doing nothing at all. Section 34.4 and the v0.5 roadmap both assume fusing the pre-touch into
the compute kernel reduces overhead; measured, the assumption is wrong. The separate stream
overlaps the walk with compute, while the fused version serialises it inside the kernel that
is trying to make progress.

That does not close the surface — it says the naive fusion is the wrong fusion. A version that
interleaves the touch into the existing load pipeline, rather than appending a second pass
after the update, is untried and is where the roadmap's intuition might still be right.

## Surface: pre-touch kernel

**`src/cuda/prefetch.cu` · `--pre-touch` · 6 strategies · frontier: unresolved**

| strategy | gain |
|---|--:|
| `scalar` | +13.5% |
| `vec4` | +15.0% |
| `vec4_ldcg` | +14.9% |
| `ptx_l2` | +13.4% |
| `warp_tile` | +14.2% |
| `partial` | +14.9% |

`ptx_l2` issues real `prefetch.global.L2` instructions — the "put this range in L2 now"
primitive section 12 assumed CUDA does not expose. PTX does.

**This axis is inside its own noise floor.** The 1.6-point spread is smaller than the 4-5%
run-to-run spread on prefetch modes, so none of the six is reliably ahead. Note also that
`partial`, which touches half the bytes, matches `vec4`, which touches all of them — the walk
is bandwidth-bound and half of it is free. Untried: warp-cooperative movement, TMA, and
touching only the tiles the recurrent kernel reads first.

## Surface: prefetch implementation

**`bench/cuda_bench.cu` · `--prefetch-impl stream|fused`**

`fused` performs the pre-touch inside the compute kernel: no second stream, no extra launch,
no cross-stream ordering. Section 34.4, and the v0.5 roadmap item. It also removes the SM
contention that makes the stream implementation noisy, so it may be the way to make the
pre-touch axis resolvable at all.

## Surface: state layout

**`bench/cuda_bench.cu` · `--state-layout` · 3 layouts · the largest single effect measured**

| layout | baseline | persist | prefetch | combined |
|---|--:|--:|--:|--:|
| `linear` | 7.60 ms | +10.5% | +15.4% | +11.3% |
| `head_interleaved` | **17.81 ms** | **-2.5%** | +10.9% | +10.3% |
| `tile_swapped` | 7.86 ms | +8.2% | **+18.7%** | +14.7% |

Access order alone is a **2.34x** effect — larger than every locality policy in this document
combined. Every layout is a bijection, so the final state is bit-identical; only the order
changes.

The important part is the interaction: `persist` is +10.5% on `linear` and **negative** on
`head_interleaved`, while `prefetch` is best on `tile_swapped`. **There is no layout-independent
best policy.** A planner that ignores layout is choosing in the dark, and a layout-aware
pre-touch — one that walks the state the way the recurrent kernel will read it — is unexplored.

---

## What does not count

- A win inside the noise floor of the mode you changed. `sweep.py` will tell you.
- A synthetic gain with no real-model number. The gate is real end-to-end decode (sections 17,
  21, 28), and `eval/decide.py` refuses to turn a synthetic result into a verdict.
- Any change to final state. 61 configurations across every axis have been verified
  bit-identical; `run_eval.py` refuses to report a result if that breaks.
- Changing the benchmark to make a number move. `bench/` and `eval/` define what is measured;
  a change there is a change to the instrument and is argued separately from the optimization
  it would score.
