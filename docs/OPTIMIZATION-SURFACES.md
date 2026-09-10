# Optimization surfaces

Where the work is, what isolates it, and what is already known.

Every surface is a named value on one enumeration, selected at run time and reported in the
benchmark's JSON. Two people can work on different mechanisms without conflicting, and any
two can be A/B'd in one process against identical state. A change that replaces a file
instead of adding an enumerator cannot be compared against what it replaced.

Sweep one axis with everything else fixed:

```bash
python3 eval/sweep.py --binary ./build/tensortransit_bench --axis prefetch-distance
python3 eval/sweep.py --binary ./build/tensortransit_bench --axis hot-set-policy -- --sequences 32
```

`sweep.py` compares the spread across an axis to that axis's own noise floor and **refuses to
name a winner inside it**, exiting non-zero. An axis that will not resolve is open, not solved.

Two benchmarks appear below and they are not interchangeable. The **real** section is
Qwen3.8-27B on a pinned SparkInfer commit; it is the one the go/no-go gate reads. The
**synthetic** section is `workloads/recurrent/synthetic/cuda_bench.cu`, which explains mechanism and decides nothing
(sections 17 and 28).

---

# Real model: Qwen3.8-27B, pinned SparkInfer, RTX 5090

`adapters/sparkinfer/` · NVFP4 checkpoint, 64 layers (48 recurrent), CUDA 13.3, sm_120 ·
control 96.08 tok/s at ctx 128 · every number is a control/candidate pair measured minutes
apart on the same box.

## Read this first: two ceilings, and the tighter one binds

```
recurrent state per token   48 layers x (3 MiB fp32 + 60 KiB bf16) x 2 (read+write) = 294 MiB
decode step                 10.34 ms at 1792 GB/s                                   =  18.5 GB
recurrent share                                                                       1.66%
throughput ceiling          f/(1-f), the currency the scorer measures                 1.69%
```

`eval/traffic_budget.py` computes it. Qwen3.8-27B is a **dense** hybrid: every weight is read
every token, and at batch 1 that is 98.34% of the traffic. Making recurrent state *free* would
be worth 1.69%, which is below the 2% threshold section 21 used to reject at — **before** any policy is
chosen, and no implementation can move it.

That is the ceiling on removing the traffic. There is a second, much tighter one on the
*persist family specifically*, because a persisting window cannot save traffic it cannot hold.
State written at layer *i* is read again at layer *i* of the **next token**, so the footprint
that must be resident is every recurrent layer for every sequence at once — 146.8 MiB at batch
1 against this device's 60 MiB persisting-L2 capacity, and 2394 MiB (39.9x) at 32 sequences:

| | traffic ceiling | resident footprint | vs capacity | persist ceiling |
|---|--:|--:|--:|--:|
| batch 1 | 1.69% | 146.8 MiB | 2.4x | 0.68% |
| concurrency 4 | 3.01% | 299.2 MiB | 5.0x | 0.59% |
| concurrency 16 | 7.44% | 1197 MiB | 19.9x | 0.35% |
| concurrency 32 | 12.71% | 2394 MiB | 39.9x | 0.28% |

**The persist ceiling falls as the traffic ceiling rises.** Weighted across the section 44
matrix the traffic ceiling is 5.22% and the persist family's is **0.52%** — below the floor at
every concurrency. `eval/traffic_budget.py --matrix configs/rtx5090-section44-ceiling.json
--bandwidth-gbs 1792` computes both, and the bound is generous: it assumes every resident byte
hits and the set-aside costs its neighbours nothing.

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

**`capture_node` IS documented as safe, and this document asserted the opposite.** CUDA's
header for `cudaStreamGetCaptureInfo` — the call the mechanism is built out of — says "All
operations other than destroy and node removal are permitted on the graph while the capture
sequence is in progress" (`cuda_runtime_api.h:2743`, CUDA 13.3, unchanged since 11.3), and the
same paragraph blesses passing the driver-owned dependency array straight to APIs that operate
on the graph. Setting a kernel-node attribute is neither a destroy nor a node removal.

Two further corrections belong here, because both were readings of a counter rather than of a
result:

- "48 of 48 nodes" was read off `windows_attached_to_node`, which counts attach CALLS that hit
  at least one node, not nodes. The counters are now separate (`window_nodes_attached`).
- Nothing had ever read an attribute back. `profiling/capture_attr_probe.cu` does: it captures N
  kernels, sets the window mid-capture exactly as the controller does, ends the capture, and
  reads it back off the finished graph and off a clone. Present and byte-correct on every
  kernel node at 1, 4, 8, 16, 32, 48, 64 and 128 nodes; zero capture invalidations;
  `compute-sanitizer --tool memcheck` clean. The instantiated graph cannot be inspected —
  CUDA 13.3 has no `cudaGraphExecGetNodes` and no exec-level attribute getter — so that half
  is behavioural: the same capture attached with a persisting window and with a streaming
  window over the same buffer replays **3.2% apart** at 48 nodes.

What survives is a precision defect rather than a legality one. `capture_node` marks EVERY
kernel node in the capture's pending dependency set, and on the non-fused convolution branch
the preceding launch is `l2_norm_qk_kernel`, which reads no convolution state — it inherits a
persisting window over memory it never touches. `capture_node_strict` marks only when exactly
one kernel node is pending, and counts the declines in `window_attach_ambiguous`. All three
modes stay opt-in and counted (`stats().capture_invalidations`), and node attachment
self-disables after a first detected invalidation.

The 32-sequence collapse is real and is now **two** things, which a summary table makes look
like one:

- **An 85% drop that is request loss.** Per-request device-memory allocation fails
  (`[qwen35] malloc: out of memory`), the runtime reports it as a per-request warning and
  continues, and the run decodes 320 or 384 tokens where a healthy one decodes 2056 — at the
  same wall time. A failure reported as a slowdown. Identified, and `real_eval.py` now refuses
  such an arm by name.
- **A 32% drop that is still open.** This is the historical one — the `prefetch` ratio 0.676
  above matches a measured 0.648, not the 0.15 that request loss produces. It completes every
  request, its per-token latency is identical to a healthy run's, and it spends 1.2 s more wall
  time somewhere outside the decode loop. It is not a decode-path fallback and it is not the
  NVFP4 per-row projection loop (forcing that loop permanently costs a *stable* 7% at 32
  sequences, not an intermittent 35%). See the changelog.

An earlier version of this document blamed it on `capture_node` mutating the graph. **That
attribution was wrong, and the code proves it.** The two arms that collapsed were `baseline`
(-14.82%) and `prefetch` (-20.09%). `src/planner.cpp:206` sets `use_persist` only for
`Persist`/`Combined`, and `src/cuda/cache_control.cu:260` gates the node-attach arming on
`plan.use_persisting_window` — so in `baseline` and `prefetch` the mechanism is **inert**. It
cannot have caused a collapse in arms that never invoke it. The dedicated 12-run probe agrees:
`capture_failed: 0` in all twelve runs, including four `capture_node` runs.

What is actually known, and one line of it has been **retracted**:

- ~~the collapse is the runtime falling off its batched decode path~~ — **it is not.** Two
  consecutive isolated runs at 32 sequences gave 910.9 and 589.9 tok/s with mean inter-token
  latencies of 19.32 and 19.31 ms. Per-token decode is identical; the lost 1.2 s of wall time
  is somewhere else. A run that had fallen onto a per-row decode path would show it in the
  latency. It is not a cache effect either;
- it costs about 28% of aggregate throughput when it happens;
- it appeared in a 2-repeat screen and in one targeted rerun, and in **zero** of twelve
  isolated runs;
- `baseline` installs no window and issues no pre-touch, so whatever causes it is not a
  locality policy at all.

The honest conclusion is that something about the eval's run *sequence* — accumulated device
state across arms is the obvious candidate — occasionally drops SparkInfer onto its
per-row path at 32 sequences, and this project has not identified it. It is an open problem,
not a settled attribution.

**The part that *is* settled is about the boundary, not the policy.** A locality library can
compute a persisting window; under graph decode it cannot deliver one without touching the
graph, because a stream attribute is host-side state a capture never records. The shortcut
that avoids changing every launch site turns out to be sanctioned after all (above); what it
costs is precision about WHICH node gets marked, not legality. Handing the window back is the
alternative, and in this integration it is a fiction: the adapter drops it.
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

Superseded by a cleaner run: 3 interleaved pairs on one box, with
`window_attach=capture_node` fixed so the persist family actually delivers a window instead of
deferring every one of them. Raw data in
[`results/rtx5090-baseline-matrix.json`](../results/rtx5090-baseline-matrix.json).

| | control | floor | ceiling | `baseline` | `persist` | `prefetch` | `combined` |
|---|--:|--:|--:|--:|--:|--:|--:|
| batch 1 | 96.67 tok/s | 0.04% | 1.69% | -0.04% | **+0.10%** | -1.27% | -1.18% |
| concurrency 4 | 334.17 tok/s | 0.09% | 3.01% | +0.00% | +0.15% | -2.07% | -2.28% |
| concurrency 16 | 790.60 tok/s | 0.13% | 7.44% | +0.06% | +0.06% | **-5.68%** | -5.81% |
| concurrency 32 | 1287.57 tok/s | 0.34% | 12.71% | -0.76% | -0.59% | -7.51% | -7.44% |

"Ceiling" is `eval/traffic_budget.py` at that step time and sequence count, with the state
bf16-compacted as the runtime does for batched decode. Weights are read once per step
whatever the batch; recurrent state once per sequence — so the share of decode traffic that
recurrent state accounts for **quadruples** from batch 1 to concurrency 16, and by 32 there is
genuinely 12.7% on the table.

`persist` is now positive and *resolved* at batch 1 (+0.10% against a 0.04% floor) — its first
real gain anywhere — and decays to negative by 32 sequences. That is the residency bound above
playing out: 0.68% available at batch 1 falling to 0.28% at 32, against a hook overhead
(`baseline`) that grows to -0.76% over the same range. The policy stops paying for itself
before the room runs out.

`prefetch` at 32 sequences carries the runtime fallback: ratios `[0.932, 0.676, 0.925]`, one
run of three collapsing 32%. It is not the locality policy — see the open problems in
[`MINING.md`](MINING.md).

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

Those collapses are the runtime falling off its batched decode path; the section above says
what is and is not known about why. A dedicated 12-run probe — 4 unhooked control, 4
`stream`, 4 `capture_node`, each run in isolation — has no collapses in it at all and gives a
usable figure:

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

`windows_attached_to_node` is non-zero under `--window-attach capture_node` and 0 under the
default, which is the whole point of the section above: telemetry is what distinguishes "the
policy did not help" from "the policy was never applied", and those are different results with
the same number.

**This sentence used to read "48 of 48, so the persisting policy really is present in the
replayed graph there", and that was an inference the counter does not support.**
`windows_attached_to_node` counts attach CALLS that marked at least one node, not nodes — the
48 was a count of hook invocations that happened to equal the node count. A counter that says
a setter returned success says nothing about what survives into a replayed graph.
`window_nodes_attached` now counts nodes, and `profiling/capture_attr_probe.cu` answers the
question the inference was standing in for by reading the attribute back off the finished
graph. See the section above.

---

# Second model: Qwen3.6-35B-A3B, sparse-MoE hybrid, same runtime and same box

`configs/qwen3.6-35b-a3b-moe-ceiling.json` · unsloth `UD-Q4_K_M` GGUF, 22.13 GB · 40 blocks, 30
of them Gated-DeltaNet · 256 experts, 8 used per token · the same pinned SparkInfer commit, the
same hook, the same RTX 5090 · raw data in
[`results/rtx5090-moe-matrix.json`](../results/rtx5090-moe-matrix.json).

This is `docs/MINING.md`'s first open surface, tested. The persist ceiling is
`2 x min(capacity, footprint) / step_traffic`; the capacity is the device's 60 MiB and cannot be
raised, so the only lever is the denominator, and `eval/traffic_budget.py` now prints the
threshold: **a decode step must move at most 6.42 GB** for a persisting window over this
footprint to reach the 2% reference threshold at all. Qwen3.8-27B moves 18.5 GB. This model
   moves 3.56 GB.

**Nothing about the integration changed.** The adapter reads the state geometry from the
runtime's own config, so the same binary brackets a 30-layer 2 MiB state as readily as a
48-layer 3 MiB one: `recurrent_layers: 30, bytes_per_layer: 2146304` in its telemetry, which is
`32 x 128 x 128 x 4 + 3 x 8192 x 2` exactly. Both formulas reproduce the pinned model's numbers,
and the geometry was read from the checkpoint's own metadata and tensor shapes rather than from
the runtime's defaults.

## Two terms move at once, which the residency ratios alone do not show

| batch 1 | Qwen3.8-27B (dense) | Qwen3.6-35B-A3B (sparse MoE) |
|---|--:|--:|
| recurrent layers | 48 | 30 |
| state per layer | 3 MiB + 60 KiB | 2 MiB + 48 KiB |
| recurrent footprint | 146.8 MiB | **61.4 MiB** |
| vs 60 MiB persisting capacity | 2.4x | **1.02x** |
| resident fraction of the state | 41% | **98%** |
| decode step traffic | 18.5 GB | **3.56 GB** |
| traffic ceiling | 1.69% | **3.75%** |
| persist-family ceiling | 0.68% | **3.67%** |

The step shrinks 5.2x *and* the footprint shrinks below the cache. Either alone would help; both
together are what takes the persist family from bounded-out to a live surface.

**How tight is that ceiling?** `traffic_budget.py` now answers rather than leaving it implied.
The checkpoint's own tensor table says a batch-1 step reads 2.63 GB of weights — 1.60 GB of
per-block non-expert weights, 0.42 GB of output head, 0.61 GB for the 8 experts one token
routes to — and with the recurrent traffic that is 2.75 GB against the 3.56 GB peak bandwidth
would allow in the measured time. **77% utilisation**, which the tool reports as `loose`: the
step is mostly but not entirely memory-bound, so a real policy lands below the ceiling for
reasons the arithmetic does not carry.

## Modes

3 interleaved pairs, control **503.18 tok/s** at ctx 128, noise floor **0.078%**.

| mode | gain | vs the dense model |
|---|--:|---|
| `baseline` (hook on, no policy) | +0.05% | the integration still costs nothing |
| `persist` | **+1.26%** | +0.10% there — **12x** |
| `prefetch` | -3.93% | -1.27% there |
| `combined` | -2.56% | -1.18% there |

Paired ratios for `persist`: 1.0137 / 1.0120 / 1.0126. Resolved, and the largest real-model gain
this repository has measured. It is a third of the 3.67% ceiling, and about 43% of what the
*configured* set-aside allows — the default `budget_fraction` 0.75 reserves 48 MiB of the 60 MiB
the device offers, which is 78% of the footprint rather than 98%.

`prefetch` costs three times what it costs on the dense model, and that is arithmetic too: the
pre-touch adds one extra read of the recurrent state, 64 MB against a 3.56 GB step is 1.8% of
the traffic where the same read was 0.35% of an 18.5 GB one, and the graph-node cost is charged
against a step 5.2x shorter.

## Surface: how large the set-aside is, as opposed to how large the constant is

`--axis set-aside-policy`. Every number in this repository before this axis reserved
`persisting_budget_fraction` of the device's persisting-L2 capacity — a constant chosen before
anything is known about the workload — and the measurements said that constant is wrong in
both directions. `SetAsidePolicy` reads the recurrent geometry the runtime already declares.

| value | what it reserves |
|---|---|
| `fixed` | `persisting_budget_fraction` of capacity, whatever the workload. **The control.** |
| `fit_footprint` | `min(windowed footprint, capacity)`. Parameter-free. |
| `residency` | `fit_footprint`, and declines outright below `--axis min-residency` (default 0.50) |

**The windowed footprint is not the token footprint, and conflating them was a defect.** A
persisting window is an address range and this library places ONE per layer, over one
sequence's slice — at concurrency the runtime hands over a device array of per-row pointers for
the pre-touch and a single host-nameable row for the window. So set-aside beyond one sequence's
footprint holds nothing, however many sequences are in flight. The token footprint is what
*competes* for the cache and counts every sequence; it is what the hot-set models count and it
is the wrong number to size a reservation from.

Batch 1 on the MoE checkpoint, three independent runs across two rebuilds. **All three resolve
and all three put the shipped constant last:**

| run | `fixed` | `fit_footprint` | `residency` | spread | floor |
|---|--:|--:|--:|--:|--:|
| 1 | +1.285% | +1.509% | +1.441% | 0.224% | 0.071% |
| 2 | +1.32% | +1.37% | +1.41% | 0.100% | 0.043% |
| 3 | +1.30% | +1.40% | +1.46% | 0.160% | 0.077% |

Worth **+0.09 to +0.22 points over the constant with no dial touched by the operator**, and it
reaches the +1.5% that previously required someone to know to set
`TENSORTRANSIT_BUDGET_FRACTION=1.00` by hand. Quote the range: between-run drift on one
configuration (0.14 points) exceeds any single run's floor.

At four sequences the axis **does not resolve** — two of three runs are inside their floor, and
the one that resolved did not replicate (the same 60 MiB reservation measured -0.99% and
-0.28% on two runs). What does resolve there is an ordering: on `--axis budget-fraction
--concurrency 4`, the largest set-aside is **last in all three pairs**, which is a 1-in-64
coincidence if a MiB of set-aside cost nothing. A sign test, not a magnitude.

`results/rtx5090-setaside.json` carries all of it, including the three integration defects that
had to be fixed before any concurrency number meant anything.

## Axes that were dead on the dense model and are live here

On Qwen3.8-27B the recurrent footprint is 2.4x the persisting capacity at batch 1 and 39.9x at
32 sequences, so no window *shape* mattered: `window-target` spanned 0.02% and `hot-set-policy`
0.01%. At 1.02x the choice is between holding most of the state and holding all of it, and the
same axes resolve.

| `--axis budget-fraction` | set-aside | gain |
|---|--:|--:|
| 0.25 | 15 MiB | +0.68% |
| 0.50 | 30 MiB | +0.96% |
| 0.75 (the shipped default) | 45 MiB | +1.28% |
| **1.00** | 60 MiB | **+1.53%** |

Span 0.85% against a 0.097% floor, 3 pairs with the opening run discarded.

| `--axis hit-ratio` | gain |
|---|--:|
| 0.25 | +0.76% |
| 0.50 | +1.13% |
| 0.75 | +1.41% |
| **1.00** | **+1.50%** |

Span 0.74% against a 0.063% floor, at the default `budget_fraction`.

| `--axis window-target` | bytes it protects | gain |
|---|--:|--:|
| `matrix` | 2 MiB/layer | **+1.31%** |
| `widest` | the same allocation | +1.31% |
| `conv` | 48 KiB/layer | +0.30% |
| `narrowest` | the same allocation | +0.28% |

Span 1.04% against a 0.080% floor — **resolved**, where on the dense model this axis spans 0.02%
and is formally open. `widest` and `narrowest` land on the matrix and conv states respectively,
which is the arithmetic working: the two pairs agree to within their own spreads. Worth noting
that the conv state is 2.3% of the recurrent bytes and still returns +0.30% by itself, because
1.4 MiB of it is entirely resident — the only recurrent state on either model whose whole
allocation fits in a set-aside with room to spare.

**Both dials are monotonic to their maximum with no interior optimum.** That is what a footprint
which nearly fits predicts, and it is the opposite of the synthetic benchmark's finding that
backing off is better under pressure — because there the footprint was many times the budget and
here it is 1.02x. The shipped defaults (0.75, 0.70) leave about a quarter of a point unclaimed.

Capture efficiency against the residency-scaled ceiling falls from 76% at `budget_fraction` 0.25
to 43% at 1.00. That is the deliberately generous bound showing itself: it assumes every resident
byte hits and that the set-aside costs its neighbours nothing, and the second assumption weakens
as the set-aside grows.

| `--axis hot-set-policy` at `budget_fraction=1.00`, `hit_ratio=1.00` | gain | spread |
|---|--:|--:|
| `fixed` | +1.631% | 0.122% |
| `quota` | +1.631% | 0.084% |
| `proportional` (the shipped heuristic) | +1.583% | 0.142% |
| `sqrt` | +1.557% | 0.084% |

Span **0.075% inside a 0.190% floor — OPEN, not solved.** `quota` ties the best figure and beats
the shipped heuristic by 0.048 points, which is inside the noise and therefore not a result.

The telemetry says why, and it is not "the policies are the same". At `budget_fraction=1.00` the
driver grants exactly **60.0 MiB** against a **61.41 MiB** footprint, so the planner does report
every layer oversubscribed and the policy branch is entered. `quota` then attaches **58 of 60**
windows with `hit_ratio_reduced: 2` — it admits 29 of the 30 recurrent layers whole and declines
one, which is `60 MiB / 2.047 MiB` exactly as designed. `proportional` instead asks all 30 layers
for 0.977x the hit ratio. **The two policies genuinely differ, over 3.3% of the state, and
0.048 points is what that difference is worth.**

So the axis is open because the lever is small here, not because the mechanism is inert. The
regime where admitting whole layers should differ materially from shaving every hit ratio is
**concurrency**, where the footprint is 8 to 16x the set-aside and `quota` would admit an eighth
of the layers rather than 29 of 30 — and that has not been measured. It is the most concrete
open item this model leaves.

Worth noting what this table does show: **the two dials compound.** +1.63% together against
+1.53% and +1.50% for each alone, and this is the best measured configuration on either model.

**`cliff` cannot be measured on this model, and the null-candidate guard is right about that.**
Sweeping the whole `hot-set-policy` axis aborts on it: at batch 1 the footprint is oversubscribed,
so `cliff` declines every window, applies no policy at all, and `real_eval.py` refuses the arm by
name — `windows_applied=0, windows_attached_to_node=0, pre_touch_launches=0`. A policy whose
answer to pressure is to do nothing *is* `baseline`. Sweep the axis with
`--values proportional,fixed,sqrt,quota`.

## Read this before quoting any number above: the gate cannot run on this checkpoint

Every figure in this section is throughput, and throughput is all it is. The exact-locality
track requires bit-identical greedy replay (sections 15 and 35), and on this checkpoint that
**cannot be established**: two *unhooked* control runs of the same binary on the same prompt
diverge at token 2.

```
control A: 13 271 760 1879 369 264 1103 314 4947 ...
control B: 13 271 760 2614 369 264 1103 314  279 ...
```

`kernels/include/sparkinfer/kernels/deterministic.h` documents the mechanism: a few ULP in the
prefill feed *discrete* top-k expert routing and int8 requant, so over 40 layers one flips an
expert and moves the argmax. `SPARKINFER_DETERMINISTIC=1` exists and does not cover this
checkpoint's Q4_K expert path — two controls still diverge with it set.

**RecurLocal is not the cause.** On the dense Qwen3.8-27B, the same binary with the same policy
at the same settings gives control, control and candidate bit-identical over every token
compared. The nondeterminism belongs to the model and the runtime, not to the locality policy.

So this whole section is an **unscorable** result: interesting, reproducible as timing, and
refused by `decide.py` because the question the gate exists to answer has no answer here. A
reproducible MoE checkpoint is the first thing a contributor to this surface needs.

## Concurrency, once the runtime can be made to batch this checkpoint

Aggregate throughput at 16 and 32 sequences first came in at 453 and 456 tok/s — *below* the
503 tok/s single-sequence rate. The adapter's packing counters said why, one isolated run per
width, and the runtime's own stderr named the cause. Above 8 rows it stops batching:
`launch_mmvq_q4k_rows` refuses `M > 8` and the bf16 `launch_mmvq_rows` dispatcher has no
chunking loop where three of its four siblings do. `SPARKINFER_PACKED_MAX_ROWS=8` makes the
engine split a wide batch into packs the GEMV accepts, and the arms become measurable. See
[`MINING.md`](MINING.md) for the full account — it is a runtime defect worth 5.4x, not a
locality result.

Every row below is measured with that cap set on **both** arms, three interleaved pairs, the
opening run discarded.

| | control | floor | footprint vs cache | traffic ceiling | persist ceiling | `baseline` | `persist` | `prefetch` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| batch 1 | 503.2 tok/s | 0.08% | **1.02x** | 3.75% | **3.66%** | +0.05% | **+1.26%** | -3.93% |
| concurrency 4 | 912.2 | 0.46% | 2.09x | 3.47% | 1.63% | -0.13% | -0.46% | -3.66% |
| concurrency 16 | 1206.9 | 0.57% | 8.38x | 4.64% | 0.53% | -0.06% | -0.17% | -2.81% |
| concurrency 32 | 1230.7 | 0.26% | 16.75x | 4.74% | 0.27% | -0.16% | -0.22% | -2.67% |

**The crossover is visible, and it is where residency crosses one half.** `persist` pays at
batch 1, where 98% of the footprint is resident, and is negative from four sequences on, where
48% is. It is not that the room disappears — the *traffic* ceiling rises from 3.75% to 4.74%
across the matrix, as it does on the dense model. It is that the fraction of that room a
persisting window can hold falls faster: 3.66% to 0.27%. Past the crossover the set-aside costs
the weight stream more than the residency returns, and `persist` sits a few hundredths below
`baseline` — which is the hook's own overhead and nothing else.

## The best configuration is workload-dependent, and the scored run shows the cost

The batch-1 axis sweeps resolve `budget_fraction=1.00, hit_ratio=1.00` as best there — +1.63%
against +1.26% at the shipped defaults. The scored matrix holds those settings on every arm, and
at concurrency they are **worse** than the defaults:

| | shipped defaults (mode axis) | tuned for batch 1 (scored run) |
|---|--:|--:|
| batch 1 | +1.26% | **+1.74%** |
| concurrency 4 | -0.46% | **-1.29%** |
| concurrency 16 | -0.17% | -0.79% |
| concurrency 32 | -0.22% | -0.75% |

Same residency story from a third direction: a set-aside sized for a footprint that fits costs
the weight stream more once it does not. Batch 1 is +1.74% across all three contexts
(+1.78 / +1.76 / +1.68 at 128 / 4096 / 16384), so the gain is not a short-context artefact.

**A planner that chose the set-aside from the measured footprint-to-capacity ratio, rather than
from a constant, is an open item** — and this is the evidence for it. Nothing in the library
currently varies `budget_fraction` with the workload; the runtime declares `TENSORTRANSIT_SEQUENCES`
and the hot-set model already computes the footprint, so the input is there and only the policy
is missing. `results/rtx5090-moe-scored.json` carries the run.

## What the whole matrix can pay, on each model

`eval/traffic_budget.py --matrix ... --persisting-l2-bytes 62914560`, weighted the way
`decide.py` weights:

| | dense Qwen3.8-27B | sparse-MoE Qwen3.6-35B-A3B |
|---|--:|--:|
| weighted traffic ceiling | 5.22% | **4.07%** |
| weighted persist-family ceiling | **0.52%** | **1.94%** |
| the share of removable traffic a persisting cache can address | 10% | **48%** |

Two things worth reading carefully. The MoE's *traffic* ceiling is **lower**, because its
concurrency steps are short and weight-light, so there is less total room. But the persist
family reaches **48% of that room instead of 10%**, and 1.94% weighted instead of 0.52%.

**1.94% weighted, and 0.52% on the model that is scored.** The best model this project has found, with both persistence
dials at maximum and a perfect replacement policy assumed, misses the significance floor by six
hundredths of a point. That is not a tuning gap and no policy closes it: the numerator is the
device's 60 MiB and the denominator is what the workload moves.

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

**`src/recurrent_planner.cpp` · `--hot-set-policy` · 5 policies · frontier: `cliff`, and `quota` is untried here**

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
refuses the window when oversubscribed loses 16 points less than it.

`quota` is a fifth policy and is not in that table, because the table predates it. Every policy
above answers oversubscription by moving one dial — the hit ratio — for every layer alike, which
models the cache as something that can keep 97% of a byte. `quota` instead admits whole layers
at the full hit ratio until the set-aside is spent and declines the window for the rest, spread
evenly and decided from the layer ordinal alone so the choice cannot move between tokens. It is
inert where the footprint is many times the budget — which is every regime the table above
measures, and why it was not worth building until a model existed whose footprint is 1.02x the
cache rather than 2.4x. Backing off the hit ratio
does not work; declining to compete does.

The residency bound at the top of this document points the same way: a window over a footprint
the cache cannot hold does not degrade gracefully, so refusing it beats shrinking it. That is a
hypothesis here rather than a result — this is the synthetic bench, which does not capture a
CUDA graph and has disagreed with the real model on four axes — and the real model shows a 0.01%
spread across all four hot-set policies at batch 1, where the footprint is only 2.4x
oversubscribed. Whether the two are the same effect is open, and settling it with hardware
counters is a real contribution.

## Surface: cache interference and QoS

**`workloads/recurrent/synthetic/cuda_bench.cu` · `--stream-bytes`, `--qos` · frontier: unexplained**

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

**`src/recurrent_planner.cpp` · `--prefetch-distance 0..8` · frontier: +21.0% at d=6**

| distance | 1 | 2 | 3 | 4 | 6 | 8 |
|---|--:|--:|--:|--:|--:|--:|
| gain | +14.7% | +15.6% | +17.0% | +18.2% | **+21.0%** | +20.5% |

Hard-coded to 0-or-1 until v0.2, leaving six points on the table. The curve has an interior
optimum: too far ahead and the state is evicted before its layer runs. That optimum depends on
state size, layer compute time and L2 pressure, so it is almost certainly not 6 on a real
model. A planner that derives it instead of taking a constant is an open problem.

## Surface: prefetch schedule

**`src/recurrent_planner.cpp` · `--prefetch-schedule` · 4 schedules · frontier: `uniform`, at batch 1**

| schedule | uniform | ramp | alternating | sparse |
|---|--:|--:|--:|--:|
| gain (d=6) | **+21.7%** | +21.5% | +14.6% | +5.9% |

Cutting pre-touch traffic costs proportionally: skipping every other layer loses 7 points,
every fourth loses 16. At batch 1 the pre-touch is not the bottleneck, so there is nothing to
save by doing less of it. Where these schedules should earn their keep is exactly where
prefetch collapsed — 32 concurrent sequences — and that has not been measured.

## Surface: prefetch implementation

**`workloads/recurrent/synthetic/cuda_bench.cu` · `--prefetch-impl` · frontier: `stream`, decisively**

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

**`executors/cuda/prefetch.cu` · `--pre-touch` · 6 strategies · frontier: unresolved**

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

**`workloads/recurrent/synthetic/cuda_bench.cu` · `--prefetch-impl stream|fused`**

`fused` performs the pre-touch inside the compute kernel: no second stream, no extra launch,
no cross-stream ordering. Section 34.4, and the v0.5 roadmap item. It also removes the SM
contention that makes the stream implementation noisy, so it may be the way to make the
pre-touch axis resolvable at all.

## Surface: state layout

**`workloads/recurrent/synthetic/cuda_bench.cu` · `--state-layout` · 3 layouts · the largest single effect measured**

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
