# TensorTransit

**TensorTransit is an engine-independent, cross-kernel, multi-tensor locality planner for AI
inference.**

> SparkInfer makes kernels fast. TensorTransit coordinates the data those kernels will need next.

Modern inference kernels can be extremely fast and still stall waiting for the next tensor.
TensorTransit builds a **future-use Transit Graph** across kernel boundaries and compiles it
into cache-residency, streaming, prefetch and overlap actions, so a runtime spends less time
waiting on memory.

It does not implement model math, own the KV cache, allocate anything, or replace an inference
server. It is a component a runtime embeds.

```text
NOT "prefetch tensors"    NOT "manage GPU memory"    NOT "another cache"

YES  a future-use graph
   + several semantic tensor classes
   + one global cross-kernel planner
   + short-timescale on-GPU locality actions
   + real inference evaluation
```

> **[`docs/VERDICT.md`](docs/VERDICT.md) — is there a scorable surface here, and if so which
> one.** Answered from measurements, with a number for each part. Read it before deciding
> whether to spend a week here; it is written to talk you out of it if the answer is no.
>
> The short version, and it is two halves: **the recurrent traffic at concurrency is 5.2% of a
> decode step at sixteen sequences and 8.6% at thirty-two, against control spreads of 0.42% and
> 1.74% — the room is real and it is ten times the noise. A persisting-L2 policy cannot reach
> it**, and on five of the ten scored cells it provably cannot be measured trying: its ceiling
> there is `2 x 60 MiB / step traffic`, which is smaller than the control's own run-to-run
> spread. One command prints the whole table:
>
> ```bash
> tools/tt-frontier generation show TTF-1 --reachable
> ```

## The answer this project already has, before you read the rest

RecurLocal — now the recurrent-state workload inside TensorTransit — asked whether a
persisting-L2 policy over recurrent state could produce a large end-to-end **throughput** gain.
It cannot, on any device this project can reach, **and the reason is arithmetic rather than
implementation**:

```text
ceiling = 2 x min(persisting capacity, footprint) / decode step traffic
```

The numerator is pinned at 60 MiB by the hardware. Weighted across the full workload matrix on
the best model found — a sparse-MoE hybrid whose decode step moves 3.56 GB instead of a dense
hybrid's 18.5 — that ceiling is **1.94%**, with both persistence dials at maximum and a bound
that already assumes every resident byte hits. On the scored dense model it is **0.52%**. The
best measured real gain is **+1.74% at batch 1**, on a checkpoint that **cannot be scored**
because the runtime is not reproducible on it.

**The 0.2 generalization does not repeal that bound.** It relocates it: the bound applies to one
planner (`recurrent_v0`) over one tensor class, rather than to the project. And it bounds
throughput specifically — it says nothing about latency, which is the other half of what a
serving frontier is. Latency is now measured; it is not yet resolvable, and one cell's p99
control spread was calibrated at **481%**. See below.

**0.2.1 draws the consequence for how work here is scored.** Until 0.2.1 a submission was
sorted into `XS`/`S`/`M`/`L`/`XL` with the lowest paying step at 2% weighted throughput gain —
*above* the 0.52% physical ceiling. A contributor could remove every recoverable byte of
recurrent traffic and score `none`. A band structure whose lowest step sits above what the
hardware can deliver is not a strict regime; it is a broken instrument telling contributors
something false about where the room is.

The bands are gone. What replaces them is a continuous **Frontier Gain**:

```text
dF = F(candidate) / F(main) - 1
```

`F` is the normalized Pareto hypervolume of the serving frontier — **goodput against p99
inter-token latency** — over a frozen matrix of workload cells whose bounds and noise floors
were calibrated on the target hardware and are published per cell.

The second objective is not decoration: the 0.52% ceiling above bounds *throughput* only. But
it is also **not yet usable**, and that is measured rather than assumed — the control p99 spread
is 0.5–3.7% per cell at short context against arm effects of 0.3–2.3%, so at three paired
repeats only two of fifteen latency figures cleared their own noise
(`results/rtx5090-0.2.1-arms.json`), and at 32 sequences the generation's own calibration froze
that cell's p99 spread at **481%**. Raising it needs repeats, not a better policy.

**The first run of the whole ten-cell generation is what put numbers on all of that**, and the
numbers are not flattering to either the policy or the instrument: the receipt read −99.5%,
two of the ten cells turned out not to be servable as concurrency cells by this runtime at all,
two more were refused by a guard dividing by the wrong thing, and one was decided at the cell
floor on the 481% axis. The last three were the evaluator's and all three are fixed;
[`results/rtx5090-ttf1-first-matrix.json`](results/rtx5090-ttf1-first-matrix.json) and
[`docs/VERDICT.md`](docs/VERDICT.md) section 8 have it. Read those,
[`frontier/README.md`](frontier/README.md) and [`docs/MINING.md`](docs/MINING.md) before
spending a week here.

## What 0.2.1 adds

0.2.0 built the core and left it in no measured path: the adapter and the synthetic benchmark
both drove the 0.1 controller directly, so **writing a planner changed nothing about the
number the evaluator prints**. 0.2.1 connects it and checks the connection rather than
asserting it.

| | |
|---|---|
| the adapter drives Registry -> Graph -> Planner -> Executor | with the 0.1 controller kept as `TENSORTRANSIT_ENGINE=v0`, so the migration is an A/B in one binary. Token-exact, 96 windows on captured graph nodes, 0 capture invalidations, gains overlapping inside their own noise floors |
| KV is registered | `recurrent_tensors: 48, kv_tensors: 16` on a live run. The second proof track could not be measured before — not "had not been" |
| a cost model that can express coordination | fitted to the hardware arms in `results/`; superlinear in residency, so concentrating beats spreading and the admission axis measures something |
| the second proof track, at model level | the global arm beats the best independent arm on all three golden traces, where under the linear model it provably could not — and `--stream-relief 0` makes the advantage disappear, which names the mechanism and would falsify it |
| Frontier Gain and its ledger | continuous `dF`, frozen generations, paired bootstrap, protected-workload guard, append-only receipts |
| a trusted, keyless, ephemeral GPU runner | plus the anti-gaming overlay, both proven by CI rather than described |
| plan replay, a gated overhead budget, live trace recording | the offline loop closes; the 0.5%-of-token budget is asserted on the real 64-layer shape |

```console
$ tools/tt-frontier generation show TTF-1
$ scripts/trusted_eval.sh --candidate <ref> --model <path> --pr 184
$ tools/tt-frontier ledger show TTF-1
```

## What 0.2 adds, and what it found

The core is five objects, and every policy decision lives on the CPU side where a test can
reach it without a GPU:

```text
TensorRegistry  ->  TransitGraph  ->  ITransitPlanner  ->  TransitPlan  ->  ITransitExecutor
   (what)             (when)            (decide)           (actions)          (CUDA)
```

Five planners (`baseline`, `recurrent_v0`, `greedy`, `budgeted`, `concurrency`), five
admission rules (`density`, `quota`, `proportional`, `reuse_order`, `role_floor`), three reuse
metrics, a trace format, a plan format, golden plan digests, and a CLI that does all of it
offline. [`docs/architecture.md`](docs/architecture.md) is the map.

### It reproduces the 0.1 bound from a recorded trace

The 0.1 ceilings were computed from a hand-written geometry file. The Transit Graph derives
them from a trace, which is a stronger statement — the rates and the state shape now come from
the same recording, so they cannot be mismatched:

```console
$ tensortransit inspect tests/golden/trace_recurrent.json
  traffic  18500000000 B per iteration

per role:
  role                  tensors     uses          bytes      removable
  recurrent_state            96       96      153944064      307888128
  model_weight               64       64    18192111872    18192111872

ceilings [MODEL, not measurements]:
  policy scope          unlimited on this device     resident     held B
  recurrent_state         +1.692%        +0.685%        40.9%   62914560
```

293.6 MiB removable, 146.8 MiB footprint, 40.9% resident, +1.69% traffic ceiling, +0.685%
persist ceiling — against 294 MiB, 146.8 MiB, 41%, 1.69% and 0.68% published before any of
this code existed. `tests/test_golden.cpp` asserts the agreement; if the graph stops
reproducing it, the graph is wrong.

### And it found a negative result about its own central claim

The specification's second proof track asks the global planner to beat naive independent
tensor policies. On a two-role trace:

```console
$ tensortransit compare tests/golden/trace_recurrent_kv.json
  arm                actions  declines    committed B    predicted
  baseline                 0       176              0      +0.000%
  recurrent_only          60       146       47185920      +0.358%
  kv_only                  8       172       47185920      +0.179%
  naive_both             224        64       47185872      +0.034%
  global                  54       149       47185920      +0.336%
```

It beats the naive both-persistent arm by **10x** — that policy shaves thirty hit ratios to a
third each and the hardware cannot keep a third of a line; under concurrency it declines
everything and disables itself. It does **not** beat the best single-role arm, and under this
cost model it cannot: total saving is `sum granted_i x density_i` under a budget, greedy-on-
density is optimal for that, and any floor that diverts budget to a lower-density role must
lose by exactly the density difference. The sweep is monotone.

**So the honest state of the multi-tensor claim is: the mechanism is built, tested and
observable; the arithmetic that would justify it is not in the cost model; and settling it
needs hardware.** The three terms a linear model cannot express — whole-line residency,
survival past the reuse distance, and interference between the streaming and persisting halves
of the cache — are named in [`docs/evaluation.md`](docs/evaluation.md), and a cost model
carrying them is the highest-value contribution available right now. It needs no GPU.

Everything above marked `predicted` is a **cost-model output**. Every artifact carrying one
declares `"basis": "model"`, and `eval/test_schemas.py` fails a plan that omits the marker.
Only `eval/decide.py --real` produces evidence about a speedup.

### The migration did not change what the policy does

Generalizing a working library is a good way to break it quietly, so the claim was measured
rather than asserted. Qwen3.8-27B on the pinned SparkInfer commit, batch 1, ctx 128, three
interleaved pairs, one RTX 5090 — through the migrated stack: `namespace tensortransit`,
`adapters/sparkinfer/`, the `TENSORTRANSIT_` environment prefix, the `TensorTransit` CMake
package.

```console
>> token-exact greedy replay gate
    control reproducible over 3 unhooked replays
    candidate vs control: identical over 64 tokens
>> pair 1/3   control: ctx128=96.11    candidate: ctx128=96.24
>> pair 2/3   control: ctx128=96.11    candidate: ctx128=96.22
>> pair 3/3   control: ctx128=96.17    candidate: ctx128=96.30

batch-1 decode: +0.13%  (noise floor 0.06%)
```

**+0.13% against a 0.06% floor** is the figure 0.1 published for this arm, and the hook
telemetry confirms the policy reached the captured decode graph — `windows_attached_to_node:
96`, `window_nodes_attached: 96`, `capture_invalidations: 0`. Raw data in
[`results/rtx5090-0.2-migration-check.json`](results/rtx5090-0.2-migration-check.json).

It is a **partial** matrix — batch 1 only, 0.40 of the section 44 weight — and `decide.py`
refuses to call it significant and names the missing 60%. That is the instrument working, and
it is why this is filed as a migration check rather than as a result.

## The first workload: recurrent state

Hybrid LLMs increasingly combine full attention with recurrent / linear-attention layers. Qwen3.8-27B, for example, has 64 language layers with a repeating pattern of three linear-attention layers followed by one full-attention layer. Its recurrent state uses FP32 and has 48 value heads of dimension 128×128, which is 3 MiB of matrix state per recurrent layer.

That state is mutable and repeatedly read/written during decode, in a known layer order —
which is what makes it a good first locality target, and why TensorTransit must not be
branded as a recurrent-state project. The `recurrent_v0` planner experiments with the CUDA
memory hierarchy rather than changing model math:

- reserve a bounded persisting-L2 set-aside when supported;
- mark hot recurrent-state windows as persisting;
- pre-touch the next recurrent layer's state asynchronously;
- rotate the hot window according to known layer execution order;
- measure whether this improves **end-to-end** decode, not just a microbenchmark.

CUDA's access-policy windows are hints, not placement guarantees. Every policy here is treated
as an experimentally measured optimization, never as an assumed win.

## Non-goals

TensorTransit does **not**:

- implement model math — attention, GEMM, Gated DeltaNet / KDA / Mamba;
- replace SparkInfer, vLLM, SGLang, TensorRT-LLM or FlashInfer;
- own KV allocation, recurrent-state allocation, or any allocation at all;
- page anything to CPU/NVMe, or transport anything between machines;
- quantize anything;
- change model outputs;
- claim any speedup before hardware measurements exist.

| TensorTransit owns | TensorTransit does not own |
|---|---|
| the future tensor-use graph | model math |
| the cross-kernel reuse model | attention / GEMM / GDN kernels |
| the global locality planner | KV allocation |
| multi-tensor L2 QoS | recurrent-state allocation |
| prefetch timing and streaming decisions | CPU/NVMe offload, distributed KV transport, RDMA |
| cache-policy lifetime and cross-stream overlap | model quantization |
| Transit Plan execution | full inference scheduling, serving API |

If the repository drifts into the right-hand column it has lost its identity. See
[`docs/design-principles.md`](docs/design-principles.md).

## Architecture

```text
inference runtime
      |
      |  register_tensor(role, bytes)         -- what exists
      |  record_use(kernel, tensor, access)   -- what will be touched, in what order
      v
+---------------------------------------------------------------+
|                          TensorTransit                        |
|   TensorRegistry -> TransitGraph -> ITransitPlanner -> Plan    |
|                                                        |      |
|                                              ITransitExecutor  |
+---------------------------------------------------------+-----+
                                                          |
                                                          v
                                              CUDA memory hierarchy
                                                  L2 <----> HBM
                                                          |
                                                          v
                                                   model kernels
```

## First real result

One RTX 5090, CUDA 13.3. **Qwen3.8-27B** (NVFP4, 64 layers, 48 of them recurrent) on a
**pinned SparkInfer commit** (`5347b27c`), through the adapter in
[`adapters/sparkinfer/`](adapters/sparkinfer/). Control and candidate are the same
binary — the hook is inert unless `TENSORTRANSIT` names a mode — run interleaved on the same box.
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

Under `stream` the policy never reaches the replayed graph, so there is nothing to measure —
and in this integration it never could: the adapter keeps the boolean and drops the window the
controller hands back, so that -0.019% is the hook's overhead, not a policy.

`capture_node` reaches the graph by setting the attribute on the node the capture just
recorded. **An earlier version of this README called that undocumented. It is not.** CUDA's
own header for `cudaStreamGetCaptureInfo` says "All operations other than destroy and node
removal are permitted on the graph while the capture sequence is in progress"
(`cuda_runtime_api.h:2743`, unchanged since CUDA 11.3), and blesses passing the driver-owned
node array directly to graph APIs. `profiling/capture_attr_probe.cu` reads the window back off the
finished graph at 1 to 128 nodes — present and byte-correct every time, memcheck-clean — and a
persisting-versus-streaming A/B over the same buffer separates by 3.2% on replay, which a
policy absent from the replay could not do.

The residual risk is precision, not legality: `capture_node` marks every kernel node the
capture has pending, and on the non-fused convolution branch that includes a kernel which
reads no recurrent state. `capture_node_strict` marks only when exactly one kernel node is
pending and counts the rest in `window_attach_ambiguous`. All three stay opt-in and counted,
and node attachment self-disables after a first detected invalidation.

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

verdict: reject   weighted gain +0.059%   status NO_FRONTIER_GAIN   significant false
  batch1          +0.184%  (w=0.40)
  concurrency16   +0.000%  (w=0.20)
  concurrency32   -0.163%  (w=0.20)
  concurrency4    +0.090%  (w=0.20)
  unresolved: ['concurrency16', 'concurrency32', 'concurrency4']
```

Batch 1 is the only arm that resolves, at +0.184% against a 0.01% noise floor — a real gain,
and roughly a quarter of the 0.68% a persisting cache can reach there. Every concurrency arm
sits inside its own run-to-run spread, and the verdict is **reject**: +0.059% weighted, far
below the noise this matrix can resolve at concurrency.

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
*f/(1−f)* — small enough to be worth knowing before any policy is
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
So the share of traffic a recurrent-state policy can address grows with concurrency. Measured on one RTX
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
the persist family specifically tops out at **0.52%** of throughput, at every
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
| measured `persist`, both dials at maximum | — | **+1.63%** |

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
+0.76% / +1.13% / +1.41% / **+1.50%** over 0.25 → 1.00, span 0.74% against a 0.063% floor. Both
dials are monotonic to their maximum with no interior optimum, which is what a footprint that
nearly fits predicts: nowhere on this model is asking for less persistence better than asking
for more. On the dense model neither dial has been swept —
the axis that *was* swept there, `window-target`, spans 0.02% and is formally open, and the
arithmetic says the persistence dials should behave the same way because no fraction of a 2.4×
oversubscribed footprint can be held. That is a prediction, not a measurement, and this
document previously stated it as one.

The two dials compound: together they give **+1.63%**, the best measured configuration on either
model. The shipped defaults (0.75, 0.70) were leaving a third of a point on the floor, and even
at the far end there are **2.0 points of headroom** to the ceiling — which is what makes batch-1
decode a surface here rather than the dead end it is on the dense model.

The new `quota` hot-set policy — admit whole layers at the full hit ratio rather than shave
every layer's — ties the best figure there (+1.631%) but the axis spans 0.075% inside a 0.190%
floor, so it is **open, not solved**. That fits: at 1.02× oversubscription there is almost
nothing for a policy to ration. The regime where it should differ is concurrency, where the
footprint is 8–16× the set-aside, and that is unmeasured.

### It does not rescue concurrency, and the crossover is at half residency

With the runtime made to batch this checkpoint (below), all four arms are measurable:

| | control | floor | footprint vs cache | persist ceiling | `persist` |
|---|--:|--:|--:|--:|--:|
| batch 1 | 503.2 tok/s | 0.08% | **1.02×** | **3.66%** | **+1.26%** |
| concurrency 4 | 912.2 | 0.46% | 2.09× | 1.63% | −0.46% |
| concurrency 16 | 1206.9 | 0.57% | 8.38× | 0.53% | −0.17% |
| concurrency 32 | 1230.7 | 0.26% | 16.75× | 0.27% | −0.22% |

`persist` pays where 98% of the footprint is resident and is negative from four sequences on,
where 48% is. The *traffic* ceiling still rises across the matrix (3.75% → 4.74%); the fraction
a persisting window can hold falls faster (3.66% → 0.27%). Past the crossover the set-aside
costs the weight stream more than residency returns.

Weighted across the whole matrix, and this is the number that decides the project:

| | dense Qwen3.8-27B | sparse-MoE Qwen3.6-35B-A3B |
|---|--:|--:|
| weighted traffic ceiling | 5.22% | 4.07% |
| **weighted persist-family ceiling** | **0.52%** | **1.94%** |
| share of removable traffic a cache can address | 10% | **48%** |

The MoE's total room is *smaller*, but the persist family reaches 48% of it instead of 10% —
**1.94% weighted, and 0.52% on the model that is scored.** The best model this work found, with
both dials at maximum and a perfect replacement policy assumed, misses **the project's own 2%
go/no-go bar** by six hundredths of a point. No policy closes that: the numerator is the
device's 60 MiB and the denominator is what the workload moves.

That bar decides whether this *research direction* continues; it does not decide whether a
submission counts, and as of 0.2.1 neither scorer lets it. `tools/tt-frontier` reports a
continuous Frontier Gain, and `eval/decide.py` calls a result significant when it is positive,
complete and clear of its own run-to-run spread — the rule the rest of the harness uses. A
threshold above the physical ceiling is a broken instrument in either scorer, and the impact
bands were removed for exactly that reason.

### The one thing that disqualifies this result: the gate cannot run here

The exact-locality track requires bit-identical model output (overview sections 15 and 35), and
on this checkpoint **that cannot be established** — not because the policy changes anything, but
because the runtime does not agree with itself. Two **unhooked** control runs of the same binary
on the same prompt diverge at token 2:

```
control A: 13 271 760 1879 369 264 1103 314 4947 ...
control B: 13 271 760 2614 369 264 1103 314  279 ...
```

The runtime documents the mechanism itself: a few ULP of difference in the prefill feed
*discrete* top-k expert routing, so over 40 layers one flips an expert and moves the argmax.
`SPARKINFER_DETERMINISTIC=1` does not cover this checkpoint's Q4_K expert path — two controls
still diverge with it set.

**The locality policy is not the cause, and that is checkable.** On the dense Qwen3.8-27B, the same
binary and the same policy at the same settings give control, control and candidate as
bit-identical over every token compared.

So the MoE numbers above are sound *as throughput* — three interleaved pairs, tight floors,
monotonic axes — and the result is **not scorable** under the project's own rules. `decide.py`
refuses it, which is correct. What was *not* correct was the first attempt's reason: it said the
candidate had changed model output. It had not, and the harness could not tell the difference.
`real_eval.py` now replays the control against itself before the candidate is compared to
anything, records `runtime_reproducible`, and reports **inconclusive** rather than accusing a
submission of a defect that belongs to the runtime.

A scorable MoE result needs a reproducible checkpoint. That is now the first blocker on this
model, ahead of any policy question.

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
weighted workload score and the ledger's status vocabulary, as a deterministic function of
measurements. It no longer applies an impact band, because there is no longer a band table:
see [`frontier/README.md`](frontier/README.md).

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

TensorTransit is a component, not an application. It installs a CMake package:

```cmake
find_package(TensorTransit 0.2 REQUIRED)
target_link_libraries(your_runtime PRIVATE TensorTransit::tensortransit_cuda)
```

The 0.1 spelling — `find_package(RecurLocal 0.1)` and the `RecurLocal::*` targets — still
resolves and CI consumes the installed package under both names. It is deprecated and will be
removed no earlier than 0.3.0; [`docs/STABILITY.md`](docs/STABILITY.md) section 6a says what
that will and will not break, and which two names are deliberately not moving.

The five-call runtime surface:

```cpp
tensortransit::TransitRuntime tt;
tt.set_device_profile(profile);
tt.set_planner("budgeted", config);
tt.set_executor(&executor);

const auto handle = tt.register_tensor({ptr, bytes, TensorRole::RecurrentState});
tt.begin_recording();
tt.record_kernel({kernel_id, order});
tt.record_use({handle.id, kernel_id, AccessKind::ReadWrite});
tt.end_recording(/*cyclic=*/true);   // a decode token is one iteration of a loop

tt.compile(state);                   // once; reused across tokens
for (each kernel) { tt.before_kernel(id); launch(); tt.after_kernel(id); }
```

Construction and every entry point are `noexcept` and report `cudaError_t`; the controller
detects CUDA Graph capture and hands the access-policy window back for the caller to attach to
its kernel node, because a stream attribute is not recorded into a graph. See
`adapters/sparkinfer/README.md` for the full contract.

## Build: CPU-only

```bash
cmake -S . -B build -DTENSORTRANSIT_BUILD_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tensortransit_info
```

Most of the frontier is reachable from this build: the planners, the graph, the plan schema,
the golden digests and the whole CLI are host code.

## The CLI

```bash
tensortransit inspect <trace.json>     # roles, reuse, and the ceiling that binds
tensortransit plan    <trace.json> --planner budgeted --admission role_floor
tensortransit compare <trace.json>     # the five policy arms over one trace
tensortransit devices                  # device profiles, read off real hardware
tensortransit planners
```

`inspect` is the one to run first. If the device-bounded ceiling for the roles your policy
may touch is under the run-to-run spread of the cells it would be measured in, nothing here
can help you — one command, no hardware. `frontier/TTF-1/reference.json` publishes both.

## Build: CUDA

```bash
cmake -S . -B build \
  -DTENSORTRANSIT_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Synthetic CUDA benchmark

```bash
./build/tensortransit_bench baseline
./build/tensortransit_bench persist
./build/tensortransit_bench prefetch
./build/tensortransit_bench combined

python3 eval/run_eval.py --binary ./build/tensortransit_bench
```

The synthetic benchmark emulates 48 recurrent layers with 3 MiB of state per layer, over 32 timed tokens preceded by 4 untimed warm-up tokens. It is a memory-locality experiment, **not** a faithful GDN model benchmark.

Benchmark options: `--layers`, `--tokens`, `--warmup-tokens`, `--inner-iters`, `--state-bytes`, `--device`.

The evaluator runs every mode `--repeats` times (default 3), interleaved so drift does not land on one mode, and compares median timings. It refuses to report a result unless every run of every mode produced the same final-state checksum, and marks the run unstable when the run-to-run spread is wider than the difference being measured — the bands below start at 2%, which a single run per mode cannot resolve. Pre-touch work is joined into the timed region, so prefetch overhead counts against prefetch modes.

Every result records what produced it — GPU, compute capability, driver and runtime version,
commit and whether the tree was dirty, and the observed graphics clock. Pass
`--pin-clock-mhz auto` (needs privileges) to lock the clock so an absolute time is
reproducible off the box, not merely same-box comparable.

## Integration contract

A runtime needs only the lifecycle hook shown in `adapters/sparkinfer/README.md`. The
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
(**5.22%** of throughput) so that nobody spends a week chasing room that is not there. That
figure bounds ONE of the frontier's two objectives; what a resident state does to a p99 tail is
unmeasured.

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
