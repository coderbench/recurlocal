# Design

## Core hypothesis

Recurrent state differs from streamed model weights: it is relatively small per layer, mutable, revisited every decode token, and accessed in a predictable layer order.

RecurLocal tests whether explicit cache policy plus next-layer warm-up can exploit this reuse.

## CUDA semantics

CUDA exposes stream-level access-policy windows with `cudaAccessPropertyPersisting` and `cudaAccessPropertyStreaming`, plus device-reported limits for L2 set-aside and access-window size.

These are hints. Oversubscription can cause thrashing, so all policies must be measured.

## Prefetch semantics

There is no general CUDA API that guarantees an arbitrary `cudaMalloc` region is placed into L2. The v0 prototype uses an asynchronous read-only pre-touch kernel. It may help or hurt; the evaluator decides.

## CUDA Graphs

Production inference captures decode into a graph, and the capture boundary is where a naive
locality layer breaks. `cudaStreamSetAttribute` sets host-side stream state; it is not an
operation the graph records, so a window installed during capture is absent from every replay
and the persist mode quietly becomes a no-op. Launching pre-touch on a second stream is worse:
a fork that is never rejoined makes the capture end invalid.

The controller therefore detects capture rather than assuming a stream. Under capture it
declines to set the stream attribute and returns the window in `LayerActions` for the caller
to attach to its kernel launch or graph node, and it forks *and* rejoins the prefetch stream
with events. Outside capture it forks only, so the pre-touch runs concurrently with the layer
instead of blocking it.

Handing the window back is not, on its own, enough: it obliges the runtime to convert its
kernel launch to `cudaLaunchKernelEx`, and a runtime with a hundred launch sites will not.
`attach_window_to_captured_node()` looks like the way out: stream capture records each launch
as a graph node and reports the nodes just appended as the current capture dependencies, so
the attribute can be set on the node the runtime has already launched, with the hook site
staying one line after the kernel.

**It is not documented as safe, and that is not a theoretical objection.** Setting an
attribute on a node of a graph that is still being captured is not an operation CUDA
sanctions. On SparkInfer's batch-1 decode capture it works, 48 nodes out of 48, and the
persisting policy is genuinely present in every replay. At 32 concurrent sequences, during the
scored evaluation, that arm twice collapsed into a runtime fallback: `cudaStreamEndCapture`
failed, SparkInfer printed `[prefill] graph capture failed -> fallback`, and it decoded row by
row from then on — 1261 → 902 aggregate tok/s, a 28% loss with no cache policy anywhere in it.

A later 12-run probe did not reproduce it: four unhooked runs, four `Stream` runs and four
`CaptureNode` runs, each in isolation at 32 sequences, all clean. So the collapse needs
something the isolated runs did not have — accumulated device state from the preceding arms
is the obvious candidate — and node mutation alone is not a demonstrated cause. It has not
been seen on the safe path or the control either, and eight clean runs do not establish a rate
for something that rare.

That is enough to decide the default and not enough to condemn the mechanism. `WindowAttach`
is an enumerator, `Stream` is the default, and `CaptureNode` stays available because knowing
what it costs is worth more than pretending it does not exist. The controller checks
`cudaStreamIsCapturing` for `Invalidated` immediately after each attach, counts it in
`stats().capture_invalidations`, and latches itself off after the first. That check is
best-effort — an invalidation that only surfaces at the runtime's own `cudaStreamEndCapture`
will not be caught by it — so a bimodal concurrency arm should be read as a fallback until
proven otherwise, whatever its median says.

The honest conclusion under the safe path: **a locality library cannot deliver a persisting
window into a captured decode graph on its own.** It can compute the window; attaching it is
the runtime's to do, at its own launch site. Pre-touch has no such problem — kernels and
events are recorded into the graph like any other work — which is why the two mechanisms have
very different integration costs even though they look symmetrical in the API.

Graph nodes are also the dominant cost of the pre-touch, which is not obvious until it is
measured. Every fork/join pair is two nodes, and a hybrid model wants one per recurrent
layer — 48 of them for Qwen3.8-27B. On the real model that ordering is worth **1.2% of a
decode step**, more than an order of magnitude larger than the locality it buys at batch 1.
`PrefetchJoin::TokenEnd` trades the per-layer ordering for one join per token; the trade is an
enumerator because which side of it wins depends on how much compute separates the recurrent
layers.

## What a hybrid model's recurrent state actually is

One state per layer is a simplification, and it is the one v0.1 made. Qwen3.8-27B carries two
per recurrent layer, in separate allocations with separate strides and different element
types: a 3 MiB fp32 matrix state and a 60 KiB bf16 convolution window. That has three
consequences the API has to carry rather than hide:

- the hot set is both, so `StateSegment` describes a layer's state as segments;
- a pre-touch expressed as `const float*` cannot touch a bf16 buffer at all, so the walk is
  byte-oriented (`pre_touch_bytes_async`) and which states it covers is a policy
  (`PreTouchCoverage`);
- only one access-policy window can be bound at a time, so which state it protects is a
  choice (`WindowTarget`). The small state is the interesting one: 2.8 MiB of convolution
  window across all 48 recurrent layers is the *only* recurrent state on this model whose
  entire allocation fits in an L2 set-aside.

At concurrency the shape changes again: the runtime holds one pair of allocations per
sequence and gathers their base pointers into a device-side array. `RowSet` and
`pre_touch_rows_async` address that directly, because the obvious alternative — a launch per
sequence — would add 3,072 kernel nodes to a captured decode graph at 32 sequences, and the
paragraph above says what graph nodes cost.

## How much is hot

The hot set decides whether a persisting window is asked for at all, and v0.1 counted the
layer about to run plus whatever the caller declared. That is wrong by a factor of the layer
count, and it was wrong in the direction that hides the problem: `persist` measured -11% at
four concurrent sequences while the planner reported *zero* oversubscription.

The reuse distance for a recurrent state is a whole token. Layer 0's state is next read after
every other recurrent layer, every attention layer, and every weight in the model has gone
past. `HotSetModel::TokenFootprint` counts every recurrent layer for every sequence;
`ReuseWindow` adds the non-recurrent traffic in between, which the library cannot see and the
runtime therefore declares. `CurrentLayer` stays as the control, because a fix that cannot be
measured against what it replaced is an assertion.

The two shapes of answer are deliberately not made to look alike: `CurrentLayer` reports bytes
that *compete with* the window, the footprint models report an *absolute* live set that
already contains it. Adding a footprint to the window would double-count the layer's own
state; subtracting it first would break as soon as `WindowScope` widens the window past one
slice.

## The ceiling

Before choosing a policy, it is worth knowing what the best possible policy would be worth.
Recurrent-state locality can only ever recover the share of decode traffic that recurrent
state accounts for. `eval/traffic_budget.py` computes it from the pinned geometry and a
measured decode rate: on Qwen3.8-27B at batch 1 it is **1.65%**, below the 2% floor the
go/no-go table rejects at, before any implementation question is asked. That share is not a
property of the implementation and no amount of tuning moves it — but it grows with
concurrency, because model weights are read once per step however many sequences are in
flight while recurrent state is read once per sequence.

## Window lifetime

An access-policy window is a stream attribute, not a per-kernel argument: it applies to everything launched on that stream until it is changed. `before_layer` installs the window and `after_layer` removes it, so the policy covers the recurrent kernel and nothing else. Leaving it installed would apply persisting/streaming policy to the following attention, GEMM and MoE kernels against a pointer they never touch, which is the interference risk the project is supposed to avoid.

## Why the surfaces are enumerations, not forks

Every tunable mechanism is a named value on one enumeration, selected at run time and
reported in the benchmark's JSON: pre-touch strategy, hot-set policy, prefetch schedule,
prefetch implementation, state layout. Two people can therefore work on different mechanisms
without conflicting, and any two mechanisms can be A/B'd in one process against identical
state. A mechanism that replaces a file instead of adding an enumerator cannot be compared
with what it replaced, which is the same as not measuring it.

The correctness contract makes this safe: every strategy is read-only, and every layout is a
bijection, so the final state is bit-identical whatever combination runs. The evaluator
enforces that rather than trusting it.

## Integration boundary

The library is embedded by a runtime, so its edges are designed for that rather than for the
bundled benchmark:

- construction and every entry point are `noexcept` and report `cudaError_t`, because a
  runtime may be built without exceptions or driven from a language binding;
- `validate(PlannerConfig)` lets a caller reject a bad config before constructing anything;
- one controller per compute stream, stated rather than implied — `concurrently_hot_bytes` is
  how a runtime declares the competing hot set, and it is only meaningful per stream;
- `bind_streams` rejects being handed the same stream twice, which would turn "prefetch" into
  the same work on the critical path;
- counters explain a null result instead of leaving the integrator to guess.

## Measurement methodology

The synthetic benchmark exists to answer a few-percent question, so its timing has to be at least that trustworthy:

- warm-up tokens run untimed, keeping module load, first-touch page mapping and cold-cache effects out of the measured region;
- the pre-touch stream is joined before the stop event, so prefetch cost is measured rather than hidden — "prefetch overhead outweighs saved HBM latency" is a stated failure criterion and an unjoined stream cannot detect it;
- the evaluator repeats and interleaves modes, compares medians, and reports run-to-run spread so an unstable result is not read as a small win.

## Correctness

RecurLocal v0 does not change model arithmetic or state representation. A lossy/compressed state representation would belong to a different evaluation contract.

The checksum is the whole guarantee that a locality change did not disturb state, so it covers every element of every layer via a grid-stride reduction, and folds partials in a fixed order so repeated runs agree bit-for-bit. A reduction whose grid is sized from the state and then clamped to the 65535-block limit would silently leave most of the default geometry unchecked.
