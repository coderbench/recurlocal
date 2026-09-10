# Design

## Core hypothesis

Recurrent state differs from streamed model weights: it is relatively small per layer, mutable, revisited every decode token, and accessed in a predictable layer order.

TensorTransit tests whether explicit cache policy plus next-layer warm-up can exploit this reuse.

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

**It IS documented as safe, and this document said the opposite for three releases.** CUDA's
own header, on the very call the mechanism is built out of, says so:

```
 * \param graph_out - Optional location to return the graph being captured into. All
 *           operations other than destroy and node removal are permitted on the graph
 *           while the capture sequence is in progress.
                    -- /usr/local/cuda/include/cuda_runtime_api.h:2743-2744 (CUDA 13.3),
                       cudaStreamGetCaptureInfo; identically cuda.h:16885 and unchanged
                       since CUDA 11.3
```

Setting a kernel-node attribute is neither destroying the graph nor removing a node. The same
paragraph goes further and blesses the exact pointer this code passes:

```
 *           The node handles may be copied out and are valid until they or the graph is
 *           destroyed. The driver-owned array may also be passed directly to APIs that
 *           operate on the graph (not the stream) without copying.
                    -- cuda_runtime_api.h:2755-2757
```

`cudaGraphKernelNodeSetAttribute` is an API that operates on the graph, and
`cudaLaunchAttributeAccessPolicyWindow` is declared "Valid for streams, graph nodes, launches"
(`driver_types.h:4017`). Neither setter documents a capture-related caveat or error code.

**And the attribute demonstrably survives.** Nothing in this repository had ever read one back;
`windows_attached_to_node` was a counter of attach CALLS being quoted as a count of nodes.
A standalone probe (`profiling/capture_attr_probe.cu`) now closes it: it captures N kernels, sets
the window mid-capture exactly as the controller does, ends the capture, and reads the
attribute back off the finished graph and off a clone. At 1, 4, 8, 16, 32, 48, 64 and 128
nodes it finds the window present and byte-for-byte correct on every kernel node, with zero
capture invalidations, and `compute-sanitizer --tool memcheck` reports zero errors.

The exec graph cannot be inspected — CUDA 13.3 has no `cudaGraphExecGetNodes` and no
exec-level attribute getter — so that half is behavioural: the same capture attached with a
*persisting* window and with a *streaming* window over the same buffer replays 3.2% apart at
48 nodes. A policy that never reached the replay could not do that.

What remains is a real but much narrower risk, and it is about precision rather than legality:
`CaptureNode` marks **every** kernel node in the capture's current dependency set, and that set
is not always one node. `WindowAttach::CaptureNodeStrict` marks only when exactly one kernel
node is pending and counts the rest in `stats().window_attach_ambiguous`.

An earlier version of this document attributed a 32-sequence throughput collapse to this
mutation. **That attribution was wrong and the code refutes it**: the arms that collapsed were
`baseline` and `prefetch`, and `plan.use_persisting_window` — the gate on arming the node
attach — is only ever true for `Persist`/`Combined`. The mechanism is inert in the arms that
failed. The collapse is a runtime fallback whose cause this project has not identified.

`Stream` remains the default, but for a different and smaller reason than the one this
document used to give: not that node attachment is illegitimate, but that it is a device the
host has not asked for. `WindowAttach` is an enumerator, and `CaptureNode` and
`CaptureNodeStrict` stay available and instrumented.

One correction the other way, because it cuts against the library: **under `Stream` this
integration delivers nothing.** The controller hands the window back in `LayerActions`, and
the SparkInfer adapter keeps only the boolean and drops `actions.window` on the floor. There
is no accessor that returns it. So `Stream`'s measured -0.019% is the hook's overhead and not
a policy at all — which is exactly what `eval/real_eval.py`'s null-candidate guard exists to
catch, and it did not, because the guard reads counters and `Stream` increments
`windows_deferred_to_caller` honestly. A runtime that wanted the sanctioned launch-site path
would need that accessor first. The
controller checks `cudaStreamIsCapturing` for `Invalidated` after every attach, counts it in
`stats().capture_invalidations`, and latches itself off after the first — best-effort, since
an invalidation that only surfaces at the runtime's own `cudaStreamEndCapture` will not be
caught by it.

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

Two tighter bounds sit under it, and both matter more than the first.

A **persisting window cannot save traffic it cannot hold**. Because the reuse distance is a
whole token, the footprint that must stay resident is every recurrent layer for every sequence
at once — 146.8 MiB on Qwen3.8-27B at batch 1 against a 60 MiB persisting capacity. So the
persist family's bound is `2 x min(capacity, footprint) / step_traffic`, and since the capacity
is the device's, the only lever is the step. `--persisting-l2-bytes` prints the inversion:
`break_even_step_traffic_bytes`, the step traffic a model has to come in under before a
persisting window is worth anything at all. 6.42 GB on this device.

A policy aimed at reuse **within** a layer is bounded far tighter still, and this is why no such
policy is shipped. The runtime's Gated-DeltaNet kernel already holds each state column in
registers across both of its passes, so 98% of the recurrent bytes are read once and written
once and there is no second touch at any distance. All that is left is the convolution window's
shift re-read — `conv x (K-2)/(K-1)` per layer, 1.97 MB per token — reported as
`within_layer_family`, a 0.011% ceiling.

## What a policy does when the footprint does not fit

Every v0.1 policy answers oversubscription by moving one dial: the requested hit ratio, for
every layer alike. That models the cache as something that can keep 97% of a byte, and it
cannot — a line is resident or it is not. `HotSetPolicy::Quota` is the alternative: admit whole
layers at the full hit ratio until the set-aside is spent, decline the window for the rest,
spread evenly, and decide from the layer ordinal alone so the choice cannot move between tokens
(a window that moved would evict exactly the state it kept last time). It needs no layer count
or sequence count from the caller, because the declared hot set is already in bytes.

Which of the two is right depends on how far over budget the footprint is, and that is a
property of the model rather than of the library — which is why both exist as enumerators
rather than one replacing the other.

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
