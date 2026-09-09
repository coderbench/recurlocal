# SparkInfer integration

The adapter that turns RecurLocal from a microbenchmark into a measurement. It brackets the
Gated-DeltaNet layers of a pinned SparkInfer commit decoding Qwen3.8-27B and produces the
`real-result.json` that `eval/decide.py --real` scores — the only metric sections 17, 21 and
28 allow to decide anything.

```
build.sh                      clone the pinned commit, install RecurLocal, patch, build
pin.json                      the commit, the checkpoint, the state geometry, the hook sites
recurlocal-hook.patch         the diff against SparkInfer: 88 lines, all insertions
recurlocal_sparkinfer.cpp     the adapter (header: include/recurlocal/sparkinfer.h)
```

```bash
integrations/sparkinfer/build.sh /path/to/workdir
```

One binary comes out. With `RECURLOCAL` unset it is the unmodified runtime; with
`RECURLOCAL=<mode>` it is the candidate. Same executable, same model load, same box — which
is the only way a locality delta can be separated from a link-order or allocator difference.

## Where the hook goes, and why there

`is_linear_layer()` in `runtime/src/models/qwen35.cpp` decides which layers carry recurrent
state, and `w.linear_attn` is set from it at two call sites. Inside a decode step, that flag
gates the block this integration brackets:

```
begin_token()          before cudaStreamBeginCapture and above every graph-replay return
  |
  +-- w.linear_attn:
  |     before_layer(L)                       plan, and pre-touch the layer d ahead
  |     ... QKV / gate / alpha / beta projections ...   <- the pre-touch's runway
  |     launch_qwen36_conv_split*             reads lin_conv_state
  |     after_launch(Conv)
  |     launch_qwen36_gdn_ar                  reads and writes lin_state
  |     after_launch(Gdn)
  |     ... gated norm, out projection ...
  |     after_layer()
  |
  +-- otherwise: attention, unchanged
  |
end_token()            beside SparkInfer's own pf_join(), for the same reason
```

Nothing in the model's arithmetic is touched; the patch contains no deletions, which CI
asserts — a patch that deletes runtime code is a fork, and a fork cannot be reapplied to
a moving upstream.

## Two recurrent states, not one

A Qwen3.8-27B layer carries **two** mutable states, in separate allocations:

| state | type | per layer | 48 recurrent layers |
|---|---|--:|--:|
| `lin_state` — Gated-DeltaNet matrix state | fp32 | 3 MiB | 144 MiB |
| `lin_conv_state` — causal-convolution window | bf16 | 60 KiB | 2.8 MiB |

RecurLocal v0.1 modelled only the first. That matters in three separate places:

- **Accounting.** The hot set is both, not the larger one.
- **Pre-touch.** A walk expressed in `float*` cannot touch a bf16 buffer at all;
  `pre_touch_bytes_async` and `PreTouchCoverage` exist for this.
- **Policy.** Only one access-policy window can be bound at a time, so which state it
  protects is a choice. The conv state is 2.8 MiB — the *only* recurrent state on this model
  whose entire allocation fits in an L2 set-aside. `WindowTarget` makes that testable.

## Why the hot-set accounting had to be fixed first

The v0.1 model counted the layer about to run. On this model that is 3 MiB against a
~48 MiB set-aside, so the planner reported no oversubscription — while the synthetic
benchmark measured `persist` at **-11%** at four concurrent sequences.

The reuse distance for a recurrent state is a **whole token**: layer 0's state is next read
after the other 47 recurrent layers, 16 attention layers and every weight in the model have
gone past. `HotSetModel::TokenFootprint` counts all of it; `ReuseWindow` adds the
non-recurrent traffic in between, which the runtime declares because the library cannot see
it. `CurrentLayer` is kept as the control, so the fix is measurable rather than asserted.

## CUDA Graphs are not optional here

SparkInfer captures the whole decode step into a graph and replays it per token. A stream
access-policy window is host-side state: setting it during capture leaves the graph without
it and the persisting hint vanishes from every replay. A `persist` run configured that way
measures the cost of the policy and none of its effect.

The controller detects capture and attaches the window to the kernel *node* the runtime just
recorded, via `cudaStreamGetCaptureInfo` — so no SparkInfer launch has to be converted to
`cudaLaunchKernelEx`. `stats().windows_attached_to_node` is what proves it happened;
`windows_deferred_to_caller > 0` with that at 0 means the run measured no policy at all.

Pre-touch is recorded into the graph the same way SparkInfer's own L2 weight prefetch is,
with an event fork and join. Those nodes are not free — SparkInfer measured one fork/join
pair at ~0.89% of a decode step, and a hybrid model wants one per recurrent layer.
`PrefetchJoin::TokenEnd` halves them, and is an axis rather than an assumption.

## Configuration

Every mechanism is an environment variable, so one binary runs the whole matrix.

| variable | values | what it selects |
|---|---|---|
| `RECURLOCAL` | `off` `baseline` `persist` `prefetch` `combined` | mode; `baseline` is the hook with no policy, i.e. its own overhead |
| `RECURLOCAL_HOT_SET_MODEL` | `current_layer` `token_footprint` `reuse_window` | what counts as competing for the set-aside |
| `RECURLOCAL_WINDOW_SCOPE` | `layer` `allocation` `ahead` | how much of the state the window covers |
| `RECURLOCAL_WINDOW_TARGET` | `matrix` `conv` `widest` `narrowest` | which recurrent state it protects |
| `RECURLOCAL_PRE_TOUCH` | `scalar` `vec4` `vec4_ldcg` `ptx_l2` `warp_tile` `partial` | how the pre-touch moves bytes |
| `RECURLOCAL_PRE_TOUCH_COVERAGE` | `matrix` `conv` `both` | which states are pre-touched |
| `RECURLOCAL_PREFETCH_DISTANCE` | `0`–`8` | recurrent layers ahead |
| `RECURLOCAL_PREFETCH_SCHEDULE` | `uniform` `ramp` `alternating` `sparse` | how the distance varies with depth |
| `RECURLOCAL_PREFETCH_JOIN` | `per_layer` `token_end` | graph nodes spent on ordering the pre-touch |
| `RECURLOCAL_HOT_SET_POLICY` | `proportional` `fixed` `sqrt` `cliff` | what to do when oversubscribed |
| `RECURLOCAL_HIT_RATIO`, `RECURLOCAL_BUDGET_FRACTION`, `RECURLOCAL_MIN_HIT_RATIO` | float | window hit ratio, L2 set-aside fraction, back-off floor |
| `RECURLOCAL_SEQUENCES` | int | sequences decoding concurrently, for the accounting |
| `RECURLOCAL_STREAMED_BYTES_PER_TOKEN` | int | non-recurrent bytes through L2 per token, for `reuse_window` |
| `RECURLOCAL_STATS` | `1` | print `RECURLOCAL_STATS {...}` to stderr at exit |

A value the adapter cannot parse disables the hook and says so, rather than silently running
the default configuration under the label of the one that was asked for.

## Measuring

```bash
# the scored number: control/candidate interleaved, token-exact greedy replay gate
eval/real_eval.py --binary $W/sparkinfer/build/runtime/qwen3_gguf_bench \
                  --generate $W/sparkinfer/build/runtime/qwen3_gguf_generate \
                  --model /path/to/Qwen3.8-27B-NVFP4 --output real-result.json \
                  --candidate RECURLOCAL=prefetch RECURLOCAL_PREFETCH_DISTANCE=2
eval/decide.py --real real-result.json

# one axis at a time, against the control arm's own noise floor
eval/real_sweep.py --binary .../qwen3_gguf_bench --model DIR --axis prefetch-distance \
                   --fixed RECURLOCAL=prefetch
```

## Concurrent decode is a second hook, not the same one

SparkInfer batches sequences through `decode_packed()`, which routes into
`qwen35_prefill.cpp` — a separate code path with its own graph capture, per-session state
allocations gathered into device-side pointer arrays, and, when `SPARKINFER_CB_GDN_STATE_B16`
is on, a state compacted to bf16 so the per-layer stride is *half* what the single-sequence
path sees. Getting that stride wrong would put the window and the pre-touch on the dead half
of the buffer and double the footprint the accounting is about, so the adapter derives it
from the runtime's own `packed_state_b16` flag rather than assuming fp32.

Both paths are bracketed. Which one a run actually used is not assumed either: the telemetry
reports `tokens_packed` and `layers_packed` beside the totals, and `max_rows_seen`. A runtime
declines to batch for reasons that have nothing to do with locality — a row set that moved, a
tail chunk of one row, an unsupported shape — and a "concurrency" measurement that silently
ran the single-sequence path is not a concurrency measurement. At concurrency 8 this
integration measures `tokens_packed: 133` of 142 and `max_rows_seen: 9`.

**Two callers of the concurrent path are deliberately not bracketed.**
`dflash_verify_short_run` is reached from `decode_packed` (bracketed) and from
`dflash_warm_verify` and `batched_forward` (not). Those two leave `packed_rows` null and do
not route through `forward_token` either, so with a DSpark draft model configured some
recurrent-layer work runs with no locality control at all — and, because the hook never fires
there, the telemetry cannot tell you it happened. Treat a DSpark-enabled measurement as
covering fewer layers than the layer counters suggest.

The pre-touch at concurrency is where `pre_touch_rows_async` earns its place: one launch per
state per layer whatever the batch size, reading each sequence's base pointer device-side. A
launch per sequence would put 3,072 extra kernel nodes into the captured decode graph at 32
sequences, and the graph-node measurement above says what that would cost.
