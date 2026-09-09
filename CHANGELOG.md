# Changelog

Notable changes to RecurLocal. Format loosely follows [Keep a Changelog](https://keepachangelog.com).

No performance claim appears here without a measurement behind it. See `docs/FRONTIER.md`.

## [Unreleased]

### Added — the real-model gate now has a producer

- **SparkInfer adapter** (`integrations/sparkinfer/`, `include/recurlocal/sparkinfer.h`). The
  metric that decides this project had no way to be measured: `eval/decide.py --real` needed a
  `real-result.json` and nothing could produce one. It can now. The adapter brackets the
  Gated-DeltaNet layers of a **pinned** SparkInfer commit (`pin.json`) decoding Qwen3.8-27B
  NVFP4 on an RTX 5090, as an 88-line insertion-only patch plus a library target (CI
  asserts the patch deletes nothing). One binary
  runs both arms: with `RECURLOCAL` unset it is the unmodified runtime.
- **Both recurrent states, not one.** A Qwen3.8-27B layer carries a 3 MiB fp32 matrix state
  *and* a 60 KiB bf16 convolution window, in separate allocations. v0.1 modelled the first and
  could not even express the second: `pre_touch_async` took a `const float*`.
  `pre_touch_bytes_async` and `PreTouchCoverage{Matrix,Conv,Both}` fix that, and the hot set
  now counts both (3,207,168 bytes per layer, confirmed by the adapter's own telemetry).
- **The hot-set accounting is fixed, and the fix is measurable.**
  `HotSetModel{CurrentLayer,TokenFootprint,ReuseWindow}`. v0.1 counted the layer about to run,
  which is how `persist` measured -11% at four concurrent sequences while the planner reported
  *zero* oversubscription. The reuse distance for a recurrent state is a whole token, so
  `TokenFootprint` counts every recurrent layer for every sequence and `ReuseWindow` adds the
  traffic that passes through L2 in between. `CurrentLayer` is retained as the control: on the
  real model the corrected model reports 48/48 layers oversubscribed where the old one reported
  0. Also exposed on the synthetic benchmark (`--hot-set-model`) and in `eval/sweep.py`.
- **`WindowAttach{Stream,CaptureNode}`, and the finding behind it.** SparkInfer captures the
  whole decode step into a graph and replays it per token, and a stream access-policy window is
  host-side state that a graph never records — so under graph decode `persist` measures its
  cost and none of its effect. `attach_window_to_captured_node()` sets the attribute on the
  kernel node the capture just recorded (`cudaStreamGetCaptureInfo`), needing no change to how
  SparkInfer launches its kernels, and it works: `windows_attached_to_node` 48/48.

  It is also **not documented as safe** — a gap in what CUDA sanctions, not an observed
  defect: no run taken has failed because of it, including twelve isolated runs at 32
  sequences. `Stream` is the default anyway, because a locality library should not mutate its
  host's graph behind its back; `CaptureNode` is opt-in, and the controller checks
  `cudaStreamIsCapturing` for `Invalidated` after each attach, counts it in
  `stats().capture_invalidations`, and latches itself off after the first.

  **Correction.** Earlier revisions of this entry, the README, `docs/DESIGN.md` and
  `docs/OPTIMIZATION-SURFACES.md` blamed a 32-sequence throughput collapse on this mutation.
  That was wrong and the code refutes it: the arms that collapsed were `baseline` (-14.8%) and
  `prefetch` (-20.1%), and node attachment is gated on `plan.use_persisting_window`, which is
  only true for `Persist`/`Combined`. The mechanism is inert in the arms that failed. The
  collapse is a runtime fallback whose cause is unidentified and remains an open problem.

  The axis separates the two cleanly at batch 1: `stream` **-0.019%**, `capture_node`
  **+0.129%**. Every point of the persist result is the delivery mechanism, not the policy —
  and the honest conclusion is about the boundary. A locality library can compute a persisting
  window; under graph decode it cannot deliver one without the runtime attaching it at its own
  launch site. Pre-touch has no such problem: its kernels and events are recorded like any
  other work.
- **`PrefetchJoin{PerLayer,TokenEnd}`.** Under capture, every fork/join pair is graph nodes,
  and a hybrid model wants one per recurrent layer. Measured on the real model, this is worth
  **1.20 points** — see below. It is the difference between the pre-touch costing 1.29% and
  costing 0.09%.
- **`WindowScope{Layer,Allocation,Ahead}` and `WindowTarget{Matrix,Conv,Widest,Narrowest}`.**
  Only one access-policy window can be bound at a time, so with two recurrent states which one
  it protects is a choice — and the conv state is the only one whose whole allocation fits in a
  set-aside. Window selection and region resolution moved to the CPU planner
  (`select_window_segment`, `resolve_window_region`), so they are unit-tested without a GPU;
  the v0.1 equivalents lived in the CUDA controller and could not be.
- **Row-major pre-touch** (`pre_touch_rows_async`) and a controller entry point for it. At
  concurrency a runtime holds one state allocation per sequence and gathers their base pointers
  into a device array. One launch per row would add 3,072 kernel nodes to a captured decode
  graph at 32 sequences; this reads the base pointer device-side, so the launch count does not
  depend on concurrency.
- `eval/real_eval.py` — the scored measurement. Control and candidate interleaved on one box,
  token-exact greedy replay as the correctness gate, per-context geometric mean, and the
  control arm's own run-to-run spread reported as the noise floor.
- `eval/real_sweep.py` — one adapter axis at a time against the real model, the real-runtime
  counterpart of `eval/sweep.py`. It refuses to name a winner inside the noise floor, and
  refuses to *estimate* a floor from fewer than three control repeats: two readings can land on
  the same number, and a zero floor would make any difference look resolved.
- `eval/traffic_budget.py` — what fraction of a decode step's memory traffic recurrent state
  actually is, which is the ceiling on everything this project does.

### Fixed — production defects, found by auditing the code and by testing it

The repository shipped 1,428 lines of CUDA with no tests at all. Writing them found real
defects; so did `compute-sanitizer`, which `CONTRIBUTING.md` had required of contributors
while nothing in the repo ran it.

- **The device-wide persisting-L2 set-aside was never given back.** `initialize()` reserved it
  with `cudaDeviceSetLimit`; `reset()`'s header promised to release it and only evicted
  persisting *lines*; the destructor cleared a bookkeeping member without telling the driver.
  Any process that ever constructed a controller — including in `RECURLOCAL=baseline`, a
  no-op control — ran the rest of its life with a smaller L2 for every other kernel. Now
  restored to its prior value, and the integration calls `shutdown()` from the model
  destructor so it happens while the CUDA context is still alive.
- **`before_layer`'s legacy overload ignored `WindowAttach::Stream`** and armed the
  undocumented graph-node mutation regardless — so a consumer asking for the documented-safe
  default still got a graph mutated mid-capture.
- **A failing pre-touch stranded the caller's graph capture.** The fork that pulls the
  prefetch stream into a capture was emitted *before* the flag recording that a join is owed,
  so any early return in between left the capture unjoined with nothing to fix it.
  `cudaStreamEndCapture` then fails and returns a null graph — which SparkInfer's `cu()`
  helper logs without aborting, after which every decode step replays a null graph and emits
  the same stale token forever. Triggerable by a *stale* error latched by unrelated code,
  because the pre-touch reports a bare `cudaGetLastError()`.
- **`configure_persisting_l2()` left the caller's current device changed.** It called
  `cudaSetDevice()` and never restored it, so merely constructing a controller repointed the
  calling thread's GPU.
- **The planner budgeted against the requested set-aside, not the granted one.** The driver
  does not honour the request: an RTX 5090 has a non-zero default (18 MiB) and returns 18 MiB
  for a 15 MiB ask. Every oversubscription decision inherited that rounding.
- **`stats().hot_set_oversubscribed` counted the wrong thing** — it keyed off
  `hit_ratio_reduced`, which `HotSetPolicy::Fixed` never sets, so it read zero under exactly
  the policy that ignores oversubscription hardest. The two are now separate counters.
- **The controller held the caller's streams past `reset()` and segfaulted at exit.** The
  streams are *borrowed*; a runtime that did the correct thing — call `shutdown()`, then
  destroy its own streams — left the controller's destructor probing a dangling handle.
  Backtrace: `cudaStreamIsCapturing <- release() <- ~Adapter <- __run_exit_handlers`, SIGSEGV
  inside libcuda. Found by the concurrent arm of the eval harness, which reported
  `cb bench exited -11` rather than a number. `reset()` now drops the handles, and the
  borrowing contract is stated on `bind_streams()`.
- **`release()` during an active capture poisoned the caller's context.** `cudaMalloc`,
  `cudaFree` and `cudaDeviceSetLimit` are all illegal while a stream is capturing;
  compute-sanitizer counted 14 such errors when a controller was destroyed mid-capture. It
  now drops its handles without touching the driver, and counts it.
- **The pre-touch consumed the host runtime's pending CUDA error.** Launch status was
  reported with `cudaGetLastError()`, which *clears* the per-thread error slot — so an
  embedded RecurLocal could swallow an error the host had not yet checked, and the host's own
  next check would report success for whatever really failed. Now `cudaPeekAtLastError()`,
  which reports without consuming.
- **`WindowScope::Ahead` widened past memory the caller never declared** when no allocation
  base was supplied — and the unit test asserted that behaviour as correct.
- **`cudaStreamCaptureStatusInvalidated` was treated as an active capture**, so the controller
  kept deferring windows into a graph that could never be instantiated.
- **`real_eval.py` scored the batch-1 arm with a different estimator from every other arm**
  and from its own documented method: a ratio of medians rather than the median of paired
  ratios, discarding the pairing that makes an interleaved same-box comparison valid. On
  drifting clocks the two disagree enough to flip a verdict. The estimator is unified,
  extracted from `main()` so it is testable without a GPU, and covered. The committed result
  was recomputed from its stored paired ratios: **+0.049% → +0.053%**, verdict unchanged
  (`reject`) — this data was tight enough that the bug did not bite.

### Added — what is the most this repository can ever pay?

- **`eval/traffic_budget.py --matrix`** answers the question one level above a single arm.
  Each workload has a physical ceiling, and the verdict is a weighted geometric mean over all
  of them, so the number that decides whether the project is worth competing on is what a
  submission scores if it hits *every* ceiling at once — one that made recurrent state
  entirely free. The weights and the bands come from `decide.py` rather than being restated,
  and a test pins that the two agree: a ceiling advertised in one currency and paid in another
  would be worse than no ceiling. Arms with no measured decode rate are excluded and named
  rather than guessed. Inputs live in `configs/rtx5090-section44-ceiling.json`, every one of
  them a baseline measurement from `results/rtx5090-real.json`.

### Added — why the persist family captures nothing, as arithmetic

- **`eval/traffic_budget.py --persisting-l2-bytes`** computes a second ceiling, far tighter
  than the traffic one and specific to the persist family. A persisting window cannot save
  traffic it cannot hold: state written at layer *i* is read again at layer *i* of the **next**
  token, so to save a byte the cache has to keep it across a full pass over the model — the
  whole per-token recurrent footprint has to be resident at once, not one layer's worth. On
  Qwen3.8-27B against this device's 60 MiB persisting capacity that footprint is 2.4x
  oversubscribed at batch 1 and **20x at 16 sequences**.

  The consequence is the answer to an open problem this repository had left standing as
  "nobody has explained why": the persist family's ceiling *falls* as concurrency rises —
  0.68% at batch 1, 0.58% at 4, 0.34% at 16 — while the traffic ceiling rises 1.68% → 7.23%
  over the same range. The room grows and the fraction of it a persisting cache can address
  shrinks faster. Measurement agrees: `persist` at 16 sequences, with the window genuinely
  attached at the graph node, is +0.06% against a 0.13% control noise floor and a predicted
  ceiling of 0.34%. The bound is deliberately generous — every resident byte hits, the
  set-aside costs its neighbours nothing — and a real policy lands below it.

### Fixed — the ceiling was quoted in the wrong currency

- **A ceiling a submission could legitimately beat.** `traffic_budget.py` reported the
  recurrent-state *share of traffic* as the ceiling. But the scorer measures
  `candidate_tps / baseline_tps`, and a step carrying *f* less traffic runs in *(1−f)* of the
  time — so throughput rises by *f/(1−f)*, which is more than *f*. Every published ceiling was
  understated: 1.65 → 1.68% at batch 1, and 11.03 → **12.40%** at 32 sequences, the surface
  `docs/MINING.md` points the competition at. The error was in the safe direction for the
  project's own verdict and the wrong direction for everyone else: a contribution could have
  exceeded a number this repository called a physical limit. The same conversion applies to
  the pre-touch cost prediction, the other way round, and was corrected with it.

### Fixed — the evaluator was not fit to arbitrate

The harness is meant to decide whether a contribution is accepted. Three ways it could quietly
fail to:

- **A candidate that applied no policy was scored as a result.** `require_hook_engaged` proved
  the hook had *loaded*, not that anything reached a kernel. Under graph decode with the safe
  `WindowAttach::Stream`, `persist` defers every window to a runtime that does not attach
  them — `windows_applied 0, windows_attached_to_node 0, pre_touch_launches 0`, and 192
  deferred. The first scored run in this repository was exactly that configuration, so its
  verdict measured the hook's overhead against the control and nothing else. `real_eval.py`
  now refuses a null candidate outright and names the counters that make it null.
- **The control arm inherited an ambient `RECURLOCAL`.** `run()` copied `os.environ` and only
  the *caller's* dict was checked for a mode, so an operator with `export RECURLOCAL=combined`
  in their shell would run a hooked "control" and the harness would report ~0% for a
  comparison of the candidate against itself. The control now scrubs every `RECURLOCAL*` name
  from the inherited environment.
- **`decide.py` never read whether the arms resolved.** `real_eval.py` computes a per-workload
  resolution against each arm's own run-to-run spread; the scorer ignored it and could return
  `significant: true` on a matrix where nothing resolved. It now refuses.
- **An omitted workload made the score better, not worse.** `decide.py` renormalises the
  weights it is given, so a workload left out of the matrix is removed from the geometric
  mean rather than averaged in — and `docs/MINING.md` points the competition at concurrency
  32, the arm with the most room and therefore the one worth not running. The scorer now
  derives coverage from the workloads it actually scored (never from a `workload_coverage`
  field in the document being scored), names what is absent together with the section 44
  weight that went with it, and refuses to mark a partial matrix `significant`. The verdict
  is still reported, and `--allow-partial` records a maintainer's decision to score one
  anyway. The repository's own published result is partial in exactly this way — it has no
  concurrency-32 arm — and now says so in the verdict rather than only in the bundle.

- **Nothing interlocked concurrent eval processes.** Two runs on one GPU do not go slower,
  they produce a number for a run that never happened — twice in this project's own history a
  VRAM race turned into a plausible-looking result. `real_eval.py` now takes an advisory
  `flock` and waits.

### Fixed — found by an adversarial audit of the finished work

Three findings survived two independent refute-by-default verifiers reading the current tree:

- **`decide.py` could not score the repository's own published result.** `results/` ships the
  matrix nested under `scored_result` alongside the axis sweeps; `score_real` only looked at
  the top level, so the exact command the README gives — `decide.py --real
  results/rtx5090-real.json` — printed "real result has no 'workloads'" and scored nothing.
  The README had illustrative output where it claimed real output. Both fixed, and a test now
  scores the committed artifact so the docs cannot drift from it again.
- **Hot-set telemetry read zero in every mode except persist.** The accounting was written
  only inside the persisting-window guard, so a prefetch-only run reported
  `hot_set_bytes: 0, hot_set_budget_bytes: 0` beside `hot_set_model: token_footprint` — a
  reader would conclude nothing was competing for L2 while 147 MiB was. The numbers describe
  the workload, not the policy that happens to be enabled, and are now reported for all four
  modes.
- **A second model instance would silently share the adapter's walk state.** The hook is a
  process-global singleton holding one model's layer ordinal, pending window and layout;
  SparkInfer's mutex is per-model, so two instances would interleave. The failure mode is not
  a crash, it is quietly wrong numbers. Rather than a mutex — which would fix the data race
  and keep the wrong logic — the adapter now detects a second compute stream and refuses,
  loudly.

### Added — the CUDA code is now tested

- `tests/test_cuda_controller.cu`: 120 device-side checks over the controller, the
  graph-capture state machine, every pre-touch strategy and the row-major path. Registered
  with `ctest`; skips cleanly with exit 0 where there is no GPU. Each regression test was
  validated by reverting its fix and confirming the test fails — one that did not was
  rewritten, and one that cannot be caught on a single-GPU box says so at runtime rather than
  reporting a green tick.
- `scripts/sanitize.sh`: memcheck, initcheck, synccheck and racecheck. All clean; memcheck
  found the release-during-capture defect above on its first run.

### Measured — the first real-model result, and it is a rejection

One RTX 5090, CUDA 13.3, Qwen3.8-27B NVFP4 on SparkInfer `5347b27c`. Control and candidate
are the same binary, interleaved. Batch-1 noise floor **0.023%** over 3 pairs.

| | ceiling | control | `baseline` | `persist` | `prefetch` | `combined` |
|---|--:|--:|--:|--:|--:|--:|
| batch 1 | 1.65% | 96.1 tok/s | -0.01% | **+0.13%** | -1.29% | -1.17% |
| concurrency 4 | 2.88% | 329 tok/s | -0.09% | +0.06% | -2.19% | -2.34% |
| concurrency 16 | 6.73% | 769 tok/s | +0.36% | -0.21% | -5.88% | -5.40% |
| concurrency 32 | — | 1262 tok/s | -1.13% | -1.13% | *unresolved* | *unresolved* |

Output is token-exact against the unhooked runtime under greedy replay.

The scored run — `prefetch` with `token_end` joins and the `ptx_l2` walk, 3 interleaved pairs,
batch 1 at three contexts plus concurrency 4 and 16 — is **-0.488% weighted**, and
`eval/decide.py --real` returns `reject`: batch1 +0.016%, concurrency4 -0.667%,
concurrency16 -1.311%. The policy demonstrably ran (188 pre-touch launches at batch 1; the
concurrency arms went through `decode_packed`, 129/138 tokens packed at c=4). It costs more
than it saves, and the cost grows with concurrency as the axis sweeps predicted.

An earlier scored run reported +0.053% for `persist` with safe window delivery; that was a
null candidate which applied no policy at all, and the harness now refuses such runs. Concurrency 32 is deliberately absent from the scored
matrix and reported as unresolved instead; `real_eval.py` leaves an unmeasured arm out and
`decide.py` renormalises the weights that remain, so a partial run reads as a partial verdict
rather than a full one with invented halves.

Three things this says that the synthetic benchmark could not:

- **At batch 1 the hypothesis cannot pass, by arithmetic.** Qwen3.8-27B is a dense hybrid;
  every weight is read every token, and recurrent state is 1.65% of the traffic. Making it
  free would be worth 1.65%, under section 21's 2% floor, before any policy is chosen.
- **At concurrency the room is real and none of it is captured.** The recurrent share
  quadruples to 6.73% by 16 sequences, and `persist` moves the *wrong way* (+0.13% to -0.21%)
  while `prefetch` gets worse in proportion to the bytes it moves.
- **The pre-touch's cost is graph nodes, not memory.** Production decode is a captured CUDA
  graph, so each fork/join is permanent: `PrefetchJoin::TokenEnd` recovers 1.20 of the
  1.29 points, and the extra full read of recurrent state that remains costs 0.09%.

Concurrency 32 is reported as unresolved rather than as a result: two of its four arms
disagreed with themselves by 30% between repeats, and one of them was `baseline`, which
installs no window and issues no pre-touch and therefore cannot cost anything. A median of
two hides that; `real_eval.py` now publishes a `resolution` block beside every workload's
gain so it cannot.

The synthetic benchmark has now disagreed with the real model on four axes — prefetch
distance (+21.0% at d=6 there, monotonically negative here), prefetch schedule (a 16-point
penalty for doing less there, a saving here), pre-touch strategy (unresolved there, `ptx_l2`
ahead by 0.70 points here) and hot-set policy (a 40-point swing there, 0.01% here). One
structural cause: it does not capture a graph, and production decode does.

`docs/OPTIMIZATION-SURFACES.md` carries the full matrix, `results/rtx5090-real.json` the raw
data, and `eval/decide.py --real` the verdict.

### Changed

- `recurlocal_cuda` is compiled whole-program instead of with relocatable device code.
  Separable compilation obliges every consumer to run a CUDA device-link step, and a runtime
  linking RecurLocal as one more static archive got `undefined reference to
  __cudaRegisterLinkedBinary_*` instead. Found by embedding it in SparkInfer, which is the
  first time anything actually consumed it as a library.

### Fixed

- The streaming access-policy window was not clamped to the device maximum (128 MiB on an
  RTX 5090), so hinting any larger weight buffer failed outright instead of hinting the
  leading window. Found by measurement, not review.

- The benchmark checksum sized its grid from the state and then clamped it to 65535 blocks,
  so at the default geometry it covered only 44% of state. The correctness gate — the whole
  guarantee that a locality change did not disturb model state — silently ignored the rest.
  It now uses a grid-stride reduction over every element, folded in a fixed order so repeated
  runs agree bit-for-bit rather than varying with `atomicAdd` ordering.
- Pre-touch work ran outside the timed region: the stop event was recorded on the compute
  stream only, so prefetch cost was never measured while its benefit was. The prefetch stream
  is now joined before the clock stops.
- `after_layer()` was a no-op, leaving the access-policy window bound to the compute stream.
  Every following non-recurrent kernel inherited persisting/streaming policy against a state
  pointer it never touches. The window is now scoped to the recurrent layer.
- The controller was destroyed after its streams in the benchmark, so it cleared an
  access-policy window on a destroyed stream.
- The planner recommended a persisting window on devices reporting no persisting-L2 support,
  where no set-aside can be reserved.
- Unchecked `cudaMalloc` for the pre-touch scratch buffer left a sticky error on the context;
  pre-touch now degrades to a no-op.
- Benchmark arguments were unvalidated: `--tokens 0` emitted `inf`, which is not valid JSON
  and crashed the evaluator.

### Added

- **Surface sweeps on real hardware** (`results/rtx5090-surfaces.json`) covering concurrency,
  cache interference, hot-set policy, prefetch schedule, prefetch implementation and state
  layout. Several overturn assumptions in the overview: kernel-integrated prefetch is 30-33
  points *worse* than a separate stream, `persist` becomes harmful at four concurrent
  sequences, and state layout alone is a 2.3x effect that changes which policy wins.
- **Two shapes of cache interference** (`--stream-mode reuse|distinct`). Re-reading one buffer
  every layer and giving each layer its own slice are different problems and give opposite
  policy answers; only the second is what a persisting window should be protected from.
- **First measured result** (`results/rtx5090-synthetic.json`): RTX 5090, CUDA 13.3, synthetic
  benchmark. persist +10.1%, prefetch +15.9%, combined +10.4%; prefetch distance 6 reaches
  +21.0%. Correctness bit-identical across 39 configurations. Synthetic only — no model, no
  runtime, and the go/no-go gate remains open.
- **Six pre-touch strategies** (`--pre-touch scalar|vec4|vec4_ldcg|ptx_l2|warp_tile|partial`).
  The kernel was a single hard-coded scalar walk. `ptx_l2` issues real `prefetch.global.L2`
  instructions — the "put this range in L2 now" primitive the overview assumed CUDA does not
  expose for arbitrary allocations; PTX does. Adding a strategy is a kernel plus an
  enumerator, and every strategy is measured against the others by one flag.
- **Four hot-set policies** (`--hot-set-policy proportional|fixed|sqrt|cliff`) for what to do
  when the recurrent state that wants to be resident exceeds the L2 set-aside. The shipped
  heuristic was one line with no alternative to compare against; `fixed` is now the naive
  control and `cliff` refuses the window rather than thrash a shared cache.
- **Concurrency in the benchmark** (`--sequences N`). The per-layer hot set now scales with
  batch, which is what makes the hot-set policy measurable at all: `hot_set_oversubscribed`
  was 0 in every previous run because the benchmark decoded a single sequence.
- **Streaming interference** (`--stream-bytes N`) between recurrent layers, standing in for
  the attention and MoE weight traffic a hybrid model pushes through the same L2. Without it
  the persisting window has nothing to defend against.
- **Kernel-integrated prefetch** (`--prefetch-impl fused`, section 34.4). The compute kernel
  performs the pre-touch itself, with no second stream and no extra launch, as an A/B against
  the stream implementation on the same mode.
- **State layout as an axis** (`--state-layout linear|head_interleaved|tile_swapped`, section
  34.3). The map is a bijection, so every layout leaves bit-identical state and only the
  access order changes — coalescing and cache-line utilisation, isolated from everything else.
- **Weight/state cache QoS** (`--qos on`, section 34.6). `before_streaming_region` hints the
  attention/MoE traffic as streaming so it passes through L2 without displacing recurrent
  state, instead of inheriting whatever window the last recurrent layer left behind.
- **Per-layer prefetch schedules** (`--prefetch-schedule uniform|ramp|alternating|sparse`,
  section 34.1). A single global distance assumes every recurrent layer has the same compute
  ahead of it to hide the walk behind, which is false in a hybrid model.
- `eval/sweep.py`: walks one surface with everything else fixed, reports gain against that
  surface's own noise floor, and refuses to name a winner inside it — so an unresolved axis
  reads as open rather than solved.
- **Prefetch distance as an open axis** (`--prefetch-distance 0..8`). It was capped at 0-or-1
  in code, against section 34.1, and was leaving about six points on the table.
- `docs/OPTIMIZATION-SURFACES.md`: each surface, the axis that isolates it, its measured
  frontier, and its noise floor — including surfaces that cannot yet be competed on.
- **Embeddable library surface.** RecurLocal is consumed by a runtime, so it now installs a
  CMake package (`find_package(RecurLocal)`, `RecurLocal::recurlocal`,
  `RecurLocal::recurlocal_cuda`) with public headers and a version header that CMake parses
  as the single source of truth. CI builds an outside consumer against the installed package.
- **CUDA Graph support.** A stream access-policy window is host-side state and is not
  recorded into a captured graph, so the persist mode silently vanished from every replay on
  any runtime using graph decode. The controller now detects capture and returns the window
  in `LayerActions` for the caller to attach to its kernel launch or graph node. Pre-touch
  forks the prefetch stream from compute with an event and rejoins it under capture — an
  unjoined fork ends the capture invalid.
- **Non-throwing integration boundary.** Construction and every entry point are `noexcept`
  and report `cudaError_t`; `initialize()`/`status()` replace a throwing constructor, and
  `validate(PlannerConfig)` lets a caller reject a bad config before building anything.
- **Controller telemetry.** `stats()` counts windows applied, windows deferred under capture,
  oversubscribed hot sets, pre-touch launches and bytes, and both stream priorities — so a
  null end-to-end result can be explained instead of guessed at. Surfaced in the eval output.
- `bind_streams` rejects the same stream for compute and prefetch, which would turn a
  prefetch into the same work on the critical path.
- Stated threading contract: one controller per compute stream, which is also the only
  arrangement in which `concurrently_hot_bytes` is meaningful.
- `recur_local_info` reports the real device when built with CUDA, instead of only
  illustrative constants.
- `run_eval.py --config` makes `configs/*.json` drive the benchmark geometry; it was
  previously a file nothing read.
- `eval/decide.py`: the go/no-go gate as a deterministic function of measurements, so the
  project can settle its own bet rather than argue it. Refuses to accept a synthetic result
  as a verdict. Covered by `eval/test_decide.py`, wired into `ctest`.
- Environment provenance in the benchmark and eval output — GPU, compute capability, SM
  count, driver and runtime version, commit and dirty-tree flag, plus observed or pinned
  graphics clock (`--pin-clock-mhz`).
- `eval/result_schema.json`, pinning the eval output shape.
- Untimed warm-up tokens (`--warmup-tokens`, default 4).
- Repeated, interleaved eval runs with median comparison and a stability verdict
  (`--repeats`, `--stability-threshold-pct`).
- Planner tests for device capability edges, window limits, concurrent hot sets, determinism
  and config validation.
- CI: Release build, JSON asset validation, and a device-less `nvcc` compile gate.
- PR template and a harness-paths guard for `eval/`.

### Changed

- Build options are spelled `RECURLOCAL_*`; the `RECURLLOCAL_*` misspelling still works and
  warns. `RECURLOCAL_WITH_CUDA` likewise, with the old macro still defined.
- Tests use always-on checks. They previously used `assert`, so a Release build reported a
  green run with nothing checked.
- CMake defaults to `Release`; the documented build commands previously produced an
  unoptimised `-O0` host build for a performance project.

## [0.1.0]

- Initial feasibility prototype: CPU locality planner, CUDA persisting-L2 controller,
  asynchronous next-state pre-touch, baseline/persist/prefetch/combined modes, synthetic
  benchmark, evaluator, CPU tests, SparkInfer integration design, GitHub CPU CI.
