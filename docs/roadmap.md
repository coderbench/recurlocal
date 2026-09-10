# Roadmap

Version numbers describe what exists, not what is hoped for. A milestone is complete when its
mechanism is implemented, tested, and either measured or explicitly marked unmeasured.

## v0.1 — RecurLocal (shipped)

Recurrent-state locality for hybrid LLM decode: persisting-L2 windows, layer-ahead pre-touch,
hot-set accounting, the SparkInfer adapter, the evaluation harness.

**Answer: the persist family is bounded below the project's own significance floor.** Ceiling
`2 x min(persisting capacity, footprint) / step traffic`; numerator pinned at 60 MiB by the
hardware; **1.94% weighted** on the best model found -- under the 2% floor the 0.1 scorer rejected at,
which is why 0.2.1 retired the floor from BOTH scorers rather than retiring the project. It
still decides the project's own go/no-go, which is a different question from whether a
submission moved the number. The instrument turned out to be worth more than the policy.

## v0.2 — Core generalization (this release)

- `TensorRegistry`, `TransitGraph`, `TransitPlan`, `ITransitPlanner`, `ITransitExecutor`,
  `TransitRuntime` — engine-independent, CPU-tested.
- Five planners: `baseline`, `recurrent_v0`, `greedy`, `budgeted`, `concurrency`.
- Five admission rules: `density`, `quota`, `proportional`, `reuse_order`, `role_floor`.
- `CudaTransitExecutor`, including graph-node window attachment, verified on hardware.
- Trace / plan / evaluation JSON schemas, golden plan digests, the `tensortransit` CLI.
- Namespace and package migration with compatibility shims that are tested, not assumed.

**Answer so far:** the Transit Graph reproduces every published 0.1 bound from a recorded
trace rather than a hand-written geometry file — 293.6 MiB removable, 146.8 MiB footprint, 41%
residency, +1.69% traffic ceiling, +0.68% persist ceiling. And a negative result worth as much:
**under a linear cost model no admission rule can beat pure density**, so the global planner
beat the naive both-persistent arm by 10x and could not beat the best single-role arm.

That negative result was about the MODEL, and 0.2.1 replaced the model. It is preserved as the
control — `--cost-model linear` still reproduces it exactly — and is no longer the answer. See
[evaluation.md](evaluation.md).

## v0.2.1 — The core in the measured path, and a scoring regime that can be reached

0.2.0 built the generalization and left it disconnected. The adapter and the synthetic
benchmark both drove the 0.1 `CudaLocalityController` directly, so `TransitRuntime`,
`TensorRegistry`, `TransitGraph`, `ITransitPlanner` and `CudaTransitExecutor` were reachable
only from the CLI and the tests. A contributor who wrote a planner changed **nothing** about
the measured end-to-end number, which made the entire competition surface decorative.

- **The adapter routes through `TransitRuntime` + `CudaTransitExecutor`**, with the 0.1
  controller kept as a second engine selected by `TENSORTRANSIT_ENGINE` so the two are
  comparable in one process against one model load. Verified: token-exact, 96 windows attached
  to captured graph nodes, zero capture invalidations, and the two engines' gains overlap
  inside their own noise floors (`results/rtx5090-0.2.1-rewiring-check.json`).
- **The synthetic benchmark routes through the same path.** `--planner`, `--admission`,
  `--cost-model` and `--window-preference` now move the number it prints.
- **KV is registered.** `declare_kv_cache` exposes the paged pools, so the second proof track
  can be measured at all — it previously could not be, and that is different from not having
  been.
- **A cost model that can express coordination.** The linear model made greedy-on-density
  provably optimal; the residency model is fitted to the hardware arms in `results/` and is
  superlinear in residency, so concentrating beats spreading and the admission axis measures
  something. `AdmissionRule::Survival` is the rule that exploits it.
- **Frontier Gain replaces the impact bands.** Continuous `dF`, Pareto hypervolume over
  goodput and p99 inter-token latency, a paired bootstrap as a qualification gate, a
  protected-workload guard, and a permanent Frontier Receipt in an append-only ledger. The
  bands went because their lowest paying step was above this hardware's physical ceiling.
- **A trusted, keyless, ephemeral GPU runner**, and the anti-gaming overlay it depends on,
  both proven by CI rather than described.
- Plan replay, a gated overhead budget, an O(log) live-set query, and live trace recording.

## v0.3 — A cost model that can express coordination

**Done in 0.2.1.** All three terms are carried, the model is fitted to every paired hardware
measurement of the `persist` arm in `results/` across two architectures, and it beats the
linear model on those points (rms 0.271 against 0.485 points; 8 of 8 arms inside their own
noise floor against 6 of 8). `eval/cost_model_fit.py` is the fit and it runs in CI.

What is still open here is the parameter the current data cannot pin: `stream_relief`, how much
of a Stream-hinted tensor's traffic actually stops interfering. It is exposed as a dial with an
optimistic default of 1.0 and it is **unmeasured**. The experiment is now runnable — a `Stream`
window reaches a captured graph node as of 0.2.1, where before it was skipped under capture and
therefore unreachable in the only regime that matters.

## v0.4 — Recurrent + KV QoS, measured

**Half done, and the half that is done changed the question.**

At MODEL level the second proof track now has an answer: under the residency cost model the
global arm beats the best independent arm on all three golden traces (+0.177%, +0.155%,
+0.129% against +0.102%), where under the linear model it provably could not. The mechanism is
isolable to one dial: set `--stream-relief 0` and the advantage disappears everywhere. So the
whole of the coordination advantage, on these traces, is the `Stream` action — which the linear
model priced at zero and which was unreachable under a captured decode graph until 0.2.1.
`results/rtx5090-second-proof-track-model.json`.

That is a prediction, not a measurement, and it names the experiment that settles it:
**measure `stream_relief`**. The prerequisites are closed — the adapter registers KV, the arms
are `TENSORTRANSIT_PRESET=<arm>` on the real model through the measured path, and a `Stream`
window now reaches a captured graph node. What is left is device time.

**Known before starting:** on the pinned dense model this contest is for less than a point of
throughput. What is not known is what any of it does to a p99 tail, which is the frontier's
other objective.

**Known before starting:** on the pinned dense model this contest is for less than a point.
The interesting regime is a model whose decode step moves under **6.42 GB**, which is what
`eval/traffic_budget.py --persisting-l2-bytes` prints as the break-even.

## v0.5 — Concurrency-aware planning

`ConcurrencyPlanner` exists and implements even-share and concentrate arbitration; neither is
measured. The measured fact it has to beat: `persist` pays at 98% residency and is negative
from four sequences on, where residency is 48%. **The crossover is at about half residency.**
Concentrating the budget on fewer requests is the obvious idea and it is untested.

## v0.6 — Automatic tracing

Traces are written by hand or by the adapter today. A `TransitTracer` over CUPTI or kernel
launch metadata would let a runtime produce one without annotation. **Note the constraint on
this hardware:** `RmProfilingAdminOnly: 1` blocks `ncu`, `nsys --gpu-metrics` and CUPTI's
profiler together, so counter-free tracing is what is available here.

## v0.7 — Offline planning

**Done in 0.2.1.** `trace.json -> planner -> plan.json -> replay` closes:
`tensortransit replay <plan.json> --trace <trace.json>` reads a serialized plan back, rebinds
its regions against a live tensor table by id, walks the kernels and reports what fired, whether
every Persist was cleared before the step ended and whether every fork was joined.

A serialized plan carries no pointer, by design — a device address from another process is
meaningless in this one — so `TransitPlan::rebind()` is what makes it executable, and it fails
loudly on a tensor the registry does not know rather than skipping it. Regions are clamped to
the descriptor's allocation, so a plan read from a file cannot widen a window past memory the
runtime owns whatever the file says.

The adapter can now also record a trace from the LIVE runtime (`TENSORTRANSIT_TRACE_OUT`),
which is what gives an offline comparison real KV block sizes and real per-layer demand instead
of a hand-written geometry.

## v0.8 — CUDA Graph node-level actions

Window attachment to captured nodes works and is verified. What is not built: placing
prefetch as a graph node rather than a stream fork, which is where the 1.20 points of
fork/join overhead live.

## v0.9 — Speculative state

A new tensor class whose future use is *probabilistic*, which is the first thing that would
require the graph to carry branch probability. Not started.

## v1.0

Only after: two independent tensor classes measured on real models, a global planner shown to
beat independent policies **on hardware**, stable evaluation, stable public API, several
planners. Three of those are open.

---

## What would end the project

The go/no-go rules are in the specification and one of them has already fired once, on the
recurrent-persist family. The rest still apply to TensorTransit as a whole:

1. real end-to-end gains stay inside benchmark noise;
2. planner or executor overhead cancels the locality benefit;
3. global multi-tensor planning does not beat independent policies;
4. the useful optimization belongs entirely inside the runtime;
5. it turns into a generic allocator or offloader;
6. it turns into a kernel library;
7. a mature project appears with the same central abstraction.

Rule 3 is **half resolved, and the half that is resolved is a model result.** Under the
residency cost model the global arm beats the best independent arm on all three golden traces,
where under the linear model it provably could not; setting `--stream-relief 0` makes the
advantage disappear on every one, which identifies the mechanism as the `Stream` action. On
hardware the arms have now been measured and the global arm ties `density` within noise — but
that measurement did NOT exercise a `Stream` action at all, because the adapter registers a
weight tensor only when the operator declares the step traffic. So the mechanism the model
names is still untested, and the experiment is one environment variable away. See
[VERDICT.md](VERDICT.md).
