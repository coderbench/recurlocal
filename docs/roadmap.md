# Roadmap

Version numbers describe what exists, not what is hoped for. A milestone is complete when its
mechanism is implemented, tested, and either measured or explicitly marked unmeasured.

## v0.1 — RecurLocal (shipped)

Recurrent-state locality for hybrid LLM decode: persisting-L2 windows, layer-ahead pre-touch,
hot-set accounting, the SparkInfer adapter, the evaluation harness.

**Answer: the persist family is bounded below the project's own significance floor.** Ceiling
`2 x min(persisting capacity, footprint) / step traffic`; numerator pinned at 60 MiB by the
hardware; **1.94% weighted against a 2.0% floor** on the best model found. The instrument
turned out to be worth more than the policy.

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
beats the naive both-persistent arm by 10x and cannot beat the best single-role arm. See
[evaluation.md](evaluation.md).

## v0.3 — A cost model that can express coordination

The concrete blocker on the second proof track, and it needs no GPU.

The linear model — saving proportional to resident share — makes greedy-on-density provably
optimal, which means the whole admission-rule axis is measuring nothing the model can see.
What is missing:

- **whole-line residency**: a line is resident or it is not;
- **survival**: a tensor whose reuse distance exceeds the budget is evicted before it pays,
  however much of it was admitted;
- **interference**: a term for what the streaming half of the cache does to the persisting
  half, which is the only place a `Stream` action can have a value.

A cost model with those terms, validated against the measurements already in `results/`, is
the highest-value contribution available right now.

## v0.4 — Recurrent + KV QoS, measured

Run the five arms on hardware and settle whether the global planner beats the independent
policies on a real workload. The mechanism is built and the arms are one command
(`tensortransit compare`, `eval/real_eval.py`); what is missing is device time and a runtime
that exposes its KV blocks to the registry.

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

`trace.json -> planner -> plan.json -> replay`. Two thirds of this exists: the CLI plans from a
trace and dumps the plan. What is missing is replay — feeding a serialized plan back into an
executor — which is what would let a contributor optimize without touching the runtime.

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

Rule 3 is currently **unresolved and honestly reported as unresolved**: the model says the
global planner ties the best independent arm, and the model cannot see the mechanisms by which
it might win. Resolving it is v0.3 and v0.4.
