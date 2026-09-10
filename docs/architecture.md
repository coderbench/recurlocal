# Architecture

TensorTransit sits between an inference runtime and the GPU memory hierarchy. It does not
implement model math, does not own any tensor's allocation, and does not schedule inference.
It answers one question, repeatedly: **given what the next kernels are going to read, what
should the memory hierarchy be doing right now.**

```text
inference runtime
      |
      |  register_tensor(role, bytes)      -- what exists
      |  record_use(kernel, tensor, access) -- what will be touched, in what order
      v
+-------------------------------------------------------------+
|                        TensorTransit                        |
|                                                             |
|  TensorRegistry -> TransitGraph -> ITransitPlanner -> Plan   |
|      (what)          (when)          (decide)      (actions) |
|                                                        |     |
|                                              ITransitExecutor|
+--------------------------------------------------------+----+
                                                         |
                                                         v
                                             CUDA memory hierarchy
                                                 L2 <---> HBM
```

Five objects, and the split between them is the whole design.

| object | owns | testable without a GPU |
|---|---|---|
| `TensorRegistry` | address -> role, size, lifetime | yes |
| `TransitGraph` | future use, reuse distance, live set | yes |
| `ITransitPlanner` | every policy decision | yes |
| `TransitPlan` | the decisions, as serializable values | yes |
| `ITransitExecutor` | turning a decision into a CUDA call | no |

**The executor decides nothing.** Which region, which hit ratio, when to clear, which stream
— all of it is in the plan, where a CPU test can reach it and a contributor can read it. This
is not tidiness. In 0.1 the region arithmetic lived inside a `.cu` file, so a window that ran
past the end of an allocation could not be caught by anything short of a device run.

## The four stages

### 1. Declaration

A runtime registers the tensors it is willing to have coordinated. The registry is
**non-owning**: TensorTransit never allocates or frees runtime memory and never dereferences
a tensor pointer on the host. Everything it does is address arithmetic and bookkeeping.

Identity is `(id, generation)`, not an address, because a runtime frees a KV block and
allocates another at the same address. Without the generation a plan compiled against the
first would place a window on the second, and nothing anywhere would notice.

### 2. Recording

The runtime declares, per kernel, which tensors it touches and how. Ordering alone is enough
for a first plan; time estimates are optional and everything degrades sensibly without them.

A recording window is **one iteration of a loop** when `cyclic` is set, and a decode token is
exactly such a loop. This matters more than it sounds: a recurrent state is used once per
token, so its only reuse edge is the one that crosses the token boundary. A graph that did
not close the loop would report the entire recurrent surface as unreused, every planner would
correctly decline all of it, and the wrong answer would look exactly like a right one.

### 3. Planning

The planner sees roles, sizes, reuse and a device profile. It never sees a model. It emits a
`TransitPlan`: actions placed before or after named kernels, plus a **decline record** saying
what did not get budget and why.

The declines are not an afterthought. Every evaluator guard in this repository exists because
a confident number hid a null result, so the generalization records the refusals as
first-class output: `no_reuse`, `budget_exhausted`, `too_large`, `reuse_too_far`,
`role_excluded`, `below_min_hit_ratio`, `not_supported`.

Plans are **reused across tokens** (spec section 33). `TransitRuntime` keys a compiled plan on
`(registry epoch, graph digest, concurrency, phase, device, config)` and reports
`last_recompile_reason` when the key moves. A decode loop that recompiled every token would
put the planner on the critical path, which is the single most likely way this layer costs
more than it saves — and without a reason code, finding that out needs a profiler.

### 4. Execution

`CudaTransitExecutor` applies actions. It refuses two things, both because they have gone
wrong here:

- a tensor whose registry entry no longer matches the plan's pointer (`stale_tensor_refs`);
- a tensor with `device == -1`, which is how `read_trace()` marks a synthetic address, so a
  plan built offline can never reach a driver.

The one mechanism that is not optional: **under CUDA Graph capture a stream access-policy
window is host-side state the graph never records.** A persisting policy set on the stream
during capture is absent from every replay while its telemetry still counts it as applied.
`attach_window_to_captured_node()` sets the attribute on the kernel node instead, and
`persist_attached_to_node` is the only counter that can distinguish a policy that ran from
one that merely reported itself.

## Two engines, one binary

The SparkInfer adapter carries both the 0.1 controller and the pipeline above, selected by
`TENSORTRANSIT_ENGINE`:

```text
TENSORTRANSIT_ENGINE=transit   TensorRegistry -> TransitGraph -> ITransitPlanner
                               -> CudaTransitExecutor          (the default)
TENSORTRANSIT_ENGINE=v0        CudaLocalityController          (the 0.1 policy, the control)
```

Both, in one binary, deliberately. Until 0.2.1 the adapter drove `v0` directly and the whole
pipeline above was reachable only from the CLI and the tests, so a contributor who wrote a
planner changed nothing about the number the evaluator prints. Replacing `v0` outright would
have made the migration unverifiable — "it changed nothing" would have been an assertion.
Keeping both makes it an A/B against one model load on one box, and
`results/rtx5090-0.2.1-rewiring-check.json` is that A/B: token-exact on both, 96 windows
attached to captured graph nodes on both, zero capture invalidations on both, and gains that
overlap inside their own noise floors.

What the adapter registers on the transit path:

| declared | as | why |
|---|---|---|
| per recurrent layer: the fp32 matrix state and the bf16 conv window | `RecurrentState`, `ReadWrite` | one global read and one global write per layer, which is what the pinned runtime's own kernel comment says |
| per full-attention layer: the K and V slices actually read this step | `KVCache`, `Read` | live bytes, not the pool stride: the pool is sized for the longest context the server will ever hold |
| per layer, when the operator declares a step figure | `ModelWeight` | the denominator and the interference. Fabricated address, `device = -1`, so the executor refuses to let it reach a driver |

One kernel per **absolute layer index**, in order, so a hybrid model's attention layers sit
between two recurrent ones where they belong — reuse distance in bytes is a question about what
runs in between, and a graph that recorded only the recurrent layers would answer it with the
attention traffic missing.

The window is delivered under `max_windows_per_kernel = 1`, because that is what the hardware
does: CUDA binds one access-policy window to a stream, a launch or a graph node at a time. A
plan marking two regions before one kernel is not describing something the device can do — the
second replaces the first, the first is silently absent, and the telemetry counts two applied
windows for one delivered policy.

## Where RecurLocal went

RecurLocal is now the recurrent-state workload inside TensorTransit, and the migration is
structural rather than a rename:

| 0.1 | 0.2 |
|---|---|
| `include/recurlocal/planner.h` | `include/tensortransit/recurrent.h` (+ shim at the old path) |
| `src/cuda/*.cu` | `executors/cuda/*.cu` |
| `workloads/recurrent/synthetic/cuda_bench.cu` | `workloads/recurrent/synthetic/cuda_bench.cu` |
| `integrations/sparkinfer/` | `adapters/sparkinfer/` |
| `namespace recurlocal` | `namespace tensortransit` (+ namespace alias) |
| the shipped policy | `planners/recurrent_v0/` |

`RecurrentV0Planner` **calls** `LocalityPlanner` rather than reimplementing it, and
`tests/test_transit_planner.cpp` asserts that the hit ratio and window size it emits are the
ones the 0.1 policy computes. A reimplementation would have been free to drift from the policy
every published number was measured under, with nothing in a diff to show it.

Every deprecated name still resolves and `tests/test_compat.cpp` includes **only** the
deprecated headers, so the shim breaking is a build failure here rather than a link failure in
somebody else's tree. See [STABILITY.md](STABILITY.md) section 8 for the removal schedule.

## What is deliberately absent

No allocator. No offload tier. No transport. No kernels of its own beyond the pre-touch walks.
No scheduler. If the repository grows any of those it has stopped being this project — the
boundary table in the specification is the one to check against.
