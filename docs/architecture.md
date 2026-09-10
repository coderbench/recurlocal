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
