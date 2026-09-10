# RecurLocal

> **This is the original project specification, kept as the historical record.** Two parts of
> it have been superseded by measurement and are marked where they appear:
>
> * **Section 26's impact-label table (XS/S/M/L/XL).** Removed in 0.2.1. Its lowest paying step
>   was 2% weighted throughput gain, and the physical ceiling for the whole shipped policy
>   family is 0.52% on the scored model and 1.94% on the best model this project has found — so
>   a submission could remove every recoverable byte of recurrent traffic and score `none`.
>   Scoring is now the continuous Frontier Gain of [`frontier/README.md`](frontier/README.md).
> * **Section 21's significance floor**, for the same reason. What replaces it is a per-cell
>   noise floor, calibrated on the target hardware and published in
>   [`frontier/TTF-1/reference.json`](frontier/TTF-1/reference.json), plus a paired bootstrap
>   that returns `INCONCLUSIVE` rather than a number it cannot support.
>
> Everything else here — the boundary table, the go/no-go rules, the correctness standard, the
> anti-gaming requirements, the security posture — is current.

> **Status as of 0.2.0.** This is the RecurLocal specification, and it remains the document
> every "section N" reference in this repository points at — the numbering is load-bearing in
> `docs/`, in `results/*.json` and in `eval/decide.py`, so the file keeps its name.
>
> It is **not** the scope authority any more. RecurLocal is now the recurrent-state workload
> inside TensorTransit, whose boundary is the table in the README and in
> `docs/design-principles.md`. Where the two disagree about what the project *is*, the newer
> one wins; where they disagree about a measurement, neither does — check `results/`.

## Detailed Project Overview and Technical Specification

**Project:** RecurLocal  
**Version:** v0.1 feasibility prototype  
**Category:** AI inference systems / CUDA performance / recurrent-state locality  
**Primary target:** Hybrid recurrent LLM inference on NVIDIA Blackwell GPUs  
**Initial integration target:** SparkInfer + Qwen3.8-27B + RTX 5090  
**License:** MIT

---

# 1. Executive Summary

**RecurLocal is a CUDA-focused performance project for improving the locality of mutable recurrent neural state in hybrid large language models.**

Its core goal is simple:

> **Keep recurrent AI state close to compute instead of repeatedly round-tripping it through high-bandwidth memory when avoidable.**

Modern hybrid LLMs increasingly combine:

- full attention;
- linear attention;
- Gated DeltaNet / GDN;
- KDA-style recurrent operators;
- Mamba-like state-space mechanisms;
- other recurrent or persistent-state layers.

These architectures carry state that behaves differently from a traditional Transformer KV cache.

A conventional Transformer mainly accumulates a KV cache whose size grows with sequence length.

A hybrid recurrent model additionally maintains fixed-size mutable state that is repeatedly read, updated, and written during generation.

Conceptually:

```text
Traditional Transformer

request
   |
   v
attention
   |
   v
KV cache
```

Hybrid model:

```text
request
   |
   +-----------------------------+
   |                             |
   v                             v
attention KV               recurrent state
                           convolution state
                           other layer state
```

RecurLocal focuses specifically on the **physical locality and memory-hierarchy behavior** of that recurrent state.

The first feasibility thesis is:

> If the next recurrent layer's state is known in advance, can software-directed L2 cache policy and asynchronous state pre-touch reduce HBM traffic enough to create a measurable end-to-end inference speedup?

RecurLocal deliberately starts with this narrow question rather than attempting to become a full inference engine.

---

# 2. What RecurLocal Is Not

RecurLocal is **not**:

- an LLM server;
- a replacement for SparkInfer;
- a replacement for vLLM or SGLang;
- a new Gated DeltaNet implementation;
- a KV-cache framework;
- a state quantization library;
- a generic model compiler;
- a checkpoint compression framework;
- a speculative-decoding engine;
- a CPU/NVMe state-paging system in v0.1.

Instead, RecurLocal is intended to become a specialized low-level component:

```text
                    inference runtime
                           |
                           v
                recurrent-state access
                           |
                           v
                    +-------------+
                    | RecurLocal  |
                    |             |
                    | locality    |
                    | prefetch    |
                    | cache hints |
                    +------+------+ 
                           |
                           v
                CUDA memory hierarchy
                   L2 <----> HBM
                           |
                           v
                    recurrent kernel
```

The inference runtime remains responsible for model execution.

RecurLocal attempts to make the data needed by that execution arrive at the right place at the right time.

---

# 3. Why This Problem Exists

## 3.1 Recurrent State Is Different From Model Weights

Large model weights often behave like a streaming workload.

For a given layer:

```text
load layer weights
      |
      v
perform computation
      |
      v
move to next layer
```

The same layer's weights may not be touched again until the next generated token.

Mutable recurrent state behaves differently.

For every decode token:

```text
load recurrent state
        |
        v
update state
        |
        v
write state
```

Then the same state is needed again on the next generated token.

Therefore the memory-access patterns have different locality characteristics.

Conceptually:

```text
MODEL WEIGHTS

large
mostly streamed
lower temporal locality


RECURRENT STATE

smaller per layer
mutable
revisited every token
high temporal importance
```

RecurLocal tries to exploit that difference.

---

# 4. Initial Qwen3.8 Motivation

The initial target is Qwen3.8-27B because it is highly relevant to the current SparkInfer/Gittensor stack and contains many recurrent/linear-attention layers.

The feasibility configuration included in the repository models:

```text
48 recurrent layers

state geometry:
48 value heads
128 value dimension
128 matrix dimension
FP32 state
```

Matrix-state size per recurrent layer:

```text
48 × 128 × 128 × 4 bytes
≈ 3 MiB
```

Across 48 recurrent layers:

```text
≈ 144 MiB
```

of matrix-state data per active sequence, before accounting for additional state or metadata.

This means recurrent-state traffic can become significant during continuous decode, especially with concurrency.

The exact production layout is determined by the inference runtime and model implementation. The repository's synthetic benchmark is intentionally an approximation used to test memory-locality ideas.

---

# 5. Core Hypothesis

RecurLocal's initial hypothesis is based on a predictable property of autoregressive inference:

> **The runtime knows which layer comes next.**

For recurrent layers:

```text
Layer N
   |
   v
Layer N+1
   |
   v
Layer N+2
```

If the runtime is currently computing layer `N`, it already knows that the state for layer `N+1` will soon be required.

Therefore RecurLocal can experiment with:

```text
while computing layer N
        |
        +----> asynchronously touch state N+1
        |
        v
finish layer N
        |
        v
start layer N+1 with potentially warmer cache state
```

At the same time, RecurLocal can use CUDA access-policy windows to hint that:

```text
current recurrent state -> persisting/hot

large streaming data -> normal or streaming
```

The goal is not to force the hardware to obey a particular placement.

CUDA cache-policy mechanisms are hints.

Therefore every policy must be judged experimentally.

---

# 6. RecurLocal's Technical Primitive

The project owns the concept of a:

> **Recurrent-State Locality Controller**

A runtime integration exposes:

```text
current recurrent-state address
current recurrent-state size
next recurrent-state address
next recurrent-state size
compute stream
prefetch stream
layer ordering
device
```

RecurLocal returns or applies locality actions:

```text
hot-window size
cache hit-ratio hint
persisting-L2 policy
next-layer pre-touch decision
future prefetch strategy
```

A conceptual API:

```cpp
recurlocal::CudaLocalityController controller(device, config);

controller.bind_streams(compute_stream, prefetch_stream);

for each recurrent layer:
    controller.before_layer(
        current_state,
        current_state_bytes,
        next_state,
        next_state_count,
        has_next_recurrent_layer
    );

    run_existing_recurrent_kernel();

    controller.after_layer();
```

The model kernel remains unchanged.

That clean boundary is important.

---

# 7. Current Repository Structure

The v0.1 repository contains:

```text
recurlocal/
├── README.md
├── LICENSE
├── SECURITY.md
├── CONTRIBUTING.md
├── CMakeLists.txt
├── repo-manifest.json
│
├── include/
│   └── recurlocal/
│       ├── planner.h
│       └── cuda_api.h
│
├── src/
│   ├── planner.cpp
│   └── cuda/
│       ├── cache_control.cu
│       └── prefetch.cu
│
├── bench/
│   └── cuda_bench.cu
│
├── eval/
│   ├── README.md
│   ├── run_eval.py
│   └── result_schema.json
│
├── tests/
│   └── test_planner.cpp
│
├── tools/
│   └── info.cpp
│
├── configs/
│   └── qwen3.8-27b-feasibility.json
│
├── integrations/
│   └── sparkinfer/
│       └── README.md
│
├── docs/
│   ├── DESIGN.md
│   └── FRONTIER.md
│
├── scripts/
│   ├── build_cpu.sh
│   ├── build_cuda.sh
│   └── run_feasibility.sh
│
└── .github/
    └── workflows/
        └── ci.yml
```

The repository is intentionally small because v0.1 is a **go/no-go feasibility test**, not yet a full production framework.

---

# 8. CPU Locality Planner

The CPU-side planner is separated from CUDA code.

This allows policy logic to be:

- unit-tested without a GPU;
- inspected independently;
- reused by runtime integrations;
- benchmarked separately;
- changed by contributors without requiring CUDA hardware.

The planner receives device capabilities:

```cpp
struct DeviceCaps {
    size_t l2_bytes;
    size_t persisting_l2_max_bytes;
    size_t access_policy_max_window_bytes;
};
```

And configuration:

```cpp
struct PlannerConfig {
    LocalityMode mode;

    double persisting_budget_fraction;
    double hit_ratio;

    int prefetch_distance;

    size_t max_hot_window_bytes;
};
```

It produces a per-layer plan:

```cpp
struct LayerPlan {
    size_t hot_window_bytes;
    double hit_ratio;

    bool use_persisting_window;
    bool prefetch_next;
};
```

---

# 9. Supported v0.1 Modes

The feasibility benchmark implements four modes.

## 9.1 Baseline

```text
baseline
```

No RecurLocal cache policy.

No next-layer pre-touch.

This represents the control case.

---

## 9.2 Persist

```text
persist
```

RecurLocal applies a CUDA access-policy window to the currently used recurrent-state region.

The intended hint is:

```text
current recurrent state
        |
        v
prefer persisting cache behavior
```

This does not guarantee that the data stays in L2.

It tells CUDA that the data has higher persistence value.

---

## 9.3 Prefetch

```text
prefetch
```

The controller asynchronously reads the next recurrent layer's state using a separate stream.

Conceptually:

```text
compute state N
     |
     +------ pre-touch state N+1
     |
     v
next layer
```

This is a feasibility mechanism.

The current pre-touch kernel is intentionally simple.

Future contributions can replace it with more advanced architecture-specific strategies.

---

## 9.4 Combined

```text
combined
```

Applies:

```text
persist current state
+
pre-touch next state
```

This is expected to be the most interesting initial experiment, but it is not assumed to be the winner.

The evaluator measures rather than assumes.

---

# 10. CUDA L2 Cache Control

CUDA exposes device properties including:

```text
l2CacheSize
persistingL2CacheMaxSize
accessPolicyMaxWindowSize
```

RecurLocal queries these at runtime.

It does not hardcode the usable L2 reservation.

The controller can request:

```cpp
cudaDeviceSetLimit(
    cudaLimitPersistingL2CacheSize,
    requested_bytes
);
```

And set a stream access-policy window:

```text
base pointer
number of bytes
hit ratio
hit property = persisting
miss property = streaming
```

This allows RecurLocal to express:

> This recurrent-state region is expected to be valuable soon and repeatedly.

The runtime must still be benchmarked because cache hints can hurt if misconfigured.

---

# 11. Why Oversubscription Matters

Suppose the device can effectively reserve:

```text
X MiB
```

for persisting behavior.

If RecurLocal simultaneously marks:

```text
3X MiB
```

as high-priority, the cache may thrash.

Therefore the planner estimates the active hot set.

If the requested hot set exceeds the configured budget, it reduces the requested hit ratio rather than pretending all data can remain resident.

Conceptually:

```text
hot set <= budget
    |
    v
high hit-ratio request


hot set > budget
    |
    v
reduce hit-ratio request
```

This is intentionally conservative.

A major future optimization frontier is finding a much better policy than the simple v0.1 heuristic.

---

# 12. Asynchronous Pre-Touch

The current CUDA prototype implements an asynchronous read-only kernel.

It reads the next recurrent-state buffer before its actual layer executes.

Important:

> **This is not a guaranteed cache prefetch primitive.**

CUDA does not expose a generic command equivalent to:

```text
place this cudaMalloc range in L2 now
```

for arbitrary data.

Instead, the prototype makes global reads on a second stream, with the goal of experimentally warming the cache hierarchy.

The pre-touch operation:

- does not change model state;
- executes asynchronously;
- uses a separate stream;
- produces a tiny observable scratch result to prevent complete compiler elimination.

This implementation is deliberately replaceable.

---

# 13. Why The Current Prefetch Kernel Is Only A Prototype

The current kernel is useful for answering:

> Is the locality idea worth pursuing at all?

It is not meant to be the final implementation.

Potential future implementations include:

- Blackwell-specific memory instructions;
- better vectorized reads;
- 128-bit / wider loads;
- warp-cooperative prefetch;
- TMA-assisted data movement where applicable;
- cache-aware tiling;
- fused prefetch inside adjacent model kernels;
- CUDA Graph node integration;
- persistent kernels;
- architecture-specific prefetch schedules;
- direct integration with GDN execution.

This is an ideal Gittensor-style optimization frontier because many different implementation strategies can be compared objectively.

---

# 14. Synthetic CUDA Benchmark

The repository contains:

```text
bench/cuda_bench.cu
```

Its purpose is to compare the four locality modes using recurrent-state-like buffers.

Default geometry:

```text
48 layers
3 MiB state per layer
32 measured tokens
4 warm-up tokens
```

Total synthetic state:

```text
48 × 3 MiB
≈ 144 MiB
```

The benchmark repeatedly updates each state block with deterministic floating-point arithmetic.

It measures:

```text
elapsed milliseconds
milliseconds per token
layer updates per second
L2 size
maximum persisting L2
maximum access-policy window
actual configured L2 set-aside
checksum
```

Example output shape:

```json
{
  "mode": "combined",
  "gpu": "NVIDIA GeForce RTX 5090",
  "layers": 48,
  "state_bytes_per_layer": 3145728,
  "total_state_bytes": 150994944,
  "tokens": 32,
  "elapsed_ms": 0.0,
  "ms_per_token": 0.0,
  "layer_updates_per_s": 0.0,
  "l2_bytes": 0,
  "persisting_l2_max_bytes": 0,
  "access_policy_max_window_bytes": 0,
  "actual_l2_set_aside_bytes": 0,
  "checksum": 0.0
}
```

The zeros above are illustrative placeholders for output fields, not benchmark results.

---

# 15. Deterministic Correctness

The locality track must not change model computation.

Therefore all synthetic modes are expected to produce the same final state.

The benchmark calculates a checksum after execution.

The evaluator checks:

```text
baseline checksum
=
persist checksum
=
prefetch checksum
=
combined checksum
```

within a strict numerical tolerance.

The purpose is to catch accidental modification or ordering problems.

In a real SparkInfer integration, stronger checks should include:

- generated-token equality;
- logit comparison;
- recurrent-state comparison;
- deterministic prompt replay.

---

# 16. Built-In Evaluation System

The repository already contains its own evaluator:

```text
eval/run_eval.py
```

This is important for Gittensor compatibility.

The evaluation logic is part of the public repository rather than being an undefined external process.

Current feasibility evaluation:

```text
run baseline
run persist
run prefetch
run combined
      |
      v
verify checksums
      |
      v
compare elapsed time
      |
      v
report best synthetic mode
```

Result:

```text
eval-result.json
```

contains:

```text
schema version
correctness
raw results
speedups
best mode
best synthetic gain
interpretation warning
```

The result explicitly states:

> synthetic feasibility only; do not publish as a model-serving speedup

This distinction is intentional.

---

# 17. The Most Important Evaluation Rule

A microbenchmark win is not enough.

Suppose:

```text
synthetic recurrent benchmark
+35%
```

but real Qwen3.8 decode changes:

```text
+0.8%
```

Then RecurLocal has not achieved its primary purpose.

The authoritative metric must be:

> **Real end-to-end inference improvement.**

Synthetic and hardware-counter measurements explain why the change works.

They do not define the final impact.

---

# 18. Real SparkInfer Evaluation

The first serious experiment should pin:

```text
Model:
Qwen3.8-27B

Runtime:
a specific SparkInfer commit

GPU:
RTX 5090

Generation:
deterministic greedy decode

Candidate:
same runtime with RecurLocal integration
```

Do not benchmark against a moving `main`.

Record exact:

```text
SparkInfer commit
RecurLocal commit
model checkpoint / digest
CUDA version
driver version
GPU
power configuration
prompt set
context length
generation length
concurrency
```

---

# 19. Real Evaluation Matrix

The first integration benchmark should include at minimum:

## Single-request decode

```text
batch/concurrency = 1
```

Measure:

```text
tokens/sec
ms/token
recurrent-kernel time
```

---

## Concurrency

Recommended:

```text
4
16
32
```

Measure:

```text
aggregate tok/s
per-request latency
recurrent-state locality behavior
```

Concurrency matters because multiple active recurrent-state working sets may compete for L2.

A locality strategy that wins at batch 1 may lose badly at batch 32.

---

## Context

Use several initial context lengths.

For example:

```text
small
medium
long
```

The exact matrix should match the pinned SparkInfer release and available memory.

---

# 20. Hardware Counter Evidence

Where possible, use Nsight Compute or CUPTI to measure:

```text
HBM read traffic
HBM write traffic
L2 hit rate
L2 sectors
L2 read/write traffic
DRAM throughput
kernel duration
cache-thrash indicators
```

The desired chain of evidence is:

```text
RecurLocal policy
      |
      v
better locality
      |
      v
less recurrent-state HBM traffic
      |
      v
faster recurrent execution
      |
      v
faster real model decode
```

If the chain breaks, the optimization needs investigation.

---

# 21. Go / No-Go Criteria

Before RecurLocal becomes a full Gittensor-targeted project, test its real end-to-end impact.

Recommended decision rule:

| Real end-to-end decode improvement | Interpretation |
|---|---|
| <2% | reject the core hypothesis |
| 2–4% | weak; probably reject |
| 4–7% | promising |
| 7–10% | strong repo candidate |
| >10% | expand immediately |

The project should be willing to fail this test.

That is important.

A technically interesting idea that does not materially improve real serving should not be forced into a Gittensor repository.

---

# 22. Why This Is Compatible With SparkInfer

SparkInfer owns model execution.

Its performance work includes areas such as:

```text
GEMM/GEMV
attention
MoE
GDN/linear-attention kernels
prefill
decode
scheduler
KV cache
server
speculative decoding
```

RecurLocal owns a different layer:

```text
recurrent-state locality
cache residency hints
prefetch timing
hot-set planning
memory hierarchy behavior
```

The boundary:

```text
                RecurLocal

       decide where/when state
       should be made hot
              |
              v
               SparkInfer

       execute model computation
              |
              v
                GPU
```

Therefore RecurLocal can remain a separate repository while directly helping SparkInfer.

---

# 23. Why This Could Be Compatible With Other Runtimes

The core API should not depend on SparkInfer-specific model code.

Future adapters could target:

```text
vLLM
SGLang
TensorRT-LLM
llama.cpp CUDA paths
custom GDN runtimes
Mamba runtimes
RWKV runtimes
```

The generic requirement is:

```text
runtime knows the recurrent-state pointer
runtime knows layer order
runtime exposes CUDA stream(s)
```

This makes RecurLocal potentially broader than a single Gittensor repository.

---

# 24. Why The Opportunity Can Continue Indefinitely

RecurLocal exposes a multi-dimensional optimization frontier.

## Hardware changes

```text
RTX 5090
RTX PRO 6000
future Blackwell variants
Rubin
future NVIDIA architectures
```

Different cache sizes and memory hierarchies can require different policies.

---

## Model changes

```text
Gated DeltaNet
KDA
Mamba
RWKV
future linear-attention architectures
```

Different recurrent-state shapes produce different locality behavior.

---

## Workload changes

```text
batch 1
high concurrency
short context
long context
speculative decoding
multimodal serving
agent serving
```

Different workloads produce different hot sets.

---

## Implementation changes

Contributors can explore:

```text
window size
hit ratio
prefetch distance
stream priority
prefetch kernel
state layout
state tiling
batch partitioning
fusion
CUDA Graph policy
device-specific paths
compiler codegen
```

There is no obvious final configuration that wins for all hardware and all models.

---

# 25. Gittensor Miner Model

RecurLocal should be operated as a **self-directed frontier repository**.

Maintainers do not need to publish issues describing optimizations.

Miner workflow:

```text
clone current main
      |
      v
run benchmark
      |
      v
profile
      |
      v
discover bottleneck independently
      |
      v
implement optimization
      |
      v
submit PR
      |
      v
authoritative evaluation
      |
      v
correctness
+
real serving gain
      |
      v
impact label
```

The repository should reward useful measurable improvements, not task completion.

---

# 26. Suggested Impact Labels

Only real serving improvements should qualify for strong impact labels.

Example project policy:

| Real end-to-end improvement | Suggested impact (SUPERSEDED -- see the note at the top of this file) |
|---|---|
| <2% | none |
| 2–4% | XS |
| 4–7% | S |
| 7–10% | M |
| 10–18% | L |
| >18% | XL |

These are proposed RecurLocal project thresholds.

They should not be presented as official Gittensor scoring unless Gittensor explicitly adopts them.

---

# 27. What A Strong Miner PR Looks Like

Example only:

```text
Baseline:

Qwen3.8-27B
RTX 5090
SparkInfer commit X

decode:
220 tok/s

recurrent-state DRAM traffic:
Y GB/s
```

Candidate:

```text
new batch-aware L2 window policy

decode:
238 tok/s

gain:
+8.2%

recurrent-state DRAM traffic:
-31%

output:
identical
```

This is an excellent RecurLocal contribution.

The engineer can explain:

```text
what changed
why locality improved
what hardware counters changed
how much real inference improved
```

Attribution is clear.

---

# 28. What Should Not Receive A Strong Reward

Example:

```text
L2 hit rate:
+40%
```

but:

```text
end-to-end decode:
+0.3%
```

This is not a strong contribution from the project's primary performance perspective.

Likewise:

```text
synthetic benchmark:
+50%
```

but:

```text
real model:
-2%
```

is a regression.

RecurLocal must remain grounded in real serving impact.

---

# 29. Example Release Format

A RecurLocal release should be easy to communicate.

If real measurements support it:

```text
RecurLocal v0.4

+8.7% Qwen3.8 decode on one RTX 5090

- 3.4× lower recurrent-state DRAM traffic
- +21% recurrent-kernel throughput
- +8.7% end-to-end decode
- -6.2% joules/token
- bit-identical greedy output
- SparkInfer integration
```

Another future release could say:

```text
RecurLocal v0.7

+19% batch-32 throughput

- new batch-aware recurrent-state hot-set planner
- lower L2 thrashing under concurrent decode
- unchanged model output
```

All values must come from authoritative measurement.

Never publish synthetic numbers as production claims.

---

# 30. Why This Is "Postable"

The project is intentionally designed so improvements map to understandable metrics.

Good release metrics include:

```text
decode tok/s
concurrent tok/s
recurrent-kernel speed
HBM bytes/token
L2 hit behavior
joules/token
```

This produces the same desirable pattern seen in strong infrastructure releases:

> new technology + clear benchmark + exact hardware + simple percentage

For example:

```text
"New layer-ahead locality planner cuts recurrent-state HBM reads by 44% and improves Qwen3.8 decode by 7.1% on one RTX 5090."
```

That is much easier to understand and share than an abstract research-only result.

---

# 31. Engineering Areas That Can Contribute

## CUDA engineers

Can improve:

```text
pre-touch kernels
memory access width
warp scheduling
stream overlap
L2 window use
TMA experiments
persistent kernels
```

---

## GPU performance engineers

Can investigate:

```text
L2 miss behavior
HBM bottlenecks
cache oversubscription
occupancy
cache/QoS interactions
```

---

## Compiler engineers

Can explore:

```text
automatic locality annotations
generated prefetch plans
kernel fusion
graph scheduling
architecture specialization
```

---

## Systems engineers

Can improve:

```text
runtime integration
multi-request hot-set allocation
stream scheduling
policy configuration
telemetry
```

---

## ML systems researchers

Can model:

```text
layer state reuse
batch locality
model-specific state geometry
workload-adaptive policies
```

---

## Numerical engineers

The exact v0 locality track does not change numerical representation, but future experimental tracks could study:

```text
compressed hot-state representation
mixed precision
error-bounded state formats
```

Those would require separate quality gates.

---

# 32. Current Limitations

RecurLocal v0.1 has important limitations.

## 32.1 No Real RTX 5090 Result Yet

The repository currently contains a feasibility benchmark.

It does **not** contain a verified Qwen3.8 speedup.

This is the most important limitation.

---

## 32.2 Pre-Touch Is Best-Effort

The current asynchronous pre-touch kernel does not guarantee data placement in L2.

It merely generates the access pattern that may warm caches.

A production-quality mechanism may need a more sophisticated implementation.

---

## 32.3 Access-Policy Windows Are Hints

CUDA's persisting-cache behavior is not a lock.

The GPU may still evict data.

Therefore RecurLocal must remain measurement-driven.

---

## 32.4 L2 Is Shared

Model weights, activations, KV data, recurrent state, and other kernels all interact with the cache hierarchy.

Improving recurrent-state locality can potentially hurt other workloads.

The evaluator must measure the whole model.

---

## 32.5 Concurrency Makes The Problem Harder

With many active requests:

```text
state A
state B
state C
...
```

all compete for the same cache.

A batch-1 policy cannot simply be applied blindly to batch 32.

This is also a major future optimization opportunity.

---

# 33. Immediate Development Plan

## Phase 0 — Feasibility

Already represented by the v0.1 repository.

Test:

```text
baseline
persist
prefetch
combined
```

on a real CUDA GPU.

---

## Phase 1 — SparkInfer Integration

Add a minimal optional hook around the current recurrent-layer execution.

Do not modify model math.

Measure:

```text
Qwen3.8-27B
RTX 5090
batch 1
```

Primary output:

```text
real decode gain
```

---

## Phase 2 — Hardware-Counter Validation

Add profiling scripts for:

```text
Nsight Compute
CUPTI
NVML
```

Measure:

```text
HBM bytes
L2 behavior
energy
kernel time
```

---

## Phase 3 — Policy Search

Replace fixed v0 policies with a configurable planner.

Search:

```text
L2 set-aside size
window size
hit ratio
prefetch distance
prefetch timing
```

The evaluator determines which policies move the real frontier.

---

## Phase 4 — Concurrent Decode

Add multi-request locality planning.

Potential concepts:

```text
per-request cache quota
hot-state rotation
batch partitioning
priority scheduling
cache-aware request grouping
```

---

## Phase 5 — Architecture-Specific Paths

Add specialized implementations for:

```text
Blackwell
future NVIDIA architectures
```

while keeping a generic CUDA path.

---

## Phase 6 — Other Recurrent Architectures

Adapters:

```text
GDN
KDA
Mamba
RWKV
```

---

# 34. Possible Future Technical Directions

These are exploration areas, not required features.

## 34.1 Layer-Aware Prefetch Distance

Instead of:

```text
always prefetch N+1
```

use:

```text
some layers -> N+1
some layers -> N+2
some layers -> no prefetch
```

depending on intervening compute time.

---

## 34.2 Batch-Aware Cache Partitioning

At high concurrency:

```text
L2 budget
   |
   +-- request A
   +-- request B
   +-- request C
```

Choose which active states deserve persisting treatment.

---

## 34.3 State Layout Transformations

Rearrange recurrent state so the access order matches kernel usage.

Possible axes:

```text
head-major
tile-major
warp-major
vectorized
interleaved
```

This would still preserve numerical semantics while improving memory behavior.

---

## 34.4 Kernel-Integrated Prefetch

Instead of a standalone pre-touch kernel:

```text
kernel N
   |
   +-- perform useful work
   +-- cooperatively touch future state
```

This may reduce overhead.

---

## 34.5 CUDA Graph Integration

Attach or configure access-policy behavior consistently inside graph-based decode paths.

This is important because production inference often relies heavily on CUDA Graphs.

---

## 34.6 Weight/State Cache QoS

Future RecurLocal could explicitly coordinate:

```text
streaming model weights
vs
persistent recurrent state
```

to reduce destructive interference.

This is a deeper version of the original thesis.

---

# 35. Recommended Project Rule

The core project rule should be:

> **Do not change model intelligence to win a locality benchmark.**

The exact-locality track should preserve model arithmetic.

If future contributors want to introduce:

```text
state quantization
state compression
approximate caching
```

those should be evaluated under a separate lossy/quality-gated track.

Do not mix them into exact locality without clearly changing the contract.

---

# 36. Why This Project Can Be Valuable Beyond Gittensor

Even if RecurLocal never becomes a Gittensor-supported repo, the technology could still be useful for:

- hybrid LLM serving;
- recurrent state-space models;
- local inference;
- high-concurrency serving;
- GPU memory-hierarchy research;
- engine development;
- kernel research;
- CUDA performance education;
- next-generation recurrent architectures.

The ideal outcome is:

```text
RecurLocal is useful on its own
        +
SparkInfer benefits from it
        +
Gittensor miners can continuously optimize it
```

not:

```text
RecurLocal exists only because rewards exist
```

---

# 37. Success Criteria

The project should be considered successful only if it demonstrates all of the following:

## Technical

- reproducible CUDA implementation;
- no model-math changes in exact mode;
- safe cache-policy control;
- measurable memory-locality change;
- stable integration with a real runtime.

## Performance

At minimum:

```text
>= 4% real end-to-end decode improvement
```

would justify continued serious development.

A larger gain is strongly preferred.

## Product

A release should be able to state:

```text
what changed
why it matters
hardware
model
baseline
new result
correctness
```

in a few sentences.

## Community

Different engineering specialties should have independently valuable optimization opportunities.

---

# 38. Failure Criteria

RecurLocal should be rejected or significantly pivoted if:

- real model speedup remains consistently below ~2%;
- locality gains simply shift bottlenecks elsewhere;
- CUDA access-policy behavior is too unstable across runs;
- prefetch overhead outweighs saved HBM latency;
- SparkInfer already achieves equivalent locality internally;
- future upstream CUDA/runtime changes make the layer redundant;
- a mature project appears with substantially the same technical primitive and better integration.

A project should not be kept alive just because implementation work has already been spent.

---

# 39. Current Repository Commands

## CPU-only

```bash
./scripts/build_cpu.sh
```

Equivalent:

```bash
cmake -S . -B build -DRECURLLOCAL_BUILD_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run information tool:

```bash
./build/recur_local_info
```

---

## CUDA

```bash
./scripts/build_cuda.sh
```

Default architecture:

```text
sm_120
```

Override:

```bash
CMAKE_CUDA_ARCHITECTURES=<arch> ./scripts/build_cuda.sh
```

---

## Synthetic Feasibility Eval

```bash
./scripts/run_feasibility.sh
```

or:

```bash
python3 eval/run_eval.py \
  --binary ./build/recur_local_cuda_bench \
  --output eval-result.json
```

---

# 40. Current Test Coverage

The CPU planner tests verify:

- recommended L2 set-aside calculation;
- combined mode;
- baseline mode;
- prefetch enable/disable;
- hot-window capping;
- invalid hit-ratio rejection.

Future tests should add:

- device capability edge cases;
- zero persisting-L2 support;
- access-window limits;
- concurrent hot-set behavior;
- planner determinism;
- CUDA controller error paths;
- multi-stream correctness;
- repeated-state checksum equivalence.

---

# 41. Security Considerations

Performance PRs can contain arbitrary CUDA code.

Authoritative evaluation should never run untrusted PRs on a persistent machine containing secrets.

Use:

```text
ephemeral GPU runner
no long-lived cloud credentials
no SSH keys
no production secrets
clean workspace
isolated model cache when possible
```

This matters particularly for Gittensor-style open contribution systems.

---

# 42. Suggested Future Repository Additions

Once the initial hardware experiment passes, add:

```text
profiling/
    nsight/
    cupti/
    nvml/

eval/
    real_model/
    concurrency/
    energy/
    regression/

integrations/
    sparkinfer/
        source adapter
        pinned commit
        automated build
        benchmark runner

bench/
    model/
    concurrency/
    layer_microbench/

.github/
    trusted-gpu-eval workflows
```

Do not build this complexity until the core hypothesis produces a meaningful real result.

---

# 43. Suggested Gittensor-Style Evaluation Flow

Future authoritative PR flow:

```text
PR
 |
 v
CPU build/test
 |
 v
CUDA build/test
 |
 v
exact correctness
 |
 v
synthetic locality benchmark
 |
 v
pinned SparkInfer + Qwen3.8
 |
 +---- batch 1
 |
 +---- concurrency 4
 |
 +---- concurrency 16
 |
 +---- concurrency 32
 |
 v
hardware counters
 |
 v
regression check
 |
 v
weighted real performance gain
 |
 v
XS / S / M / L / XL
```

The trusted evaluator should compare:

```text
current main
vs
candidate PR
```

on the same hardware.

---

# 44. Suggested Weighted Performance Score

Once real model benchmarking exists, a project-specific score could use a geometric mean across workloads.

Example:

```text
batch 1 decode       weight 0.40
concurrency 4        weight 0.20
concurrency 16       weight 0.20
concurrency 32       weight 0.20
```

For each:

```text
ratio = candidate_tps / main_tps
```

Then:

```text
weighted_gain =
geometric_mean(ratios)
```

Hard regression guard:

```text
no important workload may regress > 2%
```

unless maintainers explicitly approve a tradeoff.

This avoids rewarding a PR that wins one cherry-picked case and loses everywhere else.

---

# 45. Why Not Score L2 Hit Rate Directly?

Because L2 hit rate is an implementation metric.

The product metric is inference performance.

For example:

```text
PR A

L2 hit:
+50%

decode:
+0.5%
```

versus:

```text
PR B

L2 hit:
+12%

decode:
+8%
```

PR B is much more valuable.

Therefore:

```text
hardware counters = explanation
real model performance = score
```

---

# 46. Potential Release Roadmap

If hardware validation succeeds:

## v0.2

**SparkInfer integration**

Headline goal:

```text
prove measurable batch-1 Qwen3.8 decode gain
```

---

## v0.3

**Adaptive L2 planner**

Headline goal:

```text
automatically choose cache window/hit ratio
```

---

## v0.4

**Batch-aware locality**

Headline goal:

```text
improve concurrent decode
```

---

## v0.5

**Kernel-integrated prefetch**

Headline goal:

```text
reduce standalone pre-touch overhead
```

---

## v0.6

**Architecture-specific Blackwell path**

Headline goal:

```text
push higher real-model improvement
```

---

## v0.7+

Potential:

```text
other GDN models
KDA
Mamba
RWKV
new NVIDIA architectures
```

---

# 47. Project Messaging

Recommended concise description:

> **RecurLocal is a CUDA locality engine for recurrent LLM state. It uses software-directed cache residency and asynchronous layer-ahead prefetch to reduce recurrent-state memory traffic in hybrid models without changing model math.**

Shorter:

> **Keep recurrent AI state close to compute.**

Technical positioning:

> **SparkInfer optimizes the recurrent computation. RecurLocal optimizes where the recurrent state lives while that computation runs.**

---

# 48. Why RecurLocal Is Currently Only A Candidate

This project was intentionally generated as a feasibility repository rather than a full ecosystem because the key unknown is physical:

> **Does the RTX 5090 + Qwen3.8 + SparkInfer workload leave enough exploitable recurrent-state locality for explicit L2 policy to create a meaningful end-to-end improvement?**

No amount of architectural discussion can answer this completely.

It requires hardware measurement.

Therefore the correct order is:

```text
idea
 |
 v
small prototype
 |
 v
RTX 5090 experiment
 |
 +---- weak result -> reject/pivot
 |
 +---- strong result -> build full repo
```

This is a strength of the current project approach, not a weakness.

---

# 49. Recommended Next Experiment

Use a real RTX 5090.

## Step 1

Build and run:

```bash
baseline
persist
prefetch
combined
```

synthetic benchmark.

Confirm:

```text
correctness
stability
cache capability
```

---

## Step 2

Integrate the controller around SparkInfer's recurrent-layer state access.

Keep all model math unchanged.

---

## Step 3

Run deterministic Qwen3.8 decode.

Measure:

```text
baseline tok/s
persist tok/s
prefetch tok/s
combined tok/s
```

---

## Step 4

Profile the winner.

Confirm whether:

```text
HBM state traffic decreased
L2 locality improved
```

---

## Step 5

Run concurrency.

```text
1
4
16
32
```

Determine whether cache contention changes the optimal strategy.

---

## Step 6

Apply the go/no-go rule.

If the real gain is not meaningful, do not expand the project.

If the real gain is strong, then build the full Gittensor evaluation infrastructure.

---

# 50. Final Vision

If the feasibility test succeeds, RecurLocal can become:

> **The open optimization frontier for recurrent-state locality in hybrid AI inference.**

SparkInfer asks:

```text
How fast can we perform the math?
```

RecurLocal asks:

```text
How little expensive memory movement
can that math require?
```

Together:

```text
               RecurLocal
                   |
        state arrives efficiently
                   |
                   v
               SparkInfer
                   |
          computation executes fast
                   |
                   v
                 GPU
```

The long-term goal is not merely a better cache hit rate.

It is measurable AI-serving improvement:

```text
more tokens/sec
more concurrent users
less HBM traffic
less energy/token
same model behavior
```

And, if the hardware confirms the hypothesis, every meaningful release should be able to communicate that progress with a simple benchmark-backed statement.

---

# 51. Current Status Summary

As of this v0.1 feasibility repository:

```text
[implemented] CPU locality planner
[implemented] CUDA persisting-L2 controller
[implemented] asynchronous next-state pre-touch
[implemented] baseline/persist/prefetch/combined modes
[implemented] deterministic synthetic benchmark
[implemented] built-in evaluator
[implemented] CPU tests
[implemented] SparkInfer integration design
[implemented] GitHub CPU CI

[not yet measured] RTX 5090 synthetic results
[not yet implemented] actual SparkInfer source adapter
[not yet measured] Qwen3.8 end-to-end gain
[not yet implemented] Nsight/CUPTI automated profiling
[not yet implemented] trusted Gittensor GPU evaluator
```

The next decision must be based on **real RTX 5090 evidence**.

---

# 52. Repository Principles

1. **Measure everything.**
2. **Never fabricate performance numbers.**
3. **Real model throughput beats microbenchmark vanity metrics.**
4. **Do not change model math in the exact-locality track.**
5. **Reject ideas that do not improve real serving.**
6. **Keep the core integration small.**
7. **Make every optimization independently measurable.**
8. **Allow miners to discover their own work.**
9. **Make releases technically explainable and easy to compare.**
10. **Stay complementary to SparkInfer rather than duplicating it.**

---

# 53. Final One-Sentence Pitch

> **RecurLocal is a CUDA performance layer that keeps mutable recurrent LLM state closer to compute through cache-residency control and layer-ahead prefetch, with the goal of increasing real hybrid-model inference throughput without changing model output.**
