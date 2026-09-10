# Design principles

Ten rules, each of which has a cost this project has already paid.

### 1. Do not become another inference engine

TensorTransit owns cross-kernel locality: future-use tracking, reuse modelling, cache priority,
prefetch timing, streaming policy, L2 QoS, cross-stream overlap, plan execution. It does not
own model math, attention or GEMM kernels, KV allocation, recurrent-state allocation, CPU/NVMe
storage, distributed transport, quantization, scheduling, model loading, or tokenization.

The boundary is not a matter of taste. A locality layer that starts allocating is a memory
manager; one that starts writing kernels is a kernel library; either one has stopped being
complementary to the runtime that embeds it.

### 2. Do not own model storage

The registry is **non-owning** and the library never dereferences a tensor pointer on the
host. Everything is address arithmetic. A locality hint that could free somebody's weights is
not a hint.

### 3. Optimize between kernels

Inside-kernel movement belongs to CUTLASS, Triton, FlashInfer, ThunderKittens and the
runtime's own kernels — and they are good at it. This project measured that directly: the
within-layer reuse surface on the pinned runtime is **0.011%**, two orders of magnitude under
the significance floor, because the Gated-DeltaNet kernel already holds each state column in
registers across both passes. Against the naive kernel it replaced, the same bound would have
been 1.66%. The reuse was real and large; somebody else already took it, in registers.

### 4. Use future tensor-use knowledge

The Transit Graph is the only thing here that a cache does not already do for itself. Hardware
replacement is reactive; a plan is not. Everything else in the library is machinery for
turning that one advantage into an action.

### 5. Prefer real-model results over microbenchmarks

The synthetic benchmark said `prefetch +15.9%`. The real model said `-1.29%`, of which 1.20
points were graph-node overhead the synthetic benchmark cannot see because it does not capture
a graph. The synthetic benchmark has now disagreed with the real model on four separate axes.
It is kept for fast iteration and mechanism testing, and `eval/decide.py` **reports it but
never tiers it**.

### 6. Do not change model math in exact mode

Bit-identical greedy replay, or the result is not scorable. When the runtime itself is not
reproducible the harness says **inconclusive** — it does not accuse the candidate. A gate that
blames the contributor for the runtime is worse than no gate.

### 7. Make plans observable and reproducible

A plan is a value: serializable, diffable, digestible, dumpable as text. `tensortransit plan`
prints exactly what a policy would do to a recorded workload, and the declines say what it
refused and why. Golden plan digests turn an unintended policy change into a failing CPU test
rather than a different number in somebody's benchmark a week later.

### 8. Allow self-directed optimization

No maintainer-created issue is required. `docs/OPTIMIZATION-SURFACES.md` maps every surface,
the flag that isolates it, and the current frontier on each — including the ones that are
measured and open, and the ones nothing measures yet.

### 9. Keep runtime integration small

Five calls, no TensorTransit type in the runtime's signatures, no model math, and one binary
that runs the whole matrix from environment variables. That last part is what makes an A/B
honest: one binary, one model load, one machine, the hook inert unless the environment names a
mode.

### 10. Fail fast when an idea does not move the real frontier

The strongest evidence that this rule is real is that the project has applied it to itself.
The recurrent-persist family is bounded at **1.94% weighted against a 2.0% floor** on the best
model found, with both dials at maximum and a bound that already assumes perfect replacement.
That is in the README's first screen, not in a footnote. The generalization to TensorTransit
does not repeal it — it widens the frontier so that the bound applies to one policy family
rather than to the project.

---

## Two rules the specification does not state, which this repository learned

### The instrument is where the bugs are

The thing that decides correctness is the most defect-prone code in the repository, and its
defects are invisible because **a broken evaluator still prints a confident number**. Every
guard in `eval/real_eval.py` and `eval/decide.py` exists because the failure it prevents
actually happened here. Do not remove or relax one without understanding which incident it
encodes; `docs/STABILITY.md` and the CHANGELOG name them.

The same rule now applies to the cost model. A predicted figure is an output of a model whose
assumptions are written down in [evaluation.md](evaluation.md), and it is marked `"basis":
"model"` in every artifact that carries one — because the moment a prediction and a
measurement appear in the same table, somebody will quote whichever is larger.

### An axis whose spread sits inside its own noise is open, not solved

`eval/sweep.py` and `eval/real_sweep.py` exit non-zero rather than name a winner. A mechanism
with no alternative to compare against gives a contributor nothing to do and cannot be
measured, so a new option is a new enumerator plus an implementation — never a file
replacement — and two mechanisms must be A/B-able in one process.
