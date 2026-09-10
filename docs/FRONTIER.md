# Frontier model

**Read this first.** The frontier described below is real but it is bounded, and the bound is
now measured rather than argued: weighted across the section 44 matrix, on the best model this
project has found, a persisting L2 window tops out at **1.94% against a 2.0% significance
floor**. That is with both dials at maximum, on a device whose 60 MiB of persisting L2 is the
numerator and cannot be raised. No contributor effort moves it. The surfaces below are worth
working because they are *measurable*, not because they are worth two percent.

RecurLocal is intended for self-directed optimization. Contributors profile current `main`, find a bottleneck, and demonstrate an improvement.

Primary release metrics should be real-model decode throughput, concurrency throughput, recurrent-state HBM traffic, recurrent-kernel time, and energy.

A future release headline may look like this **only after measurement**:

> RecurLocal v0.x: +8.4% Qwen3.8 decode on one RTX 5090, with 3.1× lower recurrent-state DRAM traffic and identical output.

Never publish invented values.

Labels are computed, never asserted: `eval/decide.py` derives the go/no-go band and impact tier
from the measurements, so the verdict does not depend on who is reading the table. A synthetic
result is reported but never tiered.

Secondary metrics such as L2 hit rate explain why a change works but do not replace real end-to-end serving metrics.

## Know the ceiling before you optimize

A recurrent-state locality policy can only recover the share of decode traffic that recurrent
state accounts for. `eval/traffic_budget.py` computes that share from a pinned state geometry
and a measured decode rate, and it should be the first thing run against a new model, a new
device or a new concurrency:

```bash
eval/traffic_budget.py --ms-per-token 10.41 --bandwidth-gbs 1792 --sequences 1
```

On Qwen3.8-27B at batch 1 the answer is **1.69%**, below the 2% floor section 21 rejects at,
before any implementation question is asked. A ceiling below the floor is not a reason to tune
harder; it is the answer.

The ceiling is quoted in throughput, not in traffic share, because that is what the scorer
measures: a step carrying *f* less traffic runs in *(1−f)* of the time, so tok/s rise by
*f/(1−f)*. The two differ by 0.03 points at batch 1 and by 1.4 at 32 sequences.

`--matrix` asks the question one level up — not what one arm's ceiling is, but what a
submission would score if it hit *every* arm's ceiling at once, weighted the way section 44
weights them. That is the most this repository can ever pay, and it is worth knowing before
recruiting anyone to compete for it:

```bash
eval/traffic_budget.py --matrix configs/rtx5090-section44-ceiling.json --bandwidth-gbs 1792
```

The share is not fixed, though, and where it moves is where the frontier is. Model weights are
read once per decode step however many sequences are in flight; recurrent state is read once
per sequence. So the recurrent share grows with concurrency, and a model with a smaller weight
footprint per token — a sparse MoE rather than a dense hybrid — starts from a larger share at
every batch size.

**That has now been tested rather than argued.** `--persisting-l2-bytes` prints the threshold a
candidate model has to clear (`break_even_step_traffic_bytes`: at most 6.42 GB per decode step
on this device), and Qwen3.6-35B-A3B clears it at 3.56 GB where Qwen3.8-27B does not at 18.5 GB.
Measured, the persist-family ceiling goes 0.68% → 3.67% and `persist` goes +0.10% → +1.63%.
Screen a model with `--matrix` before integrating it; the geometry can live in the matrix spec
so a second model's rates cannot be scored against the first one's state shape.

## What the real integration has already overturned

The synthetic benchmark has now disagreed with the pinned real model on four axes. It is kept
because it explains mechanism cheaply, but it decides nothing:

| axis | synthetic says | real model says |
|---|---|---|
| prefetch distance | +21.0% at d=6 | -1.15% at d=8, monotonic, never positive |
| prefetch schedule | doing less costs 16 points | doing less *saves* up to 1 point |
| pre-touch strategy | unresolved inside a 4-5% floor | `ptx_l2` ahead by 0.70 points |
| hot-set policy | a 40-point swing at 16 sequences | 0.01% across all four at batch 1 |

The structural reason is one thing: production decode is captured into a CUDA graph and
replayed, and `workloads/recurrent/synthetic/cuda_bench.cu` is not. Under capture every fork and join the pre-touch
needs is a permanent graph node, and on this model those nodes cost more than the locality
they buy.
