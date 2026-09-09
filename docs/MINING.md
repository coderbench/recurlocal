# Competing on this repository

What is scored, why it is scored there, and what does not count.

This document exists because the obvious answers are the wrong ones, and they have changed
twice.

On the scored model — Qwen3.8-27B, a **dense** hybrid — batch-1 decode is a dead surface: there
is not enough recurrent traffic in a step to win. Concurrent decode has four to seven times the
room, which is where this document used to point, but the policy family the library ships cannot
reach that room either, for a reason that is arithmetic rather than implementation.

On a **sparse-MoE** hybrid it inverts. A decode step that moves 3.56 GB instead of 18.5 GB puts
the recurrent footprint at 1.02x the persisting cache instead of 2.4x, and batch-1 `persist`
measures **+1.53%** there against +0.10% on the dense model. Concurrency is the arm that cannot
be measured on that checkpoint, because the runtime stops batching above 8 rows.

Both are below the 2% floor as weighted matrices. The difference is that one is bounded out by
arithmetic and the other has 2.1 points of headroom to a ceiling above the floor. Read the
section for the model you intend to work on.

---

## What is scored, and what the most it can pay is

A recurrent-state locality policy can only ever recover the share of decode traffic that
recurrent state accounts for. Model weights are read once per decode step however many
sequences are in flight; recurrent state is read once *per sequence*. So the share — and the
whole opportunity — grows with concurrency:

| regime | control | noise floor | traffic ceiling | best measured | persist-family ceiling |
|---|--:|--:|--:|--:|--:|
| batch 1 | 96.67 tok/s | 0.04% | 1.69% | +0.10% | 0.68% |
| concurrency 4 | 334.17 tok/s | 0.09% | 3.01% | +0.15% | 0.59% |
| concurrency 16 | 790.60 tok/s | 0.13% | 7.44% | +0.06% | 0.35% |
| concurrency 32 | 1287.57 tok/s | 0.34% | **12.71%** | −0.59% | 0.28% |

Every number measured on one RTX 5090, three interleaved pairs per arm, in
`results/rtx5090-baseline-matrix.json`. Ceilings from
`eval/traffic_budget.py --matrix configs/rtx5090-section44-ceiling.json`.

**Weighted across the section 44 matrix, the highest score physically available is 5.22%** — a
submission that removed *all* recurrent-state traffic on every arm. That is impact `S`,
go/no-go "promising". The bands above it — `M`, `L`, `XL` — are unreachable on this model and
this device however good the policy, and no amount of contributor effort changes that. Know it
before you spend a week here.

### The ceiling that actually binds

The traffic ceiling assumes all the recurrent traffic can be removed. A **persisting window
cannot remove traffic it cannot hold**, and that is a far tighter limit.

State written at layer *i* is read again at layer *i* of the **next token**. To save a single
byte, the cache has to keep it across a full pass over the model — so the footprint that must
be resident is *every recurrent layer for every sequence at once*, not one layer's worth.
Against this device's 60 MiB persisting-L2 capacity (of a 96 MiB L2):

| regime | resident footprint needed | vs capacity | persist ceiling |
|---|--:|--:|--:|
| batch 1 | 146.8 MiB | 2.4× | 0.68% |
| concurrency 4 | 299.2 MiB | 5.0× | 0.59% |
| concurrency 16 | 1197 MiB | 19.9× | 0.35% |
| concurrency 32 | 2394 MiB | **39.9×** | 0.28% |

**The persist family's ceiling falls as concurrency rises, while the traffic ceiling rises.**
The room grows and the fraction a persisting cache can address shrinks faster. Weighted across
the matrix the persist family tops out at **0.52%** — below the 2% floor, so it cannot produce
a scorable result at any concurrency, on this model, on this device.

That bound is deliberately generous: it assumes a perfect replacement policy in which every
resident byte hits and the set-aside costs its neighbours nothing. Measurement agrees with it —
`persist` is +0.10% at batch 1 (resolved, against a 0.68% ceiling) and decays from there,
exactly as the arithmetic says it must.

Do not send a persist-family submission expecting it to clear the floor. The room at
concurrency is real, but a persisting L2 window is not the instrument that reaches it.

## What batch 1 is for — on the dense model

A regression guard. Do not make it worse; you cannot meaningfully make it better. The
arithmetic:

```
recurrent state per token   48 x (3 MiB fp32 + 60 KiB bf16) x 2  =  294 MiB
decode step                 10.34 ms x 1792 GB/s                 =   18.5 GB
recurrent share                                                      1.66%
throughput ceiling          f/(1-f)                                  1.69%
```

Qwen3.8-27B is a *dense* hybrid: every weight is read every token. Making recurrent state free
would be worth 1.69%, below the floor `eval/decide.py` rejects at, before any policy is chosen.

**On the MoE checkpoint the same arithmetic says the opposite**, which is why surface 1 below is
the one to read first: 30 recurrent layers x (2 MiB + 48 KiB) x 2 = 123 MiB against a 3.56 GB
step is 3.62%, a 3.75% throughput ceiling, and the footprint fits the cache. Batch 1 is the live
surface there and the only one that can currently be measured.

## Reproducing the measurement

```bash
integrations/sparkinfer/build.sh $WORK        # pinned commit, patched, one binary
eval/real_eval.py --binary  $WORK/sparkinfer/build/runtime/qwen3_gguf_bench \
                  --generate $WORK/sparkinfer/build/runtime/qwen3_gguf_generate \
                  --cb-binary $WORK/sparkinfer/build/runtime/qwen3_gguf_cb_bench \
                  --model $MODEL --contexts 128,4096,16384 --concurrency 4,16,32 --repeats 3 \
                  --candidate RECURLOCAL=<your mode> RECURLOCAL_WINDOW_ATTACH=capture_node \
                  --output real-result.json
eval/decide.py --real real-result.json
```

One binary runs both arms: the hook is inert unless `RECURLOCAL` names a mode, so control and
candidate differ only by environment. Never compare two separately linked binaries.

Two flags there are not decoration. **`--concurrency 4,16,32` and the three contexts fill the
whole section 44 matrix**, and `decide.py` will not call a partial one significant — a missing
arm is renormalised away, not averaged in, so leaving one out changes the score. **The
`capture_node` attach is what makes the persist family apply a policy at all** under graph
decode; with the default the windows are all deferred to a runtime that never attaches them,
and `real_eval.py` correctly refuses the arm as a null candidate. The first attempt at this
repository's own baseline died exactly there.

## What does not count

- **A gain inside its own run-to-run spread.** `real_eval.py` reports a `resolution` block per
  workload and `decide.py` will not call an unresolved matrix significant. Take more repeats.
- **A candidate that applied no policy.** A window that is computed and handed back has not
  reached a kernel. `real_eval.py` refuses a run whose telemetry shows
  `windows_applied + windows_attached_to_node + pre_touch_launches == 0`, by name. This is not
  hypothetical: the first scored run in this repository was exactly that, and its +0.053%
  measured hook overhead.
- **An incomplete workload matrix.** A workload that was not run is not averaged in as a
  zero; it is renormalised away, so omitting an arm removes it from the mean. That makes
  concurrency 32 — the arm with the most room — the one a submission profits most from not
  running. `decide.py` derives coverage from the workloads it actually scored, names what is
  absent and the weight that went with it, and refuses to call a partial matrix significant.
  It says so even when the document claims full coverage.

- **A synthetic gain with no real-model number.** `bench/cuda_bench.cu` has disagreed with the
  real model on four axes, for one structural reason: it does not capture a CUDA graph and
  production decode does. `decide.py` refuses to turn a synthetic result into a verdict.
- **Any change to model output.** The gate is token-exact greedy replay. A faster run that
  changed the output scores nothing.
- **Changing the instrument.** `bench/` and `eval/` define what is measured. The evaluator runs
  them from the base commit, not from the submission — otherwise a submission can win by
  editing the ruler, and a one-line change to a noise floor or an estimator does not look like
  cheating in a diff. `eval/run_from_base.sh <base-ref> <worktree>` enforces it and prints
  what the submission tried to change, rather than silently dropping it:

  ```bash
  eval/run_from_base.sh main /path/to/submission -- --binary ... --candidate RECURLOCAL=...
  ```

  Propose instrument changes separately from the optimization they would score.

## Where the open problems are

`docs/OPTIMIZATION-SURFACES.md` maps every surface with the flag that isolates it and what is
already known. Reordered by what is still genuinely open:

1. **A model with less weight traffic per token.** The only direction that moves both terms the
   right way. A sparse MoE reads a fraction of its weights per token, so the same recurrent
   state is a much larger share of traffic at every batch size — *and* the footprint that has
   to stay resident is unchanged, so the residency bound above does not tighten with it. Run
   `traffic_budget.py --matrix` on a candidate model before integrating it; that is a
   contribution on its own, whichever way it comes out.

2. **Reuse the cache can actually serve — CLOSED, and the answer is no.** Every shipped policy
   targets reuse across a token, which is a full model pass away and 2.4–40x too large to hold.
   Reuse *within* a layer is a distance L2 serves for free, so the only question was whether
   there are any bytes at that distance. Bounded rather than built, from the pinned runtime's
   own kernels:

   - **The matrix state contributes zero.** `gdn_ar_fast_kernel` holds each state column in
     registers across both of its passes — "ONE global read + ONE global write of the 2 MB/layer
     state", in its own comment — and the batched kernel the concurrency path uses is the same
     shape. 98% of the recurrent bytes are touched exactly twice, once each way. There is no
     second touch for a cache policy to catch.
   - **The conv window's shift is the whole of it.** `conv_split_kernel` reads the K−1 window
     entries to convolve and then re-reads K−2 of them to shift the window forward, so the
     reusable bytes are `conv_state × (K−2)/(K−1)` per layer — **1.97 MB per token**, against an
     18.5 GB step.

   `eval/traffic_budget.py` reports it as `within_layer_family`: a **0.011%** ceiling on the
   dense model and **0.028%** on the MoE, two to three orders of magnitude under the floor. The
   bound is generous — each thread reads its own window entries, so most of those re-reads never
   leave a register in the first place.

   Worth knowing what this does *not* say: it is a property of this runtime, not of Gated
   DeltaNet. The naive kernel SparkInfer replaced read the state twice and wrote it twice, so
   against a runtime like that the within-layer reusable bytes would be one extra read plus one
   extra write of the whole matrix state — **288 MiB per token**, 153x the conv-shift figure and
   a 1.66% ceiling, which is essentially the entire traffic ceiling. The reuse was real and it
   was large. Somebody else already took it, in registers, where it belongs.

3. **The runtime falling off its batched decode path — one instance now IDENTIFIED, and it is
   worth 5.4x.** On Qwen3.6-35B-A3B the fallback is not intermittent at all, it is
   deterministic, and the runtime prints its own cause. Measured with the adapter's packing
   counters, one isolated run per width:

   | concurrency | tokens packed | max rows seen | aggregate |
   |---|--:|--:|--:|
   | 4 | 129/138 (93%) | 5 | 907.5 tok/s |
   | 8 | 133/151 (88%) | 9 | **2456.0 tok/s** |
   | 16 | **127/2173 (6%)** | 17 | 452.3 tok/s |
   | 32 | **127/4205 (3%)** | 32 | 456.1 tok/s |

   Above 8 rows the runtime stops batching and decodes one row at a time, so aggregate
   throughput falls *below* the 503 tok/s single-sequence rate. Its stderr says why:

   ```
   [dflash-verify] mmvq_rows refused type=12 N=16 n_out=8192 K=2048
   [dflash-verify] declined at layer=0 (linear_attn=1) N=16
   ```

   `launch_mmvq_q4k_rows` refuses `M > 8` (`kernels/csrc/cuda/gemm/gemv.cu`), and the bf16
   `launch_mmvq_rows` dispatcher has no chunking loop — while its own `_f32` sibling, eight
   lines below, chunks `M` into groups of 8 for exactly this reason. So the first Q4_K
   projection of a wider batch is refused, `dflash_verify_short_run` declines at layer 0, and
   `decode_packed` returns false for the whole batch. `type=12` is Q4_K, and `n_out=8192,
   K=2048` is `attn_qkv.weight` — the linear-attention projection, on every recurrent layer.

   This is SparkInfer's, not RecurLocal's, and it is stated here because this document promised
   it was worth more than anything the library does. It is: 5.4x of aggregate throughput on the
   runtime's own SOTA speed target, against fractions of a percent for a cache policy.
   `integrations/sparkinfer/EXPERIMENTS.md` carries the repro and the measured effect of the
   one-loop fix.

   **The dense model's intermittent 32-sequence collapse is a different observation and is
   still unexplained.** There, `prefetch` ratios came in `[0.932, 0.676, 0.925]` — one run of
   three collapsing 32% — and it hit `baseline`, which installs no window and issues no
   pre-touch. Same family (the runtime declining to batch), no established common cause: this
   one is deterministic and quantisation-shaped, that one is intermittent on an NVFP4
   checkpoint that takes a different projection path. What is new is that it can no longer
   pass unnoticed: `real_eval.py` refuses a concurrency arm whose telemetry shows the packed
   path was not used, and names the counters.

4. **Delivering a persisting window under graph decode.** A locality library cannot attach one
   without the runtime's cooperation; the shortcut that avoids that (`capture_node`) is
   undocumented in CUDA. Worth solving for correctness — every persist measurement here depends
   on it — but note it no longer "unlocks the `persist` family". The residency bound above says
   the family tops out at 0.52% weighted even with perfect delivery.

**Closed, and stated here so nobody re-opens it:** "the current policies capture none of the
concurrency room and nobody has explained why." They cannot. The reuse distance is a full model
pass and the footprint is up to 40x the persisting cache; the ceiling falls as the room grows.
The explanation was the contribution, and it is above.

## Prerequisites the operator must provide

Not code, and not optional:

- **An ephemeral, secretless GPU runner.** Overview section 41 is explicit, and evaluating
  arbitrary CUDA from strangers on a long-lived box with an SSH key violates it. A submission
  can run any kernel it likes.
- **Enough repeats that the floor is real.** The batch-1 noise floor on the reference box is
  0.023% over 3 pairs; concurrency arms are noisier and 2 repeats cannot estimate a floor at
  all. Budget for it.
- **Clock discipline.** Graphics clocks could not be pinned on the reference box, so only
  paired same-box deltas are trustworthy. `real_eval.py` is built around that; do not compare
  across boxes or across days.

## Deliberately not built

Ledgers, dashboards, attestation and copycat detection (overview section 42). None of them is
what stands between this repository and a working competition.
