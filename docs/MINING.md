# Competing on this repository

What is scored, why it is scored there, and what does not count.

This document exists because the obvious answers are the wrong ones. RecurLocal's headline
workload has always been batch-1 decode, and **batch-1 decode is a dead surface** — there is
not enough of it to win. Concurrent decode has four to seven times the room, which is where
this document used to point. But the policy family the library ships cannot reach that room
either, for a reason that is arithmetic rather than implementation, and the section below
states it before asking anyone to spend time here.

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

## What batch 1 is for

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

2. **Reuse the cache can actually serve.** Every shipped policy targets reuse across a token,
   which is a full model pass away and 2.4–40x too large to hold. Reuse *within* a layer — the
   conv state feeding the matrix state, or state a kernel touches more than once — is a
   distance the cache can serve. Nobody has measured whether there is anything there. Note the
   conv state is only 1.9–3.8% of the recurrent bytes, so bound it before building it.

3. **The 32-sequence runtime fallback.** Something occasionally drops SparkInfer onto its
   per-row decode path at 32 sequences. Reproduced on a second box: `prefetch` ratios
   `[0.932, 0.676, 0.925]` — one run of three collapsed 32%. It is not the locality policy; it
   has now hit `baseline` (which installs no window and issues no pre-touch) and `prefetch`, on
   different boxes, while `combined` — which pre-touches identically — stayed clean. Cause
   unidentified. This is worth more than it looks: it is a 32% cliff in the runtime, not a
   fraction of a percent in a cache policy.

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
