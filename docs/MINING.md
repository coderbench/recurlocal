# Competing on this repository

What is scored, why it is scored there, and what does not count.

This document exists because the obvious answer is the wrong one. RecurLocal's headline
workload has always been batch-1 decode, and **batch-1 decode is a dead surface** — there is
not enough of it to win. Pointing a competition at it would pay for noise.

---

## What is scored: concurrent decode

A recurrent-state locality policy can only ever recover the share of decode traffic that
recurrent state accounts for. Model weights are read once per decode step however many
sequences are in flight; recurrent state is read once *per sequence*. So the share — and the
whole opportunity — grows with concurrency:

| regime | state share of traffic | ceiling | best measured to date | unclaimed |
|---|--:|--:|--:|--:|
| batch 1 | 1.65% | **1.68%** | +0.13% | — below the 2% floor; nothing to win |
| concurrency 4 | 2.88% | 2.97% | +0.06% | ~2.9 points |
| concurrency 16 | 6.74% | 7.23% | +0.36% | ~6.9 points |
| concurrency 32 | 11.03% | **12.40%** | ~0% (unresolved) | **~12 points** |

Ceilings from `eval/traffic_budget.py` on Qwen3.8-27B NVFP4 / RTX 5090, with the state
bf16-compacted as the runtime does for batched decode. "Best measured" is the strongest arm in
`results/rtx5090-real.json`.

The two left-hand columns differ on purpose, and the gap grows with concurrency. Traffic share
is what fraction of the step's bytes the state accounts for; the ceiling is what removing them
is worth in the throughput the scorer actually measures. A step carrying *f* less traffic runs
in *(1−f)* of the time, so tok/s rise by *f/(1−f)*. Quoting the share as the ceiling understates
it by 1.4 points at 32 sequences — enough that a submission could beat a number this repository
had called a ceiling.

Read that table as the competition brief. **At 32 concurrent sequences there are roughly
eleven points on the table and the shipped implementation captures none of them** — its best
scored configuration measures *negative* there. That gap is the surface.

Run `eval/traffic_budget.py` before optimizing for any new model, device or concurrency. A
ceiling below the 2% floor is an answer, not an invitation to tune harder.

## What batch 1 is for

A regression guard. Do not make it worse; you cannot meaningfully make it better. The
arithmetic:

```
recurrent state per token   48 x (3 MiB fp32 + 60 KiB bf16) x 2  =  294 MiB
decode step                 10.41 ms x 1792 GB/s                 =   18.6 GB
recurrent share                                                      1.65%
throughput ceiling          f/(1-f)                                  1.68%
```

Qwen3.8-27B is a *dense* hybrid: every weight is read every token. Making recurrent state free
would be worth 1.68%, below the floor `eval/decide.py` rejects at, before any policy is chosen.

## Reproducing the measurement

```bash
integrations/sparkinfer/build.sh $WORK        # pinned commit, patched, one binary
eval/real_eval.py --binary  $WORK/sparkinfer/build/runtime/qwen3_gguf_bench \
                  --generate $WORK/sparkinfer/build/runtime/qwen3_gguf_generate \
                  --cb-binary $WORK/sparkinfer/build/runtime/qwen3_gguf_cb_bench \
                  --model $MODEL --concurrency 4,16,32 --repeats 3 \
                  --candidate RECURLOCAL=<your mode> ... --output real-result.json
eval/decide.py --real real-result.json
```

One binary runs both arms: the hook is inert unless `RECURLOCAL` names a mode, so control and
candidate differ only by environment. Never compare two separately linked binaries.

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
already known. The ones with the most room, in order:

1. **Concurrent decode.** The recurrent share quadruples from batch 1 to 16 sequences and the
   current policies capture none of it — `persist` moves the *wrong way* (+0.13% → -0.21%) as
   the room grows, and `prefetch` gets worse in proportion to the bytes it moves. Nobody has
   explained why. That explanation, with hardware counters behind it, is the contribution.
2. **The 32-sequence runtime fallback.** Something in the eval's run sequence occasionally
   drops SparkInfer onto its per-row decode path at 32 sequences, costing 28%. It is not the
   locality policy — it happens in arms that install no window and issue no pre-touch. Cause
   unidentified.
3. **Delivering a persisting window under graph decode.** A locality library cannot attach one
   without the runtime's cooperation; the shortcut that avoids that (`capture_node`) is
   undocumented in CUDA. A safe mechanism here unlocks the whole `persist` family.
4. **A model with less weight traffic per token.** A sparse MoE reads a fraction of its weights
   per token, so the same recurrent state is a far larger share at every batch size. Run
   `traffic_budget.py` on a candidate before integrating it.

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
