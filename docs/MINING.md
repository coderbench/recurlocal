# Competing on this repository

> Start with [`VERDICT.md`](VERDICT.md): whether there is a surface here worth your week, in
> numbers, including the parts where the answer is no. This document is the how; that one is
> the whether.

What is scored, why it is scored there, and what does not count.

This document exists because the obvious answers are the wrong ones, and they have changed
twice.

On the scored model — Qwen3.8-27B, a **dense** hybrid — batch-1 decode is a dead surface: there
is not enough recurrent traffic in a step to win. Concurrent decode has four to seven times the
room, which is where this document used to point, but the policy family the library ships cannot
reach that room either, for a reason that is arithmetic rather than implementation.

On a **sparse-MoE** hybrid it inverts. A decode step that moves 3.56 GB instead of 18.5 GB puts
the recurrent footprint at 1.02x the persisting cache instead of 2.4x, and batch-1 `persist`
measures **+1.63%** there against +0.10% on the dense model. Concurrency is the arm that cannot
be measured on that checkpoint, because the runtime stops batching above 8 rows.

As weighted matrices both are small — 0.52% for the dense model against **1.94%** for the MoE,
whose ceiling is 3.7x higher. (Until 0.2.1 those numbers were compared against a 2% scoring
floor and both were under it; there is no floor any more, and what a submission is measured
against is its cells' own noise, published per cell in `frontier/TTF-1/reference.json`.) And the MoE
result is not merely below the floor, it is **not scorable at all**: that checkpoint is not
reproducible against itself, no checkpoint that this runtime can load and that fits 32 GB is,
and the dense control is only reproducible because the gate is 64 tokens long. Read the section for
the model you intend to work on, and read the correctness caveat before either.

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

**Weighted across the section 44 matrix, the highest THROUGHPUT gain physically available is
5.22%** — a submission that removed *all* recurrent-state traffic on every arm. That is the
bound on one of the frontier's two objectives, it is a property of this model and this device,
and no amount of contributor effort changes it. Know it before you spend a week here.

### How a submission is scored

**There are no impact bands any more.** A submission gets one continuous number:

```text
Frontier Gain: dF = F(candidate) / F(main) - 1
```

`F` is the normalized Pareto hypervolume of the serving frontier — goodput against p99
inter-token latency — aggregated over the frozen workload cells of a benchmark generation. The
whole system is [`frontier/README.md`](../frontier/README.md), the scorer is `eval/frontier/`
and the command is `tools/tt-frontier`.

**Why the bands went away, since the answer matters to anyone deciding whether to spend a
week here.** Until 0.2.1 the scorer sorted weighted throughput gain into `XS`/`S`/`M`/`L`/`XL`
with the lowest paying step at 2%. The numbers in the two tables above are why that was
indefensible: the physical ceiling for the whole shipped policy family is **0.52% weighted on
this model** and **1.94% on the best model this project has ever found**. A submission could
remove every recoverable byte of recurrent traffic and score `none`. A band structure whose
lowest paying step sits above what the hardware can deliver is not a strict regime — it is a
broken instrument that tells contributors the room is bigger than it is, and that anything
smaller does not count.

`dF` is continuous, so a real 0.3% expansion is reported as a real 0.3% expansion. What
replaces the bands is *calibration*: `frontier/TTF-1/reference.json` publishes, per cell, the
measured control rate and the measured run-to-run spread, so the size of the prize and the
noise you have to beat are both on the page before you start.

A receipt has to clear four things, in this order:

1. **Correctness.** Token-exact greedy replay, against a control replayed against itself first.
2. **Coverage.** A cell that was not run is a MISSING cell; the receipt says `PARTIAL` on its
   face rather than renormalising the hole away.
3. **Confidence.** A paired bootstrap over interleaved repeats, frozen seed and resample
   count. A 99% lower bound at or below zero is `INCONCLUSIVE`, and the observed figure is not
   published as a contribution.
4. **The protected-workload guard.** A regression past the generation's limit on a protected
   cell is `REGRESSION_GUARD_FAIL` however positive the aggregate is.

Statuses describe evaluation state — `FRONTIER_GAIN`, `NO_FRONTIER_GAIN`, `INCONCLUSIVE`,
`CORRECTNESS_FAIL`, `REGRESSION_GUARD_FAIL`, `BUILD_FAIL`, `EVAL_ERROR` — and none of them
categorises impact magnitude.

**The second objective is new, and the first thing measured about it is that this instrument
cannot see it yet.** The old regime scored throughput and nothing else, so the 0.52% ceiling
above was the whole story. The frontier is two-dimensional, and the tail-latency dimension has a
mechanism the throughput ceiling does not bound in the same way: a resident state removes a
variable-latency HBM round trip from the critical path, which moves a p99 more than it moves an
average.

It has now been measured, and the answer is **not yet**: the control p99 spread on this box is
0.5–3.7% per cell against arm effects of 0.3–2.3%, so at three paired repeats only two of
fifteen latency figures cleared their own noise (`results/rtx5090-0.2.1-arms.json`). That is a
fact about the harness rather than about any policy, and it is what a contributor needs to know
before choosing to work on latency: **more repeats first**. The generation allows up to nine.

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
the matrix the persist family tops out at **0.52%** of throughput on this model and device, at
any concurrency. That is above the batch-1 cells' own noise (0.14%) and below the c32 cells'
(1.7-39%), which is exactly the sort of thing a per-cell noise floor tells you and a single
project-wide floor could not.

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

## The workflow

No maintainer-created issue is required, and there are deliberately no bounty-style
optimization issues to claim.

```text
clone main
  -> tensortransit inspect <trace>         # is there room for the policy you have in mind?
  -> tools/tt-frontier generation show TTF-1
                                           # what is scored, and reference.json says what the
                                           # measured control and the noise floor are per cell
  -> change a planner / admission rule / cost model / executor / adapter
  -> tensortransit compare <trace>         # what your plan does, offline, no GPU
  -> tensortransit replay <plan.json> --trace <trace.json>
                                           # and that an executor would actually fire it
  -> ctest                                 # golden plans, schemas, the compat shim,
                                           # the frontier scorer, the overhead budget
  -> tools/tt-frontier run ...             # paired interleaved A/B over the generation
  -> tools/tt-frontier compute ...         # the Frontier Receipt, computed not typed
  -> open a PR
```

**Everything above the `tt-frontier run` line needs no GPU**, and that is deliberate: a trace
is a file, a plan is a function of that file, two plans can be diffed, and a plan can be
replayed against a recording executor. Write the planner, see exactly what it would do to a
recorded workload, and only then ask for hardware.

Your configuration reaches the measured path through the adapter's environment:

```bash
TENSORTRANSIT=persist                       # the mode
TENSORTRANSIT_WINDOW_ATTACH=capture_node    # without this the window never reaches a captured graph
TENSORTRANSIT_ENGINE=transit                # Registry -> Graph -> Planner -> Executor (default)
TENSORTRANSIT_PLANNER=budgeted              # or baseline | recurrent_v0 | greedy | concurrency
TENSORTRANSIT_ADMISSION=survival            # or density | quota | proportional | reuse_order | role_floor
TENSORTRANSIT_PRESET=global                 # or baseline | recurrent_only | kv_only | naive_both
TENSORTRANSIT_TRACE_OUT=/tmp/live.json      # record the real graph, for offline work
```

`TENSORTRANSIT_ENGINE=v0` runs the 0.1 controller instead, in the same binary. That is the
control: if a result only appears on one engine, it is about the engine.

**One setting is not optional if your planner reads survival.** The adapter registers the
state and the KV it can see — about 350 MB a step — and it cannot know the model's weight
traffic, which on this model is 18.5 GB. Left undeclared, every reuse distance in the recorded
graph is roughly fifty times too short and a residency cost model will think everything
survives:

```bash
TENSORTRANSIT_STREAMED_BYTES_PER_TOKEN=18500000000   # the measured step, batch 1
```

The adapter says so on stderr, once, when a survival-sensitive configuration runs without it.
It does not affect a measurement and it does not affect `recurrent_v0`, whose accounting counts
the resident footprint and ignores streamed bytes.

The first step is the one people skip. `tensortransit inspect` prints the device-bounded
ceiling for the roles a policy is allowed to touch, and if that number is under 2% no planner
in this repository can help you. It costs one command and no hardware.

Evaluation runs on an **ephemeral** GPU worker with a fresh workspace, no SSH keys, no cloud
credentials, read-only model artifacts and a limited token. Performance PRs contain arbitrary
code and the runner is treated as a hostile execution environment.

## Reproducing the measurement

```bash
adapters/sparkinfer/build.sh $WORK        # pinned commit, patched, one binary
eval/real_eval.py --binary  $WORK/sparkinfer/build/runtime/qwen3_gguf_bench \
                  --generate $WORK/sparkinfer/build/runtime/qwen3_gguf_generate \
                  --cb-binary $WORK/sparkinfer/build/runtime/qwen3_gguf_cb_bench \
                  --model $MODEL --contexts 128,4096,16384 --concurrency 4,16,32 --repeats 3 \
                  --candidate TENSORTRANSIT=<your mode> TENSORTRANSIT_WINDOW_ATTACH=capture_node \
                  --output real-result.json
eval/decide.py --real real-result.json
```

One binary runs both arms: the hook is inert unless `TENSORTRANSIT` names a mode, so control
and candidate differ only by environment. Never compare two separately linked binaries.

Every setting is readable under the deprecated `RECURLOCAL_` prefix too, and the commands in
`results/*.json` still reproduce verbatim for that reason. The evaluator scrubs **both**
prefixes from the control environment; a scrub that knew only one would leave an operator with
`export TENSORTRANSIT=combined` comparing the candidate against itself.

Two flags there are not decoration. **`--concurrency 4,16,32` and the three contexts fill the
whole section 44 matrix**, and `decide.py` will not call a partial one significant — a missing
arm is renormalised away, not averaged in, so leaving one out changes the score. **The
`capture_node` attach is what makes the persist family apply a policy at all** under graph
decode; with the default the windows are all deferred to a runtime that never attaches them,
and `real_eval.py` correctly refuses the arm as a null candidate. The first attempt at this
repository's own baseline died exactly there.

## What the statuses mean when your receipt comes back

```text
FRONTIER_GAIN           verified marginal expansion. This is the one that counts.
NO_FRONTIER_GAIN        confidently no expansion. A result, not a failure -- most of this
                        repository's own measurements are here.
INCONCLUSIVE            the observed figure is inside its own confidence interval. Add paired
                        repeats up to the generation's maximum; if it is still inconclusive
                        there, the effect is smaller than this instrument can resolve on this
                        box, and reference.json's per-cell spreads will have told you so.
CORRECTNESS_FAIL        the output changed. A faster run that changed the output scores nothing.
REGRESSION_GUARD_FAIL   a protected workload regressed past the generation's limit. A positive
                        aggregate does not buy that back.
BUILD_FAIL / EVAL_ERROR the candidate did not build, or the evaluator did not finish.
```

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

- **A synthetic gain with no real-model number.** `workloads/recurrent/synthetic/cuda_bench.cu` has disagreed with the
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

1. **A model with less weight traffic per token — TESTED, and it works at batch 1.** The persist
   ceiling is `2 × min(capacity, footprint) / step_traffic`. The capacity is the device's and
   cannot be raised, so the only lever is the denominator, and `eval/traffic_budget.py
   --persisting-l2-bytes` now prints the threshold rather than leaving it implied: **a decode
   step must move at most 6.42 GB** before a persisting window over this footprint reaches the
   2% threshold the break-even is defined against. Qwen3.8-27B moves 18.5 GB.

   `configs/qwen3.6-35b-a3b-moe-ceiling.json` screens Qwen3.6-35B-A3B — same architecture
   family, same runtime, same hook, 256 experts with 8 used per token. It moves **3.56 GB** per
   batch-1 step, and **two terms move, not one**: the step shrinks 5.2× *and* the footprint
   shrinks to 61.4 MiB, because this model has 30 recurrent layers of 2 MiB where the dense one
   has 48 of 3 MiB. 1.02× the cache instead of 2.4×, so 98% of the state can be resident
   instead of 41%.

   | batch 1 | dense Qwen3.8-27B | sparse-MoE Qwen3.6-35B-A3B |
   |---|--:|--:|
   | traffic ceiling | 1.69% | **3.75%** |
   | persist-family ceiling | 0.68% | **3.67%** |
   | measured `persist`, defaults | +0.10% | **+1.26%** |
   | measured `persist`, both dials at maximum | — | **+1.63%** |

   Measured, not projected: 3 interleaved pairs, control 503.2 tok/s, noise floor 0.078%, paired
   ratios 1.0137 / 1.0120 / 1.0126. That is the largest real-model gain in this repository and it
   leaves **2.0 points of headroom** to a ceiling above the floor — which is what makes this a
   surface rather than a result.

   An earlier revision of this list said a sparse MoE would help because "the footprint that has
   to stay resident is unchanged, so the residency bound does not tighten with it". That was
   wrong in the safe direction: the footprint is not unchanged, it is 2.4× smaller, and that is
   where most of the improvement comes from. Screen a candidate model with
   `traffic_budget.py --matrix` before integrating it — the geometry can live in the matrix spec,
   so a second model's rates cannot be scored against the first model's state shape.

   **And the reason it is still not a scorable result.** The exact-locality gate requires
   bit-identical greedy replay, and on this checkpoint that cannot be established: two
   **unhooked** control runs diverge at token 2, because a few ULP in the prefill feed discrete
   top-k expert routing and one flipped expert moves the argmax (the runtime documents this in
   `kernels/include/sparkinfer/kernels/deterministic.h`). `SPARKINFER_DETERMINISTIC=1` does not
   cover its Q4_K expert path. RecurLocal is not the cause — on the dense checkpoint the same
   binary and policy give control, control and candidate bit-identical — but the gate cannot
   answer, so `decide.py` refuses the run.

   **That has now been searched, and there is no reproducible MoE checkpoint to find.** Four
   were screened on hardware and none passes; the runtime accepts only F32, F16, Q8_0, Q4_K,
   Q5_K and Q6_K expert tensors (`qwen35.cpp:114`), which rules out every MXFP4 and IQ variant
   at load, and the two Q8_0/BF16 files that would be genuinely different quantizations exceed
   32 GB of VRAM. Six environment configurations were tried on the checkpoint that does load —
   including `SPARKINFER_PREFILL_BATCHED=0`, which replaces the batched prefill the runtime's
   own header blames, and a pin of every split-K, PDL and n-splits switch at once — and none is
   reproducible. What the pinned set *does* fix is the prefill seed token; a 64-token
   generation still forks in every replay. That does not locate the residual source in decode
   — a ULP difference in prefill that leaves the seed argmax alone while perturbing the
   recurrent state produces the same observation — and separating the two needs a logits dump
   the tools do not emit. The full survey is in the changelog.

   Do not spend a week looking for a checkpoint. The surface is unscorable on this runtime and
   this device, and weighted the persist family reaches 1.94% here, which was under the 2% floor this project used to score against and is now simply a small number with a published noise floor beside it — so
   even a scorable version of this result would land just under. See below.

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

   **The diagnosis is proven, and it needed no patch.** The engine already splits a batch wider
   than its packed-row cap into chunks of that cap, and the cap is an environment variable. Set
   it to the width the GEMV accepts and nothing else changes:

   | | default cap (32) | `SPARKINFER_PACKED_MAX_ROWS=8` |
   |---|--:|--:|
   | c=16 | 5.8% packed, 452.8 tok/s | **94.4% packed, 1204.4 tok/s** |
   | c=32 | 3.0% packed, 455.6 tok/s | **97.3% packed, 1226.8 tok/s** |

   The only variable is the row width handed to the GEMV, and packing goes from 3% to 97%. That
   settles it.

   Two numbers, and they are different questions. **The cliff is 5.4x**: crossing from 8 rows
   (2456 tok/s) to 16 (452) at the default cap. **The workaround recovers 2.7x** and not the
   whole cliff, because chunking re-reads the weights once per chunk — four chunks of 8 at 32
   sequences is four weight passes where one 32-row kernel would be one. So the cap is a
   usable workaround and a wider-row kernel is still the fix.

   This is SparkInfer's, not RecurLocal's, and it is stated here because this document promised
   the surface was worth more than anything the library does. It is: 5.4x of aggregate
   throughput on the runtime's own SOTA speed target, against fractions of a percent for a
   cache policy. `results/rtx5090-moe-matrix.json` carries the counters and the account.

   **It is an omission, not a design choice.** Several multi-row launchers in that file chunk
   `M > 8` into groups of 8 — `launch_gemv_rows2`, `launch_gemv_nvfp4_rows_dp4a`,
   `launch_gemv_nvfp4_rows_dp4a2` and `launch_mmvq_rows_f32`, the last of them seven lines
   below `launch_mmvq_rows` in the same file. `launch_mmvq_rows` does not.
   `docs/UPSTREAM-SPARKINFER-MMVQ.md` is the report, and it establishes the part that decides
   the fix: the 8-row limit is **not** a kernel constraint. `MMAX` is a template parameter
   sizing one accumulator array and one shared-memory buffer, the 8-row instantiation costs 60
   registers with zero spill, and the shared-memory budget does not bind until `MMAX = 64`.

   **A correction, and it undoes a conclusion this document drew.** This section previously
   named `launch_gemv_nvfp4_rows` among the chunkers and concluded from that the dense model
   "never reaches the refusal", so its collapse must be a different bug. `launch_gemv_nvfp4_rows`
   does **not** chunk — `gemv.cu:3518` is `if (M < 2 || M > 8) return false;` with no loop.

   What the dense model actually has is a **third** tier the MoE path does not
   (`qwen35_prefill.cpp:3202-3216`): the dp4a rows kernel, which chunks; then
   `launch_gemv_nvfp4_rows`, which refuses; and then a bare per-row loop that runs the
   projection one row at a time and **returns true**. So the dense model does not decline — it
   silently pays per-row weight traffic inside a forward that is still counted as packed. No
   `[dflash-verify]` line is printed, `tokens_packed` still increments, and `real_eval.py`'s
   packed-path guard cannot see it: that guard catches `decode_packed` refusing the batch,
   which is the MoE's failure and not this one.

   That is a mechanism consistent with every observation about the dense collapse — it is
   intermittent, it hit `baseline` which installs no window at all, it costs about 28%, and it
   is invisible to the counters. It is a **candidate**, not a proof: the first tier is a
   process-static environment switch, so something else would have to make
   `launch_gemv_nvfp4_rows_dp4a` return false on some runs and not others. The decisive
   experiment is cheap and is in the changelog.

4. ~~**Delivering a persisting window under graph decode.**~~ **CLOSED, and the premise was
   wrong.** This repository said `capture_node` was undocumented in CUDA. It is not:
   `cudaStreamGetCaptureInfo`'s own header says "All operations other than destroy and node
   removal are permitted on the graph while the capture sequence is in progress"
   (`cuda_runtime_api.h:2743`, unchanged since CUDA 11.3), and the same paragraph blesses
   passing the driver-owned node array straight to graph APIs. A standalone probe confirms the
   window is present and correct on the finished graph at every node count from 1 to 128,
   memcheck-clean, and a persisting-versus-streaming A/B over the same buffer separates by
   3.2% on replay. See `docs/DESIGN.md`.

   What is left is narrower: `capture_node` marks every kernel node the capture has pending,
   which is not always the one the hook fired for. `capture_node_strict` marks only when there
   is exactly one and counts the rest. Neither changes the arithmetic — the persist family
   still tops out at 0.52% weighted on the dense model with perfect delivery.

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

---

# The TensorTransit surfaces (0.2.0)

Everything above is about **one policy family on one tensor class**, and it is bounded. The
0.2 generalization does not repeal that bound — it widens the frontier so the bound applies to
`recurrent_v0` rather than to the project. These are the surfaces that opened, and what is
known about each.

## Surfaces that need no GPU

These are the ones worth taking first, because they are where the current blocker actually is.

### 1. The cost model — WORKED, and here is what it opened

**The problem is closed and the surface it created is open.** Through 0.2.0 the model was
`saved = reused_bytes x (granted/bytes) x hit_ratio` — linear in the resident share — which
made greedy-on-density *provably* optimal, so no admission rule could beat `density` and the
whole admission axis measured nothing.

`CostModel::Residency` carries the three missing terms and is fitted to every paired hardware
measurement of the `persist` arm in `results/`, across two architectures:

```text
survival(t) = min(1, (resident(t) / reuse_distance_bytes(t)) ^ 0.1108)
saved(t)    = reused_bytes(t) x (resident(t)/bytes) x survival(t)
              -  0.00086 x (resident_total / L2) x step_traffic
```

It beats the linear model on those points (rms 0.271 against 0.485 points; 8 of 8 arms inside
their own noise floor against 6 of 8) and predicts both arms that actually *resolved* to within
a fifth of their noise floor. `eval/cost_model_fit.py` is the fit and it runs in CI.

**What it says, which is a prediction about hardware you can go and test.** `saved` goes as
`resident^(1+beta)` — superlinear — so for a fixed budget spread over `n` tensors the total
goes as `n^(-beta)`. Concentrating beats spreading. The shipped recurrent policy *spreads*:
`recurrent_v0` hardcodes `HotSetPolicy::Proportional`. On the golden KV trace that is 23x worse
than concentrating under this model against 10.5x under the linear one.

Three things are open here and none needs a GPU to start:

- **`stream_relief` is unmeasured.** How much of a Stream-hinted tensor's traffic actually
  stops interfering is a dial with an optimistic default of 1.0. The experiment is newly
  possible: a `Stream` window reaches a captured graph node as of 0.2.1, where before it was
  skipped under capture and therefore unreachable in the only regime that matters.
- **A better functional form.** A power law was chosen because an exponential cannot fit the
  dense and MoE batch-1 arms at once. Two parameters against eight arms is a fit that can fail;
  a form that fits the concurrency arms as well as it fits batch 1 would be worth more than a
  new admission rule.
- **A rule that exploits the convexity.** `AdmissionRule::Survival` stops when the marginal
  admission stops paying. That is the obvious exploitation; it is not the best one — and as
  shipped it is not an exploitation at all. At the fitted `beta = 0.1108` the stopping rule
  never fires and its plan is **byte-identical to `density`'s** on all three golden traces; a
  little under `beta = 0.2` the first candidate stops paying and it admits **nothing**. There is
  no useful middle. `tests/test_golden.cpp` pins both ends, so a rule that actually differs is a
  contribution that will show up as a changed digest rather than as an argument.
- **A per-window term, which the model does not have.** In the five measured arms the gain is
  monotone in `windows_attached_to_node` — 48, 77, 144, 144, 144 — in both the per-cell and the
  aggregated view, and the model prices bytes and residency with no term for how many windows
  hold them. `density` and `reuse_order` are the pair that separates the two explanations: on
  every golden trace they commit the **same** bytes and hold the **same** resident bytes, at 30
  windows against 4, and the model says `density` wins by 1.7x while the window-count reading
  says `reuse_order` does. They disagree in sign. See docs/VERDICT.md section 3.3.

**One of the five arms abstains at concurrency, and you should know before you read a table
that includes it.** `naive_both` is Proportional admission over both tensor classes, and
Proportional answers a budget shortfall by shaving every candidate's hit ratio alike. At four
concurrent requests the concurrency trace offers 512 candidates against a 47 MB budget, every
shaved ratio falls under `min_hit_ratio`, and all 512 are declined: an **empty plan**,
indistinguishable on the device from the hook with no policy. On the recorded batch-1 trace the
same preset emits 256 actions. So the straw man does not demonstrate that unarbitrated
persistence hurts at the concurrencies the frontier scores — it declines to play. An admission
rule that spends a shared budget badly *and still spends it* would be a better straw man, and
building one is a contribution.

### 1b. The model it replaced, kept because it is still the control

`--cost-model linear` is `saved = reused_bytes x (granted / bytes) x hit_ratio`. It is not
deprecated and it is not going away: it is how you check whether a result is about your policy
or about the model. If an improvement only appears under `residency`, it is a claim about
`beta` and `eta`, not about a cache.

A test asserts that `AdmissionRule::Survival` reduces to `density` *exactly* under it, so a
comparison against `density` is not a comparison against a moving target. (Under `residency` at
the fitted beta it reduces to `density` too — see the bullet above; that is a fact about the
fitted parameters rather than about the linear model.)

### 2. A new admission rule — MEASURED, and this is the surface

One enumerator plus an implementation in `planners/budgeted/`. Comparable against every other
rule on the same trace with no hardware, and — as of 0.2.1 — on the real model too, because
`TENSORTRANSIT_ADMISSION` is on the measured path.

**It moves the number.** At sixteen concurrent sequences, control 564.9 tok/s, noise floor
0.283% (`results/rtx5090-0.2.1-arms.json`):

| arm | throughput gain | resolved |
|---|--:|:--:|
| `budgeted` / `density` | **+0.389%** | **yes** |
| `global` preset | **+0.372%** | **yes** |
| `budgeted` / `proportional` | +0.142% | no |
| `budgeted` / `quota` | +0.089% | no |
| `recurrent_v0` (shipped) | +0.071% | no |

Aggregated over c1, c4 and c16 by `tt-frontier`, `density` (+0.056%, 99% CI −0.26…+0.33) and
`quota` (−0.642%, −0.94…−0.39) have **non-overlapping** intervals. One admission rule is
confidently worse than another, end to end, on a real model. That is the whole of what a
competition surface is, and this repository did not have one before.

Read the per-cell table for what it is, though: **no pair of arms separates at that cell.** The
largest gap between two arms there is 0.247 points against a 0.283% floor, so the ordering
inside the table is not evidence. Only `density` and `global` separate from the *control*, and
what orders every arm in both views is the number of windows it attached — not which admission
rule produced them. docs/VERDICT.md section 3.3 has the table and the experiment that settles
it.

**Two things to know before you start.** First, everything below sixteen sequences is negative
for every arm, so a rule that only helps at batch 1 is helping in a regime where the family
costs 0.3–0.4%. Second, and more useful: **the cost model's ranking is contradicted by this
measurement.** It predicts `quota` ≈ `density`; measured, `quota` is the worst of the three.
Reconciling the model with that is worth more than another rule, and it needs no GPU:

```bash
tensortransit compare tests/golden/trace_recurrent_kv.json                  # what it predicts
tensortransit compare tests/golden/trace_recurrent_kv.json --cost-model linear   # the control
python3 eval/cost_model_fit.py                                              # against results/
```

### 3. A new reuse metric or a better graph

`ReuseMetric` has three enumerators and they disagree. Nothing yet uses `Time` for prefetch
placement except `PrefetchTiming::BandwidthAware`, and nothing has measured whether it beats
`FixedDistance`.

(`TransitGraph::live_bytes_at` was O(profiles x edges x uses) per call, with `build()` calling
it once per kernel. Fixed in 0.2.1: `build()` accumulates the whole live-set curve in O(edges)
and the query is a binary search into it.)

### 4. Trace fidelity — the prerequisite is closed, the work is not

`tests/golden/*.json` carry the **measured** recurrent geometry of Qwen3.8-27B and a
**synthetic** KV block size, with the weight traffic divided evenly across layers. Every
offline comparison is therefore sharp about the recurrent half and approximate about
everything else.

The adapter can now record a trace from the LIVE runtime — `TENSORTRANSIT_TRACE_OUT=<path>`
writes the graph once, after the first compile, with real KV slice sizes and real per-layer
demand. Recording one on each of the generation's cells and replacing the hand-written traces
is a contribution that needs one GPU run and then no hardware at all.

## Surfaces that need a GPU

### 5. The second proof track, measured

Run the five arms on hardware and settle whether coordination beats independent policies.
**The prerequisite is closed**: the adapter registers KV as of 0.2.1, so the arms are
`TENSORTRANSIT_PRESET=baseline|recurrent_only|kv_only|naive_both|global` on the real model
through the measured path, and `tools/tt-frontier run` takes them as a portfolio.

**Known before starting:** on the pinned dense model this contest is for less than a point of
throughput, and the interesting regime is a model whose decode step moves under **6.42 GB**.
What is *not* known is what any of it does to a p99 tail, which is the frontier's other
objective and has never been measured.

### 6. `WindowBinding::Sticky`

Bind once for the whole step instead of once per consumer. Only one window can be bound to a
stream at a time, so the planner gives it to the single densest admitted candidate and
everything else falls back to per-consumer. Implemented, plan-validated, **never measured**.

### 7. Concurrency arbitration

`ConcurrencyPlanner` implements even-share and concentrate. The measured fact to beat:
`persist` pays at 98% residency and is negative from four sequences on, where residency is
48%. **The crossover is at about half residency.** Concentrating the budget on fewer requests
is the obvious idea and nobody has run it.

### 8. `Stream`, which was dead code until 0.2.1

Telling the weight stream to get out of the way was unreachable under graph decode for two
independent reasons, both now fixed: `emit_stream_hints` required a tensor with *no reuse*, and
over a cyclic decode window nothing has none; and the CUDA executor *skipped* a Stream action
under capture, so even an emitted one was absent from every replay.

Both are fixed, so the arm that is supposed to manage the other half of a shared cache budget
can finally do something. Nobody has measured what. `stream_relief` — how much of a hinted
tensor's traffic actually stops interfering — is a dial with an optimistic default and no
measurement behind it.

### 9. Prefetch as a graph node

Window attachment to captured graph nodes works and is verified on hardware
(`tests/test_cuda_executor.cu` reads the attribute back off the finished graph). Prefetch is
still a stream fork, and a fork plus a join is a permanent graph node pair costing ~0.027% of
a decode step each — 1.20 points of the 1.29 that `prefetch` lost on the real model was that
ordering, not the memory.

## What still does not count

Unchanged from 0.1, and now with a second clause: a submission that improves a **predicted**
figure has improved a model, not a runtime. Say so in the PR. `tensortransit plan` output is
evidence about a planner; only `eval/decide.py --real` is evidence about a speedup.
