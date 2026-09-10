# Is there a scorable surface here, and if so which one

This document answers the question a maintainer has to answer before handing a repository to
contributors: **is there something here worth winning?** It is written from measurements, it
gives a number for each part of the answer, and where the answer is no it says so.

Short version:

> **The surface is real and it is large. The mechanism this project ships cannot reach it, and
> on half the matrix it provably never could.**
>
> Per cell, a persisting-L2 policy is bounded by `2 x persisting-L2 / step traffic`, and the
> numerator is 60 MiB of hardware. Against each cell's own measurable floor, **seven of TTF-1's
> ten cells cannot be measured to be won by that family** — the floor there is larger than a
> *perfect* policy's ceiling. `ctx128-c16` is one of them: ceiling **0.249%**, published spread
> **0.424%**, and it is the cell at which every headline figure in this repository was measured.
> On the three cells that remain, the shipped policy measures **−0.289%, −0.179% and −0.361%**.
>
> Removing *all* recurrent traffic is worth **5.2%** at sixteen sequences and **8.6%** at
> thirty-two against those same spreads. The room at concurrency is one to two orders of
> magnitude above the noise. It is simply not addressable by a 60 MiB carve-out.
>
> `tools/tt-frontier generation show TTF-1 --reachable` prints the whole table, computed from
> the pinned geometry and each cell's own measured control. Sections 3, 8 and 9 give each claim
> a number and name the measurement it came from.

---

## 1. What was closed before this release

The recurrent-persist family is bounded by arithmetic, not by implementation:

```text
ceiling = 2 x min(persisting capacity, footprint) / decode step traffic
```

The numerator is pinned at 60 MiB by the hardware. Weighted across the workload matrix the
family tops out at **0.52%** of throughput on the scored dense model and **1.94%** on the best
model ever found — and that better model is **not scorable**, because two *unhooked* control
replays of it diverge at token 2, so the exactness gate cannot attribute any difference to a
candidate. Four checkpoints were screened; none passes.

That is settled and no contributor effort moves it. What 0.2.1 changed is everything about
whether anything *else* here is worth moving.

## 2. The blocker that made the question unanswerable

Through 0.2.0 the adapter and the synthetic benchmark drove the 0.1 controller directly. The
whole 0.2 core — registry, graph, planners, admission rules, executor — was reachable only from
the CLI and the tests. **A contributor who wrote a planner changed nothing about the number the
evaluator prints.** The competition surface was not small; it was disconnected.

It is connected now, and the connection was checked rather than asserted: both engines in one
binary, one model load, one box, the only variable being `TENSORTRANSIT_ENGINE`
(`results/rtx5090-0.2.1-rewiring-check.json`).

## 3. What the measurement says

Five configurations against a scrubbed control, paired and interleaved, three repeats, on the
three TTF-1 cells whose control spread resolves (`results/rtx5090-0.2.1-arms.json`):

**Per cell, throughput, at 16 concurrent sequences** (control 564.9 tok/s, noise floor 0.283%):

| arm | gain | resolved |
|---|--:|:--:|
| `budgeted` / `density` | **+0.389%** | **yes** |
| `global` preset (`role_floor`) | **+0.372%** | **yes** |
| `budgeted` / `proportional` | +0.142% | no |
| `budgeted` / `quota` | +0.089% | no |
| `recurrent_v0` (the shipped policy) | +0.071% | no |

**Aggregated by the frontier scorer over c1, c4 and c16** — a partial matrix, so every receipt
says `PARTIAL`:

| arm | dF | 99% CI | status |
|---|--:|:--:|---|
| `density` | +0.056% | −0.26 … +0.33% | INCONCLUSIVE |
| `global` | −0.157% | −0.37 … +0.26% | INCONCLUSIVE |
| `recurrent_v0` | −0.429% | −1.50 … +0.32% | INCONCLUSIVE |
| `proportional` | −0.485% | −0.80 … −0.19% | NO_FRONTIER_GAIN |
| `quota` | −0.642% | −0.94 … −0.39% | NO_FRONTIER_GAIN |

Four things follow, and only the first is good news.

### 3.1 The admission axis separates two policies from each other — and no arm from the control

Aggregated over c1, c4 and c16, `density` (+0.056%, 99% CI −0.26…+0.33) and `quota` (−0.642%,
CI −0.94…−0.39) have **non-overlapping** intervals. One admission rule is confidently worse
than another, on a real model, end to end. That is the first time a change a contributor can
make has been measured to move anything at all, and it could not have been produced before
0.2.1 — the adapter drove the 0.1 controller, so no admission rule was in the measured path.

It is also the *ceiling* on what that axis can be worth. Section 7 shows the arithmetic: on
seven of ten cells the whole persist family is bounded below the floor, so no admission rule can
be measured to win them. Rearranging a budget worth less than the noise is a real difference
between two policies and not a route to a Frontier Gain.

**The per-cell headline does not survive the generation's own calibration, and this corrects an
earlier version of this document.** The table above reports `density` at +0.389% at ctx128-c16
against that *run's* floor of 0.283%. `frontier/TTF-1/reference.json` froze that cell's control
spread at **0.42%** when the generation was calibrated, and 0.389 < 0.42. The run-specific floor
was smaller only because that session happened to be quieter, and the rule this repository
applies everywhere else — an axis whose effect sits inside its own noise is open, not solved —
applies to it too.

The same arm measured twice, two hours apart on the same box, makes the point without any
statistics: `persist` (which is `recurrent_v0`, the shipped default) came back **+0.071%** in
the arms sweep and **−0.388%** in the full TTF-1 matrix at that cell. The gap between those two
is 0.46%, and the cell's published spread is 0.42%.

So the honest statement of what is measured is narrower than the one this section used to make:
two admission rules differ from each other by more than the noise; **no arm has been shown to
beat the control by more than the noise of the cell it was measured in.**

### 3.2 Below sixteen sequences the whole family is negative

Every arm is negative at one and four concurrent sequences: −0.29% to −0.43% and −0.27% to
−0.54%, most of them resolved. This is not in tension with the +0.145% the migration check
measures at batch 1 — that is the single-sequence bench, while these cells run the
continuous-batching engine with the generation's long-prefill injection alongside. Two serving
mixes, two answers, and the frontier scores the one that looks like serving.

### 3.3 The cost model ranks admission rules on an axis the measurement does not separate — and the one it does separate is an axis the model has no term for

The residency cost model fits the *magnitude* of the persist family better than the linear one
it replaced — rms 0.271 against 0.485 points, with both resolved arms predicted to within a
fifth of their own noise floor. Its **ranking** is a different question, and the answer is
sharper than "contradicted".

It predicts `quota` ≈ `density`, both far ahead of `proportional`, because saving goes as
`resident^(1+beta)` and `quota` keeps tensors whole. The measured point estimates run
`density` > `proportional` > `quota`. But **no pair of arms separates at that cell**: the
largest gap between two arms is 0.247 points against a 0.283% floor, so the per-cell table
above orders arms it cannot distinguish, and reading a ranking off it is reading noise. What
*does* separate is the aggregate, where `quota` (−0.642%, CI −0.94…−0.39) and `density`
(+0.056%, CI −0.26…+0.33) do not overlap.

The axis that orders every arm is in the telemetry rather than in the model:

| arm | `windows_attached_to_node` | at ctx128-c16 | aggregated |
|---|--:|--:|--:|
| `density` | **48** | +0.389% | +0.056% |
| `global` | **77** | +0.372% | −0.157% |
| `recurrent_v0` | 144 | +0.071% | −0.429% |
| `proportional` | 144 | +0.142% | −0.485% |
| `quota` | 144 | +0.089% | −0.642% |

Both views order the arms the same way the counter does. The two that resolve against the
control are the two with the fewest windows; the three that are mutually indistinguishable are
the three that attach the same number. The residency model prices bytes, whole-line residency,
survival against interference, and what the reservation costs the traffic it displaces. It has
**no per-window term**, so it cannot express this and did not predict it.

That is correlational, and the five arms are confounded. **The experiment that was supposed to
separate them did not**, and the reason is worth more than the experiment: `density` and
`reuse_order` emit 29 and 15 persist actions on the recorded trace, over the *same 15 kernels*,
and the device carries one access-policy window per kernel node — so both reported
`windows_attached_to_node: 48` and nothing varied
(`results/rtx5090-0.2.1-null-control.json`).

**A better explanation turned up while re-recording the traces, and it is mechanistic rather
than correlational.** Two graphs recorded from the live adapter, at one sequence and at sixteen:

| | c=1 | c=16 |
|---|--:|--:|
| recurrent state declared | 153.9 MB | **78.4 MB** |
| KV declared | 134.7 MB | **4.7 MB** |
| `step_traffic_bytes` | 18.94 GB | **18.66 GB** |
| computed ceiling | 1.00669 | **1.00679** |

Sixteen sequences move *sixteen times* the recurrent state. The graph declares one sequence's
slices at both concurrencies, so **the planner cannot tell the two workloads apart**. Three
causes, none of them in the core: `TensorDesc::request_local` — "footprint scales with
concurrency" — is set by every adapter and **read by nothing**; the hook hardcodes `rows = 1`
for KV; and `TENSORTRANSIT_STREAMED_BYTES_PER_TOKEN` is an operator-declared batch-1 constant.
Only `recurrent_v0` reads `RuntimeState::active_requests`, and every arm of the admission axis
uses `budgeted`, which does not.

That is enough to explain the whole disagreement. A window covers one address range — row 0's
slice — so at sixteen sequences it saves a *sixteenth* of the state traffic while *sixteen*
sequences' worth flows past between two uses of it. The benefit divides by concurrency and the
interference multiplies by it, and the survival term is exactly the ratio of the two. With
concurrency out of both halves the model says the family helps at concurrency; the measurement
says it does not; and `AdmissionRule::Survival` never fires because, counted for one sequence,
everything survives.

**Reproducible with no GPU.** `tests/golden/trace_live_c1.json` and `trace_live_c16.json` are
both committed, `tensortransit plan` on each is the whole experiment, and
`results/rtx5090-0.2.1-concurrency-blind-graph.json` has the numbers. It is not fixed here on
purpose: scaling a request-local tensor's footprint and reuse distance by `active_requests`
moves every digest in `tests/golden/plans.golden`, and a change that large belongs in a release
that can measure its effect rather than in the one that found it. The golden tests pin the
current behaviour on both traces, so the fix will show up as a digest change nobody can miss.

### 3.4 The tail-latency objective cannot be resolved yet, and nine repeats do not rescue it

The frontier scores goodput *and* p99 inter-token latency, and the second is where a locality
policy has a mechanism the throughput ceiling does not bound — a resident line removes a
variable-latency round trip from the critical path. It has now been measured, and the answer is
**not yet**: control p99 spreads are 0.5–3.7% per cell against arm effects of 0.3–2.3%, so at
three paired repeats only two of fifteen latency figures cleared their own noise.

That is a fact about the harness, not about any policy. The generation allows up to nine
repeats, and they have now been spent on the worst cell.

**`ctx128-c32`, nine paired repeats** (`results/rtx5090-0.2.1-c32-latency.json`). The paired p99
ratio is worse for the candidate in **seven of nine** pairs, median **1.083** — a sign test puts
that at p ≈ 0.18. The maximum-gap ratio shows nothing at all: median 0.969, worse in four of
nine. The control's own nine p99 values span 45.1 to 240.1 ms, which is 432% and confirms the
calibration's 481%.

The mechanism is visible in the pairs and is worth knowing before anyone tries again: every run
carries one ~673 ms stall — the generation's long-prefill request landing on a decode step — and
with 2016 timed gaps per run that single stall sits at the maximum and cannot move a 99th
percentile. In three of nine candidate runs and two of nine control runs it is **absent**, and
p99 then climbs to whatever the next tail is: 72, 116, 157 ms. **p99 at this cell is bimodal on
whether one stall lands in the window.** More repeats sharpen the estimate of a bimodal
distribution; they do not make it measure a policy.

What the same nine repeats *do* resolve is goodput: **−1.20% median, nine of nine pairs
negative**, a paired sign test at p = 0.004. (Peak-to-peak against that cell's published 1.738%
spread does not clear; the two rules disagree and both are reported.)

## 4. What the second proof track says, and what it does not

At model level the global arm now beats the best independent arm on all three golden traces,
where under the linear model it provably could not — and setting `--stream-relief 0` makes the
advantage disappear on every one, which identifies the mechanism as the `Stream` action
(`results/rtx5090-second-proof-track-model.json`).

**It also holds on geometry nobody wrote by hand.** `tests/golden/trace_live_c1.json` is
recorded from the live adapter — the runtime's real KV slice sizes, its real layer count, the
measured step traffic — and gives the same three-way answer: linear −0.022 points behind the
best independent arm, residency **+0.054 ahead**, residency with `--stream-relief 0` −0.008
behind again. The synthetic fixtures model 176 kernels against the runtime's 64 and a KV block
1.5x too large; the conclusion does not depend on either.

**That was not tested by the measurement above.** `stream_applied` and `stream_deferred` are
zero on every measured arm, because the adapter registers a `ModelWeight` tensor only when the
operator declares `TENSORTRANSIT_STREAMED_BYTES_PER_TOKEN`, and this run did not. The measured
`global` differs from `density` by role-floor arbitration alone — and measures the same as it
within noise.

So the coordination claim has a model-level answer with an identified mechanism and a control
that would falsify it, and the experiment that settles it is one environment variable away.

## 5. What the scoring regime had to become, and why

Until 0.2.1 a submission was sorted into `XS`/`S`/`M`/`L`/`XL` by weighted throughput gain,
lowest paying step 2%. Section 1's ceiling is **0.52%**. A contributor could remove every
recoverable byte of recurrent traffic and score `none`; the +0.389% measured above would have
scored `none`; the 0.32-point separation between two admission rules would have scored `none`
twice.

A band structure whose lowest step sits above what the hardware can deliver is not a strict
regime — it is a broken instrument, and what it tells contributors is false. It is replaced by
a continuous **Frontier Gain** with per-cell bounds and per-cell noise floors calibrated on the
target hardware, so that the size of the prize and the noise to beat are both on the page
before anyone starts. `frontier/README.md`.

Two cells of the intended 4x3 matrix are not in the generation at all, because calibration
found this device cannot run them: `ctx16384-c16` loses 30 of its requests to a device OOM in
every repeat, and `ctx16384-c32` cannot load the model. The c32 cells are scored but **not
protected by the regression guard**, because `ctx4096-c32`'s own control spread is 39% — a 5%
guard against a 39% arm rejects good submissions at random.

## 6. The honest comparison

The largest number this project has measured anywhere near itself is not its own. Above eight
rows the pinned runtime stops batching and decodes one row at a time, costing **5.4x** of
aggregate throughput; the cause is a missing chunking loop in one GEMV dispatcher, the
diagnosis needed no patch, and `docs/UPSTREAM-SPARKINFER-MMVQ.md` is the report. That is
SparkInfer's to fix and it is worth more than everything in this repository put together.

Saying so is the point. A locality planner that pretended otherwise would be competing for a
contributor's week under false pretences.

## 7. What is reachable, per cell, before anyone measures anything

The ceiling for a persisting-L2 policy is arithmetic, and section 1 gives it weighted across
the matrix. Asked of each cell separately — from the pinned state geometry and that cell's own
calibrated control rate — it says something section 1's single number cannot:

```text
    cell               persist  any mech.    bw    spread    floor   verdict
    ctx128-c1           0.491%     1.210%   71%    0.144%   0.144%   measurable by the persist family
    ctx4096-c1          0.395%     0.971%   57%    0.179%   0.179%   measurable by the persist family
    ctx16384-c1         0.159%     0.390%   23%    0.000%   0.221%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx128-c4           0.391%     1.983%   57%    0.090%   0.090%   measurable by the persist family
    ctx4096-c4          0.113%     0.564%   17%    0.156%   0.156%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx16384-c4         0.052%     0.262%    8%    0.000%   0.168%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx128-c16          0.249%     5.217%   40%    0.424%   0.424%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx4096-c16         0.065%     1.304%   10%    0.204%   0.204%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx128-c32          0.200%     8.649%   36%    1.738%   1.738%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx4096-c32         0.040%     1.615%    7%   39.008%  39.008%   PERSIST FAMILY UNWINNABLE; and so is any
```

`tools/tt-frontier generation show TTF-1 --reachable`. `persist` is `2 x persisting-L2 / step
traffic` — what a policy that held every byte it could and lost nothing would be worth.
`any mech.` is what removing *all* recurrent traffic would be worth. `spread` is how far the
control moved between repeats of itself when the generation was calibrated, published in
`reference.json` since it was frozen.

These are smaller than section 1's **0.52%**, and the difference is the workload rather than a
disagreement. Section 1 weights the section 44 matrix, whose rates come from the single-sequence
and packed benches at their own step times — 10.34 ms at batch 1. TTF-1's cells run the
continuous-batching bench with a long-prefill request injected alongside, so the same batch-1
cell takes 14.37 ms and its step therefore *carries more traffic*. A ceiling that is a share of
the step falls when the step grows. Both numbers are right about their own workload, and the one
that matters for a submission is the one computed from the cells it will be scored in.

`floor` is the larger of the published spread and what the bench can *resolve*: aggregate
throughput is printed to one decimal, so half of that last digit is the smallest difference
visible at all — 0.22% of a 22.6 tok/s cell, 0.006% of a 909 tok/s one. Two cells report a
0.000% spread because three repeats printed the same number, and their real floor is the
resolution.

**That is arithmetic, and it has now been confirmed by a null control.** `survival` and
`density` compile to the *same plan* — identical digests on every golden trace, asserted by
`tests/test_golden.cpp` — so the paired difference between them is noise with the policy held
constant. Over five interleaved repeats it is a median of **+0.230%** at `ctx128-c16`, with a
peak-to-peak of **1.29%**, against that cell's 0.249% ceiling
(`results/rtx5090-0.2.1-null-control.json`). Two arms running the same bytes differ by more than
a perfect policy could ever deliver there.

**Where the floor exceeds the persist ceiling, nothing this project ships can be measured to
win.** Not a better admission rule, not a better window shape, not a better hot-set heuristic:
the room is smaller than the noise, and that is arithmetic rather than an implementation
problem. It is true of **seven of the ten cells**, including every cell above four sequences.

The three that remain are `ctx128-c1`, `ctx4096-c1` and `ctx128-c4` — and on all three, the full
matrix in section 8 measures the shipped policy at **−0.289%, −0.179% and −0.361%**, each
resolved against that cell's own published spread. So on the only cells where this family can be
measured at all, it measurably loses.

**`ctx128-c16` is on that list, and it is the cell this repository has quoted from all
along** — section 3's table, `docs/MINING.md`'s table, the 0.2.1 release note. Its persist
ceiling is 0.249% and the arms sweep reported `density` at +0.389%. A measurement above its own
physical ceiling is not a small result; it is noise, and the cell's published spread of 0.424%
says the same thing twice.

**And the opposite half of the finding matters more.** At sixteen sequences 5.2% of the step is
recurrent traffic and at thirty-two it is 8.6%, against spreads of 0.42% and 1.74%. That room
is five to twelve times the noise and it is exactly where a serving frontier is scored. It is
unreachable by a 60 MiB carve-out for a reason that has nothing to do with policy quality: the
per-token recurrent footprint at sixteen sequences is 2.46 GB and the device's persisting
partition is 0.06 GB.

**What could address it, and what each costs.** This list is short on purpose — a vague "another
mechanism has room" would be a phantom, and the point of this document is not to manufacture
one:

| lever | what it changes | why it is not in this repository |
|---|---|---|
| more persisting cache | the numerator of `2 x L2 / step` | hardware. 60 MiB is what the device reports and no software raises it. |
| fewer live sequences | the footprint, linearly | that is the workload, not the policy, and the frontier scores the workload as given. |
| a smaller state representation | the footprint, by the compression ratio | **changes model output**, so it fails the exact-locality gate. The runtime already does the free half of it — bf16 on the packed path, which is why the concurrency footprint is 2.46 GB and not 4.9 GB. Going further is a tradeoff track and has to be argued for as one. |
| recomputing state instead of reading it | trades bandwidth for arithmetic | plausible, exact, and untouched here. It needs a kernel, not a planner, and it is a change to SparkInfer rather than to TensorTransit. |

Two of those four are not software, one is not exact, and the fourth is somebody else's
codebase. **That is the honest shape of the opportunity**, and it is why this document's answer
is "no" for the mechanism this project ships rather than "not yet". A contributor who wants the
5–9% should know before starting that no admission rule reaches it.

One caveat, in the tool's own words: the `bw` column is how much of peak bandwidth each step
actually used, and **every TTF-1 cell is below the 80% at which a traffic ceiling is tight**. So
`any mech.` is a loose upper bound — part of those steps is latency and occupancy rather than
bytes. The `persist` column does not depend on it.

**And one thing this whole section does not bound.** Every number above is about throughput. A
traffic budget says nothing about a *tail*, so there is no computed ceiling for `p99_itl_ms` at
all, and the only published constraint on it is the spread column — 0.94% at `ctx128-c1`, 6.7%
at `ctx128-c16`, 481% at `ctx128-c32`. TTF-1 scores that objective equally with goodput. It is
the half of the frontier that is **not** closed by arithmetic, and a policy that moved a tail by
more than a cell's spread would score whatever the traffic budget says.

That is an opening rather than a result. Measured, the shipped policy moves it the wrong way:
+1.246% at `ctx128-c1` against a 0.94% spread — resolved, and worse. Section 8 has the rest.

## 8. The first full TTF-1 matrix, and what it says about the matrix

Every earlier measurement here used three cells. This is the whole generation: ten cells,
`control` against `persist`, three paired repeats, interleaved, token-exact with three
reproducible control replays.

**Six of ten cells produced a paired operating point. Four did not**, and they are exactly the
long-context concurrency cells:

| cell | control tok/s | control's scaling vs its own c=1 | candidate |
|---|--:|--:|---|
| `ctx128-c4` | 221.7 | 3.20x for 4 sequences | measured |
| `ctx128-c16` | 566.4 | 8.17x for 16 | measured |
| `ctx128-c32` | 904.3 | 13.05x for 32 | measured |
| `ctx4096-c4` | 63.9 | **1.15x for 4** | fell off the batched path |
| `ctx4096-c16` | 145.9 | **2.61x for 16** | fell off the batched path |
| `ctx4096-c32` | 180.5 | **3.23x for 32** | fell off the batched path |
| `ctx16384-c4` | 29.7 | **1.32x for 4** | fell off the batched path |

The guard that refused those four reads the *adapter's* packing counters, and the control is
unhooked and emits none — so only the candidate could be seen to fail. Each was re-run with the
hook installed and **no window**, which emits the same counters and applies no policy, and the
four are not one thing:

| cell | baseline, no policy | `persist` | whose |
|---|---|---|---|
| `ctx4096-c4` | `max_rows_seen 0`, 0 decode steps packed | identical | **the runtime's** |
| `ctx16384-c4` | `max_rows_seen 0`, 0 packed, also at 4x the decode length | identical | **the runtime's** |
| `ctx4096-c16` | `max_rows_seen 16`, **63 of 64** decode steps packed | identical | **the evaluator's** |
| `ctx4096-c32` | `max_rows_seen 32`, **63 of 64** decode steps packed | identical | **the evaluator's** |

The last two batched every decode step at full width. They were refused because the guard
divided by the adapter's `tokens`, which counts the ~70 prefill chunks a 4096-token prompt
produces — 63 of 133 is 47.4% against a 50% threshold. The guard now counts decode steps
against `max_new` and checks `max_rows_seen` first.

So **two** of TTF-1's ten cells cannot be measured as concurrency cells on this runtime, and the
control's own arithmetic says the same: a cell that returns 1.15x for four concurrent sequences
was not serving four concurrent sequences in either arm. The other two are measurable and were
lost to a defect in the ruler.

**On the six cells that did batch, the shipped default policy is a regression**, and five of
the six are negative:

| cell | goodput | resolves against the cell's published spread |
|---|--:|:--:|
| `ctx128-c1` | −0.289% | yes (0.14%) |
| `ctx128-c4` | −0.361% | yes (0.09%) |
| `ctx4096-c1` | −0.179% | yes (0.18%) |
| `ctx128-c16` | −0.388% | no (0.42%) |
| `ctx128-c32` | −1.128% | no (1.74%) |
| `ctx16384-c1` | ±0.000% | no (0.00%) |

**And the receipt read −99.5%**, which is the instrument rather than the submission. Two
defects produced it, both now fixed and both now named on the receipt's own face:

1. Four cells were charged to the candidate at the generation's cell floor: two the runtime
   cannot batch, and two the guard mis-refused by dividing decode steps by a count that
   includes prefill chunks. Both halves are fixed — the denominator is the decode length the
   harness asked for, and a serving loss is now a question before it is a verdict:
   `tt-frontier run` re-runs such a cell on the **baseline** binary with the hook installed and
   no window, and a failure that reproduces there is the runtime's.
2. `ctx128-c32` was driven to the floor by a p99 change of 183% against a control spread that
   `reference.json` **froze at 481%** when the generation was calibrated. Nine repeats of that
   cell, run afterwards, confirm the calibration and retract the reading: the paired p99 ratio
   is worse for the candidate in seven of nine pairs, median 1.083, sign test p ≈ 0.18, and the
   control's own nine values span 432%. Every receipt now reports, per cell and per objective,
   what moved against what the generation published, and names a floor decision taken inside
   that spread.

Neither fix moves a score by itself. What they change is whether a number can be published
without saying what it rests on — and the first full run of this generation could not have
been.
## 9. The answer

**Is there a scorable surface?** There is a *surface* — 5.2% of the step at sixteen sequences,
8.6% at thirty-two, against control spreads of 0.42% and 1.74%. **The persisting-L2 family
cannot reach it**, and on seven of the ten cells it provably cannot be measured trying: its
ceiling there is smaller than the control's own run-to-run spread. Section 7 has the table.

So the honest answer is in two halves, and reporting only the first would be the mistake this
document exists to avoid:

- **For the mechanism this project ships:** no. The one axis that ever separated two policies
  from each other — admission — separates them by less than the cell's published noise, and the
  cell it was measured in has a physical ceiling *below* the figure that was reported. Five of
  ten cells are unwinnable by arithmetic. The shipped default is negative on every cell of the
  full matrix the runtime can batch.
- **For a mechanism that is not bounded by `2 x 60 MiB / step`:** yes, and by one to two orders
  of magnitude. The recurrent traffic is there, at the concurrencies a serving frontier is
  scored at, well outside the noise. Nothing in this repository addresses it, and nothing about
  the instrument prevents someone from doing so — the graph, the planner interface, the
  executor and the scorer are all indifferent to which action a plan emits.

**Should the bands have been shipped?** No, and they have not been. Nothing on this model and
device could ever have reached them — and the same defect turned out to be in `eval/decide.py`,
where a 2% floor gated `significant` until 0.2.1.

**Is the repository ready for contributors?** Yes, on these terms and no others:

- the surface a contributor changes is in the measured path, and that was checked rather than
  asserted;
- what is scored is continuous, and what is *reachable* is published per cell, with its noise
  and its ceiling, in one command;
- the instrument runs from the base commit, keyless and ephemeral, and CI proves both;
- the evaluator's own defects are named in the CHANGELOG with the incident each one caused,
  including five found by the first full run of its own generation;
- four things are open and named — the cost model has no per-window term and the measurement is
  ordered by window count; the `Stream` mechanism is unmeasured; the latency objective does not
  resolve at three repeats; and whether any mechanism *outside* the persist family can reach the
  5–9% at concurrency is untouched. None needs a maintainer's permission to start.

**What would change the answer.** For the persist family: a model whose decode step moves under
6.42 GB *and* is reproducible against itself. Four were screened and none is. For everything
else: nothing needs to change. The room is measured, published, and larger than the noise.

