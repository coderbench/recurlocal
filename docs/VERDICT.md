# Is there a scorable surface here, and if so which one

This document answers the question a maintainer has to answer before handing a repository to
contributors: **is there something here worth winning?** It is written from measurements, it
gives a number for each part of the answer, and where the answer is no it says so.

Short version:

> **Yes, and it is narrow.** On the scored model and device, the surface is the **admission
> axis at concurrency**, it is worth about **0.3 to 0.4 points of throughput** against a noise
> floor of **0.28%**, and it is the only place where a change a contributor can make has been
> measured to move the end-to-end number outside its own noise. Everything below sixteen
> sequences is negative. The tail-latency half of the frontier cannot currently be resolved at
> all. And the largest number anywhere near this project — 5.4x — belongs to the runtime, not
> to this library.

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

### 3.1 The admission axis is real, and it is the surface

At sixteen sequences, changing one enumerator moves the measured number by **0.32 points**
against a **0.283%** floor — `density` at +0.389% resolved against the shipped policy's +0.071%
unresolved. Aggregated, `density` (+0.056%, CI −0.26…+0.33) and `quota` (−0.642%, CI
−0.94…−0.39) have **non-overlapping** intervals: one admission rule is confidently worse than
another, on a real model, end to end.

That is the first measured evidence in this repository that the competition surface is a
surface. It could not have been produced before 0.2.1.

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

That is correlational, and the five arms are confounded — they differ in *what* they persist as
well as in *how many* windows they use. The experiment that separates them needs no new
mechanism and is the highest-value open problem this release leaves: hold the admission rule at
`density` and vary **only** the window count (`--max-windows-per-kernel`, or a cap on how many
tensors are admitted) so committed bytes and modelled residency stay matched. If the gain still
tracks window count, the model needs a per-window cost term and *fewer, larger windows* is the
direction to tune. If it does not, the correlation is an artifact of what these five arms
happen to persist.

### 3.4 The tail-latency objective cannot be resolved yet

The frontier scores goodput *and* p99 inter-token latency, and the second is where a locality
policy has a mechanism the throughput ceiling does not bound — a resident line removes a
variable-latency round trip from the critical path. It has now been measured, and the answer is
**not yet**: control p99 spreads are 0.5–3.7% per cell against arm effects of 0.3–2.3%, so at
three paired repeats only two of fifteen latency figures cleared their own noise.

That is a fact about the harness, not about any policy. The generation allows up to nine
repeats; nobody has spent them.

## 4. What the second proof track says, and what it does not

At model level the global arm now beats the best independent arm on all three golden traces,
where under the linear model it provably could not — and setting `--stream-relief 0` makes the
advantage disappear on every one, which identifies the mechanism as the `Stream` action
(`results/rtx5090-second-proof-track-model.json`).

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

## 7. The answer

**Is there a scorable surface?** Yes: the admission axis at concurrency, worth about 0.3–0.4
points of throughput against a 0.28% floor, measured and resolved. It is the only one, it is
narrow, and it is real.

**Should the bands have been shipped?** No, and they have not been. Nothing on this model and
device could ever have reached them.

**Is the repository ready for contributors?** Yes, on these terms and no others:

- the surface a contributor changes is in the measured path, and that was checked;
- what is scored is continuous, and what is reachable is published per cell with its noise;
- the instrument runs from the base commit, keyless and ephemeral, and CI proves both;
- three things are open and named — the cost model's ranking, the `Stream` mechanism, and
  whether more repeats can reach the latency objective — and none of them needs a maintainer's
  permission to start.

**What would change the answer.** A model whose decode step moves under 6.42 GB *and* is
reproducible against itself. Four were screened and none is. If one appears, the persist family
goes from 0.52% to 3.67% and this document is rewritten.
