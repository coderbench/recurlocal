# Changelog

Notable changes to TensorTransit (RecurLocal through 0.1). Format loosely follows
[Keep a Changelog](https://keepachangelog.com).

No performance claim appears here without a measurement behind it. See `docs/FRONTIER.md`.

## [0.2.1] - The core in the measured path, and a ruler that says what it cannot measure

0.2.0 built the generalization and left it disconnected from every measured number. This
release connects it, and then changes what the numbers are scored against — because once the
surface was reachable it became obvious that the scoring regime pointing at it was describing
a prize that does not exist on this hardware.

Two things came out of connecting it, and neither was the plan:

- **Asked per cell instead of weighted, the ceiling closes the question.** A persisting-L2
  policy cannot be *measured* to win seven of TTF-1's ten cells — its ceiling there is smaller
  than the floor — including the cell every headline in this repository was measured in. A null
  control confirms it from the other side: two byte-identical plans differ by more than that
  cell's entire ceiling. The recurrent traffic at concurrency is still 5–9% of the step and
  still five to twelve times the noise; reaching it is not a planner problem, and
  [`docs/VERDICT.md`](docs/VERDICT.md) section 7 names the four things that could.
- **The first full run of the generation found five defects in the evaluator, not in the
  submission.** Its receipt read −99.5%. Each is fixed, tested, and named below with the
  incident it caused.

### Fixed — the blocker: the new core was in no measured path

`adapters/sparkinfer/tensortransit_sparkinfer.cpp` and
`workloads/recurrent/synthetic/cuda_bench.cu` both drove the 0.1 `CudaLocalityController`
directly. `TransitRuntime`, `TensorRegistry`, `TransitGraph`, `ITransitPlanner` and
`CudaTransitExecutor` were reachable only from the CLI and the tests. **A contributor who wrote
a planner or an admission rule changed nothing about the measured end-to-end number**, so the
entire competition surface was decorative.

Both now route through Registry -> Graph -> Planner -> Executor. The 0.1 controller is kept as
a second engine, `TENSORTRANSIT_ENGINE=v0`, in the same binary — not out of caution, but
because it is the only way to check the migration rather than assert it.

**The check, on hardware, in one session, one box, one model load** — the only variable between
the two rows is `TENSORTRANSIT_ENGINE` (`results/rtx5090-0.2.1-rewiring-check.json`):

| engine | gain | noise floor | resolved | token-exact | windows attached to node | capture invalidations |
|---|--:|--:|:--:|:--:|--:|--:|
| `v0` (the 0.1 controller) | +0.115% | 0.075% | yes | yes | 96 | 0 |
| `transit` (Registry -> Graph -> Planner -> Executor) | +0.145% | 0.109% | yes | yes | 96 | 0 |

0.2.0 published +0.13% at a 0.06% floor for this arm. The two engines overlap inside their own
noise floors, and `windows_attached_to_node: 96` pins the delivered policy rather than its
telemetry: 2 graph captures x 48 recurrent layers, one window each. Persisting both the matrix
and the convolution state would read 192.

On the synthetic benchmark the planner choice now moves the measured number:

```console
$ tensortransit_bench persist --engine transit --planner budgeted --admission quota ...
  recurrent_v0/density     ms/tok=1.0397 actions=96 declines=48 digest=c806496b6cc08583
  budgeted/quota           ms/tok=1.0497 actions= 8 declines=92 digest=74c88ab783857793
  budgeted/proportional    ms/tok=1.0703 actions=96 declines=48 digest=a24d47d772c47fa3
```

### Added — `tt-frontier generation show --reachable`, and the answer it gives is no

One command for the question a contributor should ask first: which cells are worth trying to
win. Per cell it prints the ceiling a **perfect** persisting-L2 policy could reach, the ceiling
any mechanism that removed *all* recurrent traffic could reach, how much of peak bandwidth that
step actually used, and how far the control moved between repeats of itself at calibration.
Every figure is computed from the pinned geometry and that cell's own measured control rate.

The answer for TTF-1 is worth stating plainly:

```text
    cell               persist  any mech.    bw    spread    floor   verdict
    ctx128-c1           0.491%     1.210%   71%    0.144%   0.144%   measurable by the persist family
    ctx128-c4           0.391%     1.983%   57%    0.090%   0.090%   measurable by the persist family
    ctx16384-c1         0.159%     0.390%   23%    0.000%   0.221%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx128-c16          0.249%     5.217%   40%    0.424%   0.424%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx128-c32          0.200%     8.649%   36%    1.738%   1.738%   PERSIST FAMILY UNWINNABLE; the traffic is there, the cache is not
    ctx4096-c32         0.040%     1.615%    7%   39.008%  39.008%   PERSIST FAMILY UNWINNABLE; and so is any
```

**Seven of ten cells cannot be MEASURED to be won by a persisting-L2 policy**: the floor there
is larger than the ceiling a perfect policy could reach, so no admission rule, window shape or
hot-set heuristic can produce a result in them that is distinguishable from noise. The floor is
the larger of the published spread and what the bench can resolve — aggregate throughput is
printed to one decimal, which is 0.22% of a 22.6 tok/s cell. `ctx128-c16` is one of
them — the cell at which every headline figure in this repository was measured, whose ceiling
is 0.249% and at which the arms sweep reported +0.389%.

**And the room at concurrency is real; it is just not this family's.** Removing all recurrent
traffic is worth 5.2% at sixteen sequences and 8.6% at thirty-two, against spreads of 0.42% and
1.74%. The persisting-L2 ceiling is `2 x persisting-L2 / step traffic` and the numerator is 60
MiB of hardware. A mechanism that is not bounded by that numerator has one to two orders of
magnitude more to play for — and docs/VERDICT.md section 7 names the four levers rather than
leaving it at that: two are hardware or workload, one fails the exactness gate, and the fourth
is a kernel change in SparkInfer. None of them is a better admission rule.

The `bw` column is why the second number carries a caveat rather than a target: every TTF-1
cell sits below the 80% utilisation at which a traffic ceiling is tight, so `any mech.` is a
loose upper bound. The `persist` column does not depend on it.

### Measured — the admission axis, and what the cell it was measured in can hold

Five configurations against a scrubbed control, paired and interleaved, three repeats,
token-exact on every one, on the three TTF-1 cells whose control spread resolves. One binary,
one model load, one box (`results/rtx5090-0.2.1-arms.json`).

At sixteen concurrent sequences, control 564.9 tok/s, that run's own floor 0.283%:

| arm | throughput gain | resolved against that run's floor |
|---|--:|:--:|
| `budgeted` / `density` | +0.389% | yes |
| `global` preset | +0.372% | yes |
| `budgeted` / `proportional` | +0.142% | no |
| `budgeted` / `quota` | +0.089% | no |
| `recurrent_v0` (shipped) | +0.071% | no |

**Read against the section above, that table retracts itself.** `ctx128-c16`'s persist-family
ceiling is **0.249%**, so +0.389% is above the most a perfect policy could deliver there; the
cell's published control spread is 0.424%, which says the same thing a second way; and the same
arm measured **−0.388%** in the full matrix two hours later. The numbers are what the box
produced — three paired interleaved repeats, token-exact — and none of them is evidence that an
admission rule moved the end-to-end number.

What survives is the comparison between arms, which is a different statistic: aggregated over
c1, c4 and c16, `density` (+0.056%, 99% CI −0.26…+0.33) and `quota` (−0.642%, −0.94…−0.39) have
non-overlapping intervals — `density` `INCONCLUSIVE`, `quota` `NO_FRONTIER_GAIN`, over a PARTIAL
matrix every receipt marks as such. One admission rule is confidently worse than another. That
much could not have been produced before this release: the adapter drove the 0.1 controller, so
no admission rule was in the measured path at all.

**A null control settles the size of the arm-to-arm floor.** `survival` and `density` compile to
the same plan — identical digests on all four golden traces, asserted in `tests/test_golden.cpp`
— so the paired difference between them is noise with the policy held constant. Over five
interleaved repeats it is a median of −0.145% at `ctx128-c1`, −0.045% at `ctx128-c4` and
**+0.230% at `ctx128-c16`**, peak-to-peak 1.29%. Against that cell's 0.249% ceiling: two arms
running the same bytes differ by more than a perfect policy could deliver
(`results/rtx5090-0.2.1-null-control.json`). At `ctx128-c1` the figure is exactly one
least-significant digit of the bench's own output, which is the printer rather than the box.

The same five-repeat run measures `density` against the control at −0.289% (c1, five of five
negative), **−0.544%** (c4, five of five, resolved) and +0.106% (c16, straddling zero). Three
admission rules, one answer.

Four findings that are not good news and are recorded with the same weight:

- **Below sixteen sequences the whole family is negative** through the continuous-batching
  path: −0.29% to −0.43% at one sequence, −0.27% to −0.54% at four, most resolved. Not in
  tension with the +0.145% at batch 1 in the rewiring check — that is the single-sequence
  bench; these cells run the CB engine with the generation's long-prefill injection alongside.
- **Every arm is ordered by the number of windows it attached**, in both the per-cell and the
  aggregated view — 48 for `density`, 77 for `global`, 144 for each of the three that are
  mutually indistinguishable. The residency cost model prices bytes, whole-line residency,
  survival and the reservation; it has no per-window term, so it cannot express that and did
  not predict it. Confounded, and the experiment that separates it is named in
  docs/VERDICT.md section 3.3.
- **The cost model's RANKING of admission rules is not validated.** It predicts
  `quota` ≈ `density` ahead of `proportional`; measured, the order is
  `density` > `proportional` > `quota`. The model fits the persist family's *magnitude* better
  than the linear one — that is what `cost_model_fit.py` establishes — but no PAIR of arms
  separates at that cell (the largest gap is 0.247 points against a 0.283% floor), so the
  per-cell ordering is not evidence either way. What the aggregate does separate, the model does
  not predict. Reconciling it is the highest-value open problem the release leaves, and it needs
  no GPU.
- **The p99 objective does not resolve at three repeats.** Control p99 spreads are 0.5–3.7% per
  cell against arm effects of 0.3–2.3%; two of fifteen latency figures cleared their own noise.
  A fact about the harness, not about a policy. The generation allows nine repeats and nobody
  has spent them.

And one about scope: `stream_applied` is zero on every measured arm, because the adapter
registers a `ModelWeight` tensor only when the operator declares
`TENSORTRANSIT_STREAMED_BYTES_PER_TOKEN`. So the model's claim that coordination pays *through
the Stream action* was **not** tested here. The measured `global` differs from `density` by
role-floor arbitration alone, and measures the same as it within noise.

[`docs/VERDICT.md`](docs/VERDICT.md) puts all of it together and answers the question a
maintainer has to answer before handing this to contributors.

### Fixed — the second scorer still had the impact bands' defect

`eval/decide.py` gated `significant` on a 2% weighted gain. That is the same number the impact
bands used for their lowest paying step, and it was removed from THEM in 0.2.1 because it sits
above the physical ceiling — 0.52% for the persist family on the scored model, 1.94% on the
best model this project has found. Left in the other scorer it meant no result achievable on
this hardware could ever be called significant, `--real` always exited non-zero, and the status
vocabulary the two scorers share meant one thing in `decide.py` and another in `tt-frontier`.

`significant` is now what the rest of the harness means by it: a positive gain, on a complete
matrix, that cleared its own run-to-run spread. The 2% threshold keeps deciding `verdict` and
`go_no_go`, which answer "should this research direction continue" — the project's decision
about itself — and the reason line now says so, and says the number is above the ceiling.

**And the removal exposed a hole the floor had been hiding.** `unresolved_workloads` was read
as a list, so a document that reported no resolution evidence at all looked identical to one
where everything resolved. Absent is not resolved: a result with no
`measurement.unresolved_workloads` is INCONCLUSIVE and says why.

### Fixed — an empty plan and a plumbing failure read the same in the 0.1 counters

`windows_applied 0, windows_attached_to_node 0, pre_touch_launches 0` is a NULL CANDIDATE — the
hook loaded, bracketed layers, and nothing reached a kernel. It is also what a planner that
**decided** to do nothing looks like, and the two are opposite facts about a run.

`naive_both` — persist both tensor classes with no arbitration — declines every one of 512
candidates as `below_min_hit_ratio` at concurrency 4, because a budget shared among that many
gives each less than the configured floor. That is the arm doing precisely what the
specification says it should do badly, and refusing it would have taken the other four arms of
the five-arm run down with it.

The transit engine's decline census tells them apart, and the incident the guard exists for
keeps its guard: a window computed and handed back to a runtime that never attached it reports
`windows_deferred_to_caller > 0`, which is checked first and still aborts. An engine that keeps
no census — the 0.1 controller — gets no benefit of the doubt.

### Fixed — the packed-path guard counted prefill chunks in its denominator

`eval/real_eval.py` refused a concurrency measurement whose `tokens_packed / tokens` fell below
half. `tokens` counts every step the hook brackets, and at long context most of them are
**prefill chunks**: on the reference box a `ctx4096-c16` run brackets 133 steps of which 63 are
decode. A run whose *every* decode step batched sixteen rows therefore reported a 47.4% "packed
share" and was refused.

Two of the four cells the first full TTF-1 matrix lost were exactly that — `max_rows_seen` 16
and 32, 63 of 64 decode steps packed, and a denominator that had nothing to do with decode.
The other two were real: `max_rows_seen` **0**, not one batched decode step at 64 new tokens or
at 256, in the candidate *and* in a baseline probe running no policy at all.

The guard now asks the two questions separately:

- **`max_rows_seen >= 2`** — did concurrent decode ever happen. Unambiguous, checked first, and
  it is what catches the two cells this runtime genuinely cannot batch.
- **`tokens_packed` against `max_new`** — did most of it happen. One decode step advances every
  live row, so the steps a concurrency arm should produce is the decode length the harness
  asked for, which is immune to how long the prefill was.

The incident that created the guard still fires: 4 of 142 steps packed at `max_rows_seen` 8 —
the Q4_K row cap, worth 5.4x — is refused, by the second test, and the message names which of
the two failures it was. Callers that do not know the decode length get the first test only,
so nothing that could not be checked is failed.

### Fixed — a cell that collapses in BOTH arms was charged to the candidate

`eval/real_eval.py` refuses a concurrency measurement whose tokens did not go through the
runtime's packed decode path, because a run that decoded one row at a time is not that cell's
number. The guard reads the adapter's own `tokens_packed`/`tokens` counters — and the control
arm is unhooked by construction, so it emits none and **cannot be seen to fail the same way**.

The first full TTF-1 matrix lost four long-context concurrency cells to it. Charged to the
candidate, each one is scored at the generation's cell floor, and one floor-decided cell moves
dF through the geometric mean further than any policy in this repository ever has — a synthetic
reproduction of the exact shape puts it at **−99.8%** against a **0%** truth.

So a serving loss is now a question before it is a verdict. `tools/tt-frontier run` re-runs each
such cell once on the **baseline** binary with the hook installed and no window — the same
telemetry, no policy — and stamps the record `runtime`, `candidate` or `unresolved`:

- `runtime`: the failure reproduced with nothing running. The cell is not scored, it is named
  in `coverage.unservable_cells` with the probe's own evidence beside it, and the receipt is
  PARTIAL on its face.
- `candidate` or `unresolved`: the loss stays where it was. `unresolved` is the default, so a
  matrix run with `--no-attribution` behaves exactly as before.

The probe runs the baseline build deliberately. A candidate that could make its own probe fail
would get a cell it lost *dropped* rather than scored, which is strictly to its advantage.

### Added — the prose is checked against the data it cites

"Do not let anyone type benchmark numbers by hand" was enforced for reports, which are
generated, and not for documents, which are not. `eval/test_schemas.py` now reads every
per-cell figure out of `docs/VERDICT.md` section 8 and compares it with
`results/rtx5090-ttf1-first-matrix.json` — thirteen numbers, in ctest, failing loudly on a
drift of more than a thousandth of a point. A figure that drifts in prose is worse than one in
a report, because prose is what a contributor reads before deciding whether to spend a week.

### Added — a frozen generation is stored twice, and a test says they are the same bytes

`eval/generations/TTF-N/` is what the scorer loads and what `eval/run_from_base.sh` overlays
from the base commit; `frontier/TTF-N/` keeps a copy beside the receipts. Both are the
definition, the checksum covers it, and a receipt whose generation moved does not verify — so
two copies that drifted would make a ledger unauditable in a way nothing else here would catch.

### Changed — the authoritative runner defaults to five repeats, not the generation's minimum

Three is a floor and it is not a recommendation. The same arm measured **+0.07%** and **−0.39%**
at `ctx128-c16` in two sessions two hours apart on the reference box, against a control spread
that generation published as 0.42% for that cell. `scripts/trusted_eval.sh` and the GPU workflow
default to five; the generation allows nine and `--repeats` is how you spend them.

### Added — the receipt says which cells the score is entitled to rest on

`frontier/TTF-1/reference.json` has always published each cell's measured control spread, per
objective, so a contributor can see the noise before spending a week. Nothing read it back.

Every receipt now carries `cell_resolution`: per cell, per objective, the change observed and
the spread the generation **froze at calibration**, and whether the first exceeds the second.
It changes no score — it says which cells a score is entitled to rest on. And a floor decision
taken inside that spread is named on the receipt's face as
`aggregation.floor_decided_inside_published_noise`, in the box, in the Markdown and in the PR
comment.

The first full TTF-1 receipt is why. It read **−99.5%**, and the cell that decided it,
`ctx128-c32`, reached the floor on a p99 change of 183% against a control spread this project
had itself calibrated at **481%**. Its p50 and p95 were identical to the control's in every
repeat. Nine repeats of that cell afterwards put the paired p99 ratio at a median of 1.083,
worse for the candidate in seven of nine pairs — p ≈ 0.18, not a result. A number like the first
one is not a result either, and a receipt should not be able to publish it without saying what
it rests on.

The same document is what says four of TTF-1's ten cells cannot be measured as concurrency
cells on this runtime at all — see `results/rtx5090-ttf1-first-matrix.json` and
docs/VERDICT.md section 8.

### Added — the control cannot report a collapse, but its own scaling can

`concurrency_scaling()` compares each cell's control throughput with the **same-context c=1
cell of the same matrix**, both measured minutes apart on one box. It needs no adapter
telemetry, which is the whole point: the packed-path guard reads the adapter's counters, the
control is unhooked, and a control that fell off the batched path cannot be seen to. Its
arithmetic can — 1.15x for four concurrent sequences at `ctx4096-c4` against 3.20x at
`ctx128-c4`. Recorded in provenance and attached to every attribution verdict.

### Fixed — the `baseline` arm was refused for applying no policy

`RECURLOCAL` and `TENSORTRANSIT` are two spellings of one variable (docs/STABILITY.md section
6a). Two *readings* of it is a bug: `is_control` read both, and the null-candidate guard read
only `RECURLOCAL`. An arm configured `TENSORTRANSIT=baseline` — the hook installed with no
window, which is how the hook's own cost is measured and is one of the five arms the
specification names — therefore looked like a policy arm that had applied no policy, and was
refused for doing exactly what it was asked to do. One reader now, `declared_mode`, and the
same for a preset or planner named `baseline`. The unhooked-run message also indexed
`env_extra['RECURLOCAL']` directly, so the guard raised `KeyError` from inside its own error
path rather than reporting.

### Fixed — one counter name, two meanings, and the `baseline` arm failed on it

`RECURLOCAL_STATS`'s `stats.layers` means *bracketed recurrent layers* for the 0.1 controller,
whatever the policy did with them — and `eval/real_eval.py` refuses a run whose `layers` is zero
as "the hook initialised but bracketed no recurrent layer", which is a genuine and load-bearing
guard: it is how a run that never reached the hook site is told apart from one that did.

The transit engine published its **windowed** layer count under that name. `TENSORTRANSIT=
baseline` brackets every layer and windows none, so it produced `layers: 0` and the guard
refused it **by construction**. That is what stopped the specification's five-arm run: it never
reached a benchmark, it failed at the correctness gate on the first arm.

`stats.layers` is now the bracketed count for both engines, which is what the mapping always
claimed. The windowed count keeps its own name, `transit.layers_windowed`, in the block that
exists for counters the 0.1 line has no field for. Two spellings of one variable is a
compatibility shim; two meanings of one counter is a bug, and this is the second one this
release found.

### Fixed — the recorded "c=16" trace was the drain, not the workload

`TENSORTRANSIT_TRACE_OUT` kept the last rebuild. A concurrency run rebuilds as requests arrive
and again as they finish — a measured c=16 run rebuilds at 1, 1, 16, 15 and 1 — so the last one
is the tail of the drain, and the first recorded c=16 trace came back byte-identical to the c=1
trace, with `active_requests: 1`. It now keeps the rebuild with the **most** sequences, and the
stats line reports `trace.sequences` and `trace.writes` so a reader can tell which graph is in
the file.

### Added — the stats line says which tensor families the graph was made of

`registry`: `recurrent_tensors`/`bytes`, `kv_tensors`/`bytes`, `weight_tensors`/`bytes`, from
the graph in force. Zero KV tensors on an arm whose preset is *about* KV is not a weak result,
it is an unmeasured one, and from outside a run the two are indistinguishable — which is what
made "the second proof track cannot be measured at all" a sentence nobody could check.

`eval/real_eval.py` now refuses such an arm as `UNMEASURABLE ARM`, in the same class as NULL
CANDIDATE, and reports it as a configuration failure rather than a serving one. A `global` or
`naive_both` arm that registered no `ModelWeight` tensor is *not* refused — that is a legitimate
configuration — but the result carries `untested_mechanisms: ["stream"]`, because the 0.2.1 arms
sweep measured `stream_applied = 0` on every arm and the number was read as evidence about
coordination until the counters said otherwise. A build too old to emit `registry` is not
accused: the guard returns without an opinion, which matters because `eval/run_from_base.sh`
runs the *base* commit's evaluator against a candidate's build.

### Fixed — the sanitizers covered the code a submission cannot change

`scripts/sanitize.sh` ran memcheck, initcheck and synccheck over the 0.1 controller and
racecheck over one bench mode. It did not touch `test_cuda_executor` — the transit executor,
which is the half that manipulates a live graph capture, drops borrowed streams and hands a
device-wide L2 partition back, and the half a submission can actually change. It now covers
both, plus racecheck on *both* bench engines and a memcheck of the transit engine with a
streaming buffer present.

The first run of the extended script found 924 uninitialised device reads. They were the
executor test's own fixture reading `cudaMalloc`'d memory nobody had written, not a library
defect — but nothing would have caught the next one either. Clean on both engines now.

### Added — KV is registered, so the second proof track can be measured at all

`declare_kv_cache()` exposes SparkInfer's paged K and V pools, and
`before_attention_layer`/`after_attention_layer` bracket the full-attention layers. Before
this no adapter exposed KV to the registry, so the specification's second proof track — does
one planner arbitrating a shared budget across two tensor classes beat two independent
policies — **could not be measured**. That is different from not having been measured, and the
difference is the point.

The live run now reports `recurrent_tensors: 48, kv_tensors: 16, kv_declared: true`.

Live bytes are declared, not the pool stride: the pool is sized for the longest context the
server will ever hold, and a decode step at 128 tokens reads a thousandth of it. Declaring the
stride would have put a KV footprint two orders of magnitude too large into every ceiling.

### Changed — a cost model that can express coordination

The 0.2.0 model was `saved = reused_bytes x (granted/bytes) x hit_ratio`, linear in the
resident share. Total saving was then a fractional knapsack, greedy-on-density was **provably**
optimal, and therefore no admission rule could beat `density` and the whole axis measured
nothing. That was an artifact of the model, not a finding about caches.

`CostModel::Residency` carries the three terms `docs/evaluation.md` named as missing:

```text
resident(t) = whole cache lines of  min(granted, bytes) x hit_ratio
survival(t) = min(1, (resident(t) / reuse_distance_bytes(t)) ^ beta)
saved(t)    = reused_bytes(t) x (resident(t)/bytes) x survival(t)
              -  eta x (resident_total / L2) x step_traffic
```

`beta = 0.1108`, `eta = 0.00086`, fitted once against every paired hardware measurement of the
`persist` arm in `results/`, across two model architectures, against the POLICY effect
(`persist` minus `baseline`, so the hook's own overhead is not charged to the model).

```console
$ python3 eval/cost_model_fit.py
  residency model : rms 0.2713pp, 8/8 arms inside their noise floor
  linear model    : rms 0.4847pp, 6/8 arms inside their noise floor
```

Both arms that actually *resolved* are predicted to within a fifth of their own noise floor:
dense batch-1 +0.1418% against +0.1393% measured, MoE batch-1 +1.2015% against +1.2079%.

**Why a power law and not an exponential**, which is the part that had to be settled rather
than chosen: an exponential cannot fit those two arms at once. Dense batch-1 sits at a
reuse-distance-to-capacity ratio of 368 and delivers 0.255 of its ceiling; MoE batch-1 sits at
71 and delivers 0.415. An exponential needs its coefficient to differ by 3.4x between them; a
power law needs `beta = 0.140` and `0.115`, the same number within the spread. It is also the
classic shape of a cache miss-ratio curve.

**The consequence is the point.** `saved` goes as `resident^(1+beta)` — superlinear — so for a
fixed budget spread over `n` tensors the total goes as `n^(-beta)`. The objective is convex,
its optimum is at a vertex, and concentrating beats spreading. The shipped recurrent policy
spreads (`HotSetPolicy::Proportional`), and on the golden KV trace that is 23x worse than
concentrating under this model against 10.5x under the linear one.

`CostModel::Linear` remains, as the control. `--cost-model linear` is how a contributor checks
whether a result is about the policy or about the model, and a test asserts that
`AdmissionRule::Survival` reduces to `density` exactly under it.

### Added — `AdmissionRule::Survival`

Greedy on MARGINAL saving under the cost model, stopping when the next admission would make
the plan worse. Under `Linear` the marginal value never decreases and it is exactly `Density`;
under `Residency` it is not, because a fractional knapsack can never say "stop".

**Measured against itself, it is currently a null control, and that is pinned rather than
described.** At the fitted `beta = 0.1108` the stopping rule never fires and the plan is
byte-identical to `Density`'s on all three golden traces; a little under `beta = 0.2` the FIRST
candidate stops paying for its share of the reservation and it admits nothing at all. There is
no useful middle, so a contributor who selects `--admission survival` expecting a third
behaviour gets one of the other two. `tests/test_golden.cpp` asserts both ends, because a
change to beta, to the reservation cost or to the stopping rule moves that boundary without
moving anything a reader would look at.

### Added — `max_windows_per_kernel`, and `WindowPreference`

CUDA binds ONE access-policy window to a stream, a launch or a graph node at a time. A plan
marking two regions before one kernel was not describing something the hardware can do: the
second replaced the first, the first was silently absent, and the executor's telemetry counted
two applied windows for one delivered policy. The plan now says which one it meant, where a
CPU test can read it, and dropped bindings become `not_supported` declines with the forgone
saving attributed.

### Fixed — `Stream` was dead code in the only regime that matters

Two independent reasons, both of which had to go:

1. `emit_stream_hints` required `!profile.has_reuse()`. Over a **cyclic** decode window every
   tensor is read again next iteration, including the weight stream, so nothing was ever
   streamable. The test is now survival, not reuse: a tensor whose reuse distance exceeds the
   whole budget cannot be kept whatever anybody does, so telling it to get out of the way is
   free. Parameter-free, and the same criterion `screen()` declines a candidate on.
2. The CUDA executor **skipped** a Stream action under graph capture. A streaming window is
   the same host-side stream attribute a persisting one is, so under capture it was absent
   from every replay. It now reaches the node, exactly as a Persist does.

Together these mean the arm that tells the weight stream to get out of the way was, until now,
telling it nothing. On the golden traces the global arm's plan is finally distinct from
`recurrent_only`'s — they had the same digest.

### Changed — Frontier Gain replaces the impact bands

**The `XS`/`S`/`M`/`L`/`XL` table is gone.** Its lowest paying step was 2% weighted throughput
gain; the physical ceiling for the whole shipped policy family is **0.52%** on the scored model
and **1.94%** on the best model this project has ever found. A submission could remove every
recoverable byte of recurrent traffic and score `none`. A band structure whose lowest step
sits above what the hardware can deliver is not strict, it is broken, and what it told
contributors was false.

What replaces it is the **Transit Frontier Ledger** (`eval/frontier/`, `tools/tt-frontier`,
`frontier/`):

```text
Frontier Gain: dF = F(candidate) / F(main) - 1
```

`F` is the normalized Pareto hypervolume of the serving frontier — goodput against p99
inter-token latency — aggregated over a frozen generation's workload cells by weighted
geometric mean. Continuous; no bands anywhere, and a golden test asserts that no rendering of a
receipt assigns one.

- **Frozen generations.** `TTF-1` freezes the model, the runtime commit, the hardware, the
  cells, the published weights, the objectives, the normalization bounds, the SLOs, the
  reference point, the repeat policy, the confidence method, the guard and the hidden-seed
  rules. Its SHA-256 is in every receipt, and `receipt verify` refuses one whose generation has
  moved.
- **Per-cell bounds, calibrated on hardware.** Cells differ in absolute throughput by more than
  ten times, so one global bound would make the geometric mean an implicit weighting by
  throughput regime that nobody chose. A maximized objective is anchored at `lo = 0`, which
  makes its contribution to `dF` exactly its own ratio — so a TTF-1 number stays comparable
  with every throughput figure in `results/`.
- **Confidence as a gate, not a multiplier.** Paired bootstrap over interleaved repeats, frozen
  seed and resample count, so a receipt is reproducible bit for bit from its raw results. A 99%
  lower bound at or below zero is `INCONCLUSIVE`.
- **A failure is the absence of an operating point.** An OOM, a timeout, or a fall off the
  batched decode path produces no point at all rather than a small positive score.
- **Append-only ledger.** A finalized receipt is never rewritten; a correction is a new receipt
  naming what it supersedes and why, and `ledger audit` proves it.
- **Generated, never typed.** `tt-frontier report` produces the Markdown report, the PR comment
  and the terminal box from the receipt, and the receipt from the raw results.

`eval/decide.py` keeps every guard it had and now speaks the same status vocabulary. Its
`GO_NO_GO` table survives, because "should this research direction continue" is the project's
decision about itself and not a label attached to somebody's submission.

### Fixed — a percentile bootstrap over three points can qualify pure jitter

Confidence was the only statistical gate. It is not enough at the repeat counts this generation
allows, and that is a property of the estimator rather than a bug: a percentile bootstrap over
three paired points has at most 27 distinct resamples, so if all three paired ratios fall on the
same side of 1.0 — which pure jitter does one time in four — every resample does too and the
lower bound clears zero however small the effect. A synthetic candidate identical to `main` to
within 0.03% scored `FRONTIER_GAIN` this way, found by the CLI pipeline test on its first run.

There are now **two gates and both must pass**: the bootstrap, and the observed `dF` exceeding
the run-to-run spread of the runs that produced it — the rule the rest of this harness has
always used. The receipt carries `noise_floor_pct`, `resolved` and `confidence_qualifies`
separately, so which gate failed is on the page. The floor is each arm's own absolute spread and
not the paired one, deliberately and conservatively; the cost is a real small effect on a noisy
box being reported `INCONCLUSIVE`, and the remedy is repeats.

### Added — a receipt says which configuration held the frontier, and where

`cells[<cell>].on_frontier` names the non-dominated configurations per variant, and the Markdown
report renders them. "The candidate gained 4%" and "the candidate's specialized planner is on the
frontier in four cells and its default is on none" are different facts, and only the second tells
a contributor where their work earned its place. A candidate does not have to beat every existing
strategy everywhere — it has to hold territory somewhere — and this is the table that says
whether it did.

### Added — the PR comment is posted from a job the submission's code never runs in

The evaluation job holds `contents: read` and nothing else, because it executes a stranger's
CUDA. Posting needs `pull-requests: write`. So they are two jobs: the GPU job produces an
artifact, and a second job on a hosted runner downloads it and posts the Markdown. The two never
share a runner.

### Fixed — the exactness gate compared the candidate against itself

The control replays ran the candidate's own binary with a scrubbed environment, so the gate asked
"does enabling the policy change the output" rather than "does this submission change the
output" — and a candidate whose *inert* path changed the output would have been compared against
its own changed output and passed. `scripts/trusted_eval.sh` has both builds and now passes both;
the receipt records which question was answered (`correctness_compared_against`).

### Fixed — a hung cell took the whole matrix with it, and a null policy did not stop one

A `subprocess.TimeoutExpired` propagated and ended a sixty-run matrix two thirds of the way
through, having spent an hour of device time and produced nothing scoreable. Serving failures —
OOM, timeout, a fall off the batched path — are now recorded as *territory lost* and the matrix
continues, which is what the frontier is built to price. Configuration failures — a candidate
that asked for a policy and whose own telemetry says none reached a kernel — still abort, because
the number there would be the hook's overhead wearing a policy's name.

The `baseline` arm, which applies no policy on purpose and is how the hook's own cost is
measured, was being refused by that same guard. The exemption now reads the environment rather
than the result.

### Added — TTF-1, and what calibration found

Ten cells, not twelve. The 4x3 matrix was cut down by **measurement**, and
`frontier/TTF-1/reference.json` carries the evidence: `ctx16384-c16` loses 30 of its requests
to a device OOM in every repeat (128 decode tokens of an expected 1024) and `ctx16384-c32`
cannot even load the model — 32 sequences of 16K context plus a 17.9 GB checkpoint does not fit
32 GB of VRAM.

Per-cell control spreads are published rather than implied. They are the reason the
protected-workload guard protects `ctx128-c1` and `ctx128-c16` and **not** the c32 arms:
`ctx4096-c32`'s own control spread is 39% across three interleaved repeats. A 5% guard against
an arm whose noise is 39% does not protect anything; it rejects good submissions at random.

### Added — a trusted, keyless, ephemeral GPU runner

`scripts/trusted_eval.sh` and `.github/workflows/frontier-eval.yml`. Fresh workspace per run,
wiped afterwards; the instrument overlaid from the BASELINE into both trees; the ledger written
outside the workspace; and a refusal to run at all with an ssh-agent or a cloud token in the
environment, because a credential in scope is a credential the submission has. Deliberately
`workflow_dispatch` and not `pull_request`: a performance PR is arbitrary CUDA from a stranger.

### Fixed — `eval/run_from_base.sh` was broken against the 0.2 layout

It diffed `eval bench` and archived `eval`. `bench/` has not existed since 0.2.0, so a
submission's changes to the workload definitions were neither discarded nor reported. The
instrument is now one list used for both the overlay and the report — `eval`,
`tools/tt-frontier`, `schemas`, `configs`, `tests/golden`, `workloads` and
`adapters/sparkinfer/pin.json` — and a **CI job proves it** by submitting a tree that relaxes a
guard, moves the runtime pin and edits a frozen generation, then asserting all three are named
and none is used.

The *other* entry point was the half nothing ran. `TT_ENTRY=frontier` routes to the staged
`tools/tt-frontier`, and until now CI only proved `--help` worked through it. A second job now
stages a submission that raises the frozen generation's `cell_floor` to 1.0 — which would turn
every cell into a floor decision — computes a real receipt through the overlay, and asserts the
receipt carries the **base** generation's checksum and is not floor-decided.

### Added — `scripts/check.sh`, the whole pre-PR list in one command

Build, `ctest`, re-fit the cost model against the measurements in `results/`, load every frozen
generation, audit every ledger; `--cuda` adds the device tests and `--sanitize` adds
compute-sanitizer over both engines. The list used to be four commands in three documents,
which is a list people run three quarters of.

### Added — plan replay, closing the offline loop

`tensortransit replay <plan.json> --trace <trace.json>`. `read_plan()` reads a serialized plan
back and `TransitPlan::rebind()` resolves its regions against a live tensor table by id — a
serialized plan carries no pointer by design, so the rebind is what makes it executable, and it
fails loudly on a tensor the registry does not know rather than skipping it. Regions are clamped
to the descriptor's allocation, so a file cannot widen a window past memory the runtime owns.

### Added — the overhead budget is gated, not described

`tests/test_overhead_budget.cpp` asserts the specification's 0.5%-of-token-latency budget on
the real 64-layer Qwen3.8-27B decode shape. It checks two different failures: the layer being
slow, and the plan cache not working so the planner runs every token — which is the single most
likely way this layer costs more than it saves and is invisible in a throughput number.
Measured: 0.045–0.078% for planner plus executor, 0.19% for a whole bracketed token.

### Fixed — `TransitGraph::live_bytes_at` was quartic on the compile path

It resolved each edge's endpoints by scanning every use, for every edge, for every profile —
and `build()` called it once per kernel. `build()` now accumulates the whole live-set curve in
O(edges) and the query is a binary search into it.

### Added — a golden trace RECORDED from the runtime, beside the ones written by hand

`tests/golden/trace_live_c1.json` is the live SparkInfer adapter's own graph: the runtime's real
KV slice sizes (4,210,688 bytes per attention layer, against the 12,582,912 the hand-written
fixture invents), its real layer count (64 kernels against 176), and the measured step traffic.
The three synthetic traces stay, and keeping both is the point — a digest that moves on the
recorded trace and not on the synthetic ones is a fact about the deployment rather than about
the planner.

It also settled what item 3 was for. The first recording came back with **zero KV tensors**,
because the runtime declares its pools after it opens the token and the trace was written once
after the first compile. The second came back with `active_requests: 1` on a c=16 run, because
the trace kept the LAST rebuild and a concurrency run rebuilds as requests drain. Both are
fixed; the file now carries 96 recurrent, 32 KV and 64 weight tensors from a real batch-1 step.

### Added — live trace recording

`TENSORTRANSIT_TRACE_OUT` writes the recorded Transit Graph once, after the first compile. The
golden traces carry measured recurrent geometry and a *synthetic* KV block size with weight
traffic divided evenly across layers; a trace recorded here has the runtime's real KV slice
sizes and real per-layer demand.

### Added — percentile inter-token latency in the runtime bench

The pinned `qwen3_gguf_cb_bench` reported a mean and a max. TTF-1 scores p99, and a tail cannot
be stood in for by a mean: a policy that halves the median while doubling the 99th makes
serving worse and the mean better. Gaps are collected into a request-local vector and merged
once when the request finishes, so the hot path takes no lock — a mutex per decoded token
across N threads would perturb the very latency being measured. Insertions only; `mean_itl_ms`
and `max_itl_ms` are untouched, so every command in `results/` still means what it meant.

## [0.2.0] - RecurLocal becomes TensorTransit

The recurrent-state locality library is now the first workload inside an engine-independent,
cross-kernel, multi-tensor locality planner. **This is a generalization, not a rename**: the
recurrent policy is unchanged, every 0.1 name still resolves, and the migration was measured
rather than asserted.

### The migration did not change what the policy does

Qwen3.8-27B on the pinned SparkInfer commit, batch 1, ctx 128, three interleaved pairs, one
RTX 5090, through the migrated stack — `namespace tensortransit`, `adapters/sparkinfer/`, the
`TENSORTRANSIT_` environment prefix, the `TensorTransit` CMake package:

```
>> token-exact greedy replay gate
    control reproducible over 3 unhooked replays
    candidate vs control: identical over 64 tokens
batch-1 decode: +0.13%  (noise floor 0.06%)
```

That is the figure 0.1 published for this arm, and the hook telemetry confirms the policy
reached the captured decode graph: `windows_attached_to_node: 96`, `window_nodes_attached: 96`,
`capture_invalidations: 0`. A partial matrix by construction (batch 1 is 0.40 of the section
44 weight), and `decide.py` refuses to call it significant and names the missing 60% — which
is why it is filed as `results/rtx5090-0.2-migration-check.json`, a check rather than a result.

### Added — the core

- **`TensorRegistry`** (`include/tensortransit/tensor.h`). Non-owning; never allocates, frees,
  or dereferences a runtime pointer. Identity is `(id, generation)` so a freed-and-reallocated
  address cannot silently inherit a compiled plan's window. Re-registering a live address
  updates in place rather than issuing a new id — a decode loop that re-declares its state
  every token must not grow the table without bound or defeat the plan cache.
- **`TransitGraph`** (`graph.h`). Future-use edges with reuse distance in three currencies —
  `Ordinal`, `Bytes`, `Time` — because residency is a bytes question and prefetch timing is a
  time question, and they disagree. A `ReadWrite` use counts its bytes **twice**, which is the
  factor of two in every ceiling this project has quoted. `set_cyclic(true)` closes the loop a
  decode token is; without it a recurrent state has no reuse edge at all and the entire
  surface reads as unreachable.
- **`TransitPlan`** (`plan.h`). Serializable, diffable, digestible actions plus a **decline
  record** — `no_reuse`, `budget_exhausted`, `too_large`, `reuse_too_far`, `role_excluded`,
  `below_min_hit_ratio`, `not_supported` — so a null result can be read rather than guessed
  at. `validate()` rejects a persist that is never cleared (spec section 32) and a fork that is
  never joined (which under graph capture ends the capture INVALID).
- **`ITransitPlanner`** and five implementations in `planners/`: `baseline` (emits nothing —
  the true control, not "the general planner with the dials at zero"), `recurrent_v0`,
  `greedy`, `budgeted`, `concurrency`.
- **Five admission rules** on `budgeted`: `density`, `quota`, `proportional`, `reuse_order`,
  `role_floor`. Every 0.1 `HotSetPolicy` enumerator survives as one of these, because each was
  measured and none is dominated on every workload.
- **`ITransitExecutor`**, with `CudaTransitExecutor` and a `RecordingExecutor` that makes the
  plan/executor contract testable on a machine with no GPU.
- **`TransitRuntime`** with a plan cache keyed on `(registry epoch, graph digest, concurrency,
  phase, device, config)` and a `last_recompile_reason`. Spec section 33: a loop that
  recompiled every token would put the planner on the critical path, and without a reason code
  finding that out needs a profiler.
- **`compute_ceiling()`** (`ceiling.h`). `eval/traffic_budget.py`'s arithmetic, derived from a
  graph instead of a hand-written geometry file — so a model's decode rates can no longer be
  paired with another model's state shape.
- **Trace and plan formats** (`schemas/*.json`), golden plan digests (`tests/golden/`), and the
  `tensortransit` CLI: `inspect`, `plan`, `compare`, `devices`, `planners`.

### Added — findings

- **The Transit Graph reproduces every published 0.1 bound from a recorded trace.** 293.6 MiB
  removable, 146.8 MiB footprint, 40.9% resident, +1.692% traffic ceiling, +0.685% persist
  ceiling — against 294 MiB, 146.8 MiB, 41%, 1.69% and 0.68%, published before any of this
  code existed. Asserted in `tests/test_golden.cpp`.
- **Negative, and it is about this project's own central claim.** Under the linear cost model
  — saving proportional to resident share — total saving is `sum granted_i x density_i` under a
  budget, greedy-on-density is optimal for that, and **no admission rule can beat `density`**.
  The `role_floor` sweep is monotonically worse as the floor grows (+0.358% -> +0.269%). So the
  global planner beats the naive both-persistent arm **10x** (+0.336% vs +0.034%; that policy
  shaves thirty hit ratios to a third each, and under concurrency declines everything and
  disables itself) and **ties, slightly below, the best single-role arm**. Spec section 38 is
  therefore **partially** satisfied and honestly reported as such. The three terms a linear
  model cannot express — whole-line residency, survival past the reuse distance, interference
  between the streaming and persisting halves of the cache — are named in
  `docs/evaluation.md`, and a cost model carrying them is the highest-value open contribution.
  It needs no GPU.

### Fixed

- **`CudaTransitExecutor::release()` kept the borrowed stream handles**, so the destructor
  probed streams the caller had already destroyed — a segfault inside `libcuda` in the
  teardown path of an optional optimization layer. Found by `tests/test_cuda_executor.cu`
  doing exactly what the header tells a caller to do. `release()` now drops them.
- **The control-contamination guard only knew one environment prefix.** The adapter reads
  `TENSORTRANSIT_X` in preference to `RECURLOCAL_X` as of this release, so a scrub that knew
  only the old prefix would let an operator with `export TENSORTRANSIT=combined` compare the
  candidate against itself and measure ~0% — silently, because a contaminated control produces
  a plausible number rather than an error. `eval/real_eval.py::scrubbed_environment` removes
  both and is asserted directly rather than only through an end-to-end run.
- **CUDA 13 compatibility in the new executor**: `cudaDeviceProp::memoryClockRate` is gone
  (the attribute API is used, and the field stays 0 rather than being fabricated when it
  fails), and `cudaStreamGetCaptureInfo` took the edge-data form.
- **The device profile in `known_device_names()` disagreed with the device.** An RTX 5090
  reports a **96 MiB** L2, not the 128 MiB a spec sheet gives. The table now carries what
  `tensortransit_info` read off the hardware, which is the reason the table exists.
- **Ceiling saturation printed as `+0.000%`.** Over a cyclic window every tensor is re-read
  next iteration, so the unlimited-cache figure saturates and `f/(1-f)` collapses to zero at
  `f = 1` — which reads as "there is nothing here" when the truth is the opposite. It now
  prints `unbounded`, above a 0.99 share, and says why.

### Changed — migration

`include/recurlocal/*.h` are shims over `include/tensortransit/*.h`; `namespace recurlocal` is
an **alias** of `tensortransit` rather than a second set of declarations, so it cannot drift.
`tests/test_compat.cpp` includes **only** the deprecated headers, so the shim breaking is a
build failure here rather than a link failure in somebody else's tree, and CI consumes the
installed package under both names. `src/cuda/` -> `executors/cuda/`, `bench/` ->
`workloads/recurrent/synthetic/`, `integrations/sparkinfer/` -> `adapters/sparkinfer/`,
`tools/capture_attr_probe.cu` -> `profiling/`.

**Two names deliberately do not move**, both for the reproduction path, and
`docs/STABILITY.md` section 6a says so: the `RECURLOCAL_STATS ` stderr line prefix (because
`eval/run_from_base.sh` runs the **base commit's** parser against a candidate's binary, and
renaming it would make every base-commit evaluator accuse a good build of being unhooked) and
the `RECURLOCAL_` environment names (because every command line in `results/*.json` sets them).

### Unchanged

The 0.1 answer, which now applies to `recurrent_v0` rather than to the project: the persist
family's weighted ceiling is **1.94% against a 2.0% floor** on the best model found, and that
model is not scorable because the runtime is not reproducible on it.

## [Unreleased]

### The verdict, and what this repository should become

Three sessions have now asked whether software-directed L2 residency for recurrent state can
produce a scorable end-to-end speedup on reachable hardware. It cannot, and the reason is
arithmetic rather than implementation. The bound is

```
persist ceiling  =  2 x min(persisting capacity, recurrent footprint) / decode step traffic
```

The numerator is pinned at **60 MiB** by the device and cannot be raised. The denominator is
what the workload moves. On the best model this project has found — a sparse-MoE hybrid whose
decode step moves 3.56 GB instead of 18.5 — the weighted ceiling across the section 44 matrix
is **1.94% against the project's own 2.0% significance floor**, with both persistence dials at
maximum and a bound that already assumes every resident byte hits and the set-aside costs its
neighbours nothing. Six hundredths of a point short, and no policy closes it.

This session tried to close the three things standing between that and "production ready".
None of them closed the way it was hoped, and two of them closed *better* than hoped by being
answered rather than removed:

1. **The MoE result is not scorable, and now that is a surveyed fact rather than an open
   question.** Four checkpoints screened on hardware; the runtime accepts only six expert
   tensor types, which rules out every MXFP4 and IQ variant at load; the two files that would
   be genuinely different quantizations exceed 32 GB of VRAM; and eight environment
   configurations on the checkpoint that does load — including one that replaces the batched
   prefill the runtime's own header blames, and a pin of every split-K, PDL and n-splits
   switch at once — leave it forking on every replay. **The dense control is not clean either**;
   it is reproducible at the gate's 64 tokens and forks by 256. There is no reproducible
   sparse-MoE hybrid on this runtime and this device, and the only surface where the persist
   family pays is a sparse-MoE hybrid.
2. **The undocumented mechanism was documented all along.** `capture_node` is sanctioned by
   the CUDA header this project was reading around, has been since 11.3, and a probe now reads
   the attribute back off the finished graph at every node count from 1 to 128,
   memcheck-clean. That blocker was a misreading of a header, and correcting it costs the
   project a caveat it had been carrying for three releases.
3. **The device matrix and the stability contract are done, and they found seven real
   defects** — including a rationing policy that over-admitted by up to 6.5x, an overflow that
   turned it into its opposite, and a capability field the planner queried and never read.
   Those are real, and none of them moves the ceiling.

And the largest effects measured this session are **not this library's**. SparkInfer's
`launch_mmvq_rows` cliff is worth 2.7x of aggregate throughput on the checkpoint that matters
(`docs/UPSTREAM-SPARKINFER-MMVQ.md`), and a dense 32-sequence run intermittently loses a third
of its wall time to something that is *not* its decode path, not request loss, and not the
per-row projection fallback — per-token latency is identical through it and every request
completes. That one is narrowed on three sides and still open; a separate and more severe
failure found while looking for it — per-request device OOM, an 85% drop — is identified and
now guarded. Against those, the best thing RecurLocal does is **+1.74% at
batch 1 on one unscorable checkpoint**, and the best thing this session added to it is
**+0.1 to +0.2 points over a constant**.

**What the repository should become.** Not a locality library that is waiting for a bigger
persisting cache. Its most valuable output has consistently been the *instrument*, not the
policy: the traffic-budget calculator that says "do not start" before anyone spends a week;
the null-candidate guard that caught this project's own first result measuring its own
overhead; the packed-path guard; the control-self-replay guard, which this session had to
strengthen because one agreeing pair had certified a runtime that forks half its replays; and
the checkpoint reproducibility screen, which found that a runtime's own determinism header
claims 36-of-36 bit-identical runs on a model where the truth is 0 of 4. Every one of those
found something. The library found 1.4% on a checkpoint that cannot be scored.

The recommendation is to keep the enumerators and the measured results exactly as they are —
they are the demonstration that the instrument works — and to describe the project as what it
has turned out to be: **a locality-measurement harness for recurrent-state inference, with a
reference policy that is honestly bounded below its own significance floor on every device
this project can reach.** Say the 1.94% in the README's first paragraph rather than in section
44. A reader deciding whether to spend a week here should learn the ceiling before they learn
the mechanism.


### Added, and measured — the set-aside is sized from the workload, not from a constant

`SetAsidePolicy`. Every number this repository has published reserved
`persisting_budget_fraction` of the device's persisting-L2 capacity — a constant chosen before
anything is known about the workload — and `results/rtx5090-moe-scored.json` already showed
that constant is wrong in both directions: `budget_fraction 1.00` is the best setting at batch 1
on the sparse-MoE checkpoint (+1.63% against +1.26%) and the worst at four concurrent sequences
(-1.29% against -0.46%). The inputs to do better were already there — the runtime declares its
sequence count and the hot-set model already computes the footprint — and only the policy was
missing.

Three values, `Fixed` the control:

- **`Fixed`** — `persisting_budget_fraction` of capacity, whatever the workload. Exactly what
  every prior number was measured under, and still the default.
- **`FitFootprint`** — `min(footprint, capacity)`. Never reserve more than the footprint can
  use: holding 60 MiB for a 10 MiB working set takes 50 MiB from the cache the rest of the step
  streams through and buys nothing with it. Parameter-free.
- **`Residency`** — `FitFootprint`, plus decline outright when the fraction of the footprint
  the capacity could hold falls below `min_residency`.

`persisting_budget_fraction` is deliberately **not** applied under the two workload-aware
values. Turning both dials would leave them unable to reach the setting the workload wants
without the caller also changing the constant they exist to replace, which is the whole defect.

**The threshold is bracketed by measurement, not fitted.** The persist family is measured to
pay at a resident fraction of 0.977 (MoE, batch 1) and not to pay at 0.478 and below (the same
model at four sequences, -0.46%, inside its own 0.46% floor). Anything in that interval orders
the evidence correctly; 0.50 sits at its lower edge, so the rule gives up as little as the
evidence allows. The cost is stated rather than hidden: the **dense** checkpoint at batch 1
sits at 0.409 and measures a resolved +0.10%, so `Residency` declines a real if tiny gain
there. `--axis min-residency` is how that prediction gets falsified.

**Measured, batch 1 on the MoE checkpoint**, control 503.4 tok/s, three interleaved pairs,
noise floor **0.071%**:

| `set_aside_policy` | set-aside asked for | gain |
|---|--:|--:|
| `fixed` (the shipped constant, at the shipped 0.75) | 45 MiB | +1.285% |
| **`fit_footprint`** | **60 MiB** | **+1.509%** |
| `residency` (threshold 0.50, footprint holds 0.977) | 60 MiB | +1.441% |

Axis spread **0.224%** against a 0.071% floor — **resolved**. Sizing the reservation from the
footprint is worth **0.22 points over the shipped constant with no dial touched by the
operator**, and it reaches the +1.5% that previously required someone to know to set
`RECURLOCAL_BUDGET_FRACTION=1.00` by hand. `fit_footprint` and `residency` differ by 0.069
points, inside the floor, which is the consistency check this measurement had to pass: at a
resident fraction of 0.977 the two rules compute the *same* reservation, so any difference
between them is noise and it is.

**Three independent runs of that sweep, across two rebuilds. All three resolve, and all three
put the shipped constant last:**

| run | `fixed` | `fit_footprint` | `residency` | axis spread | floor |
|---|--:|--:|--:|--:|--:|
| 1 | +1.285% | **+1.509%** | +1.441% | 0.224% | 0.071% |
| 2 | +1.32% | +1.37% | **+1.41%** | 0.100% | 0.043% |
| 3 | +1.30% | +1.40% | **+1.46%** | 0.160% | 0.077% |

**The advantage is real and replicated; its size is bounded rather than pinned.** Against
`fixed`, `residency` is +0.11 to +0.16 points and `fit_footprint` +0.09 to +0.22, in every run,
with no dial touched by the operator. But the between-run drift on a single configuration
(`fit_footprint` moved 0.14 points between runs 1 and 3) is larger than any one run's noise
floor, so quote the range and not the best of the three. Only same-run interleaved deltas are
trustworthy here, which is what this repository has said about this box all along; three runs
is what makes the ordering a result rather than one of them.

**At concurrency the axis does not resolve, and the one run that did resolve did not
replicate.** Four sequences, `SPARKINFER_PACKED_MAX_ROWS=8` on both arms, one warm-up
discarded, three interleaved pairs, three independent runs across two rebuilds:

| run | reservation `fit_footprint` took | `fixed` | `fit_footprint` | axis spread | floor | verdict |
|---|---|--:|--:|--:|--:|---|
| 1 | 60 MiB (sized from the TOTAL footprint) | -0.07% | **-0.99%** | 0.92% | 0.41% | resolved, `fixed` |
| 2 | 31 / 60 MiB alternating | +0.07% | -0.07% | 0.15% | 0.41% | **open** |
| 3 | 60 MiB (latched) | -0.18% | -0.28% | 0.09% | 0.13% | **open** |

Runs 1 and 3 held the same 60 MiB reservation and measured -0.99% and -0.28%. The between-run
spread on one configuration is therefore about 0.7 points while each run's own noise floor is
0.13-0.41% — so **the concurrency arm on this checkpoint is not reproducible across rebuilds**,
and the -0.99% that motivated half of this design is one unreplicated observation. It is
reported here rather than quietly dropped because it is what the design was built on.

The honest statement of what this mechanism is worth:

| arm | replication | result |
|---|---|---|
| batch 1 | 3 runs, all resolved, same ordering | workload-aware beats the constant by **+0.05 to +0.22 points** |
| concurrency 4 | 3 runs, 1 resolved and not replicated | **indistinguishable** from the constant |

`Residency` declines the reservation outright at four sequences (resident fraction 0.478,
below its 0.50 threshold), and `real_eval.py`'s null-candidate guard refuses an arm that
installed no policy, **by name** — correctly, because a policy whose response to pressure is
to do nothing *is* `baseline`. So its value on that arm is `baseline` by construction, and the
guard's refusal is the observation that it declined.

**The marginal value of set-aside at concurrency, on one dial, for the first time.** Every
prior comparison at concurrency moved TWO dials at once (`budget_fraction` 0.75 → 1.00 *and*
`hit_ratio` 0.70 → 1.00), so no marginal value could be extracted from it. `--axis
budget-fraction --concurrency 4`, four sequences, three interleaved pairs:

The **magnitude does not resolve**. One of the three control runs came in 14% slow (467.8 tok/s
against 546.3 and 544.0, with inter-token latency 4.54 ms against 3.89 — slower per token, not
fewer tokens, so a different failure from the OOM above), which puts the arm's own noise floor
at **14.4%** and swallows an axis spread of 1.18%. `real_sweep.py` refuses to name a winner,
correctly.

**The ordering does resolve, and it is the one the design predicts.** Within each pair the four
candidates share a control, so their ordering is a clean comparison even when the magnitude is
not:

| pair | 0.25 | 0.50 | 0.75 | 1.00 | worst |
|--:|--:|--:|--:|--:|---|
| 1 | 1.16588 | **1.16674** | 1.16460 | 1.15733 | **1.00** |
| 2 | **0.99634** | 0.99231 | 0.99249 | 0.98957 | **1.00** |
| 3 | 0.99412 | **1.00533** | 0.99540 | 0.99357 | **1.00** |

**The largest set-aside is last in all three pairs.** If a MiB of set-aside cost nothing at
concurrency, one specific value landing last three times running is a 1-in-64 coincidence.
That is a sign test, not a magnitude, and it is the appropriate strength of claim for this
arm — but it is the first direct evidence that the *cost* side of the trade is real, as
opposed to being inferred from two configurations that differed on two dials.

**One implementation fact that decides how much of this is even reachable.** A persisting
window is an address range and this library places ONE per layer, over one sequence's slice —
at concurrency the runtime hands over a device array of per-row pointers for the pre-touch and
a single host-nameable row for the window. So set-aside beyond one sequence's footprint holds
nothing. `FitFootprint` and `Residency` now size from that *windowed* footprint rather than
from the token footprint, which is what competes for the cache and counts every sequence.
Conflating the two was a real defect: it reserved the whole 60 MiB at every concurrency, which
is the setting run 1 measured worst.

Three integration defects were fixed alongside it, each of which would have made a concurrency
measurement meaningless:

- The adapter declared `sequences = 1` to `declare_geometry()` on the packed path.
  `begin_token_packed` borrows `begin_token` for its stream setup and only *afterwards*
  overwrites the sequence count, so every workload-aware rule sized the reservation for a
  workload that was not running.
- The controller compared a fresh request against the bytes the driver **granted**. The driver
  rounds up, so any target that is not exactly the device maximum never matched and the L2
  partition was re-carved on every token.
- A runtime does not declare the same geometry on every token: SparkInfer packs most of them
  and runs the tail unpacked, and the packed path compacts the matrix state to bf16, so the
  declared bytes per layer **alternate** — 64 of 80 tokens packed at four sequences. Acting on
  each declaration re-carved the partition several times a second, evicting on every change
  exactly the state it had just kept. A target now has to hold for eight consecutive
  declarations. That is "first stable geometry wins", not "the majority wins", and the
  distinction is documented because it is visible in the telemetry: at four sequences the
  reservation latches during the opening unpacked tokens and holds there for the rest of the
  run.
- The telemetry reported the set-aside taken at `initialize()`, so every resize was invisible
  to every sweep; and reporting the value *in force* instead reports 0, because shutdown has
  already given the partition back by the time an `atexit` handler runs. It now reports the
  peak held during the run, and the widest geometry declared rather than the last one.

`results/rtx5090-setaside.json` carries the data. Registered on `eval/real_sweep.py` as
`--axis set-aside-policy` and `--axis min-residency`, on `eval/sweep.py`, and on the synthetic
benchmark as `--set-aside-policy`.

The set-aside is requested once at `initialize()`, before any workload is known, so the
controller gained `declare_geometry()`: it recomputes the reservation when the geometry
changes, is a no-op under `Fixed`, a no-op when the answer has not moved, and a no-op while
the compute stream is capturing — `cudaDeviceSetLimit` is a device-wide operation and a graph
capture is not the place for one.

### Corrected — `capture_node` is documented as safe, and this repository said otherwise

Every persist number here comes from `WindowAttach::CaptureNode`, which sets an access-policy
attribute on a kernel node of a graph that is **still being captured**. Four files asserted
that CUDA does not sanction that. CUDA's own header, on the very call the mechanism is built
out of, says the opposite:

```
 * \param graph_out - Optional location to return the graph being captured into. All
 *           operations other than destroy and node removal are permitted on the graph
 *           while the capture sequence is in progress.
 ...
 *           The node handles may be copied out and are valid until they or the graph is
 *           destroyed. The driver-owned array may also be passed directly to APIs that
 *           operate on the graph (not the stream) without copying.
        -- /usr/local/cuda/include/cuda_runtime_api.h:2743 and :2755, CUDA 13.3;
           identically cuda.h:16885, and unchanged since CUDA 11.3
```

Setting a kernel-node attribute is neither destroying the graph nor removing a node, and
`cudaGraphKernelNodeSetAttribute` is an API that operates on the graph.
`cudaLaunchAttributeAccessPolicyWindow` is declared "Valid for streams, graph nodes, launches"
(`driver_types.h:4017`). Neither setter carries a capture-related caveat or error code. The
same sentence is on NVIDIA's live documentation site and in the 11.3 archive, and the
Programming Guide's "Prohibited and Unhandled Operations" list contains no entry for it —
every item there is a stream-side operation.

**And nothing had ever checked that the attribute survives.** `windows_attached_to_node`
counts attach CALLS that marked at least one node; "48 of 48 nodes" was read off a counter
that was counting calls. `grep` for `cudaGraphGetNodes`, `cudaGraphKernelNodeGetAttribute` or
`cudaGraphExecGetNodes` over the whole repository returned nothing.

`tools/capture_attr_probe.cu` closes it. It captures N kernels, sets the window mid-capture
exactly as `CudaLocalityController::attach_window_to_captured_node` does, ends the capture,
and reads the attribute back off the finished graph and off a clone:

| nodes | attribute set | capture invalidated | present on the finished graph | matches what was set | clone carries it |
|--:|--:|--:|--:|--:|--:|
| 1, 4, 8, 16, 32, 48, 64, 128 | N of N | never | **N of N** | **N of N** | **N of N** |

`compute-sanitizer --tool memcheck` reports **0 errors** at 48 nodes, and `racecheck` likewise.

The instantiated graph cannot be inspected — CUDA 13.3 has no `cudaGraphExecGetNodes` and no
exec-level attribute getter, so the last link is necessarily behavioural. The same capture
attached with a **persisting** window and with a **streaming** window over the same buffer,
timed against each other: at 48 nodes, SparkInfer's own batch-1 decode capture size,
**+3.00% and +2.86%** on two independent runs. A policy absent from the replay cannot do that.
At 8 and 16 nodes the microbenchmark does not resolve — that is its sensitivity, not the
mechanism's, and the API half passes at every node count.

**What survives is a precision defect, not a legality one.** `CaptureNode` marks *every*
kernel node in the capture's pending dependency set, and that set is not always one node: on
the non-fused convolution branch the launch immediately before the hook is
`l2_norm_qk_kernel`, which reads no convolution state and inherits a persisting window over
memory it never touches.

### Added — `WindowAttach::CaptureNodeStrict`, and two counters that were one

`CaptureNodeStrict` marks the node only when the capture has exactly one kernel node pending
— when the node it is about to mark is unambiguously the kernel the hook fired for — and
counts the rest in `window_attach_ambiguous`. `windows_attached_to_node` (attach calls) and
`window_nodes_attached` (nodes) are now separate, so the number that was misread cannot be
misread again. Registered on `--axis window-attach`.

**Measured, and declining the ambiguous attaches is free.** Batch 1 on the MoE checkpoint,
three interleaved pairs, control 503.1 tok/s, noise floor 0.053%: `capture_node` +1.268%,
`capture_node_strict` +1.320%, axis spread 0.053% — **inside the floor, so OPEN and no winner
named**. `window_attach_ambiguous` was 0 throughout, meaning on this model the two modes made
the identical choice on every attach, which is why they measure the same. The mode exists for
the case where they do not, and the counter is how a run says which case it was in.

`Stream` stays the default, but for a smaller reason than before: not that node attachment is
illegitimate, but that it is a device the host has not asked for.

**And one correction that cuts against the library.** Under `Stream` this integration delivers
nothing at all. The controller hands the window back in `LayerActions`; the SparkInfer adapter
keeps the boolean and drops `actions.window` on the floor, and no adapter entry point returns
it. `Stream`'s measured -0.019% is the hook's overhead, not a policy. The sanctioned
launch-site path (`cudaLaunchKernelEx` with `cudaLaunchAttributeAccessPolicyWindow`) would
need that accessor before it could be built: 5 launchers, 8 triple-chevron sites and 5 runtime
call sites in the pinned SparkInfer tree, about 130 lines across 4 files, all of it inside the
kernel library the hook was designed to stay out of. A cheaper sanctioned route exists and is
recorded in `docs/DESIGN.md`: copy the node handles out during capture — which the same header
paragraph explicitly blesses — and set the attributes after `cudaStreamEndCapture`, which is
three inserted lines in two files because SparkInfer's `EndCapture` and `Instantiate` are
adjacent statements. Attaching to an already-instantiated graph is **not** available at all:
CUDA 13.3 has no exec-level attribute setter and `cudaGraphExecUpdate` rejects attribute
changes outright.

### Found — a failure mode at concurrency 32 that every guard here passed, and it is not the collapse

Looking for the dense model's intermittent 32-sequence collapse turned up something else
first, and it is worth having on its own. Eight identical isolated runs of
`qwen3_gguf_cb_bench` on Qwen3.8-27B at 32 sequences, full output captured:

| rep | decode tokens | wall | aggregate | mean ITL | `out of memory` messages |
|--:|--:|--:|--:|--:|--:|
| 1 | **320** | 2.239 s | 142.9 tok/s | 20.77 ms | **48** |
| 2 | **384** | 2.330 s | 164.8 tok/s | 24.26 ms | **47** |
| 3-8 | 2056 | 2.215-2.235 s | 919.8-928.4 tok/s | 19.25-20.54 ms | 0 |

```
[qwen35] malloc: out of memory
[warn] request error: device out of memory (not a capacity/queue condition -- requires operator attention)
```

Per-request device-memory allocation fails, the runtime reports it as a per-request warning and
carries on. **The wall time is unchanged** — 2.24 s against a healthy 2.22 s — so the arm did
not run slowly; it ran *less*, and `agg_tok_s` is tokens over wall time. A run that lost four
fifths of its requests reports an 85% throughput collapse that is entirely a failure.

Every guard in this harness passed it: the hook ran, the packed path was used, `agg_tok_s`
parsed. **`real_eval.py` now refuses such an arm by name** —
`require_requests_completed` compares the decoded token count against what the workload asked
for, refuses below 90%, and quotes the count and the out-of-memory message count. The lesson
generalises past this bug: a throughput harness that reads only a rate cannot tell a slow run
from a short one, and this repository had been averaging short runs into control/candidate
ratios for three releases.

**This is NOT the collapse this repository has been carrying**, and saying so matters because
the two look alike in a summary table. The historical evidence is `prefetch` ratios
`[0.932, 0.676, 0.925]` — a **32%** drop. This failure is an **85%** drop. The ratio that
matches 0.676 is the one below, which has a full token count.

### Still open — the 32% collapse itself, now bounded on three sides

Two consecutive identical isolated runs, earlier in the same session:

| | run 1 | run 2 |
|---|--:|--:|
| aggregate | **910.9 tok/s** | **589.9 tok/s** (ratio **0.648**) |
| decode tokens | 2056 | **2056** |
| mean inter-token latency | 19.32 ms | **19.31 ms** |
| wall | 2.257 s | **3.486 s** |

Full token count, identical per-token latency, and 1.2 seconds of extra wall time. That ratio
of 0.648 is the historical `0.676`, and **none of the three obvious explanations survives**:

- **Not request loss.** Both runs completed all 2056 tokens; the new guard would not fire.
- **Not a decode-path fallback.** A run that had dropped onto per-row decode would show it in
  the inter-token latency, and per-token decode is identical to two decimal places.
- **Not the NVFP4 per-row projection loop.** That silent third tier exists
  (`qwen35_prefill.cpp:3202-3216`: the dp4a rows kernel which chunks, then
  `launch_gemv_nvfp4_rows` which refuses `M > 8`, then a bare per-row loop that returns
  success), and `SPARKINFER_QWEN38_NVFP4_DP4A_PROJ=0` forces it permanently. Forcing it costs
  about 19% at 8 sequences and 7% at 32 and is **stable** — 847.5 and 846.4 tok/s, within
  0.1% — where the collapse is 35% and intermittent. Forcing the suspect *removes* the
  variance rather than reproducing it.

It also did not reproduce in the eight-run set above, so it is rarer than one in eight. What is
left is 1.2 seconds of wall time, outside the per-token decode loop, on a run that completed
every request — the long-prefill phase, the scheduler, or a stall the timeline would show. That
is a much narrower question than "the runtime falls off its batched decode path", which is what
this repository used to say, and `docs/OPTIMIZATION-SURFACES.md` no longer says it. The
measurement that would close it is a timeline profile, and see the section below for why that
cannot be taken on this box.

### Blocked — the spec's chain of evidence cannot be closed on this box

Section 20 of the overview asks for HBM read and write traffic, L2 hit rate, L2 sectors and
DRAM throughput as the middle link of a five-stage chain of evidence. **Nothing in this
repository has ever produced one of those numbers**; every locality claim here is inferred
from end-to-end throughput, and that is the weakest link in the chain.

Nsight Compute 2026.2.1.0 is installed at `/usr/local/cuda/bin/ncu` and is **unusable on this
box**. `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`, and every on-device
query returns `ERR_NVGPUCTRPERM` — including as root, because the restriction is a driver
module parameter set on the host, not a permission inside the container.

**The block is not specific to `ncu`, and that is worth checking before someone reaches for the
obvious alternative.** Nsight Systems 2026.1.3 is also installed, and
`nsys profile --gpu-metrics-devices=help` answers:

```
GPU Metrics: None of the installed GPUs are supported:
    Blackwell GB202 | NVIDIA GeForce RTX 5090 - Insufficient privilege, see ERR_NVGPUCTRPERM
```

`RmProfilingAdminOnly` gates the driver's hardware performance counters, so it takes `ncu`,
`nsys --gpu-metrics` and CUPTI's profiling API together. What survives is counter-free tracing —
`nsys` kernel and API timelines, CUPTI activity records — which can time a stall but cannot
report a byte of DRAM traffic. Item E is blocked before any command can be written, and no
amount of work inside this environment moves it.

Two things worth recording for whoever has a box where it is not blocked:

- **GB202 exposes eviction-class-tagged L2 counters** —
  `lts__t_sectors_aperture_device_evict_last*`, `_evict_first*`, `_evict_normal*`, 528
  sector-level variants. Those would prove *directly*, and immune to cache flushing, that a
  persisting window reached the hardware with the `hitRatio` that was asked for. No throughput
  number in this repository can make that claim, and `tools/capture_attr_probe.cu` can only
  make it about the graph, not about the cache.
- **And there is a limit no permission fixes.** `--graph-profiling node` (per-kernel
  attribution) and `--graph-profiling graph` (cross-launch cache residency) are mutually
  exclusive in ncu. The recurrent state written at layer *i* is re-read at layer *i* of the
  **next token**, so the effect this project is about lives *between* kernel launches. The
  counters could validate the traffic denominator every ceiling here rests on, and could prove
  the policy was delivered — but they can never, in one measurement, show "these kernels' DRAM
  bytes fell because the window kept their state resident". That is a property of the
  instrument, and it should be stated before someone spends a week on it expecting otherwise.

### Measured — no reproducible sparse-MoE hybrid checkpoint fits this device

The MoE result was unscorable because two unhooked control runs diverge. The task was to find
a checkpoint that does not. Four were screened on hardware, and the answer is no.

| checkpoint | size | loads on the pinned runtime | two unhooked greedy replays agree |
|---|--:|---|---|
| `Qwen3.6-35B-A3B-UD-Q4_K_M` | 22.1 GB | yes | **no** — 4 of 4 replays distinct |
| `Qwen3.6-35B-A3B-MXFP4_MOE` | 21.7 GB | **no** — `unsupported ggml type 39 for blk.0.ffn_gate_exps.weight` | — |
| `Qwen3.6-35B-A3B-UD-Q5_K_M` | 26.5 GB | loads, then emits `0 0 0 0 0 0 0 0` — `layer 34 expert qtypes 13/13/8 unsupported -> token loop` | unusable |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL` | 22.4 GB | yes | **no** — 4 of 4 distinct, with and without `SPARKINFER_DETERMINISTIC=1` |
| `Qwen3.6-35B-A3B-Q8_0` | 36.9 GB | exceeds 32 GB of VRAM | — |
| every `IQ*`, `Q2_K`, `Q3_K` variant | — | **no** — `ggml_dequant_supported` accepts only F32, F16, Q8_0, Q4_K, Q5_K, Q6_K (`qwen35.cpp:114`) | — |
| `BF16` | ~70 GB | exceeds 32 GB of VRAM | — |

**It is the runtime, not the checkpoint.** Six configurations were tried on the Q4_K_M
checkpoint and none is reproducible:

| configuration | replays |
|---|---|
| default | distinct |
| `SPARKINFER_DETERMINISTIC=1` | distinct |
| `SPARKINFER_DETERMINISTIC=1` + `SPARKINFER_PREFILL_BATCHED=0` (the token-loop prefill) | distinct |
| `SPARKINFER_PREFILL_BATCHED=0` alone | distinct |
| `SPARKINFER_DETERMINISTIC=1` + `SPARKINFER_PREFILL_MOE_QB=0` | distinct |
| `SPARKINFER_DETERMINISTIC=1` + `SPARKINFER_PREFILL_LEGACY=1` | distinct |
| `SPARKINFER_MUSE_FFN_Q3A=0` (± `DETERMINISTIC=1`) | distinct |
| every split-K, PDL and n-splits switch pinned at once | distinct |

The runtime's own `kernels/include/sparkinfer/kernels/deterministic.h` claims 36 of 36 runs of
**this exact model** are bit-identical with the mode on. They are not. What the mode *does*
fix is narrower and worth reporting upstream: with every split switch pinned, the prefill seed
token becomes stable — five single-token replays of a 256-token prompt returned `25001` five
times where the unpinned run returned `8894, 8894, 25001, 25001, 8894`. A 64-token generation
from the same prompt still forks in every replay.

**What that does and does not establish.** It establishes that the residual source survives
every switch the runtime exposes, and that it is not the two `atomicAdd` sites
`deterministic.h` names. It does **not** establish that the source is in decode. A ULP
difference in prefill that leaves the seed argmax unchanged while perturbing the Gated-DeltaNet
recurrent state or the bf16 KV writes produces exactly this observation — a stable first token
and a forked continuation — and nothing measured here separates that from a decode-side
source. Distinguishing them needs a logits or state dump, which `qwen3_gguf_generate` does not
emit. What can be said is that `deterministic_mode()` has no reach into the per-token path at
all: its only decode call site (`qwen35.cpp:880`) recomputes a logprob normaliser on the host
after the argmax has already been taken.

**The dense control is not clean either, and this repository said it was.** "On the dense
checkpoint the same binary is bit-identical" is true at the gate's 64 tokens and false past
them. Four unhooked replays of Qwen3.8-27B from the same prompt, by generation length:

| tokens generated | distinct outputs of 4 replays |
|--:|--:|
| 64 | **1** — the length the gate runs at |
| 256 | 2 |
| 512 | 1 |
| 1024 | 3 |

The difference between the two models is amplification, not presence: a few ULP stay numerical
on a dense FFN and flip a discrete top-8 expert choice on a sparse one, so the MoE forks by
token 2 and the dense model needs a few hundred. The dense checkpoint is not a reproducible
runtime either; it is a runtime whose divergence the 64-token gate is too short to see. That
is the strongest argument for replaying the control more than once, and it is why
`--gate-control-replays` defaults to 3 rather than 2.

**The conclusion for the exact-locality gate.** It requires a reproducible runtime. On this
runtime, on this device, there is no sparse-MoE hybrid checkpoint that provides one — and the
only surface where the persist family pays is a sparse-MoE hybrid. The MoE throughput numbers
stand as throughput and remain unscorable, and that is now a surveyed result rather than an
open question.

### Fixed — seven planner defects a device-capability matrix finds and one device never could

RecurLocal has run on exactly one device: sm_120, one RTX 5090, 60 MiB of persisting L2 out of
96, `accessPolicyMaxWindowSize` 128 MiB — a cap that has never once bound. `DeviceCaps` is
queried rather than hardcoded, but nothing had ever asked what the policies do when the
numbers come back different. Fabricating the caps needs no GPU: `DeviceCaps` is a plain struct
and `LocalityPlanner` is arithmetic. Doing it found seven defects, two of them reachable on the
RTX 5090 itself.

1. **`HotSetPolicy::Quota` over-admitted by up to 6.5x.** Its even-spread pattern
   `(i * admissible) % units` selects exactly `admissible` of `units` ordinals — over ONE
   period of `units`. Where the window spans more than one layer's slice, `units` falls below
   the number of recurrent layers the runtime walks, the pattern repeats, and the policy
   admits a multiple of the budget. `WindowScope::Ahead` produces exactly that today, from the
   shipped integration, through an environment variable. The period is now the recurrent-layer
   count the geometry declares — falling back to the byte-derived count, which is right
   whenever the window is one layer's slice — and the ordinal is reduced modulo it. The
   admitted count is now `budget / window` (whole windows the set-aside holds) rather than an
   ordinal share of the hot set; the two are the same number only while the window is one
   slice, and where they differ the share-derived count spends `admitted x window` bytes
   against a budget sized for something smaller.
2. **A saturated hot set made `Quota` decline everything.** `hot + window - 1` is unchecked, and
   `hot` saturates to `SIZE_MAX` by design — "a wrong answer here must never be a small one".
   The addition wrapped, `units` became 0, and the largest representable hot set produced the
   smallest possible admission: Quota silently became Cliff. Written as a division plus a
   remainder, it does not.
3. **`Quota` with no layer ordinal was `Fixed` and denied it.** Both public `plan_for_layer`
   overloads default `layer_index = -1`; ordinal 0 is always admitted, so every layer of an
   unindexed caller got a full-hit-ratio window with `hit_ratio_reduced = false` — the naive
   control, under the rationing policy's name, with the telemetry saying nothing had backed
   off. `LayerPlan::hot_set_policy` now reports the policy that was **applied**, in the idiom
   `hot_set_model` already uses, and a caller who cannot be rationed is told so.
4. **The `min_hit_ratio` floor could ask for more residency than the set-aside holds.** The
   header documents it as "below this a window is not worth asking for"; the code implemented
   it as a `std::clamp` lower bound that raises the request back up and installs the window
   anyway. On a device whose persisting L2 is smaller than one layer's state that asked for
   **77x** the reservation. The request is now capped at what the reservation can physically
   hold. It does not bind on any arm this repository has measured — the RTX 5090's budget is
   22 windows wide — so no published number moves.
5. **`hot_set_oversubscribed` gave opposite answers for identical outcomes.** For the same
   148 MiB footprint it read **false** on a device with no persisting L2 — the maximally
   oversubscribed case — and **true** on one that merely refused the access-policy window. The
   guard was `budget > 0 &&`. The comment above it already said these numbers describe the
   workload rather than the policy; now they do. *Telemetry contract change*: a zero-budget arm
   that used to emit `hot_set_oversubscribed: 0` now emits the truth, so that field is not
   comparable across this change on such an arm.
6. **`recommended_l2_set_aside()` was undefined behaviour at `SIZE_MAX` capacity.**
   `static_cast<std::size_t>` of a double that rounds to 2^64 is UB; the observed answer was
   **0**, i.e. persist silently off on the most capable device the type can express. Real
   hardware cannot reach it — `cudaDeviceProp` reports an `int` — but a fabricated `DeviceCaps`
   can, and a policy that is undefined on an input a test can construct is a policy nobody can
   check.
7. **`DeviceCaps::l2_bytes` was queried, printed, and read by nothing.** Zero policy sites
   consumed it, so a device whose two capacity numbers disagree — an emulator, a MIG slice, a
   stubbed query — got a request for a set-aside larger than its entire cache with nothing to
   say so. A set-aside is carved out of L2 and is now clamped to it when it is reported.

Not defects, checked and cleared: `saturating_mul`/`saturating_add` are arithmetically exact,
so the footprint product does **not** overflow — the suspected multiplication bug was one line
downstream, in Quota's unit count. The window **is** clamped to `access_policy_max_window_bytes`
and the clamp survives to the driver. `hit_ratio` can never exceed 1.0 or go negative.

The tests fabricate an A100-like device (40 MiB L2, 30 MiB persisting), an H100-like one
(50/40), a consumer part whose persisting L2 is smaller than one layer's state, a device with
more persisting L2 than the whole footprint, one that reports zero, and one whose numbers are
incoherent. They land either side of the residency threshold and the arithmetic says which:
the batch-1 MoE footprint is 61.4 MiB, so the A100's 30 MiB holds 0.489 of it and is declined
while the H100's 40 MiB holds 0.651 and is admitted. 296 planner checks, up from 178.

### Added — an API and ABI stability contract, and the code changes that make it true

`docs/STABILITY.md`. The library is embedded by inference runtimes and its contract is a set
of enumerator values other people switch on; there was no statement of what survives a version
bump. The document says, per symbol class, what a consumer may rely on — enumerator names,
enumerator values, and the strings they round-trip through, since those strings are the
sweep-axis and environment-variable vocabulary that every result file in `results/` names —
what it may not, and, in its last section, **where the contract is only advice rather than
something the build enforces**.

The checkable claims were made checkable:

- Every public enum now says `: int` outright. A scoped enumeration already has that
  underlying type by the language rule, so this changes nothing the compiler does and turns
  the guarantee into a line anyone can grep and a future edit cannot silently undo. The
  suspected "adding an enumerator changes `sizeof`" defect is **not present**.
- `static_assert`s in `planner.h` pin the size of all six by-value structs and the offsets of
  three fields. `PlannerConfig` has interior padding at offsets 4, 52 and 92, so a four-byte
  enum dropped into a hole would change no size at all and move no offset — which is why the
  rule is "append after the last member" rather than "keep `sizeof` stable", and why offsets
  are asserted and not only sizes. These fire in a **consumer's** build, not only in ours.
- The two counters added this release were appended to the end of `ControllerStats` rather
  than grouped by meaning in the middle, which is where they were first written — that had
  moved `sizeof` from 128 to 144 and shifted four offsets, with nothing in the build to notice.
- `RECURLOCAL_VERSION_AT_LEAST(maj, min, pat)`, so a consumer can guard a new enumerator.
- `to_string(SetAsidePolicy)` returned `"fixed"` — the **control arm's own name** — for a value
  outside the enumeration, where every other `to_string` returns `"unknown"`. It would have put
  the control's label on a candidate nobody could identify.
- `include/recurlocal/sparkinfer.h` declared `GdnStateLayout` and `GdnPackedLayout` twice, with
  `cudaStream_t compute` in the CUDA branch and `void* compute` in the fallback: two different
  types with the same name, an ODR violation that nothing would diagnose if a CUDA-built and a
  non-CUDA-built translation unit were linked together. Both branches now name the same type.

What the document does **not** claim: there is no shared library, so no `SOVERSION` is promised
(and the `VERSION` property is set on a target CMake ignores it for). `PlannerConfig` is passed
**by value** into three entry points, so adding a field is an ABI break; the answer is not a
`cbSize` field but a narrow one — build from source, in the same build as the runtime that
embeds you, which is exactly what `integrations/sparkinfer/build.sh` already does. Section 9
says plainly that today that is a convention rather than something the linker enforces, and
names the inline-namespace ABI tag that would make it enforceable.

### Fixed — one agreeing control replay is a sample, not evidence

`real_eval.py` replayed the control against itself **once** and called the runtime reproducible
if that pair agreed. The measurement above shows why that is not enough: at a 256-token prompt
the unhooked runtime returned `8894, 8894, 25001, 25001, 8894`, so a single pair drawn from
that agrees more often than not, and the gate would have certified a runtime that is not
reproducible at all. `--gate-control-replays` (default 3, floor 2) replays until one disagrees
or all agree, and `runtime_reproducible` now requires **all** of them. The result records how
many replays it took to find out — a 2 means the runtime forked immediately, a 5 means four
agreed first, which is exactly the case the old check would have passed.


### Measured — the MoE thesis, tested on hardware, and the batch-1 bound moves

The open problem this repository handed over was whether *a model with less weight traffic per
token* moves the terms the persist bound is made of. It does, by more than the residency ratios
alone suggest.

The bound is `2 x min(capacity, footprint) / step_traffic`. The capacity is the device's 60 MiB
and cannot be raised, so the only lever is the denominator — and `eval/traffic_budget.py` now
inverts the bound at the significance floor and prints the threshold outright: **a decode step
must move at most 6.42 GB** before a persisting window over this footprint reaches 2% at all.
Qwen3.8-27B moves 18.5 GB.

**Qwen3.6-35B-A3B** is the same architecture family — Gated DeltaNet plus full attention, the
same pinned SparkInfer commit, the same hook, the same box — with a sparse MoE FFN reading 8 of
256 experts per token. Nothing about the integration changed: the adapter reads the state
geometry from the runtime's config, so it brackets a 30-layer 2 MiB state as readily as a
48-layer 3 MiB one (`recurrent_layers: 30, bytes_per_layer: 2146304` in its own telemetry, which
is `32 x 128 x 128 x 4 + 3 x 8192 x 2` exactly). The geometry was read from the checkpoint's
metadata and tensor shapes, not from the runtime's defaults, and both formulas reproduce the
pinned model's numbers.

Two terms move at once, which is the part the residency ratios do not show:

| batch 1 | Qwen3.8-27B (dense) | Qwen3.6-35B-A3B (sparse MoE) |
|---|--:|--:|
| recurrent footprint | 146.8 MiB | **61.4 MiB** |
| vs 60 MiB persisting capacity | 2.4x | **1.02x** |
| resident fraction of the state | 41% | **98%** |
| decode step traffic | 18.5 GB | **3.56 GB** |
| traffic ceiling | 1.69% | **3.75%** |
| persist-family ceiling | 0.68% | **3.67%** |
| measured `persist` | +0.10% | **+1.26%** |

Three interleaved pairs, control 503.18 tok/s, noise floor 0.078%, paired ratios
1.0137 / 1.0120 / 1.0126 — resolved, and the largest real-model gain this repository has
measured; tuning the two persistence dials to their maximum takes it to **+1.63%**. That is
44% of the ceiling and leaves 2.0 points of headroom, which is what makes batch-1 decode a
surface here rather than the dead end it is on the dense model.
`results/rtx5090-moe-matrix.json` carries the data.

It does **not** rescue concurrency, and measuring concurrency on this checkpoint first required
finding a runtime defect (below). With that worked around, all four arms are measurable:

| | control | floor | footprint vs cache | persist ceiling | `persist` |
|---|--:|--:|--:|--:|--:|
| batch 1 | 503.2 tok/s | 0.08% | **1.02x** | **3.66%** | **+1.26%** |
| concurrency 4 | 912.2 | 0.46% | 2.09x | 1.63% | -0.46% |
| concurrency 16 | 1206.9 | 0.57% | 8.38x | 0.53% | -0.17% |
| concurrency 32 | 1230.7 | 0.26% | 16.75x | 0.27% | -0.22% |

**The crossover is where residency crosses one half.** `persist` pays at batch 1, where 98% of
the footprint is resident, and is negative from four sequences on, where 48% is. The traffic
ceiling still rises across the matrix (3.75% → 4.74%); the fraction a persisting window can
hold falls faster (3.66% → 0.27%). Past the crossover the set-aside costs the weight stream
more than residency returns, and `persist` sits a few hundredths below `baseline` — the hook's
own overhead and nothing else.

### The number that decides the project

Weighted across the section 44 matrix, both models, both bounds:

| | dense Qwen3.8-27B | sparse-MoE Qwen3.6-35B-A3B |
|---|--:|--:|
| weighted traffic ceiling | 5.22% | **4.07%** |
| weighted persist-family ceiling | **0.52%** | **1.94%** |
| share of removable traffic a persisting cache can address | 10% | **48%** |

The MoE's total room is *smaller* — its concurrency steps are short and weight-light, so there
is less traffic to recover. But the persist family reaches **48% of that room instead of 10%**,
which is the whole content of the residency thesis, and it lands at **1.94% against the 2.0%
significance floor**.

Six hundredths of a point short, on the best model this project has found, with both
persistence dials at their maximum and a bound that already assumes every resident byte hits
and the set-aside costs its neighbours nothing. No policy closes that gap: the numerator is the
device's 60 MiB of persisting L2 and the denominator is what the workload moves. **A larger
persisting cache, or a model that moves less per step than 3.56 GB, would; nothing in this
repository will.**

### Measured — the set-aside is the dial, and the shipped default was not it

`--axis budget-fraction`, batch 1 on the MoE checkpoint, 3 interleaved pairs with the opening
run discarded, control 503.2 tok/s, noise floor 0.097%:

| `budget_fraction` | set-aside | gain |
|---|--:|--:|
| 0.25 | 15 MiB | +0.68% |
| 0.50 | 30 MiB | +0.96% |
| 0.75 (the shipped default) | 45 MiB | +1.28% |
| **1.00** | 60 MiB | **+1.53%** |

Span 0.85% against a 0.10% floor — resolved, monotonic, and no interior optimum: every extra
MiB of set-aside is another MiB of a 61.4 MiB footprint that stays resident, and nothing in the
range is yet costing the weight stream more than it returns. The residency model predicting its
own measurement, on an axis that spans 0.02% on the dense model where the footprint is 2.4x the
cache and no fraction of it can help.

Capture efficiency against the residency-scaled ceiling falls from 76% at 0.25 to 43% at 1.00,
which is the deliberately generous bound showing itself: it assumes every resident byte hits and
that the set-aside costs its neighbours nothing, and neither is quite true.

`--axis hit-ratio` says the same thing from the other side, and resolves just as cleanly —
0.25 → +0.76%, 0.50 → +1.13%, 0.75 → +1.41%, **1.00 → +1.50%**, span 0.74% against a 0.063%
floor, measured at the default `budget_fraction`. Both dials are monotonic to their maximum and
neither has an interior optimum, which is what a footprint that nearly fits predicts: there is
no point on this model at which asking for less persistence is better than asking for more.
That is expected to be the opposite of the dense model, where no fraction of a
2.4x-oversubscribed footprint can be held — but **that is a prediction, not a measurement, and
this entry originally stated it as one.** Neither dial has been swept on the dense checkpoint.
The 0.02% figure belongs to `window-target`, a different axis, which is formally open there.

**`cliff` cannot be measured on this model, and the guard is right about that.** Sweeping the
hot-set-policy axis aborted on it: at batch 1 the footprint is oversubscribed, so `cliff`
declines every window, applies no policy at all, and `real_eval.py` refuses it as a null
candidate by name — `windows_applied=0, windows_attached_to_node=0, pre_touch_launches=0`. A
policy whose response to pressure is to do nothing *is* `baseline`, and scoring it would report
the hook's overhead as a locality result. The axis is swept without it.

### Found — the exact-locality gate cannot run on the MoE checkpoint, and the harness misread it

The scored MoE matrix came back `REJECT` with the reason *"exact-locality track requires
bit-identical model output"*. That reason was wrong, and finding out why produced two results.

**The checkpoint is not reproducible.** Two **unhooked** control runs of the same binary on the
same prompt diverge at token 2:

```
control A: 13 271 760 1879 369 264 1103 314 4947 ...
control B: 13 271 760 2614 369 264 1103 314  279 ...
```

The runtime documents the mechanism in `kernels/include/sparkinfer/kernels/deterministic.h`: a
few ULP of difference in the prefill feed *discrete* top-k expert routing and int8 requant, so
over 40 layers one flips an expert and that moves the argmax. `SPARKINFER_DETERMINISTIC=1` does
not cover this checkpoint's Q4_K expert path — two controls still diverge with it set.

**RecurLocal is not the cause, and that is checkable.** On the dense Qwen3.8-27B the same binary
and the same policy at the same settings give control, control and candidate bit-identical over
every token compared. So the MoE throughput numbers stand as throughput and the result is **not
scorable** under sections 15 and 35 — the gate cannot answer the question it exists to answer.

**The harness defect is the part worth fixing.** A gate that compares one control run to one
candidate run cannot distinguish "the policy changed the output" from "this runtime is not
reproducible on this model", and it reported the second as the first — an accusation against a
candidate that had changed nothing. `real_eval.py` now replays the control **against itself**
before the candidate is compared to anything, records `runtime_reproducible` and
`control_first_divergence`, and leaves `output_identical` as `None` rather than `False` where
the question is unanswerable. `decide.py` still refuses to score either case — that part was
right — but now says which one it is, and says explicitly that an irreproducible control is
*not the candidate's fault*.

That distinction matters beyond this model: on a repository that scores outside submissions, a
gate that blames the contributor for the runtime's nondeterminism is worse than no gate.

### Found — why the runtime falls off its batched decode path, worth 5.4x

`docs/MINING.md` said this surface was worth two orders of magnitude more than anything the
library does. It is, and one instance of it is now identified rather than observed.

On the MoE checkpoint the fallback is deterministic. Measured with the adapter's own packing
counters, one isolated run per width: concurrency 8 packs 133 of 151 tokens and reaches
**2456 tok/s**; concurrency 16 packs **127 of 2173** and reaches 452; concurrency 32 packs
**127 of 4205** and reaches 456 — *below* the 503 tok/s single-sequence rate, because above 8
rows the runtime decodes one row at a time. The runtime prints its own cause:

```
[dflash-verify] mmvq_rows refused type=12 N=16 n_out=8192 K=2048
[dflash-verify] declined at layer=0 (linear_attn=1) N=16
```

`launch_mmvq_q4k_rows` refuses `M > 8`, and the bf16 `launch_mmvq_rows` dispatcher has no
chunking loop — while its own `_f32` sibling, eight lines away, chunks `M` into groups of 8 for
exactly this reason. So the first Q4_K projection of a wider batch is refused,
`dflash_verify_short_run` declines at layer 0, and `decode_packed` returns false for the whole
batch. `type=12` is Q4_K; `n_out=8192, K=2048` is `attn_qkv.weight`, the linear-attention
projection on every recurrent layer.

**Proven without a patch.** The engine already splits a batch wider than its packed-row cap
into chunks of that cap, and the cap is an environment variable, so setting it to the width the
GEMV accepts isolates the cause exactly — nothing else changes:

| | default cap (32) | `SPARKINFER_PACKED_MAX_ROWS=8` |
|---|--:|--:|
| c=16 | 5.8% packed, 452.8 tok/s | **94.4% packed, 1204.4 tok/s** |
| c=32 | 3.0% packed, 455.6 tok/s | **97.3% packed, 1226.8 tok/s** |

Two numbers, two questions. **The cliff is 5.4x** — crossing from 8 rows (2456 tok/s) to 16
(452) at the default cap. **The cap recovers 2.7x**, not the whole cliff, because chunking
re-reads the weights once per chunk: four chunks of 8 at 32 sequences is four weight passes
where one 32-row kernel would be one. A usable workaround; a wider-row kernel is still the fix.

It also makes the MoE's concurrency arms measurable, so the scored matrix for that model is
complete rather than partial — with the cap set on **both** arms, because it is a property of
the workload rather than of the candidate.

**An omission, not a design choice.** Several multi-row launchers in that file chunk `M > 8`
into groups of 8 — `launch_gemv_rows2`, `launch_gemv_nvfp4_rows_dp4a`,
`launch_gemv_nvfp4_rows_dp4a2` and `launch_mmvq_rows_f32`, the last of them seven lines below
`launch_mmvq_rows` in the same file. `launch_mmvq_rows` does not.

**One name in that list was wrong, and correcting it undoes a conclusion.** This entry
previously named `launch_gemv_nvfp4_rows` as a chunker. It is not:
`gemv.cu:3518` reads `if (M < 2 || M > 8) return false;` and there is no loop. See the
correction below — it changes what the dense model's collapse is.

~~That also settles that the dense model's collapse is a different bug~~ — **the conclusion was
right and the reasoning was not.** The dense collapse IS a different thing from this refusal,
but not because the NVFP4 path has a chunking loop: `launch_gemv_nvfp4_rows` does not have one
(`gemv.cu:3518`). It is different because it is not a decode-path fallback at all — per-token
latency is unchanged through it, which no fallback can manage. Two later sections take it
apart: a per-request device-OOM failure found while looking for it, and the 32% drop itself,
which completes every request and is still open. The `prefetch` ratios
`[0.932, 0.676, 0.925]` belong to the second.

### Closed — reuse the cache can actually serve, bounded rather than built

`docs/MINING.md`'s second open surface asked whether reuse *within* a recurrent layer, a
distance L2 serves for free, has any bytes at it. Almost none, and `eval/traffic_budget.py`
now reports the bound as `within_layer_family`:

- **The matrix state contributes zero.** `gdn_ar_fast_kernel` holds each state column in
  registers across both of its passes — "ONE global read + ONE global write of the 2 MB/layer
  state", in its own comment — and the batched kernel the concurrency path uses is the same
  shape. 98% of the recurrent bytes are touched exactly twice, once each way.
- **The conv window's shift is the whole of it.** `conv_split_kernel` reads the K-1 window
  entries to convolve and then re-reads K-2 of them to shift the window, so the reusable bytes
  are `conv_state x (K-2)/(K-1)` per layer: **1.97 MB per token** against an 18.5 GB step.

A **0.011%** ceiling on the dense model and 0.028% on the MoE — two to three orders of magnitude
under the floor, and the bound is generous because each thread reads its own window entries and
most of those re-reads never leave a register. It is a property of this runtime, not of Gated
DeltaNet: the naive kernel SparkInfer replaced read the state twice and wrote it twice, so
against a runtime like that the within-layer reusable bytes would be an extra read and an extra
write of the whole matrix state — **288 MiB per token**, 153x the conv-shift figure and a 1.66%
ceiling, essentially the entire traffic ceiling. The reuse was real and it was large; somebody
else already took it, in registers, where it belongs.

### Added — a hot-set policy that admits whole layers

`HotSetPolicy::Quota`. Every policy before it answers oversubscription by moving one dial — the
hit ratio — for every layer alike, which models the cache as something that can keep 97% of a
byte. It cannot: a line is resident or it is not. `Quota` admits whole layers at the full hit
ratio until the set-aside is spent and declines the window for the rest, spread evenly and
decided from the layer ordinal alone so the choice cannot move between tokens — a window that
moved would evict exactly the state it kept last time.

It is inert where the footprint is many times the budget, which is every regime the dense model
offers, and that is why it was not worth building until a model existed whose footprint is 1.02x
the cache rather than 2.4x.

**Measured, and the axis does not resolve.** At `budget_fraction=1.00, hit_ratio=1.00`: `quota`
+1.631%, `fixed` +1.631%, `proportional` +1.583%, `sqrt` +1.557% — span 0.075% inside a 0.190%
floor. `quota` ties the best and beats the shipped heuristic by 0.048 points, which is inside
the noise and therefore not a result. OPEN, not solved — but the telemetry says the mechanism
is working rather than inert: at `budget_fraction=1.00` the driver grants exactly 60.0 MiB
against a 61.41 MiB footprint, every layer is reported oversubscribed, and `quota` attaches
**58 of 60** windows with `hit_ratio_reduced: 2`, admitting 29 of 30 layers whole and declining
one — `60 MiB / 2.047 MiB`, as designed, where `proportional` asks all 30 for 0.977x the ratio.
The two differ over 3.3% of the state and 0.048 points is what that is worth. The regime where
the difference should be material is concurrency, where the footprint is 8 to 16x the set-aside
and quota would admit an eighth of the layers rather than 29 of 30; that is the most concrete
open item this work leaves. What the sweep does show is
that the two dials **compound**: +1.63% together against +1.53% and +1.50% alone, the best
measured configuration on either model.

Registered on `eval/sweep.py` and `eval/real_sweep.py`, which also
gain `budget-fraction` and `hit-ratio` axes: on a model whose footprint is close to the cache,
the difference between reserving 45 MiB and 60 MiB is the difference between three quarters of
the state resident and all of it.

### Added — the evaluator can no longer score a batch that never happened

- **A concurrency arm whose runtime fell off the batched path is refused by name.** The adapter
  has always counted `tokens_packed` and `max_rows_seen`; nothing read them. `real_eval.py` now
  does, and refuses an arm that packed under half its tokens — aggregate throughput from a run
  that decoded one row at a time is not that workload's number, and a median over repeats turns
  one such run into a plausible-looking result for whatever configuration happened to be
  running. Stated limitation: the control arm is unhooked by construction and emits no
  telemetry, so a collapse *there* is still invisible. Both observed collapses were in the
  candidate.
- **A sweep can discard its opening run.** `--warmup-runs N`, default 0 so every result
  published before it reproduces exactly. The first concurrency-4 control on the MoE checkpoint
  came in at 841.9 aggregate tok/s against 910.9 and 917.6 after it — every candidate in pair 1
  was compared against a slow control, and the arm's own noise floor became 8.3%, which no
  fraction-of-a-percent difference can resolve against. Re-measured with one run discarded, the
  same arm's floor is **0.46%** — eighteen times tighter, and it resolves. It also changed the
  answer: `persist` at 4 sequences read +0.43% unresolved and reads **-0.46%** resolved.

### Added — a ceiling a reader can calibrate, and a screen for the next model

- **`--matrix` accepts a model geometry.** A second model measured on the same runtime is a
  different state shape against different decode rates; pairing one model's rates with
  another's geometry would produce a confident, wrong ceiling with nothing in the output to
  show it. The spec that supplies the rates may now supply the shape, and the result records
  which was used.
- **`break_even_step_traffic_bytes`.** The bound inverted at the significance floor: the step
  traffic a candidate model has to come in under before a persisting window is worth anything
  at all. Turns "try a sparser model" from advice into a threshold — 6.42 GB on this device,
  against Qwen3.8-27B's 18.5 GB and Qwen3.6-35B-A3B's 3.56 GB.
- **`bandwidth_bound_check`.** `implied_total_bytes_per_token` is measured time times peak
  bandwidth: what the step *could* have moved, not what it did. Against the bytes the
  checkpoint says the step reads, the dense model sits near 1.0 and its ceiling is nearly
  achievable; the MoE sits at 77% and its ceiling is a looser upper bound. Publishing the
  second kind without saying which it is would be the ceiling-in-the-wrong-currency mistake
  again, in the other direction.


### Added — the real-model gate now has a producer

- **SparkInfer adapter** (`integrations/sparkinfer/`, `include/recurlocal/sparkinfer.h`). The
  metric that decides this project had no way to be measured: `eval/decide.py --real` needed a
  `real-result.json` and nothing could produce one. It can now. The adapter brackets the
  Gated-DeltaNet layers of a **pinned** SparkInfer commit (`pin.json`) decoding Qwen3.8-27B
  NVFP4 on an RTX 5090, as an 88-line insertion-only patch plus a library target (CI
  asserts the patch deletes nothing). One binary
  runs both arms: with `RECURLOCAL` unset it is the unmodified runtime.
- **Both recurrent states, not one.** A Qwen3.8-27B layer carries a 3 MiB fp32 matrix state
  *and* a 60 KiB bf16 convolution window, in separate allocations. v0.1 modelled the first and
  could not even express the second: `pre_touch_async` took a `const float*`.
  `pre_touch_bytes_async` and `PreTouchCoverage{Matrix,Conv,Both}` fix that, and the hot set
  now counts both (3,207,168 bytes per layer, confirmed by the adapter's own telemetry).
- **The hot-set accounting is fixed, and the fix is measurable.**
  `HotSetModel{CurrentLayer,TokenFootprint,ReuseWindow}`. v0.1 counted the layer about to run,
  which is how `persist` measured -11% at four concurrent sequences while the planner reported
  *zero* oversubscription. The reuse distance for a recurrent state is a whole token, so
  `TokenFootprint` counts every recurrent layer for every sequence and `ReuseWindow` adds the
  traffic that passes through L2 in between. `CurrentLayer` is retained as the control: on the
  real model the corrected model reports 48/48 layers oversubscribed where the old one reported
  0. Also exposed on the synthetic benchmark (`--hot-set-model`) and in `eval/sweep.py`.
- **`WindowAttach{Stream,CaptureNode}`, and the finding behind it.** SparkInfer captures the
  whole decode step into a graph and replays it per token, and a stream access-policy window is
  host-side state that a graph never records — so under graph decode `persist` measures its
  cost and none of its effect. `attach_window_to_captured_node()` sets the attribute on the
  kernel node the capture just recorded (`cudaStreamGetCaptureInfo`), needing no change to how
  SparkInfer launches its kernels, and it works: `windows_attached_to_node` 48/48.

  It is also **not documented as safe** — a gap in what CUDA sanctions, not an observed
  defect: no run taken has failed because of it, including twelve isolated runs at 32
  sequences. `Stream` is the default anyway, because a locality library should not mutate its
  host's graph behind its back; `CaptureNode` is opt-in, and the controller checks
  `cudaStreamIsCapturing` for `Invalidated` after each attach, counts it in
  `stats().capture_invalidations`, and latches itself off after the first.

  **Correction.** Earlier revisions of this entry, the README, `docs/DESIGN.md` and
  `docs/OPTIMIZATION-SURFACES.md` blamed a 32-sequence throughput collapse on this mutation.
  That was wrong and the code refutes it: the arms that collapsed were `baseline` (-14.8%) and
  `prefetch` (-20.1%), and node attachment is gated on `plan.use_persisting_window`, which is
  only true for `Persist`/`Combined`. The mechanism is inert in the arms that failed. The
  collapse is a runtime fallback whose cause is unidentified and remains an open problem.

  The axis separates the two cleanly at batch 1: `stream` **-0.019%**, `capture_node`
  **+0.129%**. Every point of the persist result is the delivery mechanism, not the policy —
  and the honest conclusion is about the boundary. A locality library can compute a persisting
  window; under graph decode it cannot deliver one without the runtime attaching it at its own
  launch site. Pre-touch has no such problem: its kernels and events are recorded like any
  other work.
- **`PrefetchJoin{PerLayer,TokenEnd}`.** Under capture, every fork/join pair is graph nodes,
  and a hybrid model wants one per recurrent layer. Measured on the real model, this is worth
  **1.20 points** — see below. It is the difference between the pre-touch costing 1.29% and
  costing 0.09%.
- **`WindowScope{Layer,Allocation,Ahead}` and `WindowTarget{Matrix,Conv,Widest,Narrowest}`.**
  Only one access-policy window can be bound at a time, so with two recurrent states which one
  it protects is a choice — and the conv state is the only one whose whole allocation fits in a
  set-aside. Window selection and region resolution moved to the CPU planner
  (`select_window_segment`, `resolve_window_region`), so they are unit-tested without a GPU;
  the v0.1 equivalents lived in the CUDA controller and could not be.
- **Row-major pre-touch** (`pre_touch_rows_async`) and a controller entry point for it. At
  concurrency a runtime holds one state allocation per sequence and gathers their base pointers
  into a device array. One launch per row would add 3,072 kernel nodes to a captured decode
  graph at 32 sequences; this reads the base pointer device-side, so the launch count does not
  depend on concurrency.
- `eval/real_eval.py` — the scored measurement. Control and candidate interleaved on one box,
  token-exact greedy replay as the correctness gate, per-context geometric mean, and the
  control arm's own run-to-run spread reported as the noise floor.
- `eval/real_sweep.py` — one adapter axis at a time against the real model, the real-runtime
  counterpart of `eval/sweep.py`. It refuses to name a winner inside the noise floor, and
  refuses to *estimate* a floor from fewer than three control repeats: two readings can land on
  the same number, and a zero floor would make any difference look resolved.
- `eval/traffic_budget.py` — what fraction of a decode step's memory traffic recurrent state
  actually is, which is the ceiling on everything this project does.

### Fixed — production defects, found by auditing the code and by testing it

The repository shipped 1,428 lines of CUDA with no tests at all. Writing them found real
defects; so did `compute-sanitizer`, which `CONTRIBUTING.md` had required of contributors
while nothing in the repo ran it.

- **The device-wide persisting-L2 set-aside was never given back.** `initialize()` reserved it
  with `cudaDeviceSetLimit`; `reset()`'s header promised to release it and only evicted
  persisting *lines*; the destructor cleared a bookkeeping member without telling the driver.
  Any process that ever constructed a controller — including in `RECURLOCAL=baseline`, a
  no-op control — ran the rest of its life with a smaller L2 for every other kernel. Now
  restored to its prior value, and the integration calls `shutdown()` from the model
  destructor so it happens while the CUDA context is still alive.
- **`before_layer`'s legacy overload ignored `WindowAttach::Stream`** and armed the
  undocumented graph-node mutation regardless — so a consumer asking for the documented-safe
  default still got a graph mutated mid-capture.
- **A failing pre-touch stranded the caller's graph capture.** The fork that pulls the
  prefetch stream into a capture was emitted *before* the flag recording that a join is owed,
  so any early return in between left the capture unjoined with nothing to fix it.
  `cudaStreamEndCapture` then fails and returns a null graph — which SparkInfer's `cu()`
  helper logs without aborting, after which every decode step replays a null graph and emits
  the same stale token forever. Triggerable by a *stale* error latched by unrelated code,
  because the pre-touch reports a bare `cudaGetLastError()`.
- **`configure_persisting_l2()` left the caller's current device changed.** It called
  `cudaSetDevice()` and never restored it, so merely constructing a controller repointed the
  calling thread's GPU.
- **The planner budgeted against the requested set-aside, not the granted one.** The driver
  does not honour the request: an RTX 5090 has a non-zero default (18 MiB) and returns 18 MiB
  for a 15 MiB ask. Every oversubscription decision inherited that rounding.
- **`stats().hot_set_oversubscribed` counted the wrong thing** — it keyed off
  `hit_ratio_reduced`, which `HotSetPolicy::Fixed` never sets, so it read zero under exactly
  the policy that ignores oversubscription hardest. The two are now separate counters.
- **The controller held the caller's streams past `reset()` and segfaulted at exit.** The
  streams are *borrowed*; a runtime that did the correct thing — call `shutdown()`, then
  destroy its own streams — left the controller's destructor probing a dangling handle.
  Backtrace: `cudaStreamIsCapturing <- release() <- ~Adapter <- __run_exit_handlers`, SIGSEGV
  inside libcuda. Found by the concurrent arm of the eval harness, which reported
  `cb bench exited -11` rather than a number. `reset()` now drops the handles, and the
  borrowing contract is stated on `bind_streams()`.
- **`release()` during an active capture poisoned the caller's context.** `cudaMalloc`,
  `cudaFree` and `cudaDeviceSetLimit` are all illegal while a stream is capturing;
  compute-sanitizer counted 14 such errors when a controller was destroyed mid-capture. It
  now drops its handles without touching the driver, and counts it.
- **The pre-touch consumed the host runtime's pending CUDA error.** Launch status was
  reported with `cudaGetLastError()`, which *clears* the per-thread error slot — so an
  embedded RecurLocal could swallow an error the host had not yet checked, and the host's own
  next check would report success for whatever really failed. Now `cudaPeekAtLastError()`,
  which reports without consuming.
- **`WindowScope::Ahead` widened past memory the caller never declared** when no allocation
  base was supplied — and the unit test asserted that behaviour as correct.
- **`cudaStreamCaptureStatusInvalidated` was treated as an active capture**, so the controller
  kept deferring windows into a graph that could never be instantiated.
- **`real_eval.py` scored the batch-1 arm with a different estimator from every other arm**
  and from its own documented method: a ratio of medians rather than the median of paired
  ratios, discarding the pairing that makes an interleaved same-box comparison valid. On
  drifting clocks the two disagree enough to flip a verdict. The estimator is unified,
  extracted from `main()` so it is testable without a GPU, and covered. The committed result
  was recomputed from its stored paired ratios: **+0.049% → +0.053%**, verdict unchanged
  (`reject`) — this data was tight enough that the bug did not bite.

### Added — what is the most this repository can ever pay?

- **`eval/traffic_budget.py --matrix`** answers the question one level above a single arm.
  Each workload has a physical ceiling, and the verdict is a weighted geometric mean over all
  of them, so the number that decides whether the project is worth competing on is what a
  submission scores if it hits *every* ceiling at once — one that made recurrent state
  entirely free. The weights and the bands come from `decide.py` rather than being restated,
  and a test pins that the two agree: a ceiling advertised in one currency and paid in another
  would be worse than no ceiling. Arms with no measured decode rate are excluded and named
  rather than guessed. Inputs live in `configs/rtx5090-section44-ceiling.json`, every one of
  them a baseline measurement from `results/rtx5090-real.json`.

### Measured — the first complete section 44 matrix, and it is still a rejection

- **`results/rtx5090-real-complete.json`.** Every arm of the matrix for the first time: batch 1
  at contexts 128/4096/16384 plus concurrency 4, 16 and 32, three interleaved pairs each, one
  box, token-exact output with `candidate_hook_active: true`. Candidate is `persist` with
  `window_attach=capture_node`, the only mode that measures positive anywhere.

  ```
  verdict: reject   weighted gain +0.059%   impact none   significant false
    batch1          +0.184%  (w=0.40)
    concurrency16   +0.000%  (w=0.20)
    concurrency32   -0.163%  (w=0.20)
    concurrency4    +0.090%  (w=0.20)
    unresolved: ['concurrency16', 'concurrency32', 'concurrency4']
  ```

  Batch 1 is the only arm that resolves: **+0.184%** against a 0.014% noise floor — a real
  gain, and about a quarter of the 0.68% a persisting cache can reach there. Every concurrency
  arm is inside its own spread. The weighted result is +0.059%, and the verdict is reject.

  The previous published result had no concurrency-32 arm, which the coverage guard above now
  reports. Both files are kept: the partial one still carries the axis sweeps and the capture
  probe that explain the mechanism.

### Fixed — a correctness record that understated what the gate proved

- **`candidate_hook_active` was false on every run that worked.** The token-exact gate reads
  the adapter's stats snapshot, which is printed at process exit — after the runtime's model
  destructor has called `shutdown()` and cleared the live `initialised` flag. The guard that
  refuses an unhooked candidate already knew this and read `ever_initialised`; the correctness
  record did not, and recorded `candidate_hook_active: false`. That reads as "the token-exact
  gate compared the control against itself", which is the opposite of what it proved. Both
  sites now share one `hook_ran()`, and a test asserts `ever_initialised` is read in exactly
  one place so the two cannot drift apart again.

### Added — a reference baseline, all four arms, one box

- **`results/rtx5090-baseline-matrix.json`** is the number a submission has to beat: control
  decode rate and run-to-run noise floor at every point of the section 44 matrix, plus what
  each shipped mode does against it, plus both ceilings. Three interleaved pairs per arm, one
  box for all four (the two RTX 5090s this project has used disagreed by 2.6% at concurrency
  16, and the ceiling is computed from a measured step time, so mixing them would put that
  disagreement inside the answer). `window_attach=capture_node` is fixed throughout, because
  with the default the persist family defers every window and `real_sweep.py` correctly refuses
  the arm — which is how the first attempt at this measurement died before writing anything.

  | | control | floor | traffic ceiling | `persist` | `prefetch` | persist ceiling |
  |---|--:|--:|--:|--:|--:|--:|
  | batch 1 | 96.67 tok/s | 0.04% | 1.69% | **+0.10%** | -1.27% | 0.68% |
  | concurrency 4 | 334.17 tok/s | 0.09% | 3.01% | +0.15% | -2.07% | 0.59% |
  | concurrency 16 | 790.60 tok/s | 0.13% | 7.44% | +0.06% | -5.68% | 0.35% |
  | concurrency 32 | 1287.57 tok/s | 0.34% | 12.71% | -0.59% | -7.51% | 0.28% |

  `persist` is positive and *resolved* at batch 1 for the first time (+0.10% against a 0.04%
  floor), and decays to negative by 32 sequences — the residency bound playing out against a
  hook overhead that grows with concurrency.

### Added — why the persist family captures nothing, as arithmetic

- **`eval/traffic_budget.py --persisting-l2-bytes`** computes a second ceiling, far tighter
  than the traffic one and specific to the persist family. A persisting window cannot save
  traffic it cannot hold: state written at layer *i* is read again at layer *i* of the **next**
  token, so to save a byte the cache has to keep it across a full pass over the model — the
  whole per-token recurrent footprint has to be resident at once, not one layer's worth. On
  Qwen3.8-27B against this device's 60 MiB persisting capacity that footprint is 2.4x
  oversubscribed at batch 1 and **20x at 16 sequences**.

  The consequence is the answer to an open problem this repository had left standing as
  "nobody has explained why": the persist family's ceiling *falls* as concurrency rises —
  0.68% at batch 1, 0.58% at 4, 0.34% at 16 — while the traffic ceiling rises 1.68% → 7.23%
  over the same range. The room grows and the fraction of it a persisting cache can address
  shrinks faster. Measurement agrees: `persist` at 16 sequences, with the window genuinely
  attached at the graph node, is +0.06% against a 0.13% control noise floor and a predicted
  ceiling of 0.34%. The bound is deliberately generous — every resident byte hits, the
  set-aside costs its neighbours nothing — and a real policy lands below it.

### Fixed — the ceiling was quoted in the wrong currency

- **A ceiling a submission could legitimately beat.** `traffic_budget.py` reported the
  recurrent-state *share of traffic* as the ceiling. But the scorer measures
  `candidate_tps / baseline_tps`, and a step carrying *f* less traffic runs in *(1−f)* of the
  time — so throughput rises by *f/(1−f)*, which is more than *f*. Every published ceiling was
  understated: 1.65 → 1.68% at batch 1, and 11.03 → **12.40%** at 32 sequences, the surface
  `docs/MINING.md` points the competition at. The error was in the safe direction for the
  project's own verdict and the wrong direction for everyone else: a contribution could have
  exceeded a number this repository called a physical limit. The same conversion applies to
  the pre-touch cost prediction, the other way round, and was corrected with it.

### Fixed — the evaluator was not fit to arbitrate

The harness is meant to decide whether a contribution is accepted. Three ways it could quietly
fail to:

- **A candidate that applied no policy was scored as a result.** `require_hook_engaged` proved
  the hook had *loaded*, not that anything reached a kernel. Under graph decode with the safe
  `WindowAttach::Stream`, `persist` defers every window to a runtime that does not attach
  them — `windows_applied 0, windows_attached_to_node 0, pre_touch_launches 0`, and 192
  deferred. The first scored run in this repository was exactly that configuration, so its
  verdict measured the hook's overhead against the control and nothing else. `real_eval.py`
  now refuses a null candidate outright and names the counters that make it null.
- **The control arm inherited an ambient `RECURLOCAL`.** `run()` copied `os.environ` and only
  the *caller's* dict was checked for a mode, so an operator with `export RECURLOCAL=combined`
  in their shell would run a hooked "control" and the harness would report ~0% for a
  comparison of the candidate against itself. The control now scrubs every `RECURLOCAL*` name
  from the inherited environment.
- **`decide.py` never read whether the arms resolved.** `real_eval.py` computes a per-workload
  resolution against each arm's own run-to-run spread; the scorer ignored it and could return
  `significant: true` on a matrix where nothing resolved. It now refuses.
- **An omitted workload made the score better, not worse.** `decide.py` renormalises the
  weights it is given, so a workload left out of the matrix is removed from the geometric
  mean rather than averaged in — and `docs/MINING.md` points the competition at concurrency
  32, the arm with the most room and therefore the one worth not running. The scorer now
  derives coverage from the workloads it actually scored (never from a `workload_coverage`
  field in the document being scored), names what is absent together with the section 44
  weight that went with it, and refuses to mark a partial matrix `significant`. The verdict
  is still reported, and `--allow-partial` records a maintainer's decision to score one
  anyway. The repository's own published result is partial in exactly this way — it has no
  concurrency-32 arm — and now says so in the verdict rather than only in the bundle.

- **Nothing interlocked concurrent eval processes.** Two runs on one GPU do not go slower,
  they produce a number for a run that never happened — twice in this project's own history a
  VRAM race turned into a plausible-looking result. `real_eval.py` now takes an advisory
  `flock` and waits.

### Fixed — found by an adversarial audit of the finished work

Three findings survived two independent refute-by-default verifiers reading the current tree:

- **`decide.py` could not score the repository's own published result.** `results/` ships the
  matrix nested under `scored_result` alongside the axis sweeps; `score_real` only looked at
  the top level, so the exact command the README gives — `decide.py --real
  results/rtx5090-real.json` — printed "real result has no 'workloads'" and scored nothing.
  The README had illustrative output where it claimed real output. Both fixed, and a test now
  scores the committed artifact so the docs cannot drift from it again.
- **Hot-set telemetry read zero in every mode except persist.** The accounting was written
  only inside the persisting-window guard, so a prefetch-only run reported
  `hot_set_bytes: 0, hot_set_budget_bytes: 0` beside `hot_set_model: token_footprint` — a
  reader would conclude nothing was competing for L2 while 147 MiB was. The numbers describe
  the workload, not the policy that happens to be enabled, and are now reported for all four
  modes.
- **A second model instance would silently share the adapter's walk state.** The hook is a
  process-global singleton holding one model's layer ordinal, pending window and layout;
  SparkInfer's mutex is per-model, so two instances would interleave. The failure mode is not
  a crash, it is quietly wrong numbers. Rather than a mutex — which would fix the data race
  and keep the wrong logic — the adapter now detects a second compute stream and refuses,
  loudly.

### Added — the CUDA code is now tested

- `tests/test_cuda_controller.cu`: 120 device-side checks over the controller, the
  graph-capture state machine, every pre-touch strategy and the row-major path. Registered
  with `ctest`; skips cleanly with exit 0 where there is no GPU. Each regression test was
  validated by reverting its fix and confirming the test fails — one that did not was
  rewritten, and one that cannot be caught on a single-GPU box says so at runtime rather than
  reporting a green tick.
- `scripts/sanitize.sh`: memcheck, initcheck, synccheck and racecheck. All clean; memcheck
  found the release-during-capture defect above on its first run.

### Measured — the first real-model result, and it is a rejection

One RTX 5090, CUDA 13.3, Qwen3.8-27B NVFP4 on SparkInfer `5347b27c`. Control and candidate
are the same binary, interleaved. Batch-1 noise floor **0.023%** over 3 pairs.

| | ceiling | control | `baseline` | `persist` | `prefetch` | `combined` |
|---|--:|--:|--:|--:|--:|--:|
| batch 1 | 1.65% | 96.1 tok/s | -0.01% | **+0.13%** | -1.29% | -1.17% |
| concurrency 4 | 2.88% | 329 tok/s | -0.09% | +0.06% | -2.19% | -2.34% |
| concurrency 16 | 6.73% | 769 tok/s | +0.36% | -0.21% | -5.88% | -5.40% |
| concurrency 32 | — | 1262 tok/s | -1.13% | -1.13% | *unresolved* | *unresolved* |

Output is token-exact against the unhooked runtime under greedy replay.

The scored run — `prefetch` with `token_end` joins and the `ptx_l2` walk, 3 interleaved pairs,
batch 1 at three contexts plus concurrency 4 and 16 — is **-0.488% weighted**, and
`eval/decide.py --real` returns `reject`: batch1 +0.016%, concurrency4 -0.667%,
concurrency16 -1.311%. The policy demonstrably ran (188 pre-touch launches at batch 1; the
concurrency arms went through `decode_packed`, 129/138 tokens packed at c=4). It costs more
than it saves, and the cost grows with concurrency as the axis sweeps predicted.

An earlier scored run reported +0.053% for `persist` with safe window delivery; that was a
null candidate which applied no policy at all, and the harness now refuses such runs. Concurrency 32 is deliberately absent from the scored
matrix and reported as unresolved instead; `real_eval.py` leaves an unmeasured arm out and
`decide.py` renormalises the weights that remain, so a partial run reads as a partial verdict
rather than a full one with invented halves.

Three things this says that the synthetic benchmark could not:

- **At batch 1 the hypothesis cannot pass, by arithmetic.** Qwen3.8-27B is a dense hybrid;
  every weight is read every token, and recurrent state is 1.65% of the traffic. Making it
  free would be worth 1.65%, under section 21's 2% floor, before any policy is chosen.
- **At concurrency the room is real and none of it is captured.** The recurrent share
  quadruples to 6.73% by 16 sequences, and `persist` moves the *wrong way* (+0.13% to -0.21%)
  while `prefetch` gets worse in proportion to the bytes it moves.
- **The pre-touch's cost is graph nodes, not memory.** Production decode is a captured CUDA
  graph, so each fork/join is permanent: `PrefetchJoin::TokenEnd` recovers 1.20 of the
  1.29 points, and the extra full read of recurrent state that remains costs 0.09%.

Concurrency 32 is reported as unresolved rather than as a result: two of its four arms
disagreed with themselves by 30% between repeats, and one of them was `baseline`, which
installs no window and issues no pre-touch and therefore cannot cost anything. A median of
two hides that; `real_eval.py` now publishes a `resolution` block beside every workload's
gain so it cannot.

The synthetic benchmark has now disagreed with the real model on four axes — prefetch
distance (+21.0% at d=6 there, monotonically negative here), prefetch schedule (a 16-point
penalty for doing less there, a saving here), pre-touch strategy (unresolved there, `ptx_l2`
ahead by 0.70 points here) and hot-set policy (a 40-point swing there, 0.01% here). One
structural cause: it does not capture a graph, and production decode does.

`docs/OPTIMIZATION-SURFACES.md` carries the full matrix, `results/rtx5090-real.json` the raw
data, and `eval/decide.py --real` the verdict.

### Changed

- `recurlocal_cuda` is compiled whole-program instead of with relocatable device code.
  Separable compilation obliges every consumer to run a CUDA device-link step, and a runtime
  linking RecurLocal as one more static archive got `undefined reference to
  __cudaRegisterLinkedBinary_*` instead. Found by embedding it in SparkInfer, which is the
  first time anything actually consumed it as a library.

### Fixed

- The streaming access-policy window was not clamped to the device maximum (128 MiB on an
  RTX 5090), so hinting any larger weight buffer failed outright instead of hinting the
  leading window. Found by measurement, not review.

- The benchmark checksum sized its grid from the state and then clamped it to 65535 blocks,
  so at the default geometry it covered only 44% of state. The correctness gate — the whole
  guarantee that a locality change did not disturb model state — silently ignored the rest.
  It now uses a grid-stride reduction over every element, folded in a fixed order so repeated
  runs agree bit-for-bit rather than varying with `atomicAdd` ordering.
- Pre-touch work ran outside the timed region: the stop event was recorded on the compute
  stream only, so prefetch cost was never measured while its benefit was. The prefetch stream
  is now joined before the clock stops.
- `after_layer()` was a no-op, leaving the access-policy window bound to the compute stream.
  Every following non-recurrent kernel inherited persisting/streaming policy against a state
  pointer it never touches. The window is now scoped to the recurrent layer.
- The controller was destroyed after its streams in the benchmark, so it cleared an
  access-policy window on a destroyed stream.
- The planner recommended a persisting window on devices reporting no persisting-L2 support,
  where no set-aside can be reserved.
- Unchecked `cudaMalloc` for the pre-touch scratch buffer left a sticky error on the context;
  pre-touch now degrades to a no-op.
- Benchmark arguments were unvalidated: `--tokens 0` emitted `inf`, which is not valid JSON
  and crashed the evaluator.

### Added

- **Surface sweeps on real hardware** (`results/rtx5090-surfaces.json`) covering concurrency,
  cache interference, hot-set policy, prefetch schedule, prefetch implementation and state
  layout. Several overturn assumptions in the overview: kernel-integrated prefetch is 30-33
  points *worse* than a separate stream, `persist` becomes harmful at four concurrent
  sequences, and state layout alone is a 2.3x effect that changes which policy wins.
- **Two shapes of cache interference** (`--stream-mode reuse|distinct`). Re-reading one buffer
  every layer and giving each layer its own slice are different problems and give opposite
  policy answers; only the second is what a persisting window should be protected from.
- **First measured result** (`results/rtx5090-synthetic.json`): RTX 5090, CUDA 13.3, synthetic
  benchmark. persist +10.1%, prefetch +15.9%, combined +10.4%; prefetch distance 6 reaches
  +21.0%. Correctness bit-identical across 39 configurations. Synthetic only — no model, no
  runtime, and the go/no-go gate remains open.
- **Six pre-touch strategies** (`--pre-touch scalar|vec4|vec4_ldcg|ptx_l2|warp_tile|partial`).
  The kernel was a single hard-coded scalar walk. `ptx_l2` issues real `prefetch.global.L2`
  instructions — the "put this range in L2 now" primitive the overview assumed CUDA does not
  expose for arbitrary allocations; PTX does. Adding a strategy is a kernel plus an
  enumerator, and every strategy is measured against the others by one flag.
- **Four hot-set policies** (`--hot-set-policy proportional|fixed|sqrt|cliff`) for what to do
  when the recurrent state that wants to be resident exceeds the L2 set-aside. The shipped
  heuristic was one line with no alternative to compare against; `fixed` is now the naive
  control and `cliff` refuses the window rather than thrash a shared cache.
- **Concurrency in the benchmark** (`--sequences N`). The per-layer hot set now scales with
  batch, which is what makes the hot-set policy measurable at all: `hot_set_oversubscribed`
  was 0 in every previous run because the benchmark decoded a single sequence.
- **Streaming interference** (`--stream-bytes N`) between recurrent layers, standing in for
  the attention and MoE weight traffic a hybrid model pushes through the same L2. Without it
  the persisting window has nothing to defend against.
- **Kernel-integrated prefetch** (`--prefetch-impl fused`, section 34.4). The compute kernel
  performs the pre-touch itself, with no second stream and no extra launch, as an A/B against
  the stream implementation on the same mode.
- **State layout as an axis** (`--state-layout linear|head_interleaved|tile_swapped`, section
  34.3). The map is a bijection, so every layout leaves bit-identical state and only the
  access order changes — coalescing and cache-line utilisation, isolated from everything else.
- **Weight/state cache QoS** (`--qos on`, section 34.6). `before_streaming_region` hints the
  attention/MoE traffic as streaming so it passes through L2 without displacing recurrent
  state, instead of inheriting whatever window the last recurrent layer left behind.
- **Per-layer prefetch schedules** (`--prefetch-schedule uniform|ramp|alternating|sparse`,
  section 34.1). A single global distance assumes every recurrent layer has the same compute
  ahead of it to hide the walk behind, which is false in a hybrid model.
- `eval/sweep.py`: walks one surface with everything else fixed, reports gain against that
  surface's own noise floor, and refuses to name a winner inside it — so an unresolved axis
  reads as open rather than solved.
- **Prefetch distance as an open axis** (`--prefetch-distance 0..8`). It was capped at 0-or-1
  in code, against section 34.1, and was leaving about six points on the table.
- `docs/OPTIMIZATION-SURFACES.md`: each surface, the axis that isolates it, its measured
  frontier, and its noise floor — including surfaces that cannot yet be competed on.
- **Embeddable library surface.** RecurLocal is consumed by a runtime, so it now installs a
  CMake package (`find_package(RecurLocal)`, `RecurLocal::recurlocal`,
  `RecurLocal::recurlocal_cuda`) with public headers and a version header that CMake parses
  as the single source of truth. CI builds an outside consumer against the installed package.
- **CUDA Graph support.** A stream access-policy window is host-side state and is not
  recorded into a captured graph, so the persist mode silently vanished from every replay on
  any runtime using graph decode. The controller now detects capture and returns the window
  in `LayerActions` for the caller to attach to its kernel launch or graph node. Pre-touch
  forks the prefetch stream from compute with an event and rejoins it under capture — an
  unjoined fork ends the capture invalid.
- **Non-throwing integration boundary.** Construction and every entry point are `noexcept`
  and report `cudaError_t`; `initialize()`/`status()` replace a throwing constructor, and
  `validate(PlannerConfig)` lets a caller reject a bad config before building anything.
- **Controller telemetry.** `stats()` counts windows applied, windows deferred under capture,
  oversubscribed hot sets, pre-touch launches and bytes, and both stream priorities — so a
  null end-to-end result can be explained instead of guessed at. Surfaced in the eval output.
- `bind_streams` rejects the same stream for compute and prefetch, which would turn a
  prefetch into the same work on the critical path.
- Stated threading contract: one controller per compute stream, which is also the only
  arrangement in which `concurrently_hot_bytes` is meaningful.
- `recur_local_info` reports the real device when built with CUDA, instead of only
  illustrative constants.
- `run_eval.py --config` makes `configs/*.json` drive the benchmark geometry; it was
  previously a file nothing read.
- `eval/decide.py`: the go/no-go gate as a deterministic function of measurements, so the
  project can settle its own bet rather than argue it. Refuses to accept a synthetic result
  as a verdict. Covered by `eval/test_decide.py`, wired into `ctest`.
- Environment provenance in the benchmark and eval output — GPU, compute capability, SM
  count, driver and runtime version, commit and dirty-tree flag, plus observed or pinned
  graphics clock (`--pin-clock-mhz`).
- `eval/result_schema.json`, pinning the eval output shape.
- Untimed warm-up tokens (`--warmup-tokens`, default 4).
- Repeated, interleaved eval runs with median comparison and a stability verdict
  (`--repeats`, `--stability-threshold-pct`).
- Planner tests for device capability edges, window limits, concurrent hot sets, determinism
  and config validation.
- CI: Release build, JSON asset validation, and a device-less `nvcc` compile gate.
- PR template and a harness-paths guard for `eval/`.

### Changed

- Build options are spelled `RECURLOCAL_*`; the `RECURLLOCAL_*` misspelling still works and
  warns. `RECURLOCAL_WITH_CUDA` likewise, with the old macro still defined.
- Tests use always-on checks. They previously used `assert`, so a Release build reported a
  green run with nothing checked.
- CMake defaults to `Release`; the documented build commands previously produced an
  unoptimised `-O0` host build for a performance project.

## [0.1.0]

- Initial feasibility prototype: CPU locality planner, CUDA persisting-L2 controller,
  asynchronous next-state pre-touch, baseline/persist/prefetch/combined modes, synthetic
  benchmark, evaluator, CPU tests, SparkInfer integration design, GitHub CPU CI.
