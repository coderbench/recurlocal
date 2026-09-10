# Evaluation

Two kinds of number appear in this repository and they must never be confused:

| | what it is | where it comes from | may it be published as a gain |
|---|---|---|---|
| **measured** | a paired, interleaved, same-box A/B of two builds | `eval/real_eval.py`, `tools/tt-frontier run` | yes, with its confidence interval |
| **predicted** | an output of a cost model | `tensortransit plan` / `compare` | **no** |

Every artifact carrying a predicted figure declares `"basis": "model"`, every CLI command that
prints one says so on the same screen, and `eval/test_schemas.py` fails if a plan omits the
marker. That is deliberate belt-and-braces: this project's recurring failure mode is a
confident number whose provenance was not on the page.

---

## The scoring regime: Frontier Gain

A performance PR earns credit for one thing — **verified marginal expansion of the current real
inference serving frontier beyond `main`** — and the result is a single continuous number:

```text
dF = F(candidate) / F(main) - 1
```

`F` is the normalized Pareto hypervolume of the serving frontier — goodput against p99
inter-token latency — aggregated over a frozen matrix of workload cells. The whole system is
in [`frontier/README.md`](../frontier/README.md); the code is `eval/frontier/` and the CLI is
`tools/tt-frontier`.

### Why the impact bands went away

Until 0.2.1 a submission was scored into `XS`/`S`/`M`/`L`/`XL` by weighted throughput gain,
with the lowest paying band at 2%. The physical ceiling for the entire shipped policy family
is **0.52%** weighted on the scored model and **1.94%** on the best model this project has
ever found. A submission could remove every recoverable byte of recurrent traffic and still
score `none`.

A band structure whose lowest paying step sits above what the hardware can deliver is not a
strict regime. It is a broken instrument, and what it communicates to a contributor is false:
that the room is bigger than it is, and that anything smaller does not count. `dF` is
continuous, so a real 0.3% expansion is reported as a real 0.3% expansion, and
`frontier/TTF-N/reference.json` publishes what the generation's own calibration says is
reachable — per cell, with the measured run-to-run spread beside it.

### What a receipt has to clear

1. **Correctness first.** Token-exact greedy replay, against a control replayed against
   *itself* first. A gate comparing one control replay to one candidate replay cannot tell
   "the policy changed the output" from "this runtime is not reproducible", and it has
   reported the second as the first.
2. **The whole matrix.** A cell that was not run is not averaged in as a zero and is not
   renormalised away — it is a missing cell, and the receipt says `PARTIAL` on its face.
   Otherwise omitting the arm with the most room is the cheapest way to raise a score.
3. **Confidence as a gate, not a multiplier.** A paired bootstrap over interleaved repeats,
   with a frozen seed and resample count. If the 99% lower bound is not above zero the status
   is `INCONCLUSIVE` and the observed figure is *not* published as a contribution.
4. **The protected-workload guard.** A regression past the generation's limit on a protected
   cell is `REGRESSION_GUARD_FAIL` even when the aggregate `dF` is positive.

---

## The cost model

A plan's predicted saving is now:

```text
resident(t) = whole cache lines of  min(granted, bytes) x hit_ratio
survival(t) = min(1, (resident(t) / reuse_distance_bytes(t)) ^ beta)
saved(t)    = reused_bytes(t) x (resident(t) / bytes) x survival(t)
              -  eta x (resident_total / L2) x step_traffic
```

`beta = 0.1108` and `eta = 0.00086`, fitted once against every paired hardware measurement of
the `persist` arm in `results/`, across two model architectures. `eval/cost_model_fit.py` is
the fit, it runs in CI, and it prints the residuals rather than only the parameters.

### What it replaced, and why that mattered

The 0.2.0 model was `saved = reused_bytes x (granted/bytes) x hit_ratio` — **linear** in the
resident share. Total saving was then `sum_i granted_i x density_i` subject to a budget: a
fractional knapsack, for which greedy-on-density is *provably* optimal. So under that model no
admission rule could beat `density`, `role_floor` was monotonically worse as its floor grew by
construction rather than by accident, and the entire admission axis was measuring nothing.

That was not a finding about caches. It was an artifact of the model, and the three terms
missing from it were all real:

| term | what it is | how it is carried now |
|---|---|---|
| whole-line residency | a line is resident or it is not; there is no 30% of a byte | grants are quantised to cache lines, and a sub-line grant is worth zero |
| survival | a tensor whose reuse distance exceeds what the partition can carry is evicted before it pays | the power law above |
| interference | the reservation is taken from the same L2 the weight and KV streams use, so it **costs** | the `eta` term, and a `Stream` action lowers a tensor's reuse distance |

### Why a power law and not an exponential

An exponential cannot fit the two arms that actually resolved. The dense batch-1 arm sits at a
reuse-distance-to-capacity ratio of 368 and delivers 0.255 of its ceiling; the MoE batch-1 arm
sits at 71 and delivers 0.415. An exponential needs its coefficient to differ by 3.4x between
them. A power law needs `beta = 0.140` and `0.115` — the same number within the spread — and it
predicts both to within 0.006 points, a fifth of either arm's own noise floor. It is also the
classic shape of a cache miss-ratio curve, so it is the form to prefer on grounds other than
the fit.

```console
$ python3 eval/cost_model_fit.py
  survival = min(1, (resident_per_tensor / reuse_distance) ^ 0.1108)
  cost     = 0.00086 x 100 x (granted / L2)  = 0.0429 points

  residency model : rms 0.2713pp, 8/8 arms inside their noise floor
  linear model    : rms 0.4847pp, 6/8 arms inside their noise floor
```

### The consequence, which is the point

`saved` goes as `resident^(1+beta)` — **superlinear**. For a fixed budget spread over `n`
tensors the total goes as `n^(-beta)`: fewer, larger grants beat more, smaller ones. The
objective is convex, its optimum is at a vertex, and **concentrating beats spreading**.

The shipped recurrent policy *spreads*: `recurrent_v0` uses `HotSetPolicy::Proportional`, which
asks every recurrent layer for a shaved hit ratio. On the golden KV trace that is 23x worse
than concentrating under this model and 10.5x worse under the linear one, so the model does not
merely change the magnitudes — it makes the admission axis a thing a contributor can move.

`AdmissionRule::Survival` is the rule that exploits it: greedy on marginal saving, stopping
when the next admission would make the plan worse. Under `--cost-model linear` it is *exactly*
`density`, and a test asserts that, so a comparison against `density` is not a comparison
against a moving target.

Both models ship. `--cost-model linear` is the control, and it is how a contributor checks
whether a result is about the policy or about the model.

---

## Know the bound before writing a planner

```bash
tensortransit inspect <trace.json> --device rtx5090
```

If the device-bounded ceiling for the roles a policy may act on is under the noise floor of the
cells it would be measured in, no planner in this repository can help, and finding that out
costs one command and no hardware. `eval/traffic_budget.py` computes the same thing from a
pinned geometry file and prints `break_even_step_traffic_bytes`, which inverts the bound: on an
RTX 5090 a decode step must move **at most 6.42 GB** before a persisting window over a
recurrent footprint can reach 2% at all.

---

## Measuring

```bash
adapters/sparkinfer/build.sh $WORK          # one binary: hook inert unless the env names a mode

# the frontier evaluation, paired and interleaved over a frozen generation
tools/tt-frontier run --generation TTF-1 --model $MODEL \
    --cb-binary $WORK/sparkinfer/build/runtime/qwen3_gguf_cb_bench \
    --generate  $WORK/sparkinfer/build/runtime/qwen3_gguf_generate \
    --main-config "control=" \
    --candidate-config "persist=TENSORTRANSIT=persist,TENSORTRANSIT_WINDOW_ATTACH=capture_node" \
    --output raw.json
tools/tt-frontier compute --generation TTF-1 --results raw.json --output receipt.json
tools/tt-frontier report receipt.json --format markdown

# the single-axis instrument, still authoritative for a targeted A/B
python3 eval/real_eval.py --binary ... --candidate TENSORTRANSIT=persist ...
python3 eval/decide.py --real result.json
```

Rules the harness enforces, each because the failure happened:

- **Interleaved pairs.** main and candidate for the same cell and the same repeat run
  adjacently, so thermal drift lands on both or neither. Graphics clocks cannot be pinned on
  this hardware; only paired same-box deltas are trustworthy.
- **The control arm's environment is scrubbed** of every `TENSORTRANSIT*` **and** `RECURLOCAL*`
  name. An operator with `export TENSORTRANSIT=combined` would otherwise compare the candidate
  against itself and measure ~0%.
- **Null candidates are refused by name.** A run whose telemetry reads `windows_applied 0,
  windows_attached_to_node 0, pre_touch_launches 0` applied no policy. The first scored run in
  this repository was exactly that.
- **Arms that lost requests to a device OOM are refused.** An arm that lost requests did not
  run slower, it ran less: aggregate tok/s is tokens over wall time. Under the frontier runner
  that becomes a *failure status* rather than an abort, because a lost operating region is
  real information about a candidate — but it never becomes a slow point.
- **Arms that fell off the batched decode path are refused**, by the adapter's own packing
  counters.
- **The control is replayed against itself first.**
- **Ceilings are quoted in throughput currency**, `f/(1-f)`, never as a traffic share `f`.

`eval/run_from_base.sh` runs the instrument from the **base** commit rather than the
submission, because a one-line change to a noise floor, an estimator, a frozen generation or
the runtime pin does not look like cheating in a diff. The instrument is `eval/`,
`tools/tt-frontier`, `schemas/`, `configs/`, `tests/golden/`, `workloads/` and
`adapters/sparkinfer/pin.json`; a submission's version of any of them is discarded for scoring
and **named** in the output rather than silently dropped. CI proves it on every push.

---

## The overhead budget

Planner plus executor host time is budgeted under **0.5% of token latency**, and
`tests/test_overhead_budget.cpp` asserts it on the real 64-layer Qwen3.8-27B decode shape
rather than on a toy graph. It checks two different failures:

- the layer being too slow per token, and
- the plan cache not working, so the planner runs every token. That is the single most likely
  way this layer costs more than it saves, and it is invisible in a throughput number.
  `reuse_rate` is what says so, and a steady decode loop must be at 1.0.

Measured on the reference build: 0.045–0.078% of a token for planner plus executor, and 0.19%
for a whole bracketed token including all 64 before/after pairs.

---

## The noise floor

An axis whose spread sits inside its own run-to-run noise is **open**, not solved.
`eval/sweep.py` and `eval/real_sweep.py` exit non-zero rather than name a winner, and the
frontier scorer returns `INCONCLUSIVE` rather than publishing an unqualified figure.

Per-cell control spreads for TTF-1 are published in
[`frontier/TTF-1/reference.json`](../frontier/TTF-1/reference.json) as `control_spread_pct`,
from three interleaved repeats. Read them before choosing what to work on. Two cells that the
4x3 matrix would have contained are not in the generation at all, because calibration found
this device cannot run them: `ctx16384-c16` loses 30 of its requests to a device OOM in every
repeat, and `ctx16384-c32` cannot even load the model.
