# Evaluation

Two kinds of number appear in this repository and they must never be confused:

| | what it is | where it comes from | may it be published as a gain |
|---|---|---|---|
| **measured** | a paired, interleaved, same-box A/B of two runs of one binary | `eval/real_eval.py` | yes, with its noise floor |
| **predicted** | an output of a cost model | `tensortransit plan` / `compare` | **no** |

Every artifact carrying a predicted figure declares `"basis": "model"`, every CLI command that
prints one says so on the same screen, and `eval/test_schemas.py` fails if a plan omits the
marker. That is deliberate belt-and-braces: this project's recurring failure mode is a
confident number whose provenance was not on the page.

## The cost model, and what it cannot see

A plan's predicted saving is:

```text
saved(tensor) = reused_bytes x (granted / bytes) x hit_ratio
```

Three assumptions, stated rather than hidden:

1. every resident byte hits;
2. replacement within the budget is perfect;
3. saving scales **linearly** with the resident share.

The first two make every figure an upper bound — the same assumption the 0.1 ceilings were
quoted under. The third is the one that matters here, and it has a consequence sharp enough to
be worth stating as a result.

## The result: under a linear model, no admission rule can beat Density

Total saving is `sum_i granted_i x density_i` subject to `sum_i granted_i <= budget`. That is a
fractional knapsack, and greedy-on-density is **optimal** for it. So any rule that diverts a
byte of budget from the highest-density candidate to a lower-density one must lose, by exactly
the density difference. The sweep says so, monotonically:

```console
$ for s in 0.0 0.1 0.25 0.5 0.75 1.0; do
    tensortransit plan tests/golden/trace_recurrent_kv.json \
      --planner budgeted --admission role_floor --role-floor-share $s | grep predicted
  done
  predicted +0.358%   # floor 0.00 -- identical to Density, as the test asserts
  predicted +0.349%   # floor 0.10
  predicted +0.336%   # floor 0.25
  predicted +0.313%   # floor 0.50
  predicted +0.291%   # floor 0.75
  predicted +0.269%   # floor 1.00
```

And the five arms of the specification's second proof track, on the same trace:

```console
$ tensortransit compare tests/golden/trace_recurrent_kv.json
  arm                actions  declines    committed B    predicted
  baseline                 0       176              0      +0.000%
  recurrent_only          60       146       47185920      +0.358%
  kv_only                  8       172       47185920      +0.179%
  naive_both             224        64       47185872      +0.034%
  global                  54       149       47185920      +0.336%
```

Read honestly, that is a **partial** pass of the second proof track:

- the global planner beats the naive both-persistent policy by **10x** (+0.336% vs +0.034%),
  and that is the arm the specification actually names. The naive policy loses because
  proportional shaving asks thirty tensors for a third of a window each and the hardware
  cannot keep a third of a line. Under concurrency it does worse than that: every hit ratio
  falls below `min_hit_ratio`, everything is declined, and the policy **disables itself** —
  0 actions, +0.000%.
- the global planner does **not** beat the best independent policy. It is 0.022 points below
  recurrent-only, and by the argument above it must be.

**No amount of planner work changes that under this cost model.** The mechanisms by which
coordination could genuinely win are precisely the ones a linear model cannot represent:

1. **Whole-line residency.** A line is resident or it is not; there is no 30% of a byte.
   `AdmissionRule::Quota` represents this and the linear model flattens the difference to
   0.001 points — where hardware measured a real, if small, separation.
2. **Survival, not just residency.** A tensor whose reuse distance exceeds the budget is
   evicted before it pays off however much of it was admitted. The model charges nothing for
   that; `--max-reuse-budgets` is the crude switch that does.
3. **The other half of the cache.** Telling the weight stream to get out of the way changes
   the *denominator*, and the model has no term for interference at all. Only the global arm
   is allowed a `Stream` action, and its value here is zero because the model cannot price it.

So the honest statement of where the multi-tensor claim stands: **the mechanism is built,
tested and observable; the arithmetic that would justify it is not in the cost model; and
settling it requires hardware.** Improving the cost model is a contribution that needs no GPU
and would move the frontier more than another admission rule would.

## The bound that comes first

Before writing a planner, run:

```bash
tensortransit inspect <trace.json> --device rtx5090
```

If the device-bounded ceiling for the roles you can act on is under the significance floor,
no planner in this repository can help you, and finding that out costs one command and no
hardware. That is what the tool is for. `eval/traffic_budget.py` computes the same thing from
a pinned geometry file and prints `break_even_step_traffic_bytes`, which inverts the bound: on
an RTX 5090 a decode step must move **at most 6.42 GB** before a persisting window over a
recurrent footprint can reach 2% at all.

## Measuring

```bash
adapters/sparkinfer/build.sh $WORK          # one binary: hook inert unless the env names a mode
python3 eval/real_eval.py --binary ... --control ... --candidate TENSORTRANSIT=persist ...
python3 eval/decide.py --real result.json   # JSON on stdout, human summary on stderr
```

Rules the harness enforces, each because the failure happened:

- **Interleaved pairs**, medians of paired ratios, so thermal drift does not land on one arm.
- **The control arm's environment is scrubbed** of every `TENSORTRANSIT*` **and** `RECURLOCAL*`
  name. An operator with `export TENSORTRANSIT=combined` would otherwise compare the candidate
  against itself and measure ~0%. As of 0.2.0 the adapter reads both prefixes, so a scrub that
  knew only the old one would leave exactly the hole this guard closes — silently, because a
  contaminated control produces a plausible number rather than an error.
- **Null candidates are refused by name.** A run whose telemetry reads
  `windows_applied 0, windows_attached_to_node 0, pre_touch_launches 0` applied no policy. The
  first scored run in this repository was exactly that.
- **Partial matrices are refused.** A missing workload is renormalised away rather than
  averaged in as a zero, so omitting the arm with the most room *raises* the score.
- **The control is replayed against itself first.** A gate comparing one control replay to one
  candidate replay cannot tell "the policy changed the output" from "this runtime is not
  reproducible", and it reported the second as the first — against a candidate that had
  changed nothing.
- **Ceilings are quoted in throughput currency**, `f/(1-f)`, never as a traffic share `f`.
- **Arms that fell off the batched path are refused**, and arms that lost requests to a device
  OOM are refused below 90% of the expected token count.

`eval/run_from_base.sh` runs the instrument from the **base** commit rather than the
submission, because a one-line change to a noise floor or an estimator does not look like
cheating in a diff.

## The significance floor

An axis whose spread sits inside its own run-to-run noise is **open**, not solved.
`eval/sweep.py` and `eval/real_sweep.py` exit non-zero rather than name a winner. The
project's floor is 2%, and the honest state of the recurrent-persist family against it is in
[MINING.md](MINING.md) and on the first screen of the README.
