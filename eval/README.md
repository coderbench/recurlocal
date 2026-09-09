# Evaluation

The synthetic evaluator compares `baseline`, `persist`, `prefetch`, and `combined` against identical state data. It requires identical checksums and reports relative timing.

Each mode is run `--repeats` times (default 3) with the repeats interleaved, and modes are compared on median elapsed time. The run is labelled `unstable` when any mode's spread exceeds `--stability-threshold-pct` (default 2.0), because a spread wider than the effect makes the mode ranking meaningless. Treat an unstable result as no result rather than as a weak one.

The output shape is pinned by `result_schema.json`.

This is only a **feasibility signal**. The result that decides anything comes from the real
model on a pinned runtime, and `real_eval.py` produces it.

## The real measurement

```bash
# both arms are the same binary; the hook is inert unless RECURLOCAL names a mode
eval/real_eval.py --binary  $W/sparkinfer/build/runtime/qwen3_gguf_bench \
                  --generate $W/sparkinfer/build/runtime/qwen3_gguf_generate \
                  --cb-binary $W/sparkinfer/build/runtime/qwen3_gguf_cb_bench \
                  --model /path/to/Qwen3.8-27B-NVFP4 --concurrency 4,16 \
                  --candidate RECURLOCAL=persist --output real-result.json
eval/decide.py --real real-result.json
```

- **Interleaved.** Control and candidate run back to back and are compared within the pair.
  Graphics clocks cannot be pinned on the measurement box, so only same-box paired deltas
  mean anything.
- **Token-exact.** The correctness gate is a greedy replay compared as a list of token ids,
  not a score. A candidate that diverges is not timed.
- **Noise floor.** The control arm's own run-to-run spread is reported next to the gain. A
  gain inside it is `resolved: false` — open, not won.
- **Partial matrices stay partial.** An arm that was not run is left out rather than filled
  in, and `decide.py` renormalises the weights that remain.

## What the ceiling is

```bash
eval/traffic_budget.py --ms-per-token 10.41 --bandwidth-gbs 1792
```

Recurrent-state locality can only ever be worth the share of decode traffic that recurrent
state accounts for. On Qwen3.8-27B at batch 1 that is **1.65%** — below the 2% floor the
go/no-go table rejects at, before any policy is chosen. It is worth computing this for a new
model or concurrency *before* optimizing for it.

Three bounds come out, not one, and the tighter ones decide more:

```bash
eval/traffic_budget.py --matrix configs/qwen3.6-35b-a3b-moe-ceiling.json                        --bandwidth-gbs 1792 --persisting-l2-bytes 62914560
```

- `ceiling_pct` — removing *all* recurrent traffic, in throughput terms.
- `persist_family.ceiling_pct` — what a persisting window can reach, given that it cannot hold
  traffic larger than the cache. `break_even_step_traffic_bytes` inverts it at the significance
  floor: the step traffic a candidate model must come in under to be worth a window at all.
- `within_layer_family.ceiling_pct` — reuse inside one layer, the only distance L2 serves for
  free. Almost nothing sits there on this runtime; the number says how little.

`bandwidth_bound_check` says whether a ceiling is tight or merely true, by comparing the bytes
the checkpoint says a step reads against measured-time × peak bandwidth. A dense model sits
near 1.0; a sparse MoE near 0.77, where the arithmetic overstates what a policy can return.

## One axis at a time

```bash
eval/real_sweep.py --binary .../qwen3_gguf_bench --model DIR \
                   --axis prefetch-join --fixed RECURLOCAL=prefetch
```

The real-runtime counterpart of `sweep.py`: it walks one adapter axis with everything else
fixed and refuses to name a winner inside the noise floor. It also refuses to *estimate* a
floor from fewer than three control repeats — two readings can land on the same number, and a
zero floor would make any difference look resolved.

Other real metrics worth recording alongside: recurrent-kernel time, HBM bytes attributable to
recurrent state, L2 hit/sector counters, joules/token.

## Deciding

`decide.py` settles the go/no-go gate mechanically. The project is meant to be willing to
fail its own test, and that only holds if the verdict does not depend on who reads the table.

```bash
python3 eval/decide.py --synthetic eval-result.json
python3 eval/decide.py --real real-result.json
```

The synthetic track reports the gain and returns `SYNTHETIC-ONLY` with no tier, whatever the
number is — a `+35%` microbenchmark alongside `+0.8%` real decode is not an achievement. It
also refuses a run whose modes disagreed on the checksum, or whose timing spread was wider
than the effect.

The real track takes a workload matrix and scores it:

```json
{
  "correctness": {"output_identical": true, "method": "greedy replay, token-exact"},
  "provenance": {"gpu": "RTX 5090", "runtime_commit": "...", "model": "..."},
  "workloads": {
    "batch1":        {"weight": 0.40, "baseline_tps": 220.0, "candidate_tps": 238.0},
    "concurrency4":  {"weight": 0.20, "baseline_tps": 610.0, "candidate_tps": 640.0},
    "concurrency16": {"weight": 0.20, "baseline_tps": 1180.0, "candidate_tps": 1205.0},
    "concurrency32": {"weight": 0.20, "baseline_tps": 1500.0, "candidate_tps": 1495.0}
  }
}
```

In order: output must be bit-identical or the verdict is `REJECT` however fast the run was;
any workload regressing more than 2% is `REGRESSION` unless a maintainer waives it; then the
weighted geometric mean across the matrix decides the band and tier. The exit status is 0
only for a scored, significant improvement, so CI can gate on it directly.

Weights default by workload name, and a geometric mean is used deliberately — an arithmetic
mean lets a 2x win cancel a 2x loss.

Suggested project-local impact bands after correctness passes:

| Real end-to-end gain | Tier |
|---|---|
| <2% | none |
| 2–4% | XS |
| 4–7% | S |
| 7–10% | M |
| 10–18% | L |
| >18% | XL |

These are not claims about official Gittensor scoring.
