# Evaluation

The synthetic evaluator compares `baseline`, `persist`, `prefetch`, and `combined` against identical state data. It requires identical checksums and reports relative timing.

This is only a **feasibility signal**. A reward-worthy result must additionally benchmark a pinned real model/runtime combination.

Recommended real metrics:

- decode tok/s at batch 1;
- aggregate tok/s at concurrency 4/16/32;
- recurrent-kernel time;
- HBM read/write bytes attributable to recurrent state;
- L2 hit/sector metrics;
- joules/token;
- output/logit equivalence.

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
