# Measurements

Every file here is a measurement or a model output, and each one says which on its first line.
The rule the whole repository turns on:

> **Only `eval/decide.py --real` and `tools/tt-frontier` produce evidence about a speedup.**
> A cost-model output carries `"basis": "model"` and may never be published as a gain.

| file | what it is |
|---|---|
| `rtx5090-baseline-matrix.json` | The reference control: decode rates and noise floors at every point of the section 44 matrix on the dense model, plus what each shipped mode does against them. Not a result about a contribution; the number a contribution has to beat. |
| `rtx5090-moe-matrix.json` | The same on the sparse-MoE checkpoint, where the persist ceiling is 3.7x higher. Also where the 5.4x batched-decode cliff was diagnosed. |
| `rtx5090-moe-scored.json` | The complete MoE matrix, and **not scorable**: two unhooked control replays of that checkpoint diverge, so the exactness gate cannot attribute anything to a candidate. Kept because it is the reason a guard exists. |
| `rtx5090-real.json`, `rtx5090-real-complete.json` | The 0.1 axis sweeps and the first complete workload matrix. |
| `rtx5090-setaside.json` | `SetAsidePolicy`, measured. Resolved at batch 1, unresolved at concurrency, and the run that did resolve did not replicate — kept because half a design was built on it. |
| `rtx5090-surfaces.json`, `rtx5090-synthetic.json` | Synthetic sweeps. They explain mechanism cheaply and decide nothing; they have disagreed with the real model on four axes. |
| `rtx5090-0.2-migration-check.json` | Did generalizing RecurLocal into TensorTransit change the recurrent policy? No. |
| `rtx5090-0.2.1-rewiring-check.json` | Did putting the 0.2 core into the measured path change it? No — both engines, one binary, one model load, the only variable being `TENSORTRANSIT_ENGINE`. |
| `rtx5090-0.2.1-arms.json` | **The five arms and the admission axis, measured** — and, in finding 8, measured in a cell whose physical ceiling is *below* two of the figures reported. What survives is that two admission rules differ from each other by more than the noise. Five findings that are not good news, recorded with the same weight as the one that is. |
| `rtx5090-cost-model-fit.json` | The residency cost model fitted to every paired hardware measurement of the `persist` arm, with its residuals and the linear model it has to beat. |
| `rtx5090-second-proof-track-model.json` | The coordination claim, at **model** level, with the control that would falsify it. `"basis": "model"`. |
| `rtx5090-0.2.1-c32-latency.json` | **The generation's nine repeats, spent on one cell.** `ctx128-c32` is what drove the first full receipt to −99.5%. Goodput resolves at −1.20% (nine of nine pairs negative); p99 does **not** (seven of nine, median 1.083, p ≈ 0.18), and the three-repeat +183% figure was sampling a bimodal distribution. Retracts a reading, not a measurement. |
| `rtx5090-0.2.1-null-control.json` | **Two byte-identical plans, measured against each other.** `survival` and `density` compile to the same plan, so the paired difference between them is noise with the policy held constant: median +0.230% at `ctx128-c16`, peak-to-peak 1.29%, against that cell's 0.249% physical ceiling. Also records an experiment that did **not** work, and why. |
| `rtx5090-ttf1-first-matrix.json` | **The first run of the whole generation**, and as much a finding about the instrument as about the policy: the receipt read −99.5%, four of ten cells turned out not to be servable as concurrency cells by this runtime at all, and one more was decided at the floor on an axis whose calibrated control spread is 481%. Contains the receipt verbatim. |

Frontier Receipts are not here. They live in [`frontier/`](../frontier), which is append-only
and where a receipt stays attached to the generation that produced it.
