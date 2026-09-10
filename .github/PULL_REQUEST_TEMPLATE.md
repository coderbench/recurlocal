<!--
Spec section 92. Nothing here is a checkbox to tick: every field is a question whose answer
this repository has, at some point, got wrong in a way that a number alone would have hidden.

Delete any section that genuinely does not apply and say why, rather than leaving it blank.
-->

## What changed?

## Which tensor-movement / locality problem does this target?

<!-- Name the surface. docs/MINING.md maps them, docs/OPTIMIZATION-SURFACES.md has the flag
that isolates each one. "General performance" is not a surface. -->

## Why should it help?

<!-- The mechanism, not the result. If the answer is "the cache should keep more", say what it
should keep, what it should stop keeping to make room, and what reads it again. -->

## Is this evidence about a planner, or about a speedup?

<!-- `tensortransit plan` and `tensortransit compare` print PREDICTED figures from a cost
model whose assumptions are in docs/evaluation.md. They are evidence about a planner, and the
cost model's own ranking of admission rules is currently CONTRADICTED by measurement
(docs/VERDICT.md section 3.3). Only `tools/tt-frontier` and `eval/decide.py --real` are
evidence about a speedup. A PR that improves a predicted figure has improved a model, which is
a legitimate contribution -- say so plainly and do not present the number as a gain. -->

## Correctness

<!-- Exact mode requires bit-identical greedy replay against the unhooked runtime. Paste the
gate's verdict. If the runtime is not reproducible against itself on your checkpoint, the
harness reports `inconclusive` and that is not your fault -- include it anyway. -->

## Baseline

<!-- Commit, model, checkpoint digest, runtime commit, GPU, driver, CUDA version. -->

## Candidate

<!-- The exact environment. One binary, hook inert unless the environment names a mode. -->

## Real-model result

<!-- The Frontier Receipt, rendered by the tool rather than retyped:

    tools/tt-frontier report <receipt.json> --format comment

Every figure in that output is generated from the receipt, and the receipt from the raw
measurements. Do not type a benchmark number into this box.

Read what the receipt says about itself before pasting it:
  - `coverage.partial` -- a matrix that did not score every cell credits nothing, whatever it
    measured, and names the cells responsible in `frontier.credit_withheld`.
  - `coverage.unservable_cells` -- cells the evaluator's own probe could not serve with no
    policy running. Not your regression, and not scored.
  - `aggregation.cells_at_floor` -- a cell whose ratio was set by the generation's floor rather
    than by a measurement. One of these can move dF by a large multiple.
  - `resolved` and the noise floor. A gain smaller than its own run-to-run spread is an open
    axis, not a result, and both gates have to pass. -->

## Supporting hardware counters

<!-- Optional and often impossible: `RmProfilingAdminOnly: 1` on some hosts blocks ncu, nsys
--gpu-metrics and CUPTI's profiler together. Counter-free tracing still times a stall.
Counters explain WHY a change works; they never replace the throughput number. -->

## Reproduction command

<!-- Everything needed to re-run this from a clean clone, including the workload matrix. -->

## Regression notes

<!-- The generation names its protected cells and its own regression limit, and the receipt
reports `regression_guard` against them; a violation is a REGRESSION_GUARD_FAIL and not a
judgement call. For anything outside that set, say which cell regressed and why the trade is
worth taking -- a tradeoff track has to be argued for, not assumed. Both objectives count: a
policy that raises goodput while tripling p99 inter-token latency has moved along the frontier,
not expanded it. -->

## Guards

<!-- If this PR touches eval/, say which guard you changed and which incident it encodes. Each
one exists because the failure it prevents actually happened here, and a broken evaluator
still prints a confident number. eval/run_from_base.sh runs the instrument from the base
commit rather than from your branch, for exactly this reason. -->
