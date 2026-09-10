# The Transit Frontier Ledger

Every authoritative performance evaluation in this repository ends here, as a permanent,
machine-readable **Frontier Receipt**. The ledger is append-only: a finalized receipt is never
silently rewritten, and a correction is a *new* receipt that names what it supersedes and why.

```text
frontier/
|-- TTF-1/
|   |-- generation.json        the frozen benchmark definition
|   |-- reference.json         the frozen normalization bounds, calibrated on hardware
|   |-- current-frontier.json  the running frontier and every receipt that moved it
|   `-- receipts/              one JSON per evaluated candidate
`-- README.md
```

## What a receipt says

One number, continuous:

```text
Frontier Gain: dF = F(candidate) / F(main) - 1
```

`F` is the normalized Pareto hypervolume of the serving frontier, aggregated over the
generation's workload cells. A PR earns credit for **verified marginal expansion of the
current real inference frontier beyond `main`** — not for being large, not for changing many
files, not for improving one internal counter, and not for a maintainer's judgement.

There are no `XS`/`S`/`M`/`L`/`XL` bands, and their absence is deliberate rather than
stylistic. The band table this repository used until 0.2.1 had its lowest paying step at 2%
weighted gain, and the *physical ceiling* for the whole shipped policy family is 0.52% on the
scored model and 1.94% on the best model ever found. A submission could have removed every
recoverable byte of recurrent traffic and still scored `none`. A scoring regime whose lowest
paying band sits above what the hardware can deliver is not a strict regime; it is a broken
instrument that tells every contributor the wrong thing about where the room is.

`reference.json` records what the generation's own calibration says is reachable — per cell,
in raw units, with the measured run-to-run spread beside it — so a contributor can see the
size of the prize, and the noise they have to beat, before spending a week.

## Integrity, and the one thing that is deliberately not done

Every receipt carries a `content_digest` — SHA-256 over each of its scored fields — and a
`generation_checksum` over the frozen definition it was scored under. `tt-frontier receipt
verify` recomputes both and also re-derives the status from the numbers beside it, so three
different ways of tampering are caught: a rewritten field, a re-scored run under a changed
generation, and a status that does not follow from its own statistics. `ledger audit` runs it
over every receipt in a generation.

**Receipts are not cryptographically signed, and that is a choice rather than an omission.**
Signing needs a private key, and the evaluation runs on a worker this repository insists must
carry no key at all — `scripts/trusted_eval.sh` refuses to start if it can see a credential,
because a credential in scope is a credential the submission has. Signing on the worker would
put a key exactly where the security posture says none may be. Signing *off* the worker is
possible and is where this should go if the ledger ever needs to be trusted by someone who does
not trust the operator: the evaluator emits the receipt, a separate signer holding the key
verifies its digest and countersigns. Until there is a second party who needs that, the digest
plus an append-only history in version control is what the threat model actually calls for, and
claiming more would be theatre.

## Statuses

```text
FRONTIER_GAIN            verified marginal expansion
NO_FRONTIER_GAIN         confidently no expansion
INCONCLUSIVE             the observed figure is inside its own confidence interval
CORRECTNESS_FAIL         the output changed; correctness precedes performance scoring
REGRESSION_GUARD_FAIL    a protected workload regressed past the generation's limit
BUILD_FAIL               the candidate did not build
EVAL_ERROR               the evaluator itself failed
```

These describe evaluation *state*. None of them categorizes impact magnitude.

## Using it

```bash
tools/tt-frontier generation show TTF-1        # what is frozen, and what it means
tools/tt-frontier ledger show TTF-1            # every receipt, and the compounded expansion
tools/tt-frontier ledger audit TTF-1           # every receipt still verifies against itself
tools/tt-frontier receipt verify <receipt>     # one receipt, against its generation
tools/tt-frontier report <receipt> --format markdown
```

An evaluation is run by the trusted evaluator, never by the candidate:

```bash
scripts/trusted_eval.sh --candidate <ref> --model <path> --pr 184
```

## Running one as an operator

The GPU workflow is `workflow_dispatch` and needs one repository variable:

| variable | what it is |
|---|---|
| `TT_MODEL_PATH` | absolute path to a **read-only** copy of the pinned checkpoint on the worker |

and a runner labelled `self-hosted, gpu, ephemeral` that carries no SSH key, no cloud
credential and no long-lived token. `SECURITY.md` is the contract; `scripts/trusted_eval.sh`
refuses to start if it can see a credential, so a misconfigured worker fails loudly instead of
leaking quietly.

To check the staging without a GPU, a model or a ten-minute build:

```bash
scripts/trusted_eval.sh --baseline main --candidate <ref> --stage-only
```

## Generations

A generation freezes everything that decides what a number *means*: the model and its digest,
the runtime commit, the hardware class, the workload cells and their published weights, the
objectives and their normalization bounds, the SLOs, the reference point, the aggregation
rule, the repeat policy, the confidence method, the regression guard and the hidden-seed
distribution.

Its SHA-256 is recorded in every receipt, and `tt-frontier receipt verify` refuses a receipt
whose generation no longer hashes to what it was scored under. If the meaning of the
evaluation changes materially, the answer is a new `TTF-N` — never an edit. Historical results
stay attached to the generation that produced them and stay auditable exactly as they were
earned.
