# Security

Treat CUDA performance PRs as untrusted code. A submission can run any kernel it likes, and
the build script it supplies runs as whatever user starts the evaluation. Do not execute
arbitrary fork PRs on persistent GPU hosts carrying SSH keys, cloud credentials, production
model credentials, writable shared data, or long-lived GitHub tokens.

Use disposable GPU runners for authoritative evaluation.

## What this repository enforces rather than asks for

`scripts/trusted_eval.sh` is the authoritative evaluation, and the posture is in the script so
that it holds wherever the script runs — not only inside a CI YAML:

- **It refuses to start with a credential in scope.** `SSH_AUTH_SOCK`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`,
  `AZURE_CLIENT_SECRET`, `HF_TOKEN`, `GITHUB_TOKEN`, `GH_TOKEN` and `DOCKER_AUTH_CONFIG` are
  checked, and a set one is a hard exit before any submission code is built. A credential this
  process can see is a credential the submission has. A non-empty `~/.ssh` is a loud warning.
  CI asserts the refusal.
- **The workspace is fresh and ephemeral.** Every run gets its own directory; `--clean` removes
  it; the GitHub job wipes it in an `always()` step. Nothing a candidate writes survives into
  the next evaluation.
- **The scoring instrument comes from the BASELINE, into both trees.** `eval/`,
  `tools/tt-frontier`, the frozen TTF generations, `schemas/`, `configs/`, `tests/golden/`,
  `workloads/` and `adapters/sparkinfer/pin.json` are overlaid from the baseline ref before
  anything is built, and what the candidate tried to change in them is *named* rather than
  silently dropped. A submission cannot win by editing the ruler, and CI proves it by staging
  a tree that relaxes a guard, moves the runtime pin and edits a frozen generation.
- **The ledger is written outside the workspace**, by the evaluator, from the receipt the
  evaluator computed. A candidate never touches the permanent history.
- **The GPU workflow is `workflow_dispatch`, not `pull_request`.** A maintainer starts it
  deliberately. Automatic execution of a stranger's CUDA on a GPU host is the thing this
  document exists to prevent.
- **The GitHub job requests `contents: read` and nothing else.** No packages, no deployments,
  no OIDC identity token. Posting the receipt to the PR needs `pull-requests: write`, so it is
  a **separate job on a hosted runner** that downloads the artifact and reads a Markdown file.
  The two never share a runner: a token in the evaluation job's environment would be a token
  the submission has.

## What an operator must still provide

None of it is code, and none of it is optional:

- An **ephemeral, keyless GPU worker** labelled `self-hosted, gpu, ephemeral`, rebuilt between
  runs. Read-only model artifacts.
- The model path as a repository variable `TT_MODEL_PATH`, pointing at a **read-only** copy.
- Enough repeats that the noise floor is real. `frontier/TTF-1/reference.json` publishes the
  measured per-cell control spread; two repeats cannot estimate one at all, and three is a floor
  rather than a recommendation — the same arm measured +0.07% and −0.39% at `ctx128-c16` in two
  sessions two hours apart on the reference box, against a published spread of 0.42% for that
  cell. The authoritative runner defaults to five and the generation allows nine.
- **Clock discipline.** Graphics clocks could not be pinned on the reference box, so only
  paired same-box deltas are trustworthy. The harness is built around that; do not compare
  across boxes or across days.

## Reporting

Open a private security advisory on the repository rather than a public issue.
