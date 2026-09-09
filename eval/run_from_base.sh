#!/usr/bin/env bash
# Score a submission with the MEASURING INSTRUMENT taken from the base commit.
#
#   eval/run_from_base.sh <base-ref> <submission-worktree> [-- real_eval.py args...]
#
# Why this exists: bench/ and eval/ define what is measured. If the evaluator runs the
# submission's copy of them, a submission can win by editing the ruler - a one-line change to
# a noise floor, a repeat count, or the estimator behind the highest-weighted arm would do it,
# and none of those look like cheating in a diff.
#
# So: the submission supplies src/, include/, integrations/ and bench/*.cu kernels it wants
# scored; this script overlays eval/ and the scoring-relevant harness from <base-ref> before
# running. It is deliberately a script and not a policy document, because a rule nothing
# enforces is a rule that holds only for honest submissions.
#
# This is methodology, not an approval workflow: it takes no view on who may submit, and
# there is no ledger, tier or payout anywhere in it.
set -euo pipefail

BASE="${1:?usage: run_from_base.sh <base-ref> <submission-worktree> [-- args...]}"
SUB="${2:?usage: run_from_base.sh <base-ref> <submission-worktree> [-- args...]}"
shift 2
[ "${1:-}" = "--" ] && shift || true

[ -d "$SUB/.git" ] || [ -f "$SUB/.git" ] || { echo "!! $SUB is not a git worktree"; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# The instrument, from base. Anything the submission changed here is discarded for scoring.
git -C "$REPO" archive "$BASE" eval | tar -x -C "$STAGE"
echo ">> instrument: eval/ from $(git -C "$REPO" rev-parse --short "$BASE")"

# Report what the submission tried to change about it, rather than silently dropping it.
CHANGED="$(git -C "$SUB" diff --name-only "$BASE" -- eval bench 2>/dev/null || true)"
if [ -n "$CHANGED" ]; then
    echo ">> NOTE: the submission modifies the instrument; these changes are NOT used for scoring:"
    echo "$CHANGED" | sed 's/^/     /'
    echo ">>       score them separately, as a change to what is measured."
fi

exec python3 "$STAGE/eval/real_eval.py" "$@"
