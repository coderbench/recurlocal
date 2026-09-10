#!/usr/bin/env bash
# Score a submission with the MEASURING INSTRUMENT taken from the base commit.
#
#   eval/run_from_base.sh <base-ref> <submission-worktree> [-- args...]
#   TT_ENTRY=frontier eval/run_from_base.sh <base-ref> <worktree> [-- tt-frontier args...]
#
# Why this exists: eval/, tools/tt-frontier, the workload definitions and the runtime pin
# define what is measured. If the evaluator runs the SUBMISSION's copy of them, a submission
# can win by editing the ruler - a one-line change to a noise floor, a repeat count, a
# normalization bound, a frozen generation, or the runtime commit the whole comparison is
# against. None of those look like cheating in a diff.
#
# So: the submission supplies src/, include/, planners/, executors/ and adapters/ - the parts
# it is being scored ON - and this script overlays the instrument from <base-ref> before
# running. It is deliberately a script and not a policy document, because a rule nothing
# enforces is a rule that holds only for honest submissions.
#
# This is methodology, not an approval workflow: it takes no view on who may submit, and there
# is no tier or payout anywhere in it.
set -euo pipefail

BASE="${1:?usage: run_from_base.sh <base-ref> <submission-worktree> [-- args...]}"
SUB="${2:?usage: run_from_base.sh <base-ref> <submission-worktree> [-- args...]}"
shift 2
[ "${1:-}" = "--" ] && shift || true

[ -d "$SUB/.git" ] || [ -f "$SUB/.git" ] || { echo "!! $SUB is not a git worktree"; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Everything that decides WHAT IS MEASURED rather than how well the candidate does it.
#
# Kept in one list, used for both the overlay and the report, so the two cannot drift: a path
# that is reported as instrument but not overlaid is a hole, and one that is overlaid but not
# reported is a silent discard. `adapters/sparkinfer/pin.json` is on it because it names the
# runtime commit and the model the whole comparison is against - moving the pin changes the
# answer without touching a line of evaluator code.
INSTRUMENT=(
    eval
    tools/tt-frontier
    schemas
    configs
    tests/golden
    workloads
    adapters/sparkinfer/pin.json
)

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# The instrument, from base. Anything the submission changed here is discarded for scoring.
# `git archive` fails loudly on a path the base ref does not contain, which is what should
# happen: a rename that was not reflected here is a hole in the anti-gaming property, and a
# silent skip is exactly how it would go unnoticed.
present=()
for path in "${INSTRUMENT[@]}"; do
    if git -C "$REPO" cat-file -e "$BASE:$path" 2>/dev/null; then
        present+=("$path")
    else
        echo ">> NOTE: $path is not in $BASE; nothing to overlay from there."
    fi
done
git -C "$REPO" archive "$BASE" "${present[@]}" | tar -x -C "$STAGE"
echo ">> instrument: ${present[*]} from $(git -C "$REPO" rev-parse --short "$BASE")"

# Report what the submission tried to change about it, rather than silently dropping it.
CHANGED="$(git -C "$SUB" diff --name-only "$BASE" -- "${INSTRUMENT[@]}" 2>/dev/null || true)"
if [ -n "$CHANGED" ]; then
    echo ">> NOTE: the submission modifies the instrument; these changes are NOT used for scoring:"
    echo "$CHANGED" | sed 's/^/     /'
    echo ">>       score them separately, as a change to what is measured."
fi

# Self-check: the staged instrument has to be runnable, or the run fails later with a stack
# trace that reads like a submission bug.
[ -f "$STAGE/eval/real_eval.py" ] || { echo "!! $BASE has no eval/real_eval.py"; exit 2; }

case "${TT_ENTRY:-real_eval}" in
    real_eval)
        exec python3 "$STAGE/eval/real_eval.py" "$@"
        ;;
    frontier)
        [ -f "$STAGE/tools/tt-frontier" ] || {
            echo "!! $BASE has no tools/tt-frontier; use TT_ENTRY=real_eval against it"; exit 2; }
        # tt-frontier resolves eval/ and the frozen generations relative to its own parent,
        # so running the staged copy runs the staged generations too - which is the point:
        # a frozen TTF-N is part of the ruler.
        exec python3 "$STAGE/tools/tt-frontier" "$@"
        ;;
    *)
        echo "!! TT_ENTRY must be real_eval or frontier, not '${TT_ENTRY}'"
        exit 2
        ;;
esac
