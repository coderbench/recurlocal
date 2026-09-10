#!/usr/bin/env bash
# The authoritative main-vs-candidate evaluation, run by the trusted evaluator.
#
#   scripts/trusted_eval.sh --candidate <ref> [--baseline <ref>] [--generation TTF-1] ...
#
# WHAT MAKES IT TRUSTED
#
# A performance PR is arbitrary CUDA from a stranger, so the runner is treated as a hostile
# execution environment and the scoring is kept outside the candidate's reach:
#
#   * The WORKSPACE is fresh and ephemeral. Every run gets its own directory under
#     $TT_WORK, and --clean removes it afterwards. Nothing a candidate writes survives into
#     the next evaluation.
#   * The INSTRUMENT comes from the BASELINE, not from the candidate. eval/, the frozen TTF
#     generations, the workload definitions and the runtime pin are overlaid from the
#     baseline ref before anything runs -- so a submission cannot win by editing the ruler,
#     and what it tried to change is reported instead of silently dropped.
#   * The LEDGER is written outside the workspace, by this script, from the receipt the
#     trusted evaluator computed. A candidate never touches it.
#   * NO SECRETS. This script reads no credential and needs none. It refuses to run with an
#     ssh-agent or a cloud token in the environment rather than exposing one to a build it is
#     about to execute.
#
# It is deliberately a script and not a GitHub Action: the Action is one caller, an operator
# on a box is another, and a rule that only exists inside a CI YAML is a rule that holds only
# where that YAML runs.
set -euo pipefail

BASELINE="main"
CANDIDATE=""
GENERATION="TTF-1"
MODEL="${TT_MODEL:-}"
REPEATS=3
CELLS=""
CLEAN=0
PR=""
ARCH="${CMAKE_CUDA_ARCHITECTURES:-120}"
WORK_ROOT="${TT_WORK:-/tmp/tt-eval}"
LEDGER="${TT_LEDGER_DIR:-}"
CANDIDATE_CONFIGS=()
MAIN_CONFIGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --candidate)        CANDIDATE="$2"; shift 2 ;;
        --baseline)         BASELINE="$2"; shift 2 ;;
        --generation)       GENERATION="$2"; shift 2 ;;
        --model)            MODEL="$2"; shift 2 ;;
        --repeats)          REPEATS="$2"; shift 2 ;;
        --cells)            CELLS="$2"; shift 2 ;;
        --pr)               PR="$2"; shift 2 ;;
        --candidate-config) CANDIDATE_CONFIGS+=("$2"); shift 2 ;;
        --main-config)      MAIN_CONFIGS+=("$2"); shift 2 ;;
        --clean)            CLEAN=1; shift ;;
        *) echo "!! unknown option $1"; exit 2 ;;
    esac
done
[ -n "$CANDIDATE" ] || { echo "!! --candidate <ref> is required"; exit 2; }
[ -n "$MODEL" ] || { echo "!! --model <path> is required (or set TT_MODEL)"; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- refuse to run next to a secret ------------------------------------------------------
# A candidate's build scripts and kernels run as this user. Anything reachable from this
# environment is reachable by them.
for var in SSH_AUTH_SOCK AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY GOOGLE_APPLICATION_CREDENTIALS \
           AZURE_CLIENT_SECRET HF_TOKEN GITHUB_TOKEN GH_TOKEN DOCKER_AUTH_CONFIG; do
    if [ -n "${!var:-}" ]; then
        echo "!! $var is set. This script executes untrusted code from a submission; a"
        echo "!! credential in its environment is a credential the submission has. Unset it"
        echo "!! and re-run, or run this on a worker that never had one."
        exit 3
    fi
done
if [ -d "$HOME/.ssh" ] && [ -n "$(ls -A "$HOME/.ssh" 2>/dev/null)" ]; then
    echo ">> WARNING: $HOME/.ssh is not empty. Overview section 41 asks for a worker with no"
    echo ">>          SSH keys; this run is not on one."
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$REPO" rev-parse --short "$CANDIDATE")"
WORK="$WORK_ROOT/$RUN_ID"
mkdir -p "$WORK"
echo ">> workspace $WORK"
if [ "$CLEAN" = 1 ]; then trap 'rm -rf "$WORK"' EXIT; fi

BASE_SHA="$(git -C "$REPO" rev-parse "$BASELINE")"
CAND_SHA="$(git -C "$REPO" rev-parse "$CANDIDATE")"
echo ">> baseline  $BASE_SHA"
echo ">> candidate $CAND_SHA"
if [ "$BASE_SHA" = "$CAND_SHA" ]; then
    echo ">> NOTE: baseline and candidate are the same commit. This measures the harness's own"
    echo ">>       noise floor, which is a useful thing to measure and is not a submission."
fi

# --- two fresh checkouts, neither of which can see the other ------------------------------
for side in baseline candidate; do
    sha="$BASE_SHA"; [ "$side" = candidate ] && sha="$CAND_SHA"
    dir="$WORK/$side"
    rm -rf "$dir"; mkdir -p "$dir"
    git -C "$REPO" archive "$sha" | tar -x -C "$dir"
    echo ">> $side tree at $dir"
done

# --- the instrument, from the BASELINE, into BOTH trees -----------------------------------
# The candidate is scored on src/, include/, planners/, executors/ and adapters/ -- what it
# is proposing. Everything that decides WHAT IS MEASURED comes from the baseline, in both
# trees, so the two builds are compared by the same ruler.
INSTRUMENT=(eval tools/tt-frontier schemas configs tests/golden workloads adapters/sparkinfer/pin.json)
present=()
for path in "${INSTRUMENT[@]}"; do
    git -C "$REPO" cat-file -e "$BASE_SHA:$path" 2>/dev/null && present+=("$path") || \
        echo ">> NOTE: $path is not in the baseline; nothing to overlay."
done
CHANGED="$(git -C "$REPO" diff --name-only "$BASE_SHA" "$CAND_SHA" -- "${INSTRUMENT[@]}" || true)"
if [ -n "$CHANGED" ]; then
    echo ">> NOTE: the candidate modifies the instrument; these changes are NOT used for scoring:"
    echo "$CHANGED" | sed 's/^/     /'
fi
for side in baseline candidate; do
    git -C "$REPO" archive "$BASE_SHA" "${present[@]}" | tar -x -C "$WORK/$side"
done
echo ">> instrument: ${present[*]} from $BASE_SHA (into both trees)"

# --- build both, each against its OWN TensorTransit ---------------------------------------
# One SparkInfer commit, one patch, one model: the ONLY difference between the two binaries
# is which TensorTransit they link. That is what makes the comparison about the submission.
export PATH="/usr/local/cuda/bin:$PATH"
export CMAKE_CUDA_ARCHITECTURES="$ARCH"
for side in baseline candidate; do
    echo ">> building $side"
    ( cd "$WORK/$side" && ./adapters/sparkinfer/build.sh "$WORK/$side/si" ) \
        > "$WORK/$side-build.log" 2>&1 || {
            echo "!! $side build FAILED; see $WORK/$side-build.log"
            tail -40 "$WORK/$side-build.log"
            [ "$side" = candidate ] && { echo "BUILD_FAIL"; exit 4; }
            exit 5
        }
done
BASE_CB="$WORK/baseline/si/sparkinfer/build/runtime/qwen3_gguf_cb_bench"
CAND_CB="$WORK/candidate/si/sparkinfer/build/runtime/qwen3_gguf_cb_bench"
CAND_GEN="$WORK/candidate/si/sparkinfer/build/runtime/qwen3_gguf_generate"

# --- the paired evaluation, driven by the BASELINE's tt-frontier --------------------------
[ ${#MAIN_CONFIGS[@]} -gt 0 ] || MAIN_CONFIGS=("control=")
[ ${#CANDIDATE_CONFIGS[@]} -gt 0 ] || \
    CANDIDATE_CONFIGS=("persist=TENSORTRANSIT=persist,TENSORTRANSIT_WINDOW_ATTACH=capture_node")

args=(run --generation "$GENERATION" --model "$MODEL"
      --cb-binary "$BASE_CB" --candidate-cb "$CAND_CB" --generate "$CAND_GEN"
      --repeats "$REPEATS" --output "$WORK/raw.json"
      --baseline-commit "$BASE_SHA" --candidate-commit "$CAND_SHA"
      --evaluator-commit "$(git -C "$REPO" rev-parse HEAD)"
      --runtime-commit "$(python3 -c "import json;print(json.load(open('$WORK/baseline/adapters/sparkinfer/pin.json'))['commit'])")"
      --model-name "$(basename "$MODEL")"
      --environment-fingerprint "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tr -d ' \n')-cuda$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')")
[ -n "$CELLS" ] && args+=(--cells "$CELLS")
for c in "${MAIN_CONFIGS[@]}"; do args+=(--main-config "$c"); done
for c in "${CANDIDATE_CONFIGS[@]}"; do args+=(--candidate-config "$c"); done

echo ">> paired evaluation"
python3 "$WORK/baseline/tools/tt-frontier" "${args[@]}"

# --- the receipt, computed by the BASELINE's scorer ---------------------------------------
echo ">> receipt"
compute=(compute --generation "$GENERATION" --results "$WORK/raw.json"
         --output "$WORK/receipt.json")
[ -n "$PR" ] && compute+=(--pr "$PR")
[ -n "$CELLS" ] && compute+=(--allow-partial)
set +e
python3 "$WORK/baseline/tools/tt-frontier" "${compute[@]}"
COMPUTE_RC=$?
set -e

python3 "$WORK/baseline/tools/tt-frontier" report "$WORK/receipt.json" --format markdown \
    --output "$WORK/report.md"
python3 "$WORK/baseline/tools/tt-frontier" report "$WORK/receipt.json" --format comment \
    --output "$WORK/comment.md"
python3 "$WORK/baseline/tools/tt-frontier" report "$WORK/receipt.json" --format box

# --- the ledger, written OUTSIDE the workspace --------------------------------------------
if [ -n "$LEDGER" ]; then
    echo ">> ledger $LEDGER"
    TT_LEDGER_DIR="$LEDGER" python3 "$REPO/tools/tt-frontier" ledger append \
        --receipt "$WORK/receipt.json" ${PR:+--id "pr-$(printf '%06d' "$PR")"}
fi

echo
echo "artifacts:"
echo "  raw       $WORK/raw.json"
echo "  receipt   $WORK/receipt.json"
echo "  report    $WORK/report.md"
echo "  comment   $WORK/comment.md"
exit $COMPUTE_RC
