"""The paired, interleaved GPU runner that produces raw frontier results.

It does not reimplement measurement. It CALLS `eval/real_eval.py`, because every guard in
that file encodes an incident this project actually had -- a contaminated control arm, a null
candidate wearing a policy's telemetry, an arm that lost requests to a device OOM and reported
the loss as a slowdown, an arm that silently fell off the batched decode path. A second
measurement path would be a second place for all of those to come back.

What this file adds is the shape the frontier needs:

* **Interleaving.** main repeat k, then candidate repeat k, then main repeat k+1. Never all of
  main and then all of the candidate: on a box whose graphics clocks cannot be pinned, thermal
  drift lands entirely on whichever arm ran last.
* **A portfolio per variant.** A cell's frontier is formed from every allowed configuration of
  that variant, so a candidate that adds a specialized planner is measured on the territory it
  creates rather than on whether it beats everything everywhere.
* **Failures priced as failures.** An OOM, a timeout or a fall off the batched path produces
  NO operating point, and the receipt says which and how many. A failed region is territory
  lost, not a slow measurement (spec section 49).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import real_eval  # noqa: E402  the instrument, imported rather than reimplemented

RESULT_SCHEMA_VERSION = 1

# A run whose guard fired produced no operating point. The reason is kept verbatim so the
# receipt can say which failure it was rather than "something went wrong".
GUARD_STATUS = (
    (re.compile(r"REQUESTS DID NOT COMPLETE|out of memory", re.I), "OOM"),
    (re.compile(r"FELL OFF THE BATCHED DECODE PATH", re.I), "UNBATCHED"),
    (re.compile(r"NULL CANDIDATE", re.I), "NULL_POLICY"),
    (re.compile(r"did not initialise|emitted no RECURLOCAL_STATS", re.I), "UNHOOKED"),
)


class RunnerError(RuntimeError):
    pass


def declares_a_policy(env) -> bool:
    """Does this configuration ASK for a policy, and therefore have to deliver one?

    `real_eval.py` refuses a run whose telemetry reads `windows_applied 0,
    windows_attached_to_node 0, pre_touch_launches 0` -- the first scored run in this
    repository was exactly that, and scoring it would have reported the hook's overhead as a
    locality result. But two legitimate configurations apply no policy on purpose: the true
    control, and the `baseline` arm, which is the hook installed with no window and is how the
    hook's own cost is measured. Refusing those would make the control unmeasurable.

    So the question is what the environment ASKED for, not what came back.
    """
    if not env:
        return False
    mode = env.get("TENSORTRANSIT") or env.get("RECURLOCAL") or ""
    preset = env.get("TENSORTRANSIT_PRESET") or ""
    planner = env.get("TENSORTRANSIT_PLANNER") or ""
    if mode in ("off", "0", "baseline"):
        return False
    if preset == "baseline" or planner == "baseline":
        return False
    return True


def classify_guard(message: str) -> str:
    for pattern, status in GUARD_STATUS:
        if pattern.search(message):
            return status
    return "EVAL_ERROR"


def parse_cell(cell_id: str):
    """`ctx4096-c16` -> (4096, 16). The generation names its cells; this reads the convention."""
    match = re.fullmatch(r"ctx(\d+)-c(\d+)", cell_id)
    if not match:
        raise RunnerError(
            f"cell id {cell_id!r} does not follow the ctx<N>-c<M> convention this runner "
            f"knows how to execute. A generation is free to name cells anything, but then it "
            f"needs a runner that knows what they mean -- a runner that guessed would measure "
            f"the wrong workload under the right label.")
    return int(match.group(1)), int(match.group(2))


def measure_cell(cb_binary, model, cell_id, env, label, *, max_new, long_prefill,
                 expect_policy, verbose=False):
    """One configuration, one cell, once. Returns (metrics, status, detail).

    Every cell goes through the continuous-batching bench, INCLUDING concurrency 1. One
    binary and one code path for the whole matrix is what makes the cells comparable: a
    matrix whose low-concurrency corner came from a different bench would have a step in it
    that no policy caused.
    """
    prompt_len, concurrency = parse_cell(cell_id)
    try:
        tps, stats, out = real_eval.measure_concurrent(
            cb_binary, model, concurrency, prompt_len, max_new, long_prefill,
            env, label, verbose)
    except SystemExit as exc:
        message = str(exc)
        status = classify_guard(message)
        # A SERVING failure and a CONFIGURATION failure are different things and must be
        # treated differently.
        #
        #   OOM, TIMEOUT, UNBATCHED   the runtime tried and could not serve this workload.
        #                             That is territory lost, it is real information about the
        #                             candidate, and the frontier is built to price it -- so it
        #                             is recorded and the matrix continues.
        #
        #   NULL_POLICY, UNHOOKED     the candidate ASKED for a policy and its own telemetry
        #                             says none reached a kernel. Nothing is wrong with the
        #                             serving; what is wrong is that the harness is about to
        #                             measure something other than what was asked for. Scoring
        #                             it would report the hook's overhead as a locality result,
        #                             which is what the first scored run in this repository
        #                             did. That aborts, as `eval/real_eval.py` aborts.
        if status in ("NULL_POLICY", "UNHOOKED") and expect_policy:
            raise RunnerError(
                f"{label}: the configuration asked for a policy and applied none. This is a "
                f"configuration failure, not a serving one -- the runtime served fine and the "
                f"number would be the hook's overhead wearing a policy's name. Check "
                f"TENSORTRANSIT_WINDOW_ATTACH: without capture_node the windows are all "
                f"deferred to a runtime that never attaches them.\n{message.strip()[:600]}")
        if status == "NULL_POLICY" and not expect_policy:
            # A configuration that declares no policy is SUPPOSED to apply none. Refusing it
            # would make the true control unmeasurable, which is the one arm every comparison
            # needs, and the `baseline` arm -- the hook with no window -- unmeasurable too.
            raise RunnerError(f"{label}: a configuration that declares no policy was refused "
                              f"as a null candidate; this is an evaluator bug, not a "
                              f"submission one")
        return {}, status, {"guard": message.strip()[:800]}
    except subprocess.TimeoutExpired as exc:
        # A hung run is a serving failure like any other, and it must not take the other
        # fifty-nine down with it. A matrix that aborts two thirds of the way through has
        # spent an hour of device time and produced nothing scoreable.
        return {}, "TIMEOUT", {"guard": f"{label}: timed out after {exc.timeout}s"}
    except Exception as exc:  # noqa: BLE001 -- deliberately broad; see below
        # Anything else the harness did not anticipate. Recorded as a failure of THIS cell
        # rather than allowed to end the run, for the same reason: the receipt can say a cell
        # produced no operating point and why, and a traceback in a log cannot.
        return {}, "EVAL_ERROR", {"guard": f"{label}: {type(exc).__name__}: {exc}"[:800]}

    metrics = {"goodput_tps": tps}
    for key, pattern in (("p99_itl_ms", r"p99_itl_ms=([0-9.]+)"),
                         ("p95_itl_ms", r"p95_itl_ms=([0-9.]+)"),
                         ("p50_itl_ms", r"p50_itl_ms=([0-9.]+)"),
                         ("mean_itl_ms", r"mean_itl_ms=([0-9.]+)"),
                         ("max_itl_ms", r"max_itl_ms=([0-9.]+)")):
        found = re.search(pattern, out)
        if found:
            metrics[key] = float(found.group(1))
    detail = {"adapter": stats}
    return metrics, "OK", detail


def run_matrix(*, generation, cells, model, variants, repeats, max_new, long_prefill,
               settle_seconds=30, verbose=True, on_record=None):
    """Interleaved paired execution of the whole matrix.

    `variants` is {"main": {...}, "candidate": {...}}, each with `cb_binary` and
    `configs` -- a list of {"config_id", "env", "expect_policy"}.
    """
    for name in ("main", "candidate"):
        if name not in variants:
            raise RunnerError(f"no {name} variant: a frontier is a comparison")
    records = []
    for repeat in range(1, repeats + 1):
        for cell in cells:
            # Interleaved at the innermost level that still costs one model load per run:
            # main and candidate for the SAME cell and the SAME repeat run adjacently, so a
            # thermal excursion lands on both or neither.
            for variant in ("main", "candidate"):
                spec = variants[variant]
                for config in spec["configs"]:
                    if settle_seconds:
                        # The box operating rule that cost this project a result: two evals
                        # racing for VRAM turn the loser into a plausible-looking number.
                        time.sleep(settle_seconds)
                    label = f"{variant}/{config['config_id']}/{cell}/r{repeat}"
                    if verbose:
                        print(f">> {label}", flush=True)
                    metrics, status, detail = measure_cell(
                        spec["cb_binary"], model, cell, config["env"], label,
                        max_new=max_new, long_prefill=long_prefill,
                        expect_policy=config.get("expect_policy", True), verbose=verbose)
                    record = {
                        "result_schema_version": RESULT_SCHEMA_VERSION,
                        "generation": generation.name,
                        "workload_id": cell,
                        "variant": variant,
                        "config_id": config["config_id"],
                        "repeat": repeat,
                        "metrics": metrics,
                        "status": status,
                        "correctness": "PASS",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "detail": detail,
                    }
                    records.append(record)
                    if on_record:
                        on_record(record)
                    if verbose:
                        summary = (f"goodput={metrics.get('goodput_tps', 0):.1f} "
                                   f"p99_itl={metrics.get('p99_itl_ms', 0):.2f}"
                                   if status == "OK" else f"** {status} **")
                        print(f"   {summary}", flush=True)
    return records


def correctness_gate(generate_binary, model, *, prompt_ids, gate_tokens, control_env,
                     candidate_envs, replays=3, verbose=True, baseline_generate=None):
    """Token-exact greedy replay, and the reproducibility check that has to precede it.

    A gate that compared ONE control replay to ONE candidate replay cannot tell "the policy
    changed the output" from "this runtime is not reproducible", and it has reported the
    second as the first. So the control is replayed against itself first, and a runtime that
    does not reproduce is reported as unscorable rather than as a candidate failure.

    `baseline_generate` is the BASELINE build's binary, and passing it is what makes this a gate
    on the SUBMISSION rather than on the policy. Without it the control replays run the
    candidate's own binary with a scrubbed environment, so a candidate whose *inert* path
    changed the model's output would be compared against its own changed output and pass. The
    trusted evaluator has both builds and passes it; a contributor iterating on one build does
    not, and gets the weaker question answered -- which the receipt says.
    """
    control_binary = baseline_generate or generate_binary
    base, _ = real_eval.greedy_replay(control_binary, model, prompt_ids, gate_tokens,
                                      control_env, "control")
    reproducible = True
    for index in range(1, max(replays, 1)):
        again, _ = real_eval.greedy_replay(control_binary, model, prompt_ids, gate_tokens,
                                           control_env, f"control replay {index}")
        if again != base:
            reproducible = False
            break
    if not reproducible:
        return {"correctness": "UNSCORABLE", "runtime_reproducible": False,
                "control_binary": control_binary,
                "compared_against": "baseline build" if baseline_generate else "candidate build",
                "method": "greedy replay, token-exact", "tokens_compared": gate_tokens,
                "note": ("Two unhooked control replays diverged. The gate cannot answer "
                         "whether a policy changed the output on this checkpoint, so no "
                         "result from it is scorable -- this is a property of the runtime "
                         "and the checkpoint, not of the candidate.")}

    for config_id, env in candidate_envs:
        out, _ = real_eval.greedy_replay(generate_binary, model, prompt_ids, gate_tokens,
                                         env, f"candidate {config_id}")
        if out != base:
            first = next((i for i, (a, b) in enumerate(zip(base, out)) if a != b),
                         min(len(base), len(out)))
            return {"correctness": "FAIL", "runtime_reproducible": True,
                    "method": "greedy replay, token-exact", "tokens_compared": gate_tokens,
                    "failing_config": config_id, "first_divergence": first,
                    "control_binary": control_binary,
                    "compared_against": ("baseline build" if baseline_generate
                                         else "candidate build")}
        if verbose:
            print(f"    {config_id}: identical over {gate_tokens} tokens", flush=True)
    return {"correctness": "PASS", "runtime_reproducible": True,
            "method": "greedy replay, token-exact", "tokens_compared": gate_tokens,
            "control_replays": replays,
            "control_binary": control_binary,
            # Which question was actually answered. Against the BASELINE build it is "this
            # submission does not change the model's output"; against the candidate's own it is
            # only "enabling the policy does not change it", which a submission whose inert path
            # changed the output would pass.
            "compared_against": "baseline build" if baseline_generate else "candidate build"}


def write_raw(path, records, provenance):
    document = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "provenance": provenance,
        "results": records,
    }
    Path(path).write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return path
