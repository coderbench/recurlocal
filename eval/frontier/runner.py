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
from collections import defaultdict
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
    (re.compile(r"UNMEASURABLE ARM", re.I), "UNMEASURABLE"),
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


# A guard that says the runtime TRIED to serve this workload and could not. Distinct from a
# configuration failure, which aborts, and from EVAL_ERROR, which is the harness's own fault
# and is never attributed away from the candidate.
SERVING_GUARDS = ("OOM", "TIMEOUT", "UNBATCHED")

# The hook with no window: the same telemetry, no policy.
#
# `RECURLOCAL_STATS` is not optional here and leaving it out cost a whole probe. The adapter
# prints its stats line only when asked, `require_hook_engaged` refuses a run that printed none,
# and the refusal classifies as UNHOOKED -- which is not a serving guard, so every probe came
# back "candidate" and every loss stayed charged. Conservative, and useless. The runner adds
# this to every candidate configuration; the probe's environment is built here and was missing
# it.
ATTRIBUTION_ENV = {"TENSORTRANSIT": "baseline", "TENSORTRANSIT_WINDOW_ATTACH": "capture_node",
                   "RECURLOCAL_STATS": "1"}


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


def wait_for_free_device(settle_seconds, *, timeout=180.0, verbose=False):
    """Wait for the previous run's weights to be released, THEN settle.

    The box operating rule is that two evals racing for VRAM turn the loser into a
    plausible-looking number, and the remedy has been a fixed sleep between runs. A fixed sleep
    is the crude form of the right idea: what has to be true before the next 18 GB allocation is
    that the device is free, and that is a condition rather than a duration. Measured on the
    reference box, a 35-second settle over a 60-run matrix is 35 minutes of a GPU reading 0%
    utilisation, and the device is usually free within two.

    So: poll until no compute process holds the device, then settle for `settle_seconds`. The
    guarantee is stronger than the old one -- the old sleep could expire while a process was
    still holding memory -- and the wall time is shorter.

    Falls back to the plain sleep wherever `nvidia-smi` cannot be run, because a harness that
    skipped the wait when it could not check would be the failure this exists to prevent.
    """
    if not settle_seconds:
        return {"waited_for_device_s": 0.0, "settled_s": 0.0, "method": "disabled"}
    started = time.time()
    method = "device-free"
    while True:
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            method = "fixed sleep (nvidia-smi unavailable)"
            break
        if done.returncode != 0:
            method = "fixed sleep (nvidia-smi failed)"
            break
        if not done.stdout.strip():
            break
        if time.time() - started > timeout:
            method = "device still busy after timeout"
            break
        time.sleep(0.5)
    waited = time.time() - started
    if verbose and waited > 1.0:
        print(f"    waited {waited:.1f}s for the device to clear", flush=True)
    time.sleep(settle_seconds)
    return {"waited_for_device_s": round(waited, 2),
            "settled_s": float(settle_seconds), "method": method}


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
        if status in ("NULL_POLICY", "UNHOOKED", "UNMEASURABLE") and expect_policy:
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
                         ("max_itl_ms", r"max_itl_ms=([0-9.]+)"),
                         # How many gaps the percentiles were computed FROM. Nearest-rank p99
                         # over N samples picks index floor(0.99N), which is the maximum for
                         # any N below about a hundred -- so without this a receipt cannot
                         # tell a tail from an extremum, and the first full TTF-1 matrix had a
                         # cell whose p99 and max were the same number in two repeats of three.
                         ("itl_samples", r"itl_samples=([0-9]+)")):
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
                    # The box operating rule that cost this project a result: two evals racing
                    # for VRAM turn the loser into a plausible-looking number. Waiting for the
                    # device to be FREE and then settling is a stronger guarantee than sleeping
                    # and hoping, and it is faster.
                    wait_for_free_device(settle_seconds, verbose=verbose)
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


def cells_needing_attribution(records, cells):
    """Cells the candidate lost to a SERVING guard in every repeat, where main produced points.

    The incident this exists for: the first full TTF-1 matrix lost four long-context
    concurrency cells to `RUNTIME FELL OFF THE BATCHED DECODE PATH`, and the control looked
    perfect at all four -- because the control is unhooked by construction, emits no packing
    telemetry, and therefore CANNOT be seen to fall off the same path. The guard fires on the
    arm that can be observed, not on the arm that failed.

    Charged to the candidate, such a cell is scored at the generation's floor, and one
    floor-decided cell moves dF through the geometric mean by more than any policy in this
    repository ever has. So a cell in this list is not yet a regression; it is a question.
    """
    lost = defaultdict(lambda: {"guarded": 0, "usable": 0})
    main_ok = defaultdict(int)
    for record in records:
        cell = record["workload_id"]
        status = record.get("status", "OK")
        if record["variant"] == "main":
            if status == "OK":
                main_ok[cell] += 1
            continue
        if status in SERVING_GUARDS:
            lost[cell]["guarded"] += 1
        elif status == "OK":
            lost[cell]["usable"] += 1
    return [cell for cell in cells
            if lost[cell]["guarded"] and not lost[cell]["usable"] and main_ok[cell]]


def concurrency_scaling(records, cells):
    """How much throughput the MAIN arm actually got for the concurrency it asked for.

    Independent of any adapter telemetry, and that is the point: the packed-path guard reads
    the adapter's counters, the control is unhooked, and so a control that also fell off the
    batched path cannot be seen to. This can see it. A cell at concurrency N is compared with
    the SAME-CONTEXT c=1 cell of the same matrix, both measured on the same box minutes apart:

        ctx128-c16   8.17x for 16 sequences        the runtime is batching
        ctx4096-c16  2.61x for 16 sequences        it is not

    Returns {cell: {"scale", "concurrency", "share_of_ideal", "baseline_cell"}} for every cell
    whose context also has a c=1 cell in the matrix. Diagnostic, never a score: it is recorded
    beside an attribution so the verdict can be read against arithmetic as well as a probe.
    """
    goodput = defaultdict(list)
    for record in records:
        if record["variant"] != "main" or record.get("status", "OK") != "OK":
            continue
        value = (record.get("metrics") or {}).get("goodput_tps")
        if value:
            goodput[record["workload_id"]].append(float(value))

    def median(values):
        ordered = sorted(values)
        n = len(ordered)
        return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])

    single = {}
    for cell, values in goodput.items():
        try:
            prompt, concurrency = parse_cell(cell)
        except RunnerError:
            continue
        if concurrency == 1:
            single[prompt] = median(values)

    out = {}
    for cell in cells:
        try:
            prompt, concurrency = parse_cell(cell)
        except RunnerError:
            continue
        if concurrency < 2 or prompt not in single or not goodput.get(cell):
            continue
        base = single[prompt]
        if not base:
            continue
        scale = median(goodput[cell]) / base
        out[cell] = {"scale": round(scale, 3), "concurrency": concurrency,
                     "share_of_ideal": round(scale / concurrency, 3),
                     "baseline_cell": f"ctx{prompt}-c1"}
    return out


# A run may exceed its group's median by this much before it is called an outlier. Generous by
# design: across 77 measured runs the healthy groups vary by at most one step and the single
# collapsed run is 9.6x its group's median, so anything between about 2 and 8 separates them.
SCHEDULING_OUTLIER_RATIO = 3.0


def scheduling_outliers(records):
    """Runs that spent their time somewhere other than decoding the workload.

    This is the signature of the c=32 collapse this repository has carried as unexplained
    since 0.1, and it is now diagnosable rather than merely observed. The adapter counts every
    step it brackets (`tokens`) and the subset that took the runtime's packed decode path
    (`tokens_packed`); the difference is prefill chunks and single-row steps. In a healthy run
    that difference is a property of the workload and is stable to within one step across
    repeats -- 77, 77, 77 at ctx128-c1; 15, 15, 15 at ctx128-c16; 14, 14, 15 at ctx128-c32.

    One run of the seventy-seven measured on the reference box did 144, against its group's
    median of 15, and returned 575 tok/s where its eight siblings returned 890-896. The decode
    work was identical: 63 packed steps in every one of the nine. The runtime scheduled the 32
    requests substantially serially, so the wall time grew while the batched work did not.

    Every other guard misses it. All requests completed, so `require_requests_completed` passes.
    63 of 64 decode steps batched at 32 rows, so `require_packed_path` passes at 98%. The hook
    ran and applied a policy. The only thing wrong is that the run took 1.6x as long, and the
    only counter that says so is this one.

    Reported, never refused: the failure is the runtime's, it hits the arm that happens to be
    running, and turning it into a cell failure would charge a candidate for it. A median over
    nine repeats survives one; a median over three does not, which is what an operator needs to
    know when they read a receipt with one of these in it.
    """
    groups = defaultdict(list)
    for record in records:
        stats = ((record.get("detail") or {}).get("adapter") or {}).get("stats") or {}
        total, packed = stats.get("tokens"), stats.get("tokens_packed")
        if not total:
            continue
        groups[(record["workload_id"], record["variant"], record["config_id"])].append(
            (record, total - (packed or 0)))

    out = []
    for (cell, variant, config), rows in sorted(groups.items()):
        if len(rows) < 3:
            continue                      # two points have no median worth comparing against
        counts = sorted(count for _, count in rows)
        median = counts[len(counts) // 2] if len(counts) % 2 else \
            0.5 * (counts[len(counts) // 2 - 1] + counts[len(counts) // 2])
        if not median:
            continue
        for record, count in rows:
            if count <= SCHEDULING_OUTLIER_RATIO * median:
                continue
            detail = {
                "cell": cell, "variant": variant, "config_id": config,
                "repeat": record["repeat"],
                "non_decode_steps": count,
                "group_median_non_decode_steps": median,
                "ratio": round(count / median, 2),
                "goodput_tps": (record.get("metrics") or {}).get("goodput_tps"),
                "group_median_goodput_tps": _median_of(
                    [(r.get("metrics") or {}).get("goodput_tps") for r, _ in rows]),
                "note": ("the run bracketed far more steps than its siblings for the same "
                         "decode work, which is the runtime scheduling the requests serially "
                         "rather than co-scheduling them. Reported, not refused: it is a "
                         "property of the runtime and it hits whichever arm is running."),
            }
            record.setdefault("detail", {})["scheduling_outlier"] = detail
            out.append(detail)
    return out


def _median_of(values):
    clean = sorted(v for v in values if v)
    if not clean:
        return None
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else 0.5 * (clean[mid - 1] + clean[mid])


def attribute_serving_losses(*, cells, model, baseline_cb_binary, max_new, long_prefill,
                             settle_seconds=30, verbose=True, scaling=None):
    """One probe per cell: the BASELINE build's bench, hook on, no window, no policy.

    Two deliberate choices.

    *The baseline build, not the candidate's.* A candidate whose own probe failed would get a
    cell it lost DROPPED instead of scored at the floor, which is strictly to its advantage.
    The probe must therefore run none of the candidate's code. In the same-binary A/B a
    contributor runs locally these are one file and the probe answers a weaker question, which
    the receipt records.

    *The hook rather than the control.* The control cannot answer the question at all -- that
    is what created the incident. `baseline` is the hook installed with no window: it emits the
    same telemetry the guard reads, and it applies no policy, so a collapse under it is the
    runtime's and nobody else's.
    """
    verdicts = {}
    for cell in cells:
        wait_for_free_device(settle_seconds, verbose=verbose)
        label = f"attribution/baseline/{cell}"
        if verbose:
            print(f">> {label}", flush=True)
        metrics, status, detail = measure_cell(
            baseline_cb_binary, model, cell, dict(ATTRIBUTION_ENV), label,
            max_new=max_new, long_prefill=long_prefill, expect_policy=False, verbose=verbose)
        verdict = "runtime" if status in SERVING_GUARDS else "candidate"
        verdicts[cell] = {
            "verdict": verdict,
            "probe_status": status,
            "probe_config": "hook installed, no window (TENSORTRANSIT=baseline)",
            "probe_binary": str(baseline_cb_binary),
            "guard": detail.get("guard", ""),
        }
        if verdict == "candidate":
            # A verdict of "candidate" costs the submission a cell, so the evidence for it
            # travels with it: what the probe measured, and the packing counters that say the
            # probe served the workload the candidate could not.
            packing = ((detail.get("adapter") or {}).get("stats") or {})
            verdicts[cell]["probe_measured"] = {
                "goodput_tps": metrics.get("goodput_tps"),
                "p99_itl_ms": metrics.get("p99_itl_ms"),
                "max_rows_seen": packing.get("max_rows_seen"),
                "tokens_packed": packing.get("tokens_packed"),
                "tokens": packing.get("tokens"),
                "decode_steps_expected": max_new,
            }
        if scaling and cell in scaling:
            # The arithmetic beside the probe. A cell whose control got 1.15x out of 4
            # sequences was not serving that workload either, whatever any telemetry says.
            verdicts[cell]["main_concurrency_scaling"] = scaling[cell]
        if verbose:
            print(f"   attribution: {cell} -> {verdict} (probe {status})", flush=True)
    return verdicts


def apply_attribution(records, verdicts):
    """Stamp each serving-guard record with what the probe found, or with `unresolved`.

    `unresolved` is the conservative value and it is the DEFAULT: a loss nobody probed stays
    charged to the candidate exactly as before. The only thing that moves a cell out of the
    score is a probe that reproduced the failure without any policy running.
    """
    stamped = 0
    for record in records:
        if record["variant"] != "candidate":
            continue
        if record.get("status", "OK") not in SERVING_GUARDS:
            continue
        found = verdicts.get(record["workload_id"])
        record["attribution"] = found["verdict"] if found else "unresolved"
        if found:
            record.setdefault("detail", {})["attribution"] = found
        stamped += 1
    return stamped


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
