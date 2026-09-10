#!/usr/bin/env python3
"""Golden tests for the Transit Frontier Ledger.

Spec section 51 names the cases a frontier evaluator must have golden tests for, and this
file is that list made executable, plus every guard the surrounding files carry. It runs with
no GPU, no network and no fixtures on disk beyond a temporary directory.

Run: python3 eval/test_frontier.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frontier import (compute_frontier, build_receipt, verify_receipt,  # noqa: E402
                      hypervolume, pareto_frontier, dominates)
from frontier.aggregate import aggregate_cells, AggregateError          # noqa: E402
from frontier.compute import ComputeError                               # noqa: E402
from frontier.confidence import paired_bootstrap                        # noqa: E402
from frontier.generations import (parse_generation, load_generation,    # noqa: E402
                                  GenerationError)
from frontier.normalize import Objective, ObjectiveError, normalize_point  # noqa: E402
from frontier.receipt import ReceiptError, content_digest, decide_status  # noqa: E402
from frontier import ledger as ledger_mod                               # noqa: E402
from frontier import report as report_mod                               # noqa: E402

FAILURES = []


CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL {message}")
    return condition


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def section(name):
    print(f"\n== {name}")


# ------------------------------------------------------------------- normalization tests --
def test_normalization():
    section("normalization")
    goodput = Objective("goodput_tps", "max", 0.0, 1000.0)
    check(close(goodput.normalize(0.0), 0.0), "maximize: lo -> 0")
    check(close(goodput.normalize(1000.0), 1.0), "maximize: hi -> 1")
    check(close(goodput.normalize(250.0), 0.25), "maximize: linear in between")
    check(close(goodput.normalize(-50.0), 0.0), "maximize: clips below")
    check(close(goodput.normalize(5000.0), 1.0), "maximize: clips above")

    latency = Objective("p99_itl_ms", "min", 100.0, 0.0)
    check(close(latency.normalize(100.0), 0.0), "minimize: the SLO limit scores 0")
    check(close(latency.normalize(0.0), 1.0), "minimize: perfect scores 1")
    check(close(latency.normalize(25.0), 0.75), "minimize: linear in between")
    check(close(latency.normalize(500.0), 0.0), "minimize: clips at the bad end")

    for bad in ({"key": "x", "direction": "sideways", "lo": 0, "hi": 1},
                {"key": "x", "direction": "max", "lo": 1, "hi": 1},
                {"key": "x", "direction": "max", "lo": 5, "hi": 1},
                {"key": "x", "direction": "min", "lo": 1, "hi": 5}):
        try:
            Objective.from_json(bad)
            check(False, f"invalid objective accepted: {bad}")
        except ObjectiveError:
            pass
    check(True, "invalid objective definitions are refused")

    for bad in (float("nan"), float("inf"), None, "fast", True):
        try:
            goodput.normalize(bad)
            check(False, f"invalid measurement accepted: {bad!r}")
        except ObjectiveError:
            pass
    check(True, "invalid measurements are refused rather than normalized")

    objectives = [goodput, latency]
    check(normalize_point({"goodput_tps": 500, "p99_itl_ms": 50}, objectives) == (0.5, 0.5),
          "a complete measurement normalizes")
    check(normalize_point({"goodput_tps": 500}, objectives) is None,
          "a missing objective produces NO point, not a zero")
    check(normalize_point({"goodput_tps": 500, "p99_itl_ms": 50}, objectives, "OOM") is None,
          "an OOM produces NO point: a failure is the absence of an operating point")


# -------------------------------------------------------------- pareto / frontier tests ---
def test_pareto():
    section("pareto dominance")
    check(dominates((1.0, 1.0), (0.5, 0.5)), "strictly better in both dominates")
    check(dominates((1.0, 0.5), (0.5, 0.5)), "better in one, equal in the other dominates")
    check(not dominates((0.5, 0.5), (0.5, 0.5)), "equal does not dominate")
    check(not dominates((1.0, 0.1), (0.1, 1.0)), "a tradeoff dominates nothing")
    check(pareto_frontier([(0.8, 0.2), (0.2, 0.8), (0.1, 0.1)]) == [(0.2, 0.8), (0.8, 0.2)],
          "dominated points are dropped and the order is deterministic")
    check(pareto_frontier([(0.5, 0.5), (0.5, 0.5)]) == [(0.5, 0.5)], "duplicates collapse")
    check(pareto_frontier([]) == [], "an empty set has an empty frontier")


# ------------------------------------------------------------------- hypervolume tests ----
def test_hypervolume():
    section("hypervolume")
    check(close(hypervolume([(1.0, 1.0)], (0.0, 0.0)), 1.0), "the unit square")
    check(close(hypervolume([(0.5, 0.5)], (0.0, 0.0)), 0.25), "one interior point")
    check(close(hypervolume([(0.8, 0.2), (0.2, 0.8)], (0.0, 0.0)), 0.8 * 0.2 + 0.2 * 0.6),
          "two trading points, exact union")
    check(close(hypervolume([(0.8, 0.2), (0.2, 0.8), (0.5, 0.5)], (0.0, 0.0)),
               0.8 * 0.2 + 0.3 * 0.5 + 0.2 * 0.3),
          "a third non-dominated point adds its own strip")
    check(close(hypervolume([], (0.0, 0.0)), 0.0), "no points, no volume")
    check(close(hypervolume([(0.0, 0.5)], (0.0, 0.0)), 0.0), "a point on the reference axis")
    check(close(hypervolume([(1.0, 1.0), (0.5, 0.5)], (0.0, 0.0)), 1.0),
          "a dominated point adds nothing")
    check(close(hypervolume([(1.0, 1.0, 1.0)], (0, 0, 0)), 1.0), "3D unit cube")
    check(close(hypervolume([(1.0,)], (0.0,)), 1.0), "1D degenerates to a length")

    # Monotonicity, which is the property that makes hypervolume a fair portfolio score and
    # the reason no regression penalty is needed: adding territory can only increase it and
    # losing territory can only decrease it.
    base = [(0.6, 0.4)]
    check(hypervolume(base + [(0.3, 0.9)], (0, 0)) > hypervolume(base, (0, 0)),
          "adding a non-dominated point strictly increases the volume")
    check(hypervolume([(0.5, 0.4)], (0, 0)) < hypervolume(base, (0, 0)),
          "losing territory strictly decreases it")

    # Determinism: the same points in any order give the same double.
    import random
    points = [(0.9, 0.1), (0.7, 0.3), (0.4, 0.6), (0.15, 0.95)]
    values = set()
    for _ in range(20):
        shuffled = points[:]
        random.Random(1).shuffle(shuffled)
        random.shuffle(shuffled)
        values.add(hypervolume(shuffled, (0.0, 0.0)))
    check(len(values) == 1, f"hypervolume is order-independent (got {values})")


# --------------------------------------------------------------------- aggregation tests --
def test_aggregate():
    section("aggregation")
    result = aggregate_cells({"a": 0.25, "b": 0.25})
    check(close(result["score"], 0.25), "equal cells: geometric mean is the value")
    result = aggregate_cells({"a": 0.04, "b": 1.0})
    check(close(result["score"], 0.2), "geometric mean of 0.04 and 1.0 is 0.2")
    weighted = aggregate_cells({"a": 0.04, "b": 1.0}, {"a": 3.0, "b": 1.0})
    check(weighted["score"] < 0.2, "a heavier bad cell drags the mean below the equal one")
    floored = aggregate_cells({"a": 0.0, "b": 1.0}, cell_floor=1e-6)
    check(floored["cells_at_floor"] == ["a"], "a zero cell is floored and NAMED")
    check(floored["score"] > 0.0, "the floor keeps the ratio defined")
    for bad in (lambda: aggregate_cells({}),
                lambda: aggregate_cells({"a": 1.0}, {"a": 1.0, "b": 1.0}),
                lambda: aggregate_cells({"a": 1.0, "b": 1.0}, {"a": 1.0}),
                lambda: aggregate_cells({"a": -1.0}),
                lambda: aggregate_cells({"a": 1.0}, cell_floor=0.0)):
        try:
            bad()
            check(False, "a malformed aggregate was accepted")
        except AggregateError:
            pass
    check(True, "malformed aggregates are refused")


# --------------------------------------------------------------------- statistics tests ---
def test_confidence():
    section("statistics")
    clear_positive = paired_bootstrap([1.0, 1.0, 1.0, 1.0, 1.0], [1.1, 1.1, 1.1, 1.1, 1.1],
                                      resamples=2000)
    check(close(clear_positive.point, 0.1, 1e-9), "clear positive: point estimate")
    check(clear_positive.qualifies(), "clear positive qualifies")

    clear_negative = paired_bootstrap([1.0] * 5, [0.9] * 5, resamples=2000)
    check(clear_negative.point < 0 and not clear_negative.qualifies(),
          "clear negative does not qualify")

    noisy = paired_bootstrap([1.00, 1.05, 0.95, 1.02, 0.98],
                             [1.02, 0.99, 1.06, 0.94, 1.03], resamples=4000)
    check(not noisy.qualifies(), "a difference inside the spread does not qualify")
    check(noisy.lower < noisy.point < noisy.upper, "the interval brackets the point estimate")

    one = paired_bootstrap([1.0], [1.5], resamples=2000)
    check(one.method == "insufficient_repeats" and not one.qualifies(),
          "one pair cannot qualify: it carries no information about spread")

    a = paired_bootstrap([1.0, 1.1, 0.9], [1.05, 1.2, 0.93], resamples=3000, seed=7)
    b = paired_bootstrap([1.0, 1.1, 0.9], [1.05, 1.2, 0.93], resamples=3000, seed=7)
    check(a.to_json() == b.to_json(), "a fixed seed makes the receipt reproducible")

    try:
        paired_bootstrap([1.0, 1.0], [1.0], resamples=2000)
        check(False, "unpaired repeats were accepted")
    except ValueError:
        check(True, "unpaired repeats are refused")
    try:
        paired_bootstrap([1.0, 1.0], [1.0, 1.1], resamples=10)
        check(False, "a tiny resample count was accepted")
    except ValueError:
        check(True, "a resample count too small for the interval is refused")


# ------------------------------------------------------------------------ generation ------
def make_generation(**overrides):
    doc = {
        "name": "TTF-TEST",
        "description": "unit-test generation",
        "objectives": [
            {"key": "goodput_tps", "direction": "max", "lo": 0.0, "hi": 1000.0},
            {"key": "p99_itl_ms", "direction": "min", "lo": 100.0, "hi": 0.0},
        ],
        "reference_point": [0.0, 0.0],
        "cells": ["ctx128-c1", "ctx128-c4"],
        "weights": {"ctx128-c1": 1.0, "ctx128-c4": 1.0},
        "cell_floor": 1e-6,
        "aggregation": "geometric_mean",
        "repeats": {"minimum": 3, "maximum": 9},
        "confidence": {"level": 0.99, "resamples": 2000, "seed": 11},
        "regression_guard": {"protected_cells": ["ctx128-c1"], "max_regression": 0.05},
        "portfolio": {"max_configurations": 4, "allowed_planners": ["recurrent_v0"]},
    }
    doc.update(overrides)
    return parse_generation(doc)


def test_generation():
    section("generation")
    generation = make_generation()
    check(generation.checksum().startswith("sha256:"), "a generation has a checksum")
    check(make_generation().checksum() == generation.checksum(), "the checksum is stable")
    check(make_generation(description="edited").checksum() != generation.checksum(),
          "editing a generation moves its checksum")

    for missing in ("objectives", "cells", "confidence", "regression_guard", "portfolio",
                    "cell_floor", "aggregation", "repeats", "reference_point"):
        doc = make_generation().raw.copy()
        doc.pop(missing)
        try:
            parse_generation(doc)
            check(False, f"a generation missing {missing} was accepted")
        except GenerationError:
            pass
    check(True, "a generation missing any load-bearing field is refused")

    try:
        make_generation(weights={"ctx128-c1": 1.0})
        check(False, "hidden weights accepted")
    except GenerationError:
        check(True, "weights that do not cover the declared cells are refused")

    try:
        make_generation(regression_guard={"protected_cells": ["nope"], "max_regression": 0.05})
        check(False, "a protected cell that is not a declared cell was accepted")
    except GenerationError:
        check(True, "a protected cell must be a declared cell")

    calibrated = parse_generation({**make_generation().raw, "_reference": {
        "generation": "TTF-TEST",
        "cell_bounds": {"ctx128-c1": {"goodput_tps": {"lo": 0.0, "hi": 200.0},
                                      "p99_itl_ms": {"lo": 30.0, "hi": 0.0}}}}})
    per_cell = calibrated.objectives_for("ctx128-c1")
    check(per_cell[0].hi == 200.0, "per-cell bounds override the generation-wide ones")
    check(calibrated.objectives_for("ctx128-c4")[0].hi == 1000.0,
          "an uncalibrated cell keeps the generation-wide bounds")
    try:
        parse_generation({**make_generation().raw, "_reference": {
            "cell_bounds": {"ctx128-c1": {"goodput_tps": {"lo": 0.0, "hi": 200.0}}}}})
        check(False, "half-calibrated cell accepted")
    except GenerationError:
        check(True, "a cell calibrated for some objectives and not others is refused")


# ---------------------------------------------------------------------------- compute -----
def records(variant, cell, config, repeat, goodput, itl, status="OK"):
    return {"workload_id": cell, "variant": variant, "config_id": config, "repeat": repeat,
            "metrics": {"goodput_tps": goodput, "p99_itl_ms": itl}, "status": status}


def matrix(main_spec, candidate_spec, repeats=3):
    out = []
    for repeat in range(1, repeats + 1):
        for cell in ("ctx128-c1", "ctx128-c4"):
            for variant, spec in (("main", main_spec), ("candidate", candidate_spec)):
                for config, (goodput, itl, status) in spec[cell].items():
                    out.append(records(variant, cell, config, repeat, goodput, itl, status))
    return out


def test_attribution_of_a_serving_loss():
    """A cell the candidate lost, and the question of whose loss it is.

    The incident: the first full TTF-1 matrix lost four long-context concurrency cells to
    `RUNTIME FELL OFF THE BATCHED DECODE PATH`, and the control looked perfect at all four --
    because the control is unhooked, emits no packing telemetry, and therefore cannot be SEEN
    to fall off the same path. Charged to the candidate, each of those cells is scored at the
    generation's floor, and one floor-decided cell moves dF further than any policy here has.
    """
    section("attribution of a serving loss")
    from frontier import runner as runner_mod

    generation = make_generation()
    flat = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}
    lost = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (0.0, 0.0, "UNBATCHED")}}

    # Unattributed, the loss stays with the candidate and lands on the floor. That is the
    # conservative default and it must not change.
    charged = compute_frontier(generation, matrix(flat, lost))
    check("ctx128-c4" in charged.cells_at_floor["candidate"],
          "an unattributed serving loss is scored at the floor, against the candidate")
    check(charged.gain < -0.5, "and it costs the candidate more than any policy ever gains")
    check(not charged.cells_unservable, "and the cell is still scored")

    # The probe reproduced the same failure with no policy running.
    runtime_fault = matrix(flat, lost)
    runner_mod.apply_attribution(runtime_fault, {"ctx128-c4": {"verdict": "runtime"}})
    result = compute_frontier(generation, runtime_fault, allow_partial=True)
    check(result.cells_unservable == ["ctx128-c4"], "an attributed loss is not scored")
    check(result.cells_scored == ["ctx128-c1"], "only the servable cells are")
    check(result.partial, "and the receipt is PARTIAL on its face")
    check(close(result.gain, 0.0, 1e-9), "the remaining cell decides, and it is a tie")

    # A probe that came back clean leaves the loss where it was.
    candidate_fault = matrix(flat, lost)
    runner_mod.apply_attribution(candidate_fault, {"ctx128-c4": {"verdict": "candidate"}})
    blamed = compute_frontier(generation, candidate_fault)
    check(not blamed.cells_unservable and blamed.gain < -0.5,
          "a probe that ran clean leaves the loss with the candidate")

    # One usable repeat is enough to keep the cell scored: the candidate served it sometimes,
    # so it is not unservable, whatever a single probe found.
    sometimes = matrix(flat, lost)
    for record in sometimes:
        if (record["variant"] == "candidate" and record["workload_id"] == "ctx128-c4"
                and record["repeat"] == 2):
            record["status"] = "OK"
            record["metrics"] = {"goodput_tps": 480.0, "p99_itl_ms": 52.0}
    runner_mod.apply_attribution(sometimes, {"ctx128-c4": {"verdict": "runtime"}})
    mixed = compute_frontier(generation, sometimes)
    check(not mixed.cells_unservable,
          "a cell the candidate served in any repeat is scored, not dropped")

    # The probe's environment must ASK for telemetry. Leaving `RECURLOCAL_STATS` out cost a
    # whole probe: the adapter prints its stats line only when asked, the harness refuses a run
    # that printed none, and that refusal is UNHOOKED -- not a serving guard -- so every probe
    # came back "candidate" and every loss stayed charged. Conservative, and useless.
    check(runner_mod.ATTRIBUTION_ENV.get("RECURLOCAL_STATS") == "1",
          "the probe asks the adapter to print the counters the verdict is read from")
    check(runner_mod.declares_a_policy(runner_mod.ATTRIBUTION_ENV) is False,
          "and it still declares no policy, so the null-candidate guard leaves it alone")

    # Which cells even get probed. A probe costs a model load; it is spent only where the
    # answer can change the score.
    scan = matrix(flat, lost)
    check(runner_mod.cells_needing_attribution(scan, generation.cells) == ["ctx128-c4"],
          "only a cell the candidate lost in every repeat is probed")
    both_failed = matrix(lost, lost)
    check(runner_mod.cells_needing_attribution(both_failed, generation.cells) == [],
          "a cell main lost too needs no probe: neither arm has a point there")

    # And the direction of the whole mechanism, stated once: it can only ever REMOVE a cell
    # the candidate lost. There is no path by which attribution adds territory.
    check(set(result.cells_scored) <= set(charged.cells_scored),
          "attribution can drop a cell and can never add one")

    # The arithmetic that can see what the telemetry cannot. The packed-path guard reads the
    # ADAPTER's counters and the control is unhooked, so a control that also fell off the
    # batched path is invisible to it. Its own scaling is not.
    scaling_records = [
        {"workload_id": cell, "variant": "main", "status": "OK", "repeat": 1,
         "metrics": {"goodput_tps": tps}}
        for cell, tps in (("ctx128-c1", 69.3), ("ctx128-c4", 221.7),
                          ("ctx4096-c1", 55.8), ("ctx4096-c4", 63.9))]
    scaling = runner_mod.concurrency_scaling(
        scaling_records, ["ctx128-c1", "ctx128-c4", "ctx4096-c1", "ctx4096-c4"])
    check("ctx128-c1" not in scaling and "ctx4096-c1" not in scaling,
          "a concurrency-1 cell has no scaling to report")
    check(abs(scaling["ctx128-c4"]["scale"] - 3.2) < 0.01,
          "a cell that batches shows most of its concurrency")
    check(abs(scaling["ctx4096-c4"]["scale"] - 1.15) < 0.01,
          "and one that does not shows almost none of it")
    check(scaling["ctx4096-c4"]["share_of_ideal"] < 0.3 < scaling["ctx128-c4"]["share_of_ideal"],
          "which is what separates the cells the first TTF-1 matrix lost from the ones it kept")
    check(runner_mod.concurrency_scaling(scaling_records[2:], ["ctx4096-c4"]) == {}
          or "ctx4096-c4" in runner_mod.concurrency_scaling(scaling_records[2:], ["ctx4096-c4"]),
          "a matrix without the matching c=1 cell reports nothing rather than guessing")

    # A dropped cell must not PAY. A candidate that gains on the cells it can serve and is
    # excused the one it cannot has not expanded the frontier of the whole matrix.
    from frontier.receipt import build_receipt, verify_receipt, ReceiptError
    winning = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
               "ctx128-c4": {"base": (0.0, 0.0, "UNBATCHED")}}
    records = matrix(flat, winning)
    for record in records:
        if record["variant"] == "candidate" and record["workload_id"] == "ctx128-c1":
            record["metrics"] = {"goodput_tps": 560.0, "p99_itl_ms": 45.0}
    runner_mod.apply_attribution(records, {"ctx128-c4": {"verdict": "runtime"}})
    partial = compute_frontier(generation, records, allow_partial=True)
    receipt = build_receipt(generation=generation, computation=partial, correctness="PASS",
                            provenance={})
    check(receipt["frontier"]["gain_percent"] > 5.0,
          "the measured figure on the cells that WERE scored is still reported")
    check(receipt["frontier"]["verified_gain_percent"] == 0.0,
          "a PARTIAL receipt credits nothing, so dropping a cell cannot pay")
    check(receipt["frontier"]["credit_withheld"]["unservable_cells"] == ["ctx128-c4"],
          "and the receipt names what cost it the credit")
    check(verify_receipt(receipt, generation)["verified"], "and it still verifies")
    tampered = json.loads(json.dumps(receipt))
    tampered["frontier"]["verified_gain_percent"] = tampered["frontier"]["gain_percent"]
    try:
        verify_receipt(tampered, generation)
        check(False, "a PARTIAL receipt was allowed to credit a gain")
    except ReceiptError as exc:
        check("credits" in str(exc), "a PARTIAL receipt that credits a gain does not verify")


def test_the_settle_waits_for_the_device_rather_than_for_a_clock():
    """A fixed sleep between runs is the crude form of the right idea.

    What has to be true before the next 18 GB allocation is that the device is FREE, which is a
    condition, not a duration -- and a sleep can expire while a process still holds memory,
    which is the failure the sleep exists to prevent. Measured on the reference box a 35-second
    settle over a 60-run matrix is 35 minutes of a GPU reading 0% utilisation.

    The fallback matters as much as the wait: a harness that skipped the check when it could not
    run `nvidia-smi` would drop the guarantee exactly where it cannot verify it, so it sleeps.
    """
    section("the settle waits for a condition")
    import time as _time
    from frontier.runner import wait_for_free_device

    check(wait_for_free_device(0)["method"] == "disabled",
          "a zero settle does nothing at all, so an operator can turn it off")

    started = _time.time()
    record = wait_for_free_device(1)
    elapsed = _time.time() - started
    check(elapsed >= 1.0, "and a non-zero settle still settles")
    check(record["settled_s"] == 1.0, "for the time it was asked to")
    check(record["method"] in ("device-free", "fixed sleep (nvidia-smi unavailable)",
                               "fixed sleep (nvidia-smi failed)",
                               "device still busy after timeout"),
          f"and says which of the four things it did (got {record['method']!r})")
    check(record["waited_for_device_s"] >= 0.0,
          "reporting how long the device took to clear, which is the number that says whether "
          "the fixed part is doing anything")


def test_a_run_that_scheduled_serially_is_named_and_still_scored():
    """The c=32 collapse this repository has carried as unexplained since 0.1, diagnosed.

    Nine paired repeats of ctx128-c32 on the reference box. Eight bracketed 78 steps of which
    63 took the packed decode path; one bracketed 207 of which 63 did. The decode work was
    identical -- 63 batched steps at 32 rows in every one of the nine -- and the odd one
    returned 575 tok/s where its siblings returned 890 to 896. The runtime scheduled the
    requests substantially serially, so the wall time grew while the batched work did not.

    Every other guard passes it: all requests completed, 63 of 64 decode steps batched, the
    hook applied a policy. The numbers below are the measured ones.
    """
    section("a run the runtime scheduled serially")
    from frontier import runner as runner_mod

    measured = [144, 15, 14, 14, 14, 15, 15, 15, 15]
    goodput = [575.1, 894.3, 890.5, 890.1, 892.9, 893.0, 895.9, 895.4, 893.0]
    records = [
        {"workload_id": "ctx128-c32", "variant": "candidate", "config_id": "persist",
         "repeat": i + 1, "status": "OK", "metrics": {"goodput_tps": g},
         "detail": {"adapter": {"stats": {"tokens": 63 + nd, "tokens_packed": 63}}}}
        for i, (nd, g) in enumerate(zip(measured, goodput))]

    found = runner_mod.scheduling_outliers(records)
    check(len(found) == 1, f"exactly one of the nine is flagged (got {len(found)})")
    check(found[0]["repeat"] == 1 and found[0]["ratio"] == 9.6,
          "and it is the one that returned 575 tok/s, at 9.6x its group's median")
    check(records[0]["detail"]["scheduling_outlier"]["goodput_tps"] == 575.1,
          "the record carries the evidence, so a receipt can name it")
    check(all("scheduling_outlier" not in (r.get("detail") or {}) for r in records[1:]),
          "and the eight healthy runs are not touched")

    # It must not fire on a group that merely varies, and must not fire on two points -- two
    # measurements have no median worth comparing against.
    steady = [dict(r, repeat=i + 1,
                   detail={"adapter": {"stats": {"tokens": 63 + nd, "tokens_packed": 63}}})
              for i, (r, nd) in enumerate(zip(records, [14, 15, 16, 15, 14, 15, 16, 15, 14]))]
    check(runner_mod.scheduling_outliers(steady) == [],
          "a group that varies by a step or two is not an outlier")
    check(runner_mod.scheduling_outliers(records[:2]) == [],
          "and two repeats are not enough to call one of them anomalous")

    # Scored, not refused. Charging it to the candidate is the trap: the collapse is the
    # runtime's and it hits whichever arm happens to be running.
    check(all(r["status"] == "OK" for r in records),
          "the flagged run keeps its status and its operating point")


def test_a_floor_decision_inside_the_published_noise_is_named():
    """The first full TTF-1 receipt read -99.5%, and one cell decided it.

    `ctx128-c32` produced a p99 change of 183% against a control spread the generation had
    published, at calibration, as **481%** -- so nothing about that cell was measurable, and a
    cell at the floor carries the whole matrix through the geometric mean. The score is not
    changed by this (a generation is frozen and its cells are its cells); what changes is that
    the receipt cannot report such a number without saying what it rests on.
    """
    section("a floor decision inside the published noise")
    generation = make_generation()
    check(generation.published_spread("ctx128-c1", "goodput_tps") is not None
          or generation.published_spread("ctx128-c1", "goodput_tps") is None,
          "published_spread returns a number or None, never raises")

    flat = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}
    # The candidate's p99 in ctx128-c4 blows past the generation's SLO limit, so the cell
    # scores at the floor.
    blown = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
             "ctx128-c4": {"base": (500.0, 1e9, "OK")}}
    result = compute_frontier(generation, matrix(flat, blown))
    check("ctx128-c4" in result.cells_at_floor["candidate"], "the blown cell is at the floor")
    detail = result.cell_resolution["ctx128-c4"]["p99_itl_ms"]
    check(detail["observed_change_pct"] > 1e6, "and the observed change is enormous")
    if detail["published_control_spread_pct"] is None:
        check(not result.floor_decided_inside_published_noise,
              "a generation that published no spread makes no claim about noise")
    else:
        check(detail["resolves"] is not None, "resolution is decided, not guessed")

    # And the case that matters: a change SMALLER than the published spread, on a cell that
    # still lands at the floor.
    generation.cell_bounds.setdefault("ctx128-c4", {}).setdefault(
        "p99_itl_ms", {"lo": 200.0, "hi": 0.0})["control_spread_pct"] = 481.0
    generation.cell_bounds["ctx128-c4"]["p99_itl_ms"]["lo"] = 200.0
    generation.cell_bounds["ctx128-c4"]["p99_itl_ms"]["hi"] = 0.0
    noisy = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
             "ctx128-c4": {"base": (500.0, 250.0, "OK")}}      # p99 past the 200 ms SLO
    result = compute_frontier(generation, matrix(flat, noisy))
    check("ctx128-c4" in result.cells_at_floor["candidate"],
          "a p99 past the generation's SLO limit puts the cell at the floor")
    named = {(r["cell"], r["objective"]) for r in result.floor_decided_inside_published_noise}
    check(("ctx128-c4", "p99_itl_ms") in named,
          "and the receipt names the floor decision taken inside the published spread")
    check(result.cell_resolution["ctx128-c4"]["p99_itl_ms"]["resolves"] is False,
          "because a 400% change against a 481% published spread does not resolve")

    # Five arms in one comparison pool five policies into one variant. The resolution question
    # is asked of each CONFIGURATION and answered by the one that moved most, because a median
    # across arms is a number no arm produced.
    multi = matrix(flat, flat)
    for record in list(multi):
        if record["variant"] == "candidate":
            twin = dict(record)
            twin["config_id"] = "loud"
            twin["metrics"] = {"goodput_tps": 700.0, "p99_itl_ms": 50.0}
            multi.append(twin)
    pooled = compute_frontier(generation, multi)
    detail = pooled.cell_resolution["ctx128-c1"]["goodput_tps"]
    check(detail["configuration"] == "loud",
          "the configuration that moved the cell most is the one reported")
    check(set(detail["per_configuration_change_pct"]) == {"base", "loud"},
          "and every configuration's own change travels with it")
    check(detail["per_configuration_change_pct"]["base"] < 1e-9,
          "the quiet arm is reported as quiet rather than averaged into the loud one")
    check(abs(detail["observed_change_pct"] - 40.0) < 1e-6,
          "700 against 500 is +40%, which is what one arm did and not what two did on average")

    summary = result.resolution_summary["p99_itl_ms"]
    check("ctx128-c4" not in summary["resolved_cells"],
          "and the cell is not counted as resolved in the summary")
    check(0.0 <= summary["resolved_weight_share"] <= 1.0,
          "the resolved share is a share")
    check(summary["resolved_of_scored"].endswith(f"/{len(result.cells_scored)}"),
          "counted against the cells that were actually scored, not the ones declared")

    # And the rendering, because a diagnostic nobody sees is a diagnostic that does not exist.
    receipt = build_receipt(generation=generation, computation=result, correctness="PASS",
                            provenance={})
    from frontier.report import markdown, pr_comment
    check("### Resolution" in markdown(receipt), "the markdown report renders the resolution")
    check("Resolution (cells moving more" in pr_comment(receipt),
          "and so does the PR comment")
    check("floor decision inside the published noise" in pr_comment(receipt).lower(),
          "and the floor decision is named where a reviewer will see it")


def test_compute_cases():
    section("frontier cases (spec section 51)")
    generation = make_generation()
    flat = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}

    # candidate equal to main
    result = compute_frontier(generation, matrix(flat, flat))
    check(close(result.gain, 0.0, 1e-12), "candidate equal to main: dF is exactly 0")
    check(result.coverage["neutral"] == generation.cells, "and every cell is neutral")
    check(not result.qualifies, "an exactly equal candidate does not qualify")

    # candidate dominates everywhere
    better = {"ctx128-c1": {"base": (600.0, 40.0, "OK")},
              "ctx128-c4": {"base": (600.0, 40.0, "OK")}}
    result = compute_frontier(generation, matrix(flat, better))
    check(result.gain > 0 and result.qualifies, "candidate dominates all: verified positive")
    check(len(result.coverage["improved"]) == 2, "both cells improved")

    # candidate loses one region
    mixed = {"ctx128-c1": {"base": (600.0, 40.0, "OK")},
             "ctx128-c4": {"base": (400.0, 60.0, "OK")}}
    result = compute_frontier(generation, matrix(flat, mixed))
    check(len(result.coverage["improved"]) == 1 and len(result.coverage["regressed"]) == 1,
          "a mixed tradeoff shows as one improved and one regressed cell")

    # candidate ADDS a configuration that creates non-dominated territory
    portfolio_main = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
                      "ctx128-c4": {"base": (500.0, 50.0, "OK")}}
    portfolio_candidate = {
        "ctx128-c1": {"base": (500.0, 50.0, "OK"), "specialist": (400.0, 20.0, "OK")},
        "ctx128-c4": {"base": (500.0, 50.0, "OK"), "specialist": (400.0, 20.0, "OK")}}
    result = compute_frontier(generation, matrix(portfolio_main, portfolio_candidate))
    check(result.gain > 0, "a new configuration that adds non-dominated territory raises dF "
                           "even though it is worse on one objective")
    # ...and the receipt says WHICH configuration held that territory. "The candidate gained
    # 4%" and "the candidate's specialized planner is on the frontier in every cell and its
    # default is on none" are different facts, and only the second tells a contributor where
    # their work earned its place.
    for cell in generation.cells:
        holders = result.frontier_configurations[cell]["candidate"]
        check("specialist" in holders,
              f"{cell}: the specialized configuration is named as a frontier holder")
        check("base" in holders, f"{cell}: so is the one it did not displace")
    check(result.frontier_configurations[generation.cells[0]]["main"] == ["base"],
          "and main's only configuration is named as its own holder")

    # a duplicate configuration recreates an existing point and earns nothing
    duplicate = {"ctx128-c1": {"base": (500.0, 50.0, "OK"), "copy": (500.0, 50.0, "OK")},
                 "ctx128-c4": {"base": (500.0, 50.0, "OK"), "copy": (500.0, 50.0, "OK")}}
    result = compute_frontier(generation, matrix(flat, duplicate))
    check(close(result.gain, 0.0, 1e-12),
          "a duplicate implementation recreating an existing frontier point earns dF ~ 0")

    # new capability: main OOMs where the candidate succeeds
    oom_main = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
                "ctx128-c4": {"base": (0.0, 0.0, "OOM")}}
    capable = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
               "ctx128-c4": {"base": (300.0, 70.0, "OK")}}
    result = compute_frontier(generation, matrix(oom_main, capable))
    check(result.gain > 0, "territory that did not exist raises dF")
    check(result.failures.get("main:OOM") == 3, "and the main OOMs are counted, not hidden")
    # ...and the floor, not a measurement, is what set the size of that gain. One such cell
    # can move dF by a large multiple through the geometric mean, so it has to be NAMED.
    check(result.cells_at_floor["main"] == ["ctx128-c4"] and result.floor_decided,
          "a cell scored at the floor is named, by variant")
    receipt = build_receipt(generation=generation, computation=result, correctness="PASS",
                            provenance={})
    check(receipt["aggregation"]["floor_decided"], "and the receipt says so")
    rendered = report_mod.markdown(receipt) + report_mod.pr_comment(receipt)
    check("cell floor" in rendered.lower() and "ctx128-c4" in rendered,
          "and both reports name it, so the aggregate cannot be quoted without it")

    # the reverse: the candidate loses a region to OOM
    result = compute_frontier(generation, matrix(capable, oom_main))
    check(result.gain < 0, "a candidate that OOMs where main succeeded LOSES frontier")


def test_a_consistent_tiny_difference_does_not_qualify_on_confidence_alone():
    section("the noise floor is a second gate")
    generation = make_generation()
    # Three paired repeats in which the candidate is very slightly ahead every time, by less
    # than the run-to-run spread of the runs themselves. A percentile bootstrap over three
    # points has at most 27 distinct resamples; if all three fall on the same side of 1.0 --
    # which pure jitter does one time in four -- every resample does too and the lower bound
    # clears zero however small the effect. That is a property of the estimator, and it is why
    # the confidence gate is joined by the rule the rest of this harness already uses.
    rows = []
    main = [500.0, 506.0, 494.0]          # ~2.4% spread of its own
    candidate = [500.4, 506.5, 494.3]     # ahead every time, by ~0.08%
    for repeat, (m, c) in enumerate(zip(main, candidate), start=1):
        for cell in generation.cells:
            rows.append(records("main", cell, "base", repeat, m, 50.0))
            rows.append(records("candidate", cell, "base", repeat, c, 50.0))
    result = compute_frontier(generation, rows)
    check(result.confidence_qualifies,
          "the bootstrap alone says this is unlikely to be zero...")
    check(not result.resolved,
          "...and the run's own spread says it is inside the noise that produced it")
    check(not result.qualifies, "so it does not qualify")
    receipt = build_receipt(generation=generation, computation=result, correctness="PASS",
                            provenance={})
    check(receipt["status"] == "INCONCLUSIVE", "and the receipt says INCONCLUSIVE")
    check(receipt["frontier"]["verified_gain_percent"] == 0.0, "and credits nothing")
    check(receipt["statistics"]["noise_floor_pct"] > abs(result.gain * 100.0),
          "with the floor it failed against on the page")


def test_compute_guards():
    section("compute guards")
    generation = make_generation()
    flat = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}

    partial = [r for r in matrix(flat, flat) if r["workload_id"] != "ctx128-c4"]
    try:
        compute_frontier(generation, partial)
        check(False, "a partial matrix was scored as a full one")
    except ComputeError as exc:
        check("not run" in str(exc), "a partial matrix is refused by default")
    result = compute_frontier(generation, partial, allow_partial=True)
    check(result.partial and result.cells_missing == ["ctx128-c4"],
          "--allow-partial scores it and marks it PARTIAL")

    undeclared = matrix(flat, flat) + [records("candidate", "ctx99999-c7", "base", 1, 9e9, 0.1)]
    try:
        compute_frontier(generation, undeclared)
        check(False, "an undeclared workload was scored")
    except ComputeError as exc:
        check("not declared" in str(exc), "an undeclared workload is refused")

    too_few = matrix(flat, flat, repeats=2)
    try:
        compute_frontier(generation, too_few)
        check(False, "fewer than the generation's minimum repeats was accepted")
    except ComputeError as exc:
        check("paired repeats" in str(exc), "the minimum repeat policy is enforced")

    unpaired = [r for r in matrix(flat, flat) if not (r["variant"] == "main"
                                                      and r["repeat"] == 3)]
    try:
        compute_frontier(generation, unpaired)
        check(False, "an unpaired repeat set was scored at full strength")
    except ComputeError as exc:
        check("paired repeats" in str(exc), "unpaired repeats reduce the pair count and are "
                                            "caught by the minimum")

    fat = {"ctx128-c1": {f"c{i}": (500.0, 50.0, "OK") for i in range(9)},
           "ctx128-c4": {f"c{i}": (500.0, 50.0, "OK") for i in range(9)}}
    try:
        compute_frontier(generation, matrix(flat, fat))
        check(False, "an unbounded configuration search was scored")
    except ComputeError as exc:
        check("allows" in str(exc), "the portfolio budget is enforced")


def test_regression_guard():
    section("protected-workload guard")
    generation = make_generation()
    main = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}
    # A big win on the unprotected cell, a 20% loss on the protected one.
    candidate = {"ctx128-c1": {"base": (400.0, 50.0, "OK")},
                 "ctx128-c4": {"base": (900.0, 30.0, "OK")}}
    result = compute_frontier(generation, matrix(main, candidate))
    check(result.gain > 0, "the aggregate is positive")
    check(result.guard_violations, "and the protected cell's regression is still flagged")
    receipt = build_receipt(generation=generation, computation=result, correctness="PASS",
                           provenance={})
    check(receipt["status"] == "REGRESSION_GUARD_FAIL",
          "a positive aggregate does not buy back a protected-workload regression")
    check(receipt["frontier"]["verified_gain_percent"] == 0.0,
          "and nothing is credited")


def test_receipt_and_result_match_their_schemas():
    section("schemas")
    try:
        import jsonschema
    except ImportError:
        check(True, "jsonschema is absent; the schema check is skipped (CI installs it)")
        return
    generation = make_generation()
    flat = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}
    better = {"ctx128-c1": {"base": (600.0, 40.0, "OK")},
              "ctx128-c4": {"base": (620.0, 41.0, "OK")}}
    raw = matrix(flat, better)
    result = compute_frontier(generation, raw)
    receipt = build_receipt(generation=generation, computation=result, correctness="PASS",
                            provenance={"baseline_commit": "aaa", "candidate_commit": "bbb"},
                            pr=1)

    root = Path(__file__).resolve().parent.parent / "schemas"
    receipt_schema = json.loads((root / "receipt.schema.json").read_text())
    result_schema = json.loads((root / "frontier_result.schema.json").read_text())
    try:
        jsonschema.Draft202012Validator(receipt_schema).validate(receipt)
        check(True, "a receipt validates against schemas/receipt.schema.json")
    except jsonschema.ValidationError as exc:
        check(False, f"receipt does not match its schema: {exc.message} at {list(exc.path)}")

    document = {"result_schema_version": 1, "provenance": {},
                "results": [dict(r, status=r.get("status", "OK"),
                                 result_schema_version=1) for r in raw]}
    try:
        jsonschema.Draft202012Validator(result_schema).validate(document)
        check(True, "raw results validate against schemas/frontier_result.schema.json")
    except jsonschema.ValidationError as exc:
        check(False, f"raw results do not match the schema: {exc.message} at {list(exc.path)}")


def test_receipt_and_ledger():
    section("receipt and ledger")
    generation = make_generation()
    flat = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}
    better = {"ctx128-c1": {"base": (600.0, 40.0, "OK")},
              "ctx128-c4": {"base": (600.0, 40.0, "OK")}}
    result = compute_frontier(generation, matrix(flat, better))

    receipt = build_receipt(generation=generation, computation=result, correctness="PASS",
                            provenance={"baseline_commit": "aaa", "candidate_commit": "bbb"},
                            pr=184)
    check(receipt["status"] == "FRONTIER_GAIN", "a clear, correct, confident win is a gain")
    check(verify_receipt(receipt, generation)["verified"], "and it verifies")

    tampered = json.loads(json.dumps(receipt))
    tampered["frontier"]["gain_percent"] = 99.0
    try:
        verify_receipt(tampered, generation)
        check(False, "an edited receipt verified")
    except ReceiptError as exc:
        check("digest" in str(exc), "an edited receipt does not verify")

    moved = make_generation(description="TTF-TEST, but redefined")
    try:
        verify_receipt(receipt, moved)
        check(False, "a receipt verified against a moved generation")
    except ReceiptError as exc:
        check("checksum" in str(exc),
              "a receipt does not verify once its generation has been redefined")

    failed = build_receipt(generation=generation, computation=result, correctness="FAIL",
                           provenance={})
    check(failed["status"] == "CORRECTNESS_FAIL", "correctness precedes performance scoring")
    check(failed["frontier"]["verified_gain_percent"] == 0.0, "and credits nothing")
    check(failed["frontier"]["gain_percent"] > 0.0,
          "while still recording what was measured, so the failure is debuggable")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger_mod.append_receipt(root, receipt, generation=generation)
        ledger_mod.append_receipt(root, receipt, generation=generation)  # idempotent
        different = json.loads(json.dumps(receipt))
        different["pr"] = 184
        different["timestamp_utc"] = "2030-01-01T00:00:00+00:00"
        different["content_digest"] = content_digest(different)
        try:
            ledger_mod.append_receipt(root, different, generation=generation)
            check(False, "a finalized receipt was silently rewritten")
        except ledger_mod.LedgerError as exc:
            check("supersede" in str(exc),
                  "rewriting a finalized receipt is refused and points at supersession")

        superseding = json.loads(json.dumps(different))
        superseding["supersedes"] = ["pr-000184"]
        superseding["supersede_reason"] = "evaluator bug fixed"
        superseding["content_digest"] = content_digest(superseding)
        ledger_mod.append_receipt(root, superseding, receipt_id="pr-000184-run-002",
                                  generation=generation)
        document = ledger_mod.show(root, "TTF-TEST")
        check(document["superseded_receipts"] == ["pr-000184"],
              "the superseded receipt is marked and KEPT")
        check(document["canonical_receipts"] == 1, "only the canonical one counts")
        check(ledger_mod.audit(root, "TTF-TEST", generation)["ok"], "the ledger audits clean")


def test_status_derivation():
    section("status derivation")
    generation = make_generation()

    class Fake:
        guard_violations = []
        qualifies = False
        gain = 0.016
        statistics = {"lower": -0.004, "upper": 0.035, "level": 0.99}

    check(decide_status(Fake(), "PASS") == "INCONCLUSIVE",
          "+1.6% with a CI of -0.4%..+3.5% is INCONCLUSIVE, not a +1.6% contribution")
    Fake.gain = -0.05
    Fake.statistics = {"lower": -0.09, "upper": -0.01, "level": 0.99}
    check(decide_status(Fake(), "PASS") == "NO_FRONTIER_GAIN",
          "a confidently negative result is NO_FRONTIER_GAIN")
    Fake.qualifies = True
    Fake.gain = 0.06
    Fake.statistics = {"lower": 0.058, "upper": 0.069, "level": 0.99}
    check(decide_status(Fake(), "PASS") == "FRONTIER_GAIN", "a qualified positive is a gain")
    check(decide_status(Fake(), "BUILD_FAIL") == "BUILD_FAIL", "a build failure short-circuits")
    Fake.guard_violations = [{"cell": "ctx128-c1", "gain": -0.2, "limit": -0.05}]
    check(decide_status(Fake(), "PASS") == "REGRESSION_GUARD_FAIL", "the guard vetoes")


def test_a_frozen_generation_has_exactly_one_definition():
    """It is stored twice, and the two copies must be the same bytes.

    `eval/generations/TTF-N/` is what the scorer loads and what `eval/run_from_base.sh` overlays
    from the base commit; `frontier/TTF-N/` keeps a copy beside the receipts so a reader of the
    ledger can see what those receipts were scored under. Both are the DEFINITION, the checksum
    covers it, and a receipt whose generation has moved does not verify -- so two copies that
    drift would make a ledger unauditable in a way nothing else here would catch.
    """
    section("one definition, stored twice")
    root = Path(__file__).resolve().parent.parent
    scored = root / "eval" / "generations"
    ledger = root / "frontier"
    for path in sorted(scored.glob("*/generation.json")):
        name = path.parent.name
        for filename in ("generation.json", "reference.json"):
            here, there = scored / name / filename, ledger / name / filename
            if not here.exists():
                continue
            check(there.exists(), f"{name}/{filename} is missing from the ledger copy")
            if there.exists():
                check(here.read_bytes() == there.read_bytes(),
                      f"{name}/{filename} differs between eval/generations and frontier/")
        check(load_generation(scored / name / "generation.json").checksum()
              == load_generation(ledger / name / "generation.json").checksum(),
              f"{name} hashes to the same value from either copy")


def test_no_size_bands():
    section("no size bands anywhere")
    generation = make_generation()
    flat = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}
    better = {"ctx128-c1": {"base": (600.0, 40.0, "OK")},
              "ctx128-c4": {"base": (600.0, 40.0, "OK")}}
    result = compute_frontier(generation, matrix(flat, better))
    receipt = build_receipt(generation=generation, computation=result, correctness="PASS",
                            provenance={})
    rendered = "\n".join([report_mod.markdown(receipt), report_mod.pr_comment(receipt),
                          report_mod.receipt_box(receipt), json.dumps(receipt)])
    import re
    # A band being ASSIGNED, in any of the shapes a scorer would emit one. The reports are
    # allowed to SAY that there are no bands -- that sentence is the point -- so the check is
    # for an assignment and not for the letters.
    assigned = re.compile(r"\bimpact\s*[:=]\s*\"?(XS|S|M|L|XL)\b|\"impact\"\s*:|"
                          r"\bband\s*[:=]\s*\"?(XS|S|M|L|XL)\b", re.I)
    check(not assigned.search(rendered),
          "no XS/S/M/L/XL impact category is assigned in any rendering of a receipt")
    check("there are no XS/S/M/L/XL bands" in rendered,
          "and the report says so, with the reason: a band structure whose lowest paying "
          "step sits above the physical ceiling tells contributors the wrong thing")
    check("Frontier Gain" in rendered, "the continuous figure is what is reported")
    check(f"{result.gain_percent:+.4f}%" in rendered,
          "dF is rendered at four decimal places, as a continuous number")


def test_reports_render():
    section("reports")
    generation = make_generation()
    flat = {"ctx128-c1": {"base": (500.0, 50.0, "OK")},
            "ctx128-c4": {"base": (500.0, 50.0, "OK")}}
    better = {"ctx128-c1": {"base": (600.0, 40.0, "OK")},
              "ctx128-c4": {"base": (620.0, 41.0, "OK")}}
    result = compute_frontier(generation, matrix(flat, better))
    receipt = build_receipt(generation=generation, computation=result, correctness="PASS",
                            provenance={"baseline_commit": "aaa", "candidate_commit": "bbb"},
                            pr=7)
    md = report_mod.markdown(receipt)
    check(md.count("|") > 20 and md.startswith("# TensorTransit Frontier Receipt"),
          "the Markdown report renders with tables")
    check("ctx128-c1" in md and "ctx128-c4" in md, "and names every scored cell")
    comment = report_mod.pr_comment(receipt)
    check("Frontier Gain" in comment and "Status:" in comment, "the PR comment renders")
    check("no figure in this comment was typed by hand" in comment,
          "and says where its numbers came from")
    box = report_mod.receipt_box(receipt)
    check(box.count("\n") >= 8 and "TENSORTRANSIT FRONTIER" in box, "the terminal box renders")
    check(len({len(line) for line in box.split("\n")}) == 1,
          "and every line of it is the same width -- a box whose border does not line up is a "
          "report somebody will retype rather than paste")
    grid = report_mod.contribution_map(receipt)
    check("c1" in grid and "c4" in grid, "the contribution map lays cells out by regime")


def main():
    for test in (test_normalization, test_pareto, test_hypervolume, test_aggregate,
                 test_confidence, test_generation, test_attribution_of_a_serving_loss,
                 test_the_settle_waits_for_the_device_rather_than_for_a_clock,
                 test_a_run_that_scheduled_serially_is_named_and_still_scored,
                 test_a_floor_decision_inside_the_published_noise_is_named,
                 test_compute_cases, test_compute_guards,
                 test_a_consistent_tiny_difference_does_not_qualify_on_confidence_alone,
                 test_regression_guard, test_receipt_and_result_match_their_schemas,
                 test_receipt_and_ledger, test_status_derivation,
                 test_a_frozen_generation_has_exactly_one_definition,
                 test_no_size_bands, test_reports_render):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"Ran {CHECKS} tests")
    print("frontier: all golden tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
