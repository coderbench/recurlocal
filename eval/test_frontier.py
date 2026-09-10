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
from frontier.generations import parse_generation, GenerationError      # noqa: E402
from frontier.normalize import Objective, ObjectiveError, normalize_point  # noqa: E402
from frontier.receipt import ReceiptError, content_digest, decide_status  # noqa: E402
from frontier import ledger as ledger_mod                               # noqa: E402
from frontier import report as report_mod                               # noqa: E402

FAILURES = []


def check(condition, message):
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
                 test_confidence, test_generation, test_compute_cases, test_compute_guards,
                 test_regression_guard, test_receipt_and_result_match_their_schemas,
                 test_receipt_and_ledger, test_status_derivation,
                 test_no_size_bands, test_reports_render):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("frontier: all golden tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
