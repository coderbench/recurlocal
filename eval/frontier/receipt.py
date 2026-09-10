"""The Frontier Receipt: the permanent, machine-readable evidence artifact.

A receipt is the only thing that survives. It has to answer, years later and without the box
that produced it: which generation, which baseline commit, which candidate commit, on what
hardware, under what runtime, against what model, with what correctness result, at what
confidence, and how much frontier was created.

It is also the thing an audit checks, so it carries a `content_digest` over its own scored
fields. `verify_receipt` recomputes that digest and re-derives the status from the numbers,
which is what turns "a finalized receipt must not be silently rewritten" from a rule into a
check. If an evaluator bug requires a correction, the answer is a SUPERSEDING receipt with a
reason, never an edit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

RECEIPT_SCHEMA_VERSION = 1

# Statuses describe evaluation STATE. None of them categorizes impact magnitude, and that is
# the point: no XS/S/M/L/XL anywhere in this system.
STATUSES = ("FRONTIER_GAIN", "NO_FRONTIER_GAIN", "INCONCLUSIVE", "CORRECTNESS_FAIL",
            "BUILD_FAIL", "REGRESSION_GUARD_FAIL", "EVAL_ERROR")


class ReceiptError(ValueError):
    """A receipt is malformed, or does not verify against its own contents."""


def decide_status(computation, correctness: str) -> str:
    """The status a receipt gets, derived from the numbers rather than chosen.

    Order matters and is the specification's: correctness precedes performance scoring, the
    protected-workload guard vetoes even a positive aggregate, and confidence qualifies rather
    than scales.
    """
    if correctness == "BUILD_FAIL":
        return "BUILD_FAIL"
    if correctness != "PASS":
        return "CORRECTNESS_FAIL"
    if computation.guard_violations:
        return "REGRESSION_GUARD_FAIL"
    if not computation.qualifies:
        # One of the two gates did not pass: the confidence lower bound is not above zero, or
        # the observed difference is inside the run-to-run spread of the runs that produced it.
        # Both are "we do not know", and publishing the observed figure as a verified
        # contribution is exactly what this status prevents.
        if computation.gain <= 0.0 and computation.statistics.get("upper", 0.0) < 0.0:
            return "NO_FRONTIER_GAIN"
        return "INCONCLUSIVE"
    return "FRONTIER_GAIN"


def content_digest(receipt: dict) -> str:
    """SHA-256 over everything except the digest and signature fields themselves."""
    scored = {k: v for k, v in receipt.items() if k not in ("content_digest", "signature")}
    canonical = json.dumps(scored, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def build_receipt(*, generation, computation, correctness, provenance, pr=None,
                  supersedes=None, reason=None, timestamp=None) -> dict:
    """Assemble the canonical receipt.

    `correctness` is "PASS", "FAIL", or "BUILD_FAIL"; a run that did not build or that changed
    the model's output has a verified dF of exactly 0 whatever it measured, and the measured
    figure is still recorded so the failure is debuggable.
    """
    status = decide_status(computation, correctness)
    verified_gain = computation.gain_percent if status == "FRONTIER_GAIN" else 0.0

    # A PARTIAL matrix credits nothing, whatever it measured, and `frontier/README.md` has
    # always said so -- "does not qualify as a full-coverage contribution" -- while the code
    # credited it anyway. It matters more now that a cell can leave the score by being
    # attributed to the runtime: dropping a cell is the cheapest way to raise a score, and a
    # rule that stops a drop from PAYING removes the incentive rather than policing it.
    #
    # The measured figure is still reported, and the status still says what the statistics say.
    # What is withheld is the credit, and the receipt names the reason.
    credit_withheld = None
    if verified_gain and getattr(computation, "partial", False):
        missing = list(getattr(computation, "cells_missing", []))
        unservable = list(getattr(computation, "cells_unservable", []))
        credit_withheld = {
            "reason": "partial coverage",
            "missing_cells": missing,
            "unservable_cells": unservable,
            "note": ("A receipt that did not score the whole matrix credits nothing. Omitting "
                     "the arm with the most room would otherwise be the cheapest way to raise "
                     "a score, and an attributed serving loss leaves the matrix by the same "
                     "door. Run the whole matrix to be credited."),
        }
        verified_gain = 0.0

    receipt = {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "benchmark_generation": generation.name,
        "generation_checksum": generation.checksum(),
        "timestamp_utc": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),

        "pr": pr,
        "baseline_commit": provenance.get("baseline_commit"),
        "candidate_commit": provenance.get("candidate_commit"),
        "evaluator_commit": provenance.get("evaluator_commit"),

        "hardware": provenance.get("hardware", {}),
        "runtime": provenance.get("runtime", {}),
        "model": provenance.get("model", {}),
        "environment_fingerprint": provenance.get("environment_fingerprint"),

        # Whether this was a contributor iterating locally or the trusted evaluator's run, and
        # whether the generation's workload instances are public or hidden. Recorded because a
        # receipt that does not say which cannot be compared with one that was scored the other
        # way -- and because "TTF-1 has no hidden instances" is a fact about the pinned bench
        # that a future generation will not share.
        "mode": provenance.get("mode", "unspecified"),
        "instances": provenance.get("instances", "unspecified"),

        "correctness": correctness,
        "correctness_method": provenance.get("correctness_method",
                                             "greedy replay, token-exact"),
        # WHICH question the gate answered. Against the baseline build it is "this submission
        # does not change the model's output"; against the candidate's own binary it is only
        # "enabling the policy does not change it", which a submission whose inert path changed
        # the output would pass. A receipt that did not distinguish them would let the weaker
        # check be read as the stronger one.
        "correctness_compared_against":
            (provenance.get("correctness") or {}).get("compared_against", "unspecified"),

        "frontier": {
            "before": computation.frontier_before,
            "after": computation.frontier_after,
            # The measured figure, always. A receipt that hid the measurement behind its
            # status would make a CORRECTNESS_FAIL undebuggable.
            "gain_percent": computation.gain_percent,
            # What the ledger credits. Zero unless the status is FRONTIER_GAIN.
            "verified_gain_percent": verified_gain,
            # Present only when a measured gain was NOT credited, and why.
            "credit_withheld": credit_withheld,
        },
        "objectives": [o.to_json() for o in generation.objectives],
        # The bounds each cell was actually scored under. Without these a reader cannot
        # reproduce a hypervolume from the raw results, and a receipt that cannot be
        # reproduced is a claim rather than evidence.
        "cell_objectives": {
            cell: [o.to_json() for o in generation.objectives_for(cell)]
            for cell in computation.cells_scored
        },
        "reference_point": list(generation.reference_point),
        "aggregation": {
            "method": generation.aggregation,
            "weights": computation.weights,
            "cell_floor": generation.cell_floor,
            # Which cells the FLOOR decided rather than a measurement. A cell whose
            # hypervolume was zero for one arm has its ratio set by this constant, and one
            # such cell can move dF by hundreds of percent through the geometric mean. Naming
            # them is the difference between a result and a number.
            "cells_at_floor": computation.cells_at_floor,
            "floor_decided": computation.floor_decided,
            # A floor decision taken on an objective whose PUBLISHED control spread swallows
            # the change observed. The generation froze that spread when it was calibrated, so
            # this is not the run grading its own homework.
            "floor_decided_inside_published_noise":
                getattr(computation, "floor_decided_inside_published_noise", []),
        },

        "coverage": {
            "improved_cells": len(computation.coverage["improved"]),
            "neutral_cells": len(computation.coverage["neutral"]),
            "regressed_cells": len(computation.coverage["regressed"]),
            "total_cells": len(generation.cells),
            "scored_cells": len(computation.cells_scored),
            "missing_cells": computation.cells_missing,
            # A cell dropped because the evaluator's own probe reproduced the failure with no
            # policy running. Dropping a cell is the cheapest way to raise a score, so the
            # drop and its evidence travel together and force PARTIAL.
            "unservable_cells": getattr(computation, "cells_unservable", []),
            "unservable_evidence": getattr(computation, "unservable_evidence", {}),
            "partial": computation.partial,
            # Runs that took far longer than their siblings for the same decode work. Scored,
            # because the failure is the runtime's, and named, because a median over three
            # repeats does not survive one of them.
            "scheduling_outliers": getattr(computation, "scheduling_outliers", []),
        },
        # Per cell, per objective: what moved, against the spread the generation published for
        # that cell. Diagnostic and never a score -- it says which cells the score is entitled
        # to rest on.
        "cell_resolution": getattr(computation, "cell_resolution", {}),
        # How much of the scored matrix this receipt is entitled to rest on, per objective.
        "resolution_summary": getattr(computation, "resolution_summary", {}),
        "cells": {
            cell: {
                "gain": computation.cell_gain[cell],
                "hypervolume": computation.cell_hypervolume[cell],
                "weight": computation.weights[cell],
                "protected": cell in generation.protected_cells,
                # Which configurations were on the frontier in this cell. This is where a new
                # planner earns its place: it does not have to beat everything everywhere, it
                # has to hold territory somewhere.
                "on_frontier": computation.frontier_configurations.get(cell, {}),
            }
            for cell in computation.cells_scored
        },
        "configurations": computation.configurations,
        "failures": computation.failures,

        "statistics": {
            "confidence_level": computation.statistics["level"],
            "lower_gain_percent": computation.statistics["lower"] * 100.0,
            "upper_gain_percent": computation.statistics["upper"] * 100.0,
            "paired_repeats": computation.statistics["repeats"],
            "unpaired_repeats": computation.unpaired_repeats,
            "method": computation.statistics["method"],
            "resamples": computation.statistics["resamples"],
            "seed": computation.statistics["seed"],
            "per_repeat_frontier": computation.per_repeat,
            # The second gate. The bootstrap says the difference is unlikely to be zero; this
            # says it is bigger than the run-to-run spread of the runs that produced it. A
            # percentile bootstrap over three paired points cannot tell a consistent tiny
            # effect from consistent tiny bias, so both have to pass.
            "noise_floor_pct": computation.noise_floor_pct,
            "main_spread_pct": computation.main_spread_pct,
            "candidate_spread_pct": computation.candidate_spread_pct,
            "resolved": computation.resolved,
            "confidence_qualifies": computation.confidence_qualifies,
        },
        "regression_guard": {
            "protected_cells": generation.protected_cells,
            "max_regression": generation.protected_max_regression,
            "violations": computation.guard_violations,
        },
        "status": status,
    }
    if supersedes:
        receipt["supersedes"] = list(supersedes)
        receipt["supersede_reason"] = reason or "unspecified"
    receipt["content_digest"] = content_digest(receipt)
    return receipt


def verify_receipt(receipt: dict, generation=None) -> dict:
    """Recompute what the receipt asserts about itself. Raises on a mismatch.

    Three checks, and each catches a different way a receipt can stop being evidence:
    a rewritten field (digest), a re-scored run under a changed generation (checksum), and a
    status that does not follow from the numbers next to it (derivation).
    """
    problems = []
    if receipt.get("receipt_schema_version") != RECEIPT_SCHEMA_VERSION:
        problems.append(f"schema version {receipt.get('receipt_schema_version')} "
                        f"!= {RECEIPT_SCHEMA_VERSION}")
    stated = receipt.get("content_digest")
    recomputed = content_digest(receipt)
    if stated != recomputed:
        problems.append(f"content digest {stated} != recomputed {recomputed}: the receipt "
                        f"has been edited since it was finalized")
    if generation is not None:
        if receipt.get("benchmark_generation") != generation.name:
            problems.append(f"receipt is for {receipt.get('benchmark_generation')}, "
                            f"checked against {generation.name}")
        elif receipt.get("generation_checksum") != generation.checksum():
            problems.append(
                f"generation checksum has moved: the receipt was scored under "
                f"{receipt.get('generation_checksum')} and {generation.name} now hashes to "
                f"{generation.checksum()}. A generation is frozen; if its meaning changed, "
                f"the answer is a new TTF-N, not an edit.")

    status = receipt.get("status")
    if status not in STATUSES:
        problems.append(f"unknown status {status!r}")
    verified = receipt.get("frontier", {}).get("verified_gain_percent")
    if status != "FRONTIER_GAIN" and verified not in (0, 0.0):
        problems.append(f"status {status} credits {verified}%; only FRONTIER_GAIN may credit "
                        f"a non-zero gain")
    if receipt.get("coverage", {}).get("partial") and verified not in (0, 0.0):
        problems.append(f"a PARTIAL receipt credits {verified}%; a matrix that did not score "
                        f"every cell credits nothing, because omitting the arm with the most "
                        f"room would otherwise be the cheapest way to raise a score")
    if status == "FRONTIER_GAIN":
        lower = receipt.get("statistics", {}).get("lower_gain_percent")
        if lower is None or lower <= 0.0:
            problems.append(f"status FRONTIER_GAIN with a lower confidence bound of {lower}: "
                            f"confidence is a qualification gate, and this one did not pass")
        if receipt.get("correctness") != "PASS":
            problems.append("status FRONTIER_GAIN with correctness not PASS")
        if receipt.get("regression_guard", {}).get("violations"):
            problems.append("status FRONTIER_GAIN with protected-workload violations")

    if problems:
        raise ReceiptError("; ".join(problems))
    return {"verified": True, "status": status, "content_digest": recomputed}
