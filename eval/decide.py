#!/usr/bin/env python3
"""RecurLocal's go/no-go gate, applied to a measurement instead of read off a table.

The whole project is a bet that is meant to be settled, not defended: the overview states
plainly that RecurLocal "should be willing to fail this test". That only works if the
decision is mechanical. The bands in sections 21, 26 and 44 are otherwise prose that each
reader applies slightly differently, and a project deciding its own fate by eye will decide
in its own favour.

Two entry points, deliberately asymmetric:

  decide.py --synthetic eval-result.json
      Reads the synthetic feasibility eval. Reports the number and refuses to turn it into
      a verdict, because a locality microbenchmark is not a serving result (sections 17 and
      28: "+35% synthetic, +0.8% real" means the project has not done its job).

  decide.py --real real-result.json
      Settles the bet: exact-output gate, per-workload regression guard, then a weighted
      geometric mean across the workload matrix (section 44).

Real-result input shape:

  {
    "correctness": {"output_identical": true, "method": "greedy replay, token-exact"},
    "provenance":  {"gpu": "...", "recurlocal_commit": "...", "runtime_commit": "...", ...},
    "workloads": {
      "batch1":       {"weight": 0.40, "baseline_tps": 220.0, "candidate_tps": 238.0},
      "concurrency4": {"weight": 0.20, "baseline_tps": ...,   "candidate_tps": ...},
      "concurrency16":{"weight": 0.20, ...},
      "concurrency32":{"weight": 0.20, ...}
    }
  }
"""
import argparse, json, math, sys
from pathlib import Path

SCHEMA_VERSION = 1

# Section 21 — go/no-go on real end-to-end decode improvement.
GO_NO_GO = [(10.0, "expand", "expand immediately"),
            (7.0, "strong", "strong repo candidate"),
            (4.0, "promising", "promising"),
            (2.0, "weak", "weak; probably reject"),
            (float("-inf"), "reject", "reject the core hypothesis")]

# Section 26 — the project's own impact bands, for describing a result once it exists.
# Explicitly not Gittensor scoring, and not a contributor reward scale.
IMPACT = [(18.0, "XL"), (10.0, "L"), (7.0, "M"), (4.0, "S"), (2.0, "XS"), (float("-inf"), "none")]

# Section 44 — default workload matrix and weights.
DEFAULT_WEIGHTS = {"batch1": 0.40, "concurrency4": 0.20, "concurrency16": 0.20, "concurrency32": 0.20}

# Section 44 — "no important workload may regress > 2%".
REGRESSION_GUARD = 0.98
# Both tables call anything under 2% noise; it is also the synthetic evaluator's
# default stability threshold, so the two agree on what counts as resolvable.
SIGNIFICANCE_PCT = 2.0


def band(value, table):
    return next(entry for entry in table if value >= entry[0])


def score_real(doc, allow_regression=False):
    correctness = doc.get("correctness") or {}
    workloads = doc.get("workloads") or {}
    if not workloads:
        raise SystemExit("real result has no 'workloads'")

    out = {"schema_version": SCHEMA_VERSION, "track": "real",
           "provenance": doc.get("provenance", {})}

    # Exact-locality track: identical model output is a precondition, not a tradeoff
    # (sections 15 and 35). A faster run that changed the output scores nothing.
    if correctness.get("output_identical") is not True:
        out.update(scored=False, impact=None, verdict="REJECT",
                   reason="exact-locality track requires bit-identical model output; "
                          f"correctness.output_identical={correctness.get('output_identical')!r}")
        return out
    out["correctness_method"] = correctness.get("method", "unspecified")

    ratios, per_workload, regressions = {}, {}, []
    for name, w in sorted(workloads.items()):
        base, cand = w.get("baseline_tps"), w.get("candidate_tps")
        if not isinstance(base, (int, float)) or not isinstance(cand, (int, float)) or base <= 0 or cand <= 0:
            raise SystemExit(f"workload {name!r}: baseline_tps and candidate_tps must be positive numbers")
        weight = w.get("weight", DEFAULT_WEIGHTS.get(name))
        if weight is None:
            raise SystemExit(f"workload {name!r}: no weight given and no default for that name")
        if weight < 0:
            raise SystemExit(f"workload {name!r}: weight must be >= 0")
        ratio = cand / base
        ratios[name] = (ratio, float(weight))
        per_workload[name] = {"weight": float(weight), "baseline_tps": float(base),
                              "candidate_tps": float(cand), "ratio": ratio,
                              "gain_pct": (ratio - 1.0) * 100.0}
        if ratio < REGRESSION_GUARD:
            regressions.append(name)
    out["workloads"] = per_workload

    total_weight = sum(w for _, w in ratios.values())
    if total_weight <= 0:
        raise SystemExit("workload weights sum to zero")
    # Geometric mean: a 2x win on one workload must not cancel a 2x loss on another.
    gm = math.exp(sum(w * math.log(r) for r, w in ratios.values()) / total_weight)
    gain_pct = (gm - 1.0) * 100.0
    out["weighted_gain_pct"] = gain_pct
    out["weighted_ratio"] = gm

    if regressions and not allow_regression:
        out.update(scored=False, impact=None, verdict="REGRESSION",
                   regressed_workloads=regressions,
                   reason=f"{', '.join(regressions)} regressed more than "
                          f"{(1 - REGRESSION_GUARD) * 100:.0f}%; maintainers must approve the tradeoff "
                          "explicitly (--allow-regression)")
        return out
    if regressions:
        out["regressed_workloads"] = regressions
        out["regression_waived"] = True

    # An arm whose gain sits inside its own run-to-run spread is not evidence. real_eval.py
    # computes that per workload; until now decide.py never read it, so a "significant"
    # verdict could rest entirely on arms that did not resolve.
    unresolved = sorted((doc.get("measurement") or {}).get("unresolved_workloads") or [])
    if unresolved:
        out["unresolved_workloads"] = unresolved

    _, decision, decision_text = band(gain_pct, GO_NO_GO)
    _, impact = band(gain_pct, IMPACT)
    out.update(scored=True, impact=impact, verdict=decision, go_no_go=decision_text,
               significant=gain_pct >= SIGNIFICANCE_PCT and not unresolved)
    if unresolved and gain_pct >= SIGNIFICANCE_PCT:
        out["reason"] = (f"weighted gain {gain_pct:.2f}% clears the significance floor, but "
                         f"{', '.join(unresolved)} did not resolve outside its own run-to-run "
                         "spread — take more repeats before treating this as a result")
    elif not out["significant"]:
        out["reason"] = (f"weighted gain {gain_pct:.2f}% is below the {SIGNIFICANCE_PCT:.0f}% "
                         "significance floor — not a verified improvement")
    return out


def score_synthetic(doc):
    """Report the synthetic number; never let it stand in for a serving result."""
    out = {"schema_version": SCHEMA_VERSION, "track": "synthetic",
           "scored": False, "impact": None,
           "best_mode": doc.get("best_mode"),
           "best_synthetic_gain_pct": doc.get("best_synthetic_gain_pct"),
           "provenance": doc.get("environment", {})}

    if doc.get("correctness") != "pass":
        out.update(verdict="REJECT",
                   reason=f"synthetic eval correctness={doc.get('correctness')!r}; "
                          "state was not preserved across modes")
        return out

    stability = doc.get("stability") or {}
    if stability.get("verdict") == "unstable":
        out.update(verdict="UNSTABLE",
                   reason=f"run-to-run spread {stability.get('max_rel_spread_pct')}% exceeds the "
                          f"{stability.get('threshold_pct')}% threshold; the mode ranking is not resolved")
        return out

    out.update(verdict="SYNTHETIC-ONLY",
               reason="a locality microbenchmark is not a serving result; the go/no-go gate is "
                      "decided only by real end-to-end decode on a pinned model and runtime "
                      "(overview sections 17 and 28). Re-run with --real once that exists.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=Path, metavar="EVAL_RESULT_JSON",
                     help="eval-result.json from run_eval.py; reported but never tiered")
    src.add_argument("--real", type=Path, metavar="REAL_RESULT_JSON",
                     help="real end-to-end workload matrix to score")
    ap.add_argument("--allow-regression", action="store_true",
                    help="score despite a workload regressing more than 2% (maintainer decision)")
    ap.add_argument("--output", type=Path, help="also write the verdict here")
    a = ap.parse_args()

    path = a.synthetic or a.real
    doc = json.loads(path.read_text())
    verdict = score_synthetic(doc) if a.synthetic else score_real(doc, a.allow_regression)

    text = json.dumps(verdict, indent=2)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    # A verdict is not a process failure: the caller reads the JSON. Only a scored,
    # significant improvement exits 0, so a release script can gate on it directly.
    return 0 if verdict.get("scored") and verdict.get("significant") else 1


if __name__ == "__main__":
    sys.exit(main())
