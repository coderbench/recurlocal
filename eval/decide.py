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


def unwrap_real(doc):
    """Accept either a bare result or the results/ bundle that embeds one.

    `results/rtx5090-real.json` carries the scored matrix under "scored_result", alongside the
    axis sweeps and probes that give it context. Until this existed, the command the README
    told you to run - `decide.py --real results/rtx5090-real.json` - printed
    "real result has no 'workloads'" and scored nothing. A scorer that cannot read the
    repository's own published result is not a scorer.
    """
    if "workloads" not in doc and isinstance(doc.get("scored_result"), dict):
        return doc["scored_result"]
    return doc


def score_real(doc, allow_regression=False, allow_partial=False):
    doc = unwrap_real(doc)
    correctness = doc.get("correctness") or {}
    workloads = doc.get("workloads") or {}
    if not workloads:
        raise SystemExit("real result has no 'workloads'")

    out = {"schema_version": SCHEMA_VERSION, "track": "real",
           "provenance": doc.get("provenance", {})}

    # Exact-locality track: identical model output is a precondition, not a tradeoff
    # (sections 15 and 35). A faster run that changed the output scores nothing.
    #
    # Two different failures, and calling them both "the candidate changed the output" is a
    # false accusation. `output_identical: False` is the candidate diverging from a control the
    # runtime CAN reproduce. `None` is the gate being unable to answer -- either it did not run,
    # or the control does not agree with itself, which happens on a sparse-MoE checkpoint where
    # a few ULP in the prefill flip a discrete top-k expert choice. Neither is scorable, and
    # only the first is the submission's fault.
    if correctness.get("output_identical") is not True:
        if correctness.get("output_identical") is False:
            reason = ("the candidate changed model output: the exact-locality track requires "
                      "bit-identical greedy replay against the control "
                      f"(first divergence at token {correctness.get('first_divergence')})")
        elif correctness.get("runtime_reproducible") is False:
            n = correctness.get("control_replays")
            which = correctness.get("control_first_divergent_replay")
            how_many = (f"{which} of {n} unhooked control replays"
                        if which and n else "two control runs")
            reason = ("INCONCLUSIVE, and not the candidate's fault: "
                      f"{how_many} of this runtime on this model disagree with each other "
                      f"(first divergence at token {correctness.get('control_first_divergence')}"
                      "), so a candidate/control difference cannot be attributed to the "
                      "candidate. The exact-locality gate needs a reproducible runtime and "
                      "checkpoint; this result cannot be scored either way")
        else:
            reason = ("the exact-locality gate did not produce a verdict; "
                      f"correctness.output_identical={correctness.get('output_identical')!r}")
        out.update(scored=False, impact=None, verdict="REJECT", reason=reason)
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

    # Coverage is derived here, from the section 44 matrix, and never read out of the
    # document being scored. A missing workload is not a neutral omission: the weights
    # that remain are renormalised, so dropping an arm removes it from the mean entirely.
    # The competition brief points at concurrency 32, which makes that the arm a
    # submission profits most from leaving out. Report what is absent and how much weight
    # went with it, and do not let a partial matrix clear the significance floor.
    scored_names = set(workloads)
    missing = sorted(set(DEFAULT_WEIGHTS) - scored_names) if scored_names & set(DEFAULT_WEIGHTS) else []
    if missing:
        out["workload_coverage"] = {
            "matrix": "section 44",
            "measured": sorted(scored_names),
            "missing": missing,
            "missing_weight_share": round(sum(DEFAULT_WEIGHTS[n] for n in missing), 4),
        }

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

    blocking = missing if not allow_partial else []
    _, decision, decision_text = band(gain_pct, GO_NO_GO)
    _, impact = band(gain_pct, IMPACT)
    out.update(scored=True, impact=impact, verdict=decision, go_no_go=decision_text,
               partial=bool(missing),
               significant=gain_pct >= SIGNIFICANCE_PCT and not unresolved and not blocking)
    if missing and allow_partial:
        out["partial_waived"] = True
    reasons = []
    if gain_pct < SIGNIFICANCE_PCT:
        reasons.append(f"weighted gain {gain_pct:.2f}% is below the "
                       f"{SIGNIFICANCE_PCT:.0f}% significance floor")
    if unresolved:
        reasons.append(f"{', '.join(unresolved)} did not resolve outside its own run-to-run "
                       "spread — take more repeats")
    if blocking:
        share = sum(DEFAULT_WEIGHTS[n] for n in blocking)
        reasons.append(f"the workload matrix is incomplete: {', '.join(blocking)} "
                       f"({share:.0%} of the section 44 weight) was not measured, and the "
                       "remaining weights were renormalised to cover the gap "
                       "(--allow-partial to score anyway)")
    if reasons:
        out["reason"] = "; ".join(reasons)
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


def summarize(v, stream=sys.stderr):
    """A one-screen version of the verdict, on stderr so stdout stays machine-readable.

    The JSON is what a release script reads; nobody reads it to find out what happened. This
    prints the same fields in the order that decides them, and every qualifier the verdict
    carries - unresolved arms, a missing workload - so a summary can never look cleaner than
    the result it summarizes.
    """
    if v.get("track") == "synthetic":
        print(f"\nverdict: {v['verdict']}   {v.get('reason', '')}", file=stream)
        return
    gain = v.get("weighted_gain_pct")
    head = f"\nverdict: {v['verdict']}"
    if gain is not None:
        head += (f"   weighted gain {gain:+.3f}%   impact {v.get('impact') or 'none'}"
                 f"   significant {str(bool(v.get('significant'))).lower()}")
    print(head, file=stream)
    for name, w in sorted((v.get("workloads") or {}).items()):
        print(f"  {name:<15s} {w['gain_pct']:+.3f}%  (w={w['weight']:.2f})", file=stream)
    if v.get("unresolved_workloads"):
        print(f"  unresolved: {v['unresolved_workloads']}", file=stream)
    cov = v.get("workload_coverage")
    if cov:
        print(f"  NOT MEASURED: {', '.join(cov['missing'])} "
              f"({cov['missing_weight_share']:.0%} of the weight)"
              f"{' -- waived' if v.get('partial_waived') else ''}", file=stream)
    if v.get("regressed_workloads"):
        print(f"  regressed: {', '.join(v['regressed_workloads'])}", file=stream)
    if v.get("reason"):
        print(f"  {v['reason']}", file=stream)


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
    ap.add_argument("--allow-partial", action="store_true",
                    help="let an incomplete section 44 workload matrix clear the significance "
                         "floor (maintainer decision); the verdict still reports what is missing")
    ap.add_argument("--output", type=Path, help="also write the verdict here")
    a = ap.parse_args()

    path = a.synthetic or a.real
    doc = json.loads(path.read_text())
    verdict = score_synthetic(doc) if a.synthetic else score_real(doc, a.allow_regression, a.allow_partial)

    text = json.dumps(verdict, indent=2)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    summarize(verdict)
    # A verdict is not a process failure: the caller reads the JSON. Only a scored,
    # significant improvement exits 0, so a release script can gate on it directly.
    return 0 if verdict.get("scored") and verdict.get("significant") else 1


if __name__ == "__main__":
    sys.exit(main())
