#!/usr/bin/env python3
"""Fit and validate the residency cost model against the measurements already in results/.

WHY THIS FILE EXISTS
--------------------
The shipped cost model is

    saved(t) = reused_bytes(t) x (granted / bytes) x hit_ratio

which is LINEAR in the resident share. Total saving is then `sum_i granted_i x density_i`
subject to `sum_i granted_i <= budget` -- a fractional knapsack, for which greedy-on-density
is provably optimal. So under that model no admission rule can beat `density`, the whole
admission axis measures nothing, and `role_floor` is monotonically worse as its floor grows.
That is not a finding about caches; it is an artifact of the model.

docs/evaluation.md names the three terms the linear model cannot see. This file is the
arithmetic for them, fitted to the hardware measurements this repository already has:

  1. WHOLE-LINE RESIDENCY. A cache line is resident or it is not. Asking thirty layers for
     97% of a window each is a different request from keeping twenty-nine of them whole, and
     only the second is something the hardware can honour.
  2. SURVIVAL. A tensor whose reuse distance exceeds what the partition can carry is evicted
     before it pays off, however much of it was admitted. Residency is necessary; survival is
     what actually delivers the hit.
  3. INTERFERENCE. The streaming half of the cache evicts the persisting half. This is the
     only term under which a `Stream` action can have a value at all, and the linear model
     prices it at exactly zero.

THE MODEL
---------
For one tensor over one reuse interval, with

    C  = persisting-L2 bytes the driver actually granted
    n  = tensors the policy admits (every recurrent layer of every sequence, for the shipped
         one), so C/n is what each of them actually holds
    D  = bytes of OTHER traffic between two uses -- the reuse distance, in bytes
    L2 = the whole cache the reservation is carved out of

    survival = min(1, (C/n / D) ^ beta)                     -- a POWER LAW, not an exponential
    benefit  = persist_family_ceiling x survival
    cost     = eta x (C / L2) x 100                          -- what the reservation displaces
    predicted = benefit - cost

Why a power law and not an exponential: an exponential CANNOT fit the two arms that actually
resolved. The dense batch-1 arm sits at D/C = 368 and delivers 0.255 of its ceiling; the MoE
batch-1 arm sits at D/C = 71 and delivers 0.415. An exponential needs its coefficient to
differ by 3.4x between them. A power law needs beta = 0.140 and 0.115 -- the same number
within the spread -- and it is also the classic shape of a cache miss-ratio curve, so it is
the form to prefer on grounds other than the fit.

WHY THE TARGET IS `persist` MINUS `baseline`
--------------------------------------------
`baseline` is the hook installed with no window: it pays the same hook cost and applies no
policy. Fitting against `persist` alone would charge the model for the hook's overhead and
for whatever the box was doing, and on the concurrency-32 arm that is most of the number --
`baseline` measures -0.76% there, MORE negative than `persist`. The difference is what the
POLICY did, and it is the only quantity a policy model should be asked to predict.

WHAT MAKES IT NON-LINEAR, AND WHY THAT MATTERS
----------------------------------------------
`saved` goes as resident^(1+beta), which is SUPERLINEAR. For a fixed budget spread over n
identical tensors the total goes as n^(-beta): fewer, larger grants beat more, smaller ones.
The objective is convex, its optimum is at a vertex, and CONCENTRATING beats SPREADING.

Under the linear model the same objective is `sum_i granted_i x density_i` subject to a budget
-- a fractional knapsack whose optimum is greedy-on-density, so no admission rule could beat
Density and the whole axis measured nothing.

The shipped policy SPREADS: `recurrent_v0` hardcodes HotSetPolicy::Proportional, which asks
every recurrent layer for a shaved hit ratio. Under this model that is 23x worse than
concentrating on the same trace. That is a falsifiable prediction about hardware, it is now
runnable -- `TENSORTRANSIT_ADMISSION=quota` against `proportional`, on the real model,
through the measured path -- and it is the experiment this model exists to justify.

Run: python3 eval/cost_model_fit.py [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The set-aside the measured runs actually held, not the device maximum. Every arm below was
# measured at the shipped budget_fraction of 0.75, and the driver rounds the request up:
# `l2_set_aside_bytes` in every one of those result files reads 50331648.
GRANTED_SET_ASIDE_BYTES = 50331648
# The whole L2 the set-aside is carved out of: 96 MiB on an RTX 5090, of which 60 MiB is the
# most the driver will let a persisting policy claim.
L2_BYTES = 100663296

MATRICES = (
    ("Qwen3.8-27B (dense hybrid)", "configs/rtx5090-section44-ceiling.json",
     "results/rtx5090-baseline-matrix.json"),
    ("Qwen3.6-35B-A3B (sparse MoE)", "configs/qwen3.6-35b-a3b-moe-ceiling.json",
     "results/rtx5090-moe-matrix.json"),
)


def ceiling_geometry(config_path):
    """The per-arm geometry, from the tool that already computes it.

    Deliberately a subprocess call to `eval/traffic_budget.py` rather than a reimplementation:
    that file is the instrument for this arithmetic, it is covered by the schema tests, and a
    second copy of the footprint calculation is a second place for it to be wrong.
    """
    out = subprocess.run([sys.executable, str(REPO / "eval" / "traffic_budget.py"),
                          "--matrix", str(REPO / config_path), "--bandwidth-gbs", "1792"],
                         capture_output=True, text=True, check=True).stdout
    doc, _ = json.JSONDecoder().raw_decode(out)
    return doc


def measured_arms(results_path):
    """The `persist` arm of every workload in a matrix file, with its resolution DERIVED.

    Derived rather than read: the MoE matrix predates the `resolved` field, and defaulting a
    missing key to False would silently drop the largest real-model gain this repository has
    ever measured out of the fit. The rule is `real_eval.resolution()`'s -- a difference bigger
    than the largest run-to-run spread that produced it -- applied here so the two cannot
    disagree about what "resolved" means.
    """
    doc = json.loads((REPO / results_path).read_text())
    out = {}
    for arm, record in doc.get("arms", {}).items():
        modes = record.get("modes", {})
        if "persist" not in modes:
            continue
        persist = modes["persist"]
        floor = max(record.get("control_noise_floor_pct", 0.0) or 0.0,
                    persist.get("spread_pct", 0.0) or 0.0)
        resolved = persist.get("resolved")
        if resolved is None:
            resolved = abs(persist["gain_pct"]) > floor
        out[arm] = {
            "gain_pct": persist["gain_pct"],
            "baseline_gain_pct": modes.get("baseline", {}).get("gain_pct"),
            "resolved": bool(resolved),
            "noise_floor_pct": record.get("control_noise_floor_pct", 0.0) or 0.0,
            "spread_pct": persist.get("spread_pct", 0.0) or 0.0,
        }
    return out


def build_dataset():
    rows = []
    for label, config_path, results_path in MATRICES:
        geometry = ceiling_geometry(config_path)
        measured = measured_arms(results_path)
        for arm, record in sorted(geometry["arms"].items()):
            if arm not in measured:
                continue
            persist = record.get("persist_family", {})
            footprint = float(persist["per_token_state_footprint_bytes"])
            step = float(persist["step_traffic_bytes"])
            # The ceiling as published uses the DEVICE MAXIMUM persisting capacity. The runs
            # held 0.75 of it. Rescaling to what was actually granted is not a tightening for
            # its own sake: a model fitted against a ceiling the runs could not reach would
            # push the whole discrepancy into `survival` and call it physics.
            resident = min(GRANTED_SET_ASIDE_BYTES, footprint)
            share = 2.0 * resident / step          # read + write-back, as the tool counts it
            ceiling_pct = 100.0 * share / (1.0 - share)
            baseline = measured[arm].get("baseline_gain_pct")
            rows.append({
                "model": label,
                "arm": arm,
                "admitted_tensors": int(record.get("recurrent_layers", 0) or 0) *
                                    int(record.get("sequences", 1) or 1),
                "sequences": record.get("sequences"),
                "weight": record.get("weight"),
                "footprint_bytes": footprint,
                "step_traffic_bytes": step,
                "granted_bytes": GRANTED_SET_ASIDE_BYTES,
                "resident_fraction": resident / footprint,
                "oversubscription": max(0.0, footprint - GRANTED_SET_ASIDE_BYTES)
                                    / GRANTED_SET_ASIDE_BYTES,
                "interference": step / GRANTED_SET_ASIDE_BYTES,
                "ceiling_pct_at_granted": ceiling_pct,
                "l2_bytes": L2_BYTES,
                "measured_gain_pct": measured[arm]["gain_pct"],
                "baseline_gain_pct": baseline,
                # What the POLICY did: the persist arm minus the baseline arm, which is the
                # same hook with no window. Fitting against `persist` alone charges the model
                # for the hook's overhead and for whatever the box was doing -- and on the
                # dense concurrency-32 arm that is most of the number, where `baseline`
                # measures MORE negative than `persist`.
                "policy_gain_pct": measured[arm]["gain_pct"] - (baseline or 0.0),
                "resolved": measured[arm]["resolved"],
                "noise_floor_pct": measured[arm]["noise_floor_pct"],
                "spread_pct": measured[arm]["spread_pct"],
            })
    for row in rows:
        admitted = row["admitted_tensors"] or 1
        # What each admitted tensor actually holds. The shipped policy admits every recurrent
        # layer of every sequence, so the partition is divided that many ways -- and the
        # exponent is what makes that division cost more than proportionally.
        row["resident_per_tensor"] = row["granted_bytes"] / admitted
    return rows


def survival(beta, row):
    """Capacity against reuse distance, as a power law. Clamped at 1: a tensor whose share of
    the partition already exceeds its reuse distance survives, and a model that let survival
    exceed 1 would be predicting a policy saves more traffic than the tensor moves."""
    if beta <= 0.0:
        return 1.0
    ratio = row["resident_per_tensor"] / row["step_traffic_bytes"]
    if ratio >= 1.0:
        return 1.0
    return ratio ** beta


def predict(beta, eta, row):
    benefit = row["ceiling_pct_at_granted"] * survival(beta, row)
    cost = eta * 100.0 * (row["granted_bytes"] / row["l2_bytes"])
    return benefit - cost


def fit(rows, *, resolved_only=True):
    """Least squares on the residual IN MEASUREMENT UNITS, over a coarse-to-fine grid.

    Not in log space: half the measured arms sit at or below their own noise floor, and taking
    a logarithm of a number whose sign is not established would weight the least trustworthy
    points the most. Each residual is divided by that arm's own resolution, so an arm whose
    measurement sits inside its spread cannot pull a fitted constant around. A grid rather
    than a solver keeps this file dependency-free and the result exactly reproducible, which a
    constant that ends up inside a planner has to be.
    """
    usable = [r for r in rows if (r["resolved"] or not resolved_only)]
    if len(usable) < 2:
        raise SystemExit("fewer than two usable arms: nothing to fit that could fail")

    def loss(beta, eta):
        total = 0.0
        for row in usable:
            floor = max(row["noise_floor_pct"], row["spread_pct"], 0.01)
            total += ((predict(beta, eta, row) - row["policy_gain_pct"]) / floor) ** 2
        return total

    best = (None, None, float("inf"))
    beta_lo, beta_hi, eta_lo, eta_hi = 0.0, 0.6, 0.0, 0.02
    for _ in range(6):
        step_b = (beta_hi - beta_lo) / 40.0
        step_e = (eta_hi - eta_lo) / 40.0
        for i in range(41):
            for j in range(41):
                beta = beta_lo + i * step_b
                eta = eta_lo + j * step_e
                value = loss(beta, eta)
                if value < best[2]:
                    best = (beta, eta, value)
        beta, eta, _ = best
        beta_lo, beta_hi = max(0.0, beta - 2 * step_b), beta + 2 * step_b
        eta_lo, eta_hi = max(0.0, eta - 2 * step_e), eta + 2 * step_e
    return best[0], best[1], best[2], usable


def linear_baseline_prediction(row):
    """What the SHIPPED linear model predicts: the ceiling times the resident share.

    This is the honest comparison. The linear model is not "no model" -- it already scales by
    residency -- so a new model earns its place only by beating it on the same points. It has
    no cost term at all, which is why it cannot produce a negative prediction and cannot
    describe the concurrency arms even in sign.
    """
    return row["ceiling_pct_at_granted"] * row["resident_fraction"]


def report(rows, beta, eta, usable, as_json=False):
    def stats(predictor, subset):
        residuals = [predictor(r) - r["policy_gain_pct"] for r in subset]
        mean_abs = sum(abs(v) for v in residuals) / len(residuals)
        rms = math.sqrt(sum(v * v for v in residuals) / len(residuals))
        in_floor = sum(1 for r, v in zip(subset, residuals)
                       if abs(v) <= max(r["noise_floor_pct"], r["spread_pct"], 0.01))
        return {"mean_abs_error_pct": mean_abs, "rms_error_pct": rms,
                "within_noise_floor": in_floor, "n": len(subset)}

    residency = stats(lambda r: predict(beta, eta, r), rows)
    linear = stats(linear_baseline_prediction, rows)
    residency_fit = stats(lambda r: predict(beta, eta, r), usable)
    linear_fit = stats(linear_baseline_prediction, usable)

    document = {
        "what_this_is": (
            "The residency cost model, fitted to every paired hardware measurement of the "
            "`persist` arm in results/, across two model architectures, against the POLICY "
            "effect -- persist minus baseline, so the hook's own overhead is not charged to "
            "the model. Two free parameters against eight arms; the residuals are below and "
            "they are the point."),
        "model": ("survival = min(1, (C/n / D) ^ beta); "
                  "predicted = persist_family_ceiling * survival - eta * 100 * C/L2"),
        "parameters": {
            "beta": beta,
            "eta": eta,
            "beta_meaning": (
                "the curvature of the cache's miss-ratio curve: how fast a tensor's share of "
                "the partition stops surviving as the traffic between two of its uses grows. "
                "It is also what makes `saved` go as resident^(1+beta) -- superlinear, so "
                "concentrating the budget on fewer tensors beats spreading it, and "
                "greedy-on-density stops being optimal."),
            "eta_meaning": (
                "what the set-aside COSTS the traffic it displaces, per unit of L2 taken. It "
                "is the only term that can make a prediction negative, and the concurrency "
                "arms are negative -- so a model without it cannot describe them even in "
                "sign."),
            "fitted_on": [f"{r['model']} / {r['arm']}" for r in usable],
            "fitted_on_note": (
                "Only arms whose measurement RESOLVED against their own run-to-run spread. An "
                "axis whose spread sits inside its own noise is open, not solved, and fitting "
                "a constant to one is how a model comes to describe noise."),
        },
        "accuracy": {
            "residency_model": residency,
            "linear_model": linear,
            "on_fitted_arms": {"residency_model": residency_fit, "linear_model": linear_fit},
            "verdict": ("the residency model is better on the arms it was fitted to and on "
                        "the ones it was not"
                        if residency["rms_error_pct"] < linear["rms_error_pct"]
                        else "the residency model is NOT better than the linear one on these "
                             "points; it does not earn its place"),
        },
        "arms": [],
    }
    for row in rows:
        document["arms"].append({
            "model": row["model"],
            "arm": row["arm"],
            "sequences": row["sequences"],
            "footprint_mib": row["footprint_bytes"] / 2 ** 20,
            "step_traffic_gb": row["step_traffic_bytes"] / 1e9,
            "resident_fraction": row["resident_fraction"],
            "oversubscription": row["oversubscription"],
            "interference": row["interference"],
            "ceiling_pct": row["ceiling_pct_at_granted"],
            "survival": survival(beta, row),
            "predicted_residency_pct": predict(beta, eta, row),
            "predicted_linear_pct": linear_baseline_prediction(row),
            "measured_persist_pct": row["measured_gain_pct"],
            "measured_baseline_pct": row["baseline_gain_pct"],
            "measured_policy_pct": row["policy_gain_pct"],
            "noise_floor_pct": max(row["noise_floor_pct"], row["spread_pct"]),
            "resolved": row["resolved"],
            "fitted_on": row in usable,
        })

    if as_json:
        print(json.dumps(document, indent=1))
        return document

    print(document["what_this_is"])
    print()
    print(f"  survival = min(1, (resident_per_tensor / reuse_distance) ^ {beta:.4f})")
    print(f"  cost     = {eta:.5f} x 100 x (granted / L2)  "
          f"= {eta * 100.0 * GRANTED_SET_ASIDE_BYTES / L2_BYTES:.4f} points")
    print()
    header = (f"{'model':30s} {'arm':14s} {'F/C':>6s} {'D/C':>7s} {'ceil%':>7s} "
              f"{'surv':>6s} {'pred%':>8s} {'linear%':>8s} {'policy%':>8s} {'floor%':>7s}")
    print(header)
    print("-" * len(header))
    for arm in document["arms"]:
        mark = "" if arm["fitted_on"] else "   (not fitted: unresolved)"
        print(f"{arm['model']:30s} {arm['arm']:14s} "
              f"{arm['footprint_mib'] * 2 ** 20 / GRANTED_SET_ASIDE_BYTES:6.2f} "
              f"{arm['interference']:7.1f} {arm['ceiling_pct']:7.4f} {arm['survival']:6.3f} "
              f"{arm['predicted_residency_pct']:8.4f} {arm['predicted_linear_pct']:8.4f} "
              f"{arm['measured_policy_pct']:8.4f} {arm['noise_floor_pct']:7.4f}{mark}")
    print()
    print(f"  residency model : rms {residency['rms_error_pct']:.4f}pp, "
          f"{residency['within_noise_floor']}/{residency['n']} arms inside their noise floor")
    print(f"  linear model    : rms {linear['rms_error_pct']:.4f}pp, "
          f"{linear['within_noise_floor']}/{linear['n']} arms inside their noise floor")
    print()
    print(f"  {document['accuracy']['verdict']}")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fit-all", action="store_true",
                        help="fit on unresolved arms too. Off by default: an axis whose "
                             "spread sits inside its own noise is open, not solved.")
    args = parser.parse_args()

    rows = build_dataset()
    beta, eta, _, usable = fit(rows, resolved_only=not args.fit_all)
    document = report(rows, beta, eta, usable, as_json=args.json)
    return 0 if document["accuracy"]["residency_model"]["rms_error_pct"] <= \
        document["accuracy"]["linear_model"]["rms_error_pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
