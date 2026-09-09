#!/usr/bin/env python3
"""Sweep one optimization surface and report it in isolation.

`run_eval.py` compares the four locality modes. That answers "does the mechanism work",
not "which implementation of it is best", and it cannot tell whether a difference is real
or noise on a surface whose noise floor differs from baseline's.

This walks a single axis with every other variable fixed, reports the spread alongside the
gain, and refuses to name a winner when the winner is inside the noise. A surface where no
setting clears the noise is an open surface, not a solved one.

  sweep.py --binary ./build/recur_local_cuda_bench --axis pre-touch
  sweep.py --binary ./build/recur_local_cuda_bench --axis sequences --mode combined
  sweep.py --binary ./build/recur_local_cuda_bench --axis prefetch-distance --repeats 9
"""
import argparse, json, statistics, subprocess, sys
from pathlib import Path

# Each axis: the benchmark flag, the values to walk, and the mode that exercises it.
AXES = {
    "pre-touch":         {"flag": "--pre-touch",         "values": ["scalar", "vec4", "vec4_ldcg",
                                                                    "ptx_l2", "warp_tile", "partial"],
                          "mode": "prefetch"},
    "prefetch-distance": {"flag": "--prefetch-distance", "values": [1, 2, 3, 4, 6, 8], "mode": "prefetch"},
    "hot-set-policy":    {"flag": "--hot-set-policy",    "values": ["fixed", "proportional", "sqrt", "cliff", "quota"],
                          "mode": "persist"},
    "sequences":         {"flag": "--sequences",         "values": [1, 2, 4, 8, 16, 32], "mode": "combined"},
    "stream-bytes":      {"flag": "--stream-bytes",      "values": [0, 16 << 20, 64 << 20, 256 << 20],
                          "mode": "persist"},
    "prefetch-schedule": {"flag": "--prefetch-schedule", "values": ["uniform", "ramp", "alternating", "sparse"],
                          "mode": "prefetch"},
    "prefetch-impl":     {"flag": "--prefetch-impl",     "values": ["stream", "fused"], "mode": "prefetch"},
    "state-layout":      {"flag": "--state-layout",      "values": ["linear", "head_interleaved", "tile_swapped"],
                          "mode": "combined"},
    "qos":               {"flag": "--qos",               "values": ["off", "on"], "mode": "persist"},
    # What counts as competing for the set-aside. Sweep it at concurrency (-- --sequences 4
    # or more): at one sequence the two models disagree about the number but the policy
    # lands in the same place, and the disagreement is the whole point.
    "hot-set-model":     {"flag": "--hot-set-model",
                          "values": ["current_layer", "token_footprint", "reuse_window"],
                          "mode": "persist"},
}


def measure(binary, mode, extra, repeats):
    times, last = [], None
    for _ in range(repeats):
        p = subprocess.run([str(binary), mode, *extra], capture_output=True, text=True)
        if p.returncode:
            raise SystemExit(f"{mode} {' '.join(extra)} failed:\n{p.stderr}")
        last = json.loads(p.stdout.strip().splitlines()[-1])
        times.append(last["elapsed_ms"])
    median = statistics.median(times)
    return {"median_ms": median, "min_ms": min(times), "max_ms": max(times),
            "rel_spread_pct": (max(times) - min(times)) / median * 100.0,
            "checksum": last["checksum"], "run": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--axis", required=True, choices=sorted(AXES))
    ap.add_argument("--mode", help="override the mode that exercises this axis")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--output", type=Path)
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="extra benchmark args after --")
    a = ap.parse_args()
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra
    axis = AXES[a.axis]
    mode = a.mode or axis["mode"]

    points = {}
    for value in axis["values"]:
        args = [*extra, axis["flag"], str(value)]
        # The baseline is re-measured per point: sequences and stream-bytes change the
        # workload itself, so a single global baseline would not be a control.
        base = measure(a.binary, "baseline", args, a.repeats)
        cand = measure(a.binary, mode, args, a.repeats)
        if cand["checksum"] != base["checksum"]:
            raise SystemExit(f"correctness failure at {a.axis}={value}: "
                             f"{cand['checksum']} != {base['checksum']}")
        points[str(value)] = {
            "gain_pct": (base["median_ms"] / cand["median_ms"] - 1.0) * 100.0,
            "baseline_ms": base["median_ms"], "candidate_ms": cand["median_ms"],
            "rel_spread_pct": cand["rel_spread_pct"],
            "hot_set_oversubscribed": cand["run"].get("hot_set_oversubscribed", 0),
        }

    gains = [p["gain_pct"] for p in points.values()]
    noise = max(p["rel_spread_pct"] for p in points.values())
    span = max(gains) - min(gains)
    best = max(points, key=lambda k: points[k]["gain_pct"])
    resolved = span > noise

    out = {"axis": a.axis, "mode": mode, "repeats": a.repeats, "extra_args": list(extra),
           "points": points, "best": best, "best_gain_pct": points[best]["gain_pct"],
           "spread_across_axis_pct": span, "noise_floor_pct": noise,
           "resolved": resolved,
           "verdict": (f"{a.axis}={best} leads by {span:.2f} points over a {noise:.2f}% noise floor"
                       if resolved else
                       f"unresolved: the axis spans {span:.2f} points inside a {noise:.2f}% noise "
                       f"floor, so no setting is measurably best - raise --repeats or leave it open")}

    print(f"{'value':<14}{'gain%':>9}{'spread%':>10}{'oversub':>9}")
    for k, v in points.items():
        print(f"{k:<14}{v['gain_pct']:>8.2f}%{v['rel_spread_pct']:>9.2f}%{v['hot_set_oversubscribed']:>9}")
    print(f"\n{out['verdict']}")
    if a.output:
        a.output.write_text(json.dumps(out, indent=2) + "\n")
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main())
