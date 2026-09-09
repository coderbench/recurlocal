#!/usr/bin/env python3
"""Walk one adapter axis on the real runtime, everything else fixed.

`eval/sweep.py` does this for the synthetic benchmark. This is the same discipline against
the pinned model: each value of one axis is measured as a control/candidate pair, so every
comparison is against a control that ran on the same box minutes earlier, and the spread
across the axis is reported next to the spread of the control arm with itself.

Like sweep.py it **refuses to name a winner inside the noise floor** and exits non-zero when
the axis will not resolve. An axis that will not resolve is open, not solved.

    eval/real_sweep.py --binary .../qwen3_gguf_bench --model DIR --axis mode
    eval/real_sweep.py --binary ... --model DIR --axis prefetch-distance \\
                       --fixed RECURLOCAL=prefetch
"""
import argparse, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_eval import measure, measure_concurrent, median, rel_spread_pct   # noqa: E402

# Each axis is one environment variable and the values the adapter accepts for it. Adding a
# mechanism to RecurLocal means adding its enumerator here; that is the whole registration.
AXES = {
    "mode":               ("RECURLOCAL", ["baseline", "persist", "prefetch", "combined"]),
    "pre-touch":          ("RECURLOCAL_PRE_TOUCH",
                           ["scalar", "vec4", "vec4_ldcg", "ptx_l2", "warp_tile", "partial"]),
    "pre-touch-coverage": ("RECURLOCAL_PRE_TOUCH_COVERAGE", ["matrix", "conv", "both"]),
    "prefetch-distance":  ("RECURLOCAL_PREFETCH_DISTANCE", ["1", "2", "3", "4", "6", "8"]),
    "prefetch-schedule":  ("RECURLOCAL_PREFETCH_SCHEDULE", ["uniform", "ramp", "alternating", "sparse"]),
    "prefetch-join":      ("RECURLOCAL_PREFETCH_JOIN", ["per_layer", "token_end"]),
    "hot-set-model":      ("RECURLOCAL_HOT_SET_MODEL", ["current_layer", "token_footprint", "reuse_window"]),
    "hot-set-policy":     ("RECURLOCAL_HOT_SET_POLICY", ["proportional", "fixed", "sqrt", "cliff", "quota"]),
    "window-scope":       ("RECURLOCAL_WINDOW_SCOPE", ["layer", "allocation", "ahead"]),
    "window-target":      ("RECURLOCAL_WINDOW_TARGET", ["matrix", "conv", "widest", "narrowest"]),
    # How the window reaches the kernel under graph capture. Not a tuning knob: capture_node
    # mutates a graph mid-capture and can invalidate it, so this axis measures a risk as much
    # as a gain. Check `capture_invalidations` in the telemetry alongside the number.
    "window-attach":      ("RECURLOCAL_WINDOW_ATTACH", ["stream", "capture_node"]),
    # How much of the device's persisting-L2 capacity to reserve. It decides nothing on a model
    # whose recurrent footprint is several times the cache -- the window is hopeless at any
    # fraction. It decides everything on one whose footprint is close to it, where the
    # difference between reserving 45 MiB and 60 MiB is the difference between three quarters
    # of the state resident and all of it.
    "budget-fraction":    ("RECURLOCAL_BUDGET_FRACTION", ["0.25", "0.50", "0.75", "1.00"]),
    "hit-ratio":          ("RECURLOCAL_HIT_RATIO", ["0.25", "0.50", "0.75", "1.00"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--axis", required=True, choices=sorted(AXES))
    ap.add_argument("--values", help="comma-separated subset of the axis to walk")
    ap.add_argument("--fixed", nargs="*", default=[], metavar="KEY=VALUE",
                    help="the rest of the candidate configuration, held constant")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int,
                    help="measure aggregate continuous-batching throughput at this concurrency "
                         "instead of single-sequence decode; --binary must then be "
                         "qwen3_gguf_cb_bench")
    ap.add_argument("--cb-prompt-len", type=int, default=128)
    ap.add_argument("--cb-max-new", type=int, default=64)
    ap.add_argument("--cb-long-prefill", type=int, default=4096)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()

    var, values = AXES[a.axis]
    if a.values:
        values = [v.strip() for v in a.values.split(",") if v.strip()]
    fixed = {}
    for item in a.fixed:
        k, v = item.split("=", 1)
        fixed[k] = v
    fixed.setdefault("RECURLOCAL", "combined")
    fixed["RECURLOCAL_STATS"] = "1"
    ctxs = [a.context]

    def one(env, label):
        """One measurement of whichever workload this sweep is about."""
        if a.concurrency:
            tps, _, _ = measure_concurrent(a.binary, a.model, a.concurrency, a.cb_prompt_len,
                                           a.cb_max_new, a.cb_long_prefill, env, label, True)
            return tps
        d, _, _ = measure(a.binary, a.model, a.tokens, ctxs, env, label, True)
        return d[str(a.context)]["decode_tps"]

    workload = (f"concurrency {a.concurrency} (aggregate)" if a.concurrency
                else f"batch 1, ctx {a.context}, {a.tokens} tokens")
    print(f"axis {a.axis} ({var}) over {values}")
    print(f"fixed: {' '.join(f'{k}={v}' for k, v in sorted(fixed.items()) if k != var)}")
    print(f"model {a.model}, {workload}, {a.repeats} interleaved pairs\n")

    control_runs = []
    rows = {v: [] for v in values}
    for rep in range(a.repeats):
        print(f"pair {rep + 1}/{a.repeats}", flush=True)
        control_runs.append(one({}, "control"))
        for v in values:
            env = dict(fixed)
            env[var] = v
            rows[v].append(one(env, f"{a.axis}={v}") / control_runs[-1])

    floor = rel_spread_pct(control_runs)
    result = {"axis": a.axis, "variable": var, "workload": workload,
              "concurrency": a.concurrency, "context": a.context, "tokens": a.tokens,
              "repeats": a.repeats, "fixed": {k: v for k, v in fixed.items() if k != var},
              "control_tps": control_runs, "control_noise_floor_pct": floor, "values": {}}
    print(f"\ncontrol {['%.2f' % t for t in control_runs]} tok/s"
          f"   noise floor {'unknown' if floor is None else f'{floor:.2f}%'}\n")
    print(f"{'value':<18}{'gain %':>10}{'spread %':>11}")
    gains = {}
    for v in values:
        gain = (median(rows[v]) - 1.0) * 100.0
        spread = rel_spread_pct(rows[v])
        gains[v] = gain
        result["values"][v] = {"paired_ratios": rows[v], "gain_pct": gain, "spread_pct": spread}
        print(f"{v:<18}{gain:>+10.2f}{(spread if spread is not None else float('nan')):>11.2f}")

    span = max(gains.values()) - min(gains.values())
    result["spread_across_axis_pct"] = span
    # Two control readings can land on the same number, which would make the peak-to-peak
    # spread exactly zero and then *any* difference across the axis would count as resolved.
    # A floor needs at least three samples to mean anything, and it can never be finer than
    # the benchmark's own reported precision (two decimals on a tok/s figure).
    quantisation = (0.005 / median(control_runs) * 100.0) if control_runs and median(control_runs) else 0.0
    estimable = len(control_runs) >= 3 and floor is not None and floor > 0.0
    effective = max(floor, quantisation) if estimable else None
    result["control_noise_floor_estimable"] = estimable
    result["quantisation_floor_pct"] = quantisation
    result["effective_floor_pct"] = effective
    resolved = effective is not None and span > effective
    result["resolved"] = resolved
    result["best"] = max(gains, key=gains.get) if resolved else None
    print()
    if resolved:
        print(f"axis spread {span:.2f}% > noise floor {effective:.3f}%  ->  best: {result['best']}")
    elif not estimable:
        print(f"axis spread {span:.2f}%, but {len(control_runs)} control repeats cannot estimate a "
              "noise floor (peak-to-peak needs at least 3). OPEN. No winner named.")
    else:
        print(f"axis spread {span:.2f}% is inside the {effective:.3f}% noise floor -- "
              "OPEN, not solved. No winner named.")
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {a.output}")
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main())
