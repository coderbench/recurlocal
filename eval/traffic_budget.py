#!/usr/bin/env python3
"""How much of a decode step's memory traffic is recurrent state?

The ceiling on any recurrent-state locality optimization is the share of decode traffic that
recurrent state accounts for. If that share is 1.6%, then making the state free is worth
1.6%, and section 21's 2% floor cannot be reached however good the policy is. That is a
fact about the model and the workload, not about the implementation, and it is worth
knowing before spending another week on the implementation.

Two independent estimates, deliberately:

  geometry     recurrent bytes per token, from the pinned state shape. Exact.
  measurement  total bytes per decode STEP, from the measured step time and the device's
               memory bandwidth. This is where the assumption lives: it treats decode as
               bandwidth-bound, which for a dense model streaming every weight per step is
               very nearly true and is checked below.

Concurrency is the interesting axis and the reason this is not a one-off calculation. Model
weights are read once per step however many sequences are in flight; recurrent state is read
once per sequence. So the recurrent share grows with the batch: on Qwen3.8-27B it is 1.65% at
batch 1 and 2.88% at four sequences with the runtime's bf16-compacted state. A ceiling below
the 2% floor is an answer; a ceiling above it is a target.

The check: a pre-touch adds a whole extra read of the recurrent state per token. On a
saturated memory system that costs about `recurrent_read_share` of the step, so comparing the
prediction against a measured `RECURLOCAL=prefetch` run tests the model —
`--measured-prefetch-cost-pct` runs it. Measure the pre-touch with its per-layer ordering
removed, or the number is dominated by graph-node overhead rather than by traffic. A measured
cost far BELOW the prediction is not a broken model: it means either the step has spare
bandwidth, or the pre-touch is working and the real read is being served from cache.

    eval/traffic_budget.py --ms-per-token 10.42 --bandwidth-gbs 1792
    eval/traffic_budget.py --ms-per-token 10.42 --bandwidth-gbs 1792 --sequences 32
"""
import argparse, json, math, sys
from pathlib import Path

MiB = 1024 * 1024


def arm_ceiling(m, sequences, ms_per_token, state_bytes_scale, bandwidth_gbs):
    """The share of one decode step's traffic that recurrent state accounts for."""
    layers = m["recurrent_layers"]
    per_layer = (m["lin_state_bytes_per_layer"] * state_bytes_scale
                 + m["lin_conv_state_bytes_per_layer"])
    # Every recurrent layer reads its state and writes it back, once per token, per sequence.
    state_read = per_layer * layers * sequences
    state_traffic = state_read * 2

    total_traffic = bandwidth_gbs * 1e9 * (ms_per_token / 1000.0)
    share = state_traffic / total_traffic
    # The share of traffic and the ceiling on THROUGHPUT are not the same number, and the
    # difference is not academic. Removing a fraction f of a bandwidth-bound step's traffic
    # shortens the step to (1-f)T, so tokens per second rise by f/(1-f) -- more than f. The
    # scorer measures candidate_tps/baseline_tps, so f/(1-f) is the ceiling in the currency
    # the verdict is paid in. Reporting f there understates it by 1.4 points at 32 sequences,
    # which would let a submission legitimately beat a number this repository called a
    # ceiling.
    ceiling = share / (1.0 - share)
    return {
        "model": m.get("label"),
        "sequences": sequences,
        "recurrent_layers": layers,
        "state_bytes_per_layer_per_sequence": per_layer,
        "state_read_bytes_per_token": state_read,
        "state_traffic_bytes_per_token": state_traffic,
        "measured_ms_per_token": ms_per_token,
        "device_bandwidth_gbs": bandwidth_gbs,
        "implied_total_bytes_per_token": total_traffic,
        "recurrent_share_of_traffic_pct": share * 100.0,
        "ceiling_pct": ceiling * 100.0,
        "note": "ceiling_pct is what removing ALL recurrent-state traffic would be worth, in "
                "the throughput terms the scorer measures: a step carrying f less traffic "
                "runs in (1-f) of the time, so tok/s rise by f/(1-f). A locality policy "
                "recovers a fraction of it, never more.",
    }


def matrix_ceiling(m, spec, bandwidth_gbs, output=None):
    """The best score the whole section 44 matrix can physically return.

    Every arm has its own ceiling, and the verdict is a weighted geometric mean over all of
    them. So the interesting number is not any one arm's ceiling but what a submission would
    score if it hit ALL of them - a submission that makes recurrent state entirely free.
    That is the most this repository can ever pay, and if it lands in a low band then no
    amount of contributor effort moves it, because the traffic is not there to recover.

    Bands and weights come from decide.py rather than being restated here: a ceiling
    expressed in bands that had drifted from the scorer's would be worse than no ceiling.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import decide

    arms, ratios = {}, {}
    for name, weight in sorted(decide.DEFAULT_WEIGHTS.items()):
        arm = (spec.get("arms") or {}).get(name)
        if arm is None:
            arms[name] = {"measured": False,
                          "note": "no measured decode rate for this arm; it cannot be given a "
                                  "ceiling, and the matrix ceiling below excludes it"}
            continue
        seqs = int(arm.get("sequences", 1))
        ms = arm.get("ms_per_token")
        if ms is None:
            # At concurrency one step advances every sequence, so the step time is
            # sequences/aggregate_tps - not the reciprocal of the aggregate rate.
            tps = arm.get("aggregate_tps")
            if not tps:
                raise SystemExit(f"arm {name!r}: give ms_per_token or aggregate_tps")
            ms = seqs / float(tps) * 1000.0
        c = arm_ceiling(m, seqs, ms, float(arm.get("state_bytes_scale", 1.0)), bandwidth_gbs)
        c["measured"] = True
        c["weight"] = weight
        c["source"] = arm.get("source", "unspecified")
        arms[name] = c
        ratios[name] = (1.0 + c["ceiling_pct"] / 100.0, weight)

    if not ratios:
        raise SystemExit("no arm in the matrix had a measured decode rate")

    total_weight = sum(w for _, w in ratios.values())
    gm = math.exp(sum(w * math.log(r) for r, w in ratios.values()) / total_weight)
    best_pct = (gm - 1.0) * 100.0
    _, impact = decide.band(best_pct, decide.IMPACT)
    _, decision, decision_text = decide.band(best_pct, decide.GO_NO_GO)

    missing = sorted(n for n, v in arms.items() if not v.get("measured"))
    out = {
        "what_this_is": "the highest weighted score the section 44 matrix can physically "
                        "return on this model and device, if a submission removed ALL "
                        "recurrent-state traffic on every arm",
        "model": m.get("label"),
        "device_bandwidth_gbs": bandwidth_gbs,
        "arms": arms,
        "arms_without_a_measured_rate": missing,
        "weights_covered": round(total_weight, 4),
        "best_possible_weighted_gain_pct": best_pct,
        "best_possible_impact": impact,
        "best_possible_verdict": decision,
        "best_possible_go_no_go": decision_text,
        "significance_floor_pct": decide.SIGNIFICANCE_PCT,
        "reachable_bands": [t for th, t in decide.IMPACT if best_pct >= th],
    }
    if best_pct < decide.SIGNIFICANCE_PCT:
        out["conclusion"] = ("no submission can clear the significance floor on this matrix; "
                             "the traffic to recover is not there")
    else:
        out["conclusion"] = (f"a submission that made recurrent state free would score "
                             f"{best_pct:.2f}% ({impact}); bands above that are unreachable "
                             "on this model and device, however good the policy")

    text = json.dumps(out, indent=2)
    if output:
        output.write_text(text + "\n")
    print(text)
    print(f"\nbest possible weighted gain: {best_pct:.2f}%  ->  impact {impact}, "
          f"go/no-go {decision}", file=sys.stderr)
    if missing:
        print(f"arms with no measured rate (excluded): {', '.join(missing)}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", type=Path,
                    default=Path(__file__).resolve().parent.parent / "integrations" / "sparkinfer" / "pin.json",
                    help="state geometry comes from the pinned integration")
    ap.add_argument("--matrix", type=Path, metavar="ARMS_JSON",
                    help="score the whole section 44 matrix instead of one arm: the highest "
                         "weighted gain any submission could physically earn. See "
                         "configs/rtx5090-section44-ceiling.json")
    ap.add_argument("--ms-per-token", type=float,
                    help="milliseconds per decode STEP, not per emitted token. At concurrency "
                         "one step advances every sequence, so this is sequences/aggregate_tps: "
                         "4 sequences at 329 aggregate tok/s is 12.16, not 3.04.")
    ap.add_argument("--bandwidth-gbs", type=float, required=True, help="device memory bandwidth, GB/s")
    ap.add_argument("--sequences", type=int, default=1)
    ap.add_argument("--state-bytes-scale", type=float, default=1.0,
                    help="1.0 for fp32 matrix state; 0.5 when the runtime compacts it to bf16 for "
                         "batched decode (SparkInfer does, under SPARKINFER_CB_GDN_STATE_B16), "
                         "which halves the footprint the whole question is about")
    ap.add_argument("--measured-prefetch-cost-pct", type=float,
                    help="measured slowdown of a full extra state read, to test the model")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.matrix is None and a.ms_per_token is None:
        ap.error("--ms-per-token is required unless --matrix is given")

    pin = json.loads(a.pin.read_text())
    m = pin["model"]

    if a.matrix is not None:
        return matrix_ceiling(m, json.loads(a.matrix.read_text()), a.bandwidth_gbs, a.output)

    out = arm_ceiling(m, a.sequences, a.ms_per_token, a.state_bytes_scale, a.bandwidth_gbs)
    share = out["recurrent_share_of_traffic_pct"] / 100.0
    read_share = out["state_read_bytes_per_token"] / out["implied_total_bytes_per_token"]

    if a.measured_prefetch_cost_pct is not None:
        # An extra read grows the step's traffic by read_share, so the step takes
        # (1+read_share) of the time and throughput falls by read_share/(1+read_share).
        predicted = read_share / (1.0 + read_share) * 100.0
        out["traffic_model_check"] = {
            "predicted_extra_read_cost_pct": predicted,
            "measured_prefetch_cost_pct": a.measured_prefetch_cost_pct,
            "residual_pct": a.measured_prefetch_cost_pct - predicted,
            "note": "a pre-touch adds one full state read per token. A residual ABOVE zero is "
                    "what the traffic model does not explain (graph nodes, SM contention); a "
                    "residual BELOW zero means the extra read cost less than the bytes imply - "
                    "spare bandwidth, or the pre-touched lines being served from cache.",
        }

    text = json.dumps(out, indent=2)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)

    ceiling_pct = out["ceiling_pct"]
    verdict = ("BELOW the 2% go/no-go floor: no locality policy on this state can reach it"
               if ceiling_pct < 2.0 else
               "above the 2% floor: a locality policy has room to matter here")
    print(f"\nrecurrent state is {share * 100:.2f}% of decode traffic at "
          f"{a.sequences} sequence(s), a throughput ceiling of {ceiling_pct:.2f}% "
          f"-- {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
