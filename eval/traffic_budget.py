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


# Above this fraction of peak bandwidth, a decode step is close enough to memory-saturated
# that "removing f of the traffic shortens the step by f" is a fair description of it, and the
# ceiling below is tight. Under it the step is spending time on something other than moving
# bytes -- launch latency, low occupancy, small-GEMV inefficiency -- and freeing traffic buys
# proportionally less than the arithmetic suggests. The ceiling stays a valid UPPER bound
# either way, because it is computed from measured time rather than from assumed bytes; what
# changes is how close a real policy could ever come to it.
BANDWIDTH_BOUND_MIN = 0.80


def arm_ceiling(m, sequences, ms_per_token, state_bytes_scale, bandwidth_gbs,
                persisting_l2_bytes=None, active_weight_bytes_per_token=None):
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
    out = {
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
    if active_weight_bytes_per_token:
        # The bytes the step HAS to move: the weights it actually reads plus the recurrent
        # traffic computed above. Composed here rather than supplied whole, so a config only
        # has to state the one quantity that is a property of the checkpoint.
        out.update(bandwidth_bound_check(active_weight_bytes_per_token + state_traffic,
                                         total_traffic, active_weight_bytes_per_token))
    out.update(within_layer_ceiling(m, layers, sequences, total_traffic))
    if persisting_l2_bytes:
        out.update(persist_family_ceiling(state_read, total_traffic, persisting_l2_bytes))
    return out


def bandwidth_bound_check(active_bytes_per_token, total_traffic, weight_bytes=None):
    """Is this step actually bandwidth-bound? The ceiling's tightness depends on it.

    `implied_total_bytes_per_token` is measured step time times peak bandwidth, so it is what
    the step COULD have moved, not what it did. Comparing it against the bytes the checkpoint
    says the step has to touch -- non-expert weights, the routed experts a token actually
    selects, the output head, the recurrent state -- says how much of the step was really
    spent moving bytes.

    A dense model streaming every weight sits near 1.0 and the ceiling is nearly achievable.
    A sparse MoE reads a small fraction of its weights and can sit near 0.5: the step is
    half latency and occupancy, and freeing recurrent traffic cannot return the whole
    arithmetic share. Reporting the ratio is the difference between a ceiling a reader can
    calibrate and one that merely looks large.
    """
    util = active_bytes_per_token / total_traffic
    bound = "tight" if util >= BANDWIDTH_BOUND_MIN else "loose"
    return {
        "bandwidth_bound_check": {
            "active_bytes_per_token": active_bytes_per_token,
            "active_weight_bytes_per_token": weight_bytes,
            "bandwidth_utilisation": util,
            "bound": bound,
            "note": ("the step moves %.2f GB of the %.2f GB peak bandwidth would allow in the "
                     "measured time, i.e. %.0f%% utilisation. " % (
                         active_bytes_per_token / 1e9, total_traffic / 1e9, util * 100.0)) +
                    ("the step is memory-saturated, so the ceiling above is tight: freeing a "
                     "fraction of the traffic really does shorten the step by about that "
                     "fraction." if bound == "tight" else
                     "the step is NOT memory-saturated, so the ceiling above is a loose upper "
                     "bound: part of the step is latency and occupancy rather than bytes, and "
                     "freeing recurrent traffic returns less than its arithmetic share. Treat "
                     "it as an upper bound to be measured against, not as a target."),
        }
    }


def within_layer_ceiling(m, layers, sequences, total_traffic):
    """A third ceiling: reuse the cache can actually serve, because it is inside one layer.

    The persist family targets reuse across a token -- a full model pass, far larger than any
    cache. The obvious next question is whether a recurrent layer touches its own state more
    than once while it runs, because that distance is microseconds and L2 serves it for free.
    This bounds the answer from the state geometry and the pinned runtime's kernels.

    Two recurrent kernels, and they are not alike:

      matrix state   `gdn_ar_fast_kernel` holds each state column in registers across both of
                     its passes, so every byte is read exactly once and written exactly once.
                     There is no second touch for any cache policy to serve. (The naive
                     kernel it replaced read the state twice and wrote it twice -- so this is
                     a property of the pinned runtime, not of Gated DeltaNet. A runtime that
                     had not made that change WOULD have reuse here, and this bound would be
                     larger for it.)

      conv state     `conv_split_kernel` reads the K-1 window entries to convolve, then reads
                     K-2 of them AGAIN to shift the window forward one step. Those re-reads
                     are the whole of the within-layer reuse on this model.

    So the bound is the conv window's shift re-read and nothing else. That is a fraction of a
    state that is already only 1.9-3.8% of the recurrent bytes, which is why this is worth
    computing before it is worth building.
    """
    conv = m.get("lin_conv_state_bytes_per_layer", 0)
    k = m.get("linear_conv_kernel_dim", 0)
    if not conv or k < 3:
        # K < 3 leaves no window to shift, so there is no second touch at all.
        reread = 0
    else:
        # The window is K-1 entries wide; the shift re-reads all but the last of them.
        reread = conv * (k - 2) / (k - 1)
    saveable = reread * layers * sequences
    f = saveable / total_traffic
    return {
        "within_layer_family": {
            "reusable_bytes_per_token": saveable,
            "share_of_traffic_pct": f * 100.0,
            "ceiling_pct": f / (1.0 - f) * 100.0 if f < 1.0 else float("inf"),
            "matrix_state_reuse_bytes": 0,
            "note": "the most a policy targeting reuse WITHIN a recurrent layer could ever "
                    "save, on the pinned runtime's kernels: the conv window's shift re-read. "
                    "The matrix state contributes zero because the GDN kernel already holds "
                    "each column in registers across both passes -- one global read, one "
                    "global write. A reuse distance the cache can serve is not worth much if "
                    "there are almost no bytes at that distance.",
        }
    }


def persist_family_ceiling(state_read, total_traffic, persisting_l2_bytes):
    """A second, much tighter ceiling: what a persisting-L2 policy specifically can reach.

    The traffic ceiling assumes all the recurrent traffic can be removed. A persisting window
    cannot remove traffic it cannot hold. State written at layer i is read again at layer i of
    the NEXT token, so to save a byte the cache has to keep it across a full pass over the
    model - which means the whole per-token recurrent footprint has to be resident at once,
    not one layer's worth. That footprint is compared here against the device's persisting-L2
    capacity, and whatever does not fit is traffic the persist family cannot address however
    the window is shaped.

    The bound is deliberately generous: it assumes a perfect replacement policy in which every
    resident byte hits, and it ignores the set-aside's cost to everything else competing for
    L2. A real policy lands below it.
    """
    resident = min(persisting_l2_bytes, state_read)
    saved = resident * 2 / total_traffic          # the resident bytes' read AND write-back
    # The whole bound has one lever. The numerator is 2*min(capacity, footprint) and the
    # capacity is fixed by the device, so on any model whose footprint already exceeds the
    # cache the only way to raise this ceiling is to shrink the DENOMINATOR -- to run a model
    # that moves fewer bytes per decode step. Inverting the bound at the significance floor
    # gives the number to screen a candidate model with, before integrating it.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import decide
    floor = decide.SIGNIFICANCE_PCT / 100.0
    break_even = resident * 2 * (1.0 + 1.0 / floor)
    return {
        "persist_family": {
            "per_token_state_footprint_bytes": state_read,
            "persisting_l2_capacity_bytes": persisting_l2_bytes,
            "footprint_over_capacity": state_read / persisting_l2_bytes,
            "resident_fraction_of_state": resident / state_read,
            "ceiling_pct": saved / (1.0 - saved) * 100.0,
            "break_even_step_traffic_bytes": break_even,
            "step_traffic_bytes": total_traffic,
            "clears_significance_floor": total_traffic <= break_even,
            "break_even_note": (
                f"a decode step must move at most {break_even / 1e9:.2f} GB for a persisting "
                f"window over this footprint to reach the {decide.SIGNIFICANCE_PCT}% floor at "
                f"all; this one moves {total_traffic / 1e9:.2f} GB. The capacity in the "
                "numerator is the device's and cannot be raised, so a model with less weight "
                "traffic per token is the only lever there is."),
            "note": "an optimistic bound on the persist family only: every resident byte "
                    "hits, and the set-aside costs nothing to the traffic competing with it. "
                    "State is reused a whole model pass later, so the footprint that must be "
                    "resident is every recurrent layer for every sequence, not one layer's.",
        }
    }


def matrix_ceiling(m, spec, bandwidth_gbs, output=None, persisting_l2_bytes=None):
    """The best score the whole section 44 matrix can physically return.

    Every arm has its own ceiling, and the verdict is a weighted geometric mean over all of
    them. So the interesting number is not any one arm's ceiling but what a submission would
    score if it hit ALL of them - a submission that makes recurrent state entirely free.
    That is the most this repository can ever pay, and if it lands in a low band then no
    amount of contributor effort moves it, because the traffic is not there to recover.

    Bands and weights come from decide.py rather than being restated here: a ceiling
    expressed in bands that had drifted from the scorer's would be worse than no ceiling.

    The geometry comes from the matrix spec when it carries one, and from the pinned
    integration otherwise. A second model measured on the same runtime is a different state
    shape against different decode rates, and pairing one model's rates with another's
    geometry produces a confident, wrong ceiling with nothing in the output to show it -- so
    the spec that supplies the rates may also supply the shape, and the result records which
    was used.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import decide

    geometry_source = "pin"
    if spec.get("model"):
        m = spec["model"]
        geometry_source = "matrix spec"

    arms, ratios, persist_ratios = {}, {}, {}
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
        c = arm_ceiling(m, seqs, ms, float(arm.get("state_bytes_scale", 1.0)), bandwidth_gbs,
                        persisting_l2_bytes or spec.get("persisting_l2_bytes"),
                        arm.get("active_weight_bytes_per_token")
                        or spec.get("active_weight_bytes_per_token"))
        c["measured"] = True
        c["weight"] = weight
        c["source"] = arm.get("source", "unspecified")
        arms[name] = c
        ratios[name] = (1.0 + c["ceiling_pct"] / 100.0, weight)
        if "persist_family" in c:
            persist_ratios[name] = (1.0 + c["persist_family"]["ceiling_pct"] / 100.0, weight)

    if not ratios:
        raise SystemExit("no arm in the matrix had a measured decode rate")

    total_weight = sum(w for _, w in ratios.values())
    gm = math.exp(sum(w * math.log(r) for r, w in ratios.values()) / total_weight)
    best_pct = (gm - 1.0) * 100.0

    _, decision, decision_text = decide.band(best_pct, decide.GO_NO_GO)

    missing = sorted(n for n, v in arms.items() if not v.get("measured"))
    out = {
        "what_this_is": "the highest weighted score the section 44 matrix can physically "
                        "return on this model and device, if a submission removed ALL "
                        "recurrent-state traffic on every arm",
        "model": m.get("label"),
        "model_geometry_source": geometry_source,
        "device_bandwidth_gbs": bandwidth_gbs,
        "arms": arms,
        "arms_without_a_measured_rate": missing,
        "weights_covered": round(total_weight, 4),
        "best_possible_weighted_gain_pct": best_pct,
        # The continuous figure, not a band. There is no band table any more; see the
        # comment where decide.IMPACT used to be, and frontier/README.md.
        "best_possible_gain_pct": best_pct,
        "best_possible_verdict": decision,
        "best_possible_go_no_go": decision_text,
        "significance_floor_pct": decide.SIGNIFICANCE_PCT,
        "clears_significance_floor": best_pct >= decide.SIGNIFICANCE_PCT,
    }
    if persist_ratios:
        pw = sum(w for _, w in persist_ratios.values())
        pgm = math.exp(sum(w * math.log(r) for r, w in persist_ratios.values()) / pw)
        p_pct = (pgm - 1.0) * 100.0
        out["persist_family_best_possible_weighted_gain_pct"] = p_pct
        out["persist_family_note"] = (
            "the persist family cannot save traffic it cannot hold resident. State is reused a "
            "whole model pass later, so the footprint that must stay in cache is every "
            "recurrent layer for every sequence. Against this device's persisting-L2 capacity "
            "that footprint is oversubscribed at every arm, and the shortfall grows with "
            "concurrency faster than the room does - which is why the persist family measures "
            "nothing at 16 sequences even though the traffic ceiling has quadrupled.")
    if best_pct < decide.SIGNIFICANCE_PCT:
        out["conclusion"] = ("no submission can clear the significance floor on this matrix; "
                             "the traffic to recover is not there")
    else:
        out["conclusion"] = (f"a submission that made recurrent state free would gain "
                             f"{best_pct:.2f}% of throughput on this matrix; that is the bound "
                             "on ONE of the frontier's two objectives, on this model and this "
                             "device, however good the policy")

    text = json.dumps(out, indent=2)
    if output:
        output.write_text(text + "\n")
    print(text)
    print(f"\nbest possible weighted THROUGHPUT gain: {best_pct:.2f}%  ->  "
          f"go/no-go {decision}. This bounds one objective; the frontier also scores p99 "
          f"inter-token latency, which this arithmetic says nothing about.", file=sys.stderr)
    if missing:
        print(f"arms with no measured rate (excluded): {', '.join(missing)}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", type=Path,
                    default=Path(__file__).resolve().parent.parent / "adapters" / "sparkinfer" / "pin.json",
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
    ap.add_argument("--persisting-l2-bytes", type=int, metavar="BYTES",
                    help="the device's persisting-L2 capacity (recur_local_info prints it as "
                         "persisting_l2_max_bytes). Adds a second, much tighter ceiling for "
                         "the persist family: it cannot save traffic it cannot hold resident.")
    ap.add_argument("--measured-prefetch-cost-pct", type=float,
                    help="measured slowdown of a full extra state read, to test the model")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.matrix is None and a.ms_per_token is None:
        ap.error("--ms-per-token is required unless --matrix is given")

    pin = json.loads(a.pin.read_text())
    m = pin["model"]

    if a.matrix is not None:
        return matrix_ceiling(m, json.loads(a.matrix.read_text()), a.bandwidth_gbs,
                              a.output, a.persisting_l2_bytes)

    out = arm_ceiling(m, a.sequences, a.ms_per_token, a.state_bytes_scale,
                      a.bandwidth_gbs, a.persisting_l2_bytes)
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
