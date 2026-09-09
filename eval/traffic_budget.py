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
import argparse, json, sys
from pathlib import Path

MiB = 1024 * 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pin", type=Path,
                    default=Path(__file__).resolve().parent.parent / "integrations" / "sparkinfer" / "pin.json",
                    help="state geometry comes from the pinned integration")
    ap.add_argument("--ms-per-token", type=float, required=True,
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

    pin = json.loads(a.pin.read_text())
    m = pin["model"]
    layers = m["recurrent_layers"]
    per_layer = (m["lin_state_bytes_per_layer"] * a.state_bytes_scale
                 + m["lin_conv_state_bytes_per_layer"])
    # Every recurrent layer reads its state and writes it back, once per token, per sequence.
    state_read = per_layer * layers * a.sequences
    state_traffic = state_read * 2

    step_seconds = a.ms_per_token / 1000.0
    total_traffic = a.bandwidth_gbs * 1e9 * step_seconds

    share = state_traffic / total_traffic
    read_share = state_read / total_traffic
    out = {
        "model": m.get("label"),
        "sequences": a.sequences,
        "recurrent_layers": layers,
        "state_bytes_per_layer_per_sequence": per_layer,
        "state_read_bytes_per_token": state_read,
        "state_traffic_bytes_per_token": state_traffic,
        "measured_ms_per_token": a.ms_per_token,
        "device_bandwidth_gbs": a.bandwidth_gbs,
        "implied_total_bytes_per_token": total_traffic,
        "recurrent_share_of_traffic_pct": share * 100.0,
        "ceiling_pct": share * 100.0,
        "note": "ceiling_pct is what removing ALL recurrent-state traffic would be worth. A "
                "locality policy recovers a fraction of it, never more.",
    }

    if a.measured_prefetch_cost_pct is not None:
        predicted = read_share * 100.0
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

    verdict = ("BELOW the 2% go/no-go floor: no locality policy on this state can reach it"
               if share * 100.0 < 2.0 else
               "above the 2% floor: a locality policy has room to matter here")
    print(f"\nrecurrent state is {share * 100:.2f}% of decode traffic at "
          f"{a.sequences} sequence(s) -- {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
