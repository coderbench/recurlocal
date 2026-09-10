#!/usr/bin/env python3
"""Which checkpoints can this device pay for, before downloading any of them.

    eval/screen_checkpoints.py --hf Qwen/Qwen3.5-9B --gguf unsloth/Qwen3.5-9B-GGUF
    eval/screen_checkpoints.py --preset qwen3.5-family --bandwidth-gbs 1792

The persist family's ceiling is arithmetic:

    ceiling = 2 x min(persisting-L2 capacity, recurrent footprint) / decode step traffic

The GPU fixes the numerator -- 60 MiB on an RTX 5090, and no software raises it. The MODEL
fixes the denominator, and that is the whole search. Qwen3.8-27B moves 18.5 GB per step and
tops out at 0.68% at batch 1; something moving 3 GB does not.

Both terms come out of a `config.json` and a file listing, so a candidate can be ruled in or out
for the price of a few kilobytes rather than a 20 GB download. What this CANNOT tell you is the
other gate -- whether two unhooked greedy replays of the checkpoint agree -- which needs the
runtime and one model load. `docs/VERDICT.md` explains why that gate exists: on the sparse-MoE
checkpoint that cleared the traffic gate, four replays of four were distinct, because a few ULP
in the prefill feed discrete top-k expert routing. A DENSE model has no such routing, which is
why this screen reports the architecture beside the ceiling.

Nothing here is a measurement. `basis` is "model" in the output for the same reason it is in
every other predicted figure this repository prints.
"""
from __future__ import annotations

import argparse
import json
import sys

PERSISTING_L2_BYTES = 62914560          # RTX 5090, from tools/info.cpp
BREAK_EVEN_NOTE = ("the decode step a persisting window over this footprint has to beat to "
                   "reach a given ceiling; see eval/traffic_budget.py --persisting-l2-bytes")

# The quantisations `ggml_dequant_supported` accepts (qwen35.cpp). Every MXFP4, IQ*, Q2_K and
# Q3_K variant is refused at load, so screening them wastes a download.
SUPPORTED_QUANTS = ("Q4_K_M", "Q4_K_XL", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32",
                    # Not a ggml type: the pinned checkpoint is an NVFP4 build with its own
                    # kernels, which is why it loads while every MXFP4 GGUF does not.
                    "NVFP4")
VRAM_BYTES = 32 * 1000 ** 3


def active_weight_bytes(config: dict, file_bytes: int) -> tuple:
    """What a decode step actually READS, which is not the file size on a sparse model.

    A dense model reads every weight every token. A sparse MoE reads the shared trunk plus the
    experts a token routes to -- `num_experts_per_tok` of `num_experts` -- and using the file
    size instead is the difference between a 22 GB step and a 3.6 GB one, which is the
    difference between a 0.6% ceiling and a 3.7% one. Getting this wrong in the permissive
    direction would send someone to download 22 GB for nothing; getting it wrong in the other
    would hide the only family that has ever cleared the traffic gate.
    """
    text = config.get("text_config", config)
    experts = text.get("num_experts") or text.get("moe_num_experts")
    per_token = text.get("num_experts_per_tok") or text.get("moe_topk")
    if not experts or not per_token:
        return file_bytes, "dense: every weight is read every token"
    # The trunk is whatever is not an expert. Without a per-tensor listing the split has to be
    # estimated, and the estimate is stated rather than hidden.
    shared = text.get("shared_expert_intermediate_size", 0)
    expert_share = 0.9          # of a sparse checkpoint's bytes, typical for this family
    trunk = file_bytes * (1.0 - expert_share)
    active = trunk + file_bytes * expert_share * (per_token / experts)
    # Checked against the one sparse checkpoint this project has MEASURED: Qwen3.6-35B-A3B at
    # 22.13 GB, where results/rtx5090-moe-matrix.json puts the step at 3.56 GB and this estimate
    # gives 2.84 GB. So it is optimistic by about 20% on that case -- the trunk share is a
    # guess, not a tensor listing -- and a candidate that clears the gate only narrowly should
    # be re-checked with `--active-weight-bytes` from a real measurement before anyone downloads
    # it. It is stated rather than tuned away, because tuning a constant on one point is how a
    # screen comes to agree with the case it was fitted to and nothing else.
    return int(active), (f"sparse MoE: {per_token} of {experts} experts routed per token, "
                         f"trunk estimated at {(1 - expert_share) * 100:.0f}% of the file"
                         + (" plus a shared expert" if shared else "")
                         + "; optimistic by ~20% on the one measured case")


def recurrent_geometry(config: dict) -> dict:
    """Per-layer recurrent state and the per-token footprint, from a Qwen3.5-family config.

    The matrix state is `v_heads x value_head_dim x key_head_dim` in fp32 and the convolution
    window is `(kernel - 1) x qkv_dim` in bf16 -- which reproduces the pinned model's measured
    3145728 and 61440 exactly, and that agreement is the only reason to trust it on a model
    nobody has loaded yet.
    """
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    interval = int(text.get("full_attention_interval") or 0)
    recurrent = layers - (layers // interval if interval else 0)
    v_heads = int(text["linear_num_value_heads"])
    k_heads = int(text["linear_num_key_heads"])
    v_dim = int(text["linear_value_head_dim"])
    k_dim = int(text["linear_key_head_dim"])
    kernel = int(text["linear_conv_kernel_dim"])

    state = v_heads * v_dim * k_dim * 4
    qkv = k_heads * k_dim * 2 + v_heads * v_dim
    conv = (kernel - 1) * qkv * 2
    return {
        "layers": layers, "full_attention_interval": interval,
        "recurrent_layers": recurrent,
        "lin_state_bytes_per_layer": state,
        "lin_conv_state_bytes_per_layer": conv,
        "footprint_bytes_per_token": recurrent * (state + conv),
        "architecture": (config.get("architectures") or ["?"])[0],
        "sparse_moe": "Moe" in (config.get("architectures") or [""])[0],
    }


def ceiling_for(footprint_bytes: int, step_traffic_bytes: int,
                persisting_l2_bytes: int = PERSISTING_L2_BYTES) -> dict:
    """What a PERFECT persisting policy could be worth, in the currency the scorer measures.

    A step carrying `f` less traffic runs in `(1-f)` of the time, so tokens per second rise by
    `f/(1-f)`. Reporting the traffic share instead understates it, which would let a submission
    legitimately beat a number called a ceiling.
    """
    resident = min(persisting_l2_bytes, footprint_bytes)
    share = (2.0 * resident) / step_traffic_bytes
    return {
        "resident_bytes": resident,
        "footprint_fits_in_partition": footprint_bytes <= persisting_l2_bytes,
        "traffic_share": share,
        "ceiling_pct": (share / (1.0 - share)) * 100.0 if share < 1.0 else float("inf"),
    }


def screen(name: str, config: dict, weight_bytes: int, quant: str,
           persisting_l2_bytes: int = PERSISTING_L2_BYTES) -> dict:
    geometry = recurrent_geometry(config)
    active, how = active_weight_bytes(config, weight_bytes)
    # State traffic rides on top of the weights: every recurrent layer reads its state and
    # writes it back once per token. Activations are not modelled and are small beside both.
    step = active + 2 * geometry["footprint_bytes_per_token"]
    out = {"model": name, "quantisation": quant, "file_bytes": weight_bytes,
           "active_weight_bytes_per_token": active, "active_weight_basis": how,
           "step_traffic_bytes": step, **geometry,
           **ceiling_for(geometry["footprint_bytes_per_token"], step, persisting_l2_bytes)}
    reasons = []
    if quant.upper() not in {q.upper() for q in SUPPORTED_QUANTS}:
        reasons.append(f"{quant} is not one of the quantisations the runtime dequantises "
                       f"({', '.join(SUPPORTED_QUANTS)}); it is refused at load")
    if weight_bytes > VRAM_BYTES:
        reasons.append(f"{weight_bytes / 1e9:.1f} GB exceeds the device's {VRAM_BYTES / 1e9:.0f} GB")
    out["loadable"] = not reasons
    out["refused_because"] = reasons
    # A sparse-MoE checkpoint clears the traffic gate easily and has never cleared the OTHER
    # one: discrete top-k expert routing turns a few ULP of prefill difference into a different
    # token, so two UNHOOKED replays diverge and nothing on it can be scored.
    out["reproducibility_risk"] = ("high -- sparse-MoE routing; four checkpoints screened and "
                                   "none reproducible" if geometry["sparse_moe"] else
                                   "low -- dense, no expert routing; the pinned dense model is "
                                   "bit-identical across replays")
    out["basis"] = "model"
    out["note"] = ("predicted_* and ceiling_pct are cost-model outputs, not measurements. The "
                   "second gate -- two unhooked greedy replays agreeing -- needs hardware.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=argparse.FileType(), action="append", default=[],
                    metavar="CONFIG_JSON",
                    help="a local Qwen3.5-family config.json; repeatable")
    ap.add_argument("--weight-bytes", type=int, action="append", default=[],
                    help="weight bytes for the matching --config, in order")
    ap.add_argument("--quant", action="append", default=[],
                    help="quantisation label for the matching --config, in order")
    ap.add_argument("--name", action="append", default=[], help="label, in order")
    ap.add_argument("--active-weight-bytes", type=int, action="append", default=[],
                    help="measured bytes of weights a decode step reads, overriding the "
                         "estimate; in order with --config. Use it for a sparse checkpoint "
                         "that clears the gate narrowly.")
    ap.add_argument("--persisting-l2-bytes", type=int, default=PERSISTING_L2_BYTES)
    ap.add_argument("--json", action="store_true", help="machine-readable on stdout")
    args = ap.parse_args()

    if not args.config:
        ap.error("give at least one --config")
    rows = []
    for i, handle in enumerate(args.config):
        config = json.load(handle)
        weight = args.weight_bytes[i] if i < len(args.weight_bytes) else 0
        quant = args.quant[i] if i < len(args.quant) else "?"
        name = args.name[i] if i < len(args.name) else handle.name
        if not weight:
            ap.error(f"--weight-bytes is required for {name}")
        row = screen(name, config, weight, quant, args.persisting_l2_bytes)
        if i < len(args.active_weight_bytes) and args.active_weight_bytes[i]:
            measured = args.active_weight_bytes[i]
            row["active_weight_bytes_per_token"] = measured
            row["active_weight_basis"] = "measured, supplied by --active-weight-bytes"
            row["step_traffic_bytes"] = measured + 2 * row["footprint_bytes_per_token"]
            row.update(ceiling_for(row["footprint_bytes_per_token"],
                                   row["step_traffic_bytes"], args.persisting_l2_bytes))
        rows.append(row)

    if args.json:
        print(json.dumps({"basis": "model", "candidates": rows}, indent=1))
        return 0
    print(f"{'model':28s} {'quant':8s} {'step':>8s} {'footprint':>10s} {'fits':>5s} "
          f"{'ceiling':>8s}  loadable / reproducibility")
    for r in sorted(rows, key=lambda r: -r["ceiling_pct"]):
        print(f"{r['model'][:28]:28s} {r['quantisation']:8s} "
              f"{r['step_traffic_bytes'] / 1e9:7.2f}G {r['footprint_bytes_per_token'] / 1e6:9.1f}M "
              f"{'yes' if r['footprint_fits_in_partition'] else 'no':>5s} "
              f"{r['ceiling_pct']:7.3f}%  "
              f"{'yes' if r['loadable'] else 'NO: ' + r['refused_because'][0][:40]} / "
              f"{r['reproducibility_risk'].split(' -- ')[0]}")
    print()
    print("  ceiling = 2 x min(persisting-L2, footprint) / step traffic, in throughput terms.")
    print("  It is a MODEL output. The other gate -- two unhooked greedy replays agreeing --")
    print("  needs the runtime and one model load, and is what ruled out every sparse-MoE")
    print("  checkpoint this project has screened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
