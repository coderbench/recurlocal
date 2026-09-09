#!/usr/bin/env python3
"""Synthetic feasibility evaluator for the four RecurLocal locality modes.

Runs each mode several times, checks that every run produced an identical final
state, and compares median timings. The go/no-go bands in the README start at 2%,
so a single timed run per mode cannot resolve them; repeats also expose the
run-to-run cache-policy instability listed as a project failure criterion.
"""
import argparse, datetime, json, re, statistics, subprocess
from pathlib import Path

SCHEMA_VERSION = 3
MODES = ["baseline", "persist", "prefetch", "combined"]

def nvidia_smi(*args):
    """Best-effort nvidia-smi; None whenever the tool is missing or unhappy."""
    try:
        p = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def supported_graphics_clocks():
    out = nvidia_smi("-q", "-d", "SUPPORTED_CLOCKS")
    return sorted({int(m) for m in re.findall(r"Graphics\s*:\s*(\d+)\s*MHz", out)}) if out else []


def pin_graphics_clock(target):
    """Lock the graphics clock so an absolute time is reproducible off this box.

    The same-box mode deltas do not need this, but a third party rebuilding the result
    does: boost wanders with temperature and power. Best-effort — it needs privileges.
    """
    if target == "auto":
        clocks = supported_graphics_clocks()
        if not clocks:
            return None
        target = clocks[-1]
    target = int(target)
    return target if nvidia_smi("-lgc", f"{target},{target}") is not None else None


def observed_graphics_clock():
    out = nvidia_smi("--query-gpu=clocks.gr", "--format=csv,noheader,nounits")
    if not out:
        return None
    try:
        return int(out.splitlines()[0].strip())
    except ValueError:
        return None


def repo_commit():
    here = Path(__file__).resolve().parent
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here,
                             capture_output=True, text=True, timeout=30)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here,
                               capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None, None
    if rev.returncode:
        return None, None
    return rev.stdout.strip(), bool(dirty.stdout.strip()) if not dirty.returncode else None


def config_args(cfg):
    """Translate a configs/*.json geometry into benchmark arguments."""
    state = cfg.get("state") or {}
    bench = cfg.get("benchmark") or {}
    args = []
    if "recurrent_layers" in cfg:
        args += ["--layers", str(int(cfg["recurrent_layers"]))]
    if "matrix_state_bytes_per_recurrent_layer" in state:
        args += ["--state-bytes", str(int(state["matrix_state_bytes_per_recurrent_layer"]))]
    for key, flag in (("tokens", "--tokens"), ("warmup_tokens", "--warmup-tokens"),
                      ("inner_iters", "--inner-iters")):
        if key in bench:
            args += [flag, str(int(bench[key]))]
    if not args:
        raise SystemExit("config contains no recognised benchmark geometry")
    return args


def run_one(binary, mode, extra):
    p = subprocess.run([str(binary), mode, *extra], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"{mode} failed:\n{p.stdout}\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])

def close(a, b, rel=1e-8, abs_=1e-6):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b), 1.0))

def timing_stats(samples):
    median = statistics.median_low(samples)
    return {"samples": samples,
            "min": min(samples),
            "median": median,
            "max": max(samples),
            "mean": statistics.fmean(samples),
            "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "rel_spread_pct": (max(samples) - min(samples)) / median * 100.0 if median > 0 else 0.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--output", default="eval-result.json", type=Path)
    ap.add_argument("--repeats", default=3, type=int,
                    help="timed runs per mode; the median run is reported (default: 3)")
    ap.add_argument("--stability-threshold-pct", default=2.0, type=float,
                    help="flag the run as unstable above this per-mode spread (default: 2.0)")
    ap.add_argument("--config", type=Path, metavar="CONFIG_JSON",
                    help="benchmark geometry from a configs/*.json file; explicit benchmark "
                         "arguments after -- still win")
    ap.add_argument("--pin-clock-mhz", metavar="MHZ",
                    help="lock the graphics clock before timing, or 'auto' for the highest "
                         "supported; needs privileges, and is recorded either way")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra
    if a.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    # A config file that nothing reads is decoration. Geometry from --config is prepended,
    # so anything passed explicitly still overrides it.
    if a.config:
        extra = config_args(json.loads(a.config.read_text())) + list(extra)

    pinned_clock = pin_graphics_clock(a.pin_clock_mhz) if a.pin_clock_mhz else None
    if a.pin_clock_mhz and pinned_clock is None:
        print("warning: could not lock the graphics clock; continuing with the observed clock")

    # Interleave repeats so thermal drift and clock behaviour land on every mode
    # rather than penalising whichever mode happens to run last.
    runs = {m: [] for m in MODES}
    clocks = []
    try:
        for _ in range(a.repeats):
            for m in MODES:
                runs[m].append(run_one(a.binary, m, extra))
                sample = observed_graphics_clock()
                if sample is not None:
                    clocks.append(sample)
    finally:
        if pinned_clock is not None:
            nvidia_smi("-rgc")

    reference = runs["baseline"][0]["checksum"]
    for m in MODES:
        for i, r in enumerate(runs[m]):
            if not close(reference, r["checksum"]):
                raise SystemExit(
                    f"correctness failure: {m} run {i} checksum {r['checksum']} != {reference}")

    for m in MODES:
        for r in runs[m]:
            if r["elapsed_ms"] <= 0.0:
                raise SystemExit(f"invalid timing: {m} reported elapsed_ms={r['elapsed_ms']}")

    timing = {m: timing_stats([r["elapsed_ms"] for r in runs[m]]) for m in MODES}
    # Report the median run itself, so `results` stays a real observation.
    results = {m: next(r for r in runs[m] if r["elapsed_ms"] == timing[m]["median"]) for m in MODES}

    base_ms = timing["baseline"]["median"]
    speedups = {m: base_ms / timing[m]["median"] for m in MODES}
    best = max((m for m in MODES if m != "baseline"), key=lambda m: speedups[m])

    worst_spread = max(timing[m]["rel_spread_pct"] for m in MODES)
    stable = worst_spread <= a.stability_threshold_pct
    interpretation = "synthetic feasibility only; do not publish as a model-serving speedup"
    if not stable:
        interpretation += (f"; timing spread {worst_spread:.2f}% exceeds the "
                           f"{a.stability_threshold_pct:.2f}% threshold, so mode differences "
                           "of that order are not resolved by this run")

    commit, dirty = repo_commit()
    probe = results["baseline"]
    environment = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "recurlocal_commit": commit,
        "recurlocal_dirty": dirty,
        "benchmark_args": list(extra),
        "gpu": probe.get("gpu"),
        "compute_capability": probe.get("compute_capability"),
        "sm_count": probe.get("sm_count"),
        "driver_version": probe.get("driver_version"),
        "runtime_version": probe.get("runtime_version"),
        "clocks_pinned": pinned_clock is not None,
        "pinned_clock_mhz": pinned_clock,
        "observed_clock_mhz": ({"min": min(clocks), "median": statistics.median_low(clocks),
                                "max": max(clocks)} if clocks else None),
    }

    out = {"schema_version": SCHEMA_VERSION, "correctness": "pass", "repeats": a.repeats,
           "environment": environment, "results": results, "timing_ms": timing,
           "speedup_vs_baseline": speedups, "best_mode": best,
           "best_synthetic_gain_pct": (speedups[best] - 1.0) * 100.0,
           "stability": {"threshold_pct": a.stability_threshold_pct,
                         "max_rel_spread_pct": worst_spread,
                         "verdict": "stable" if stable else "unstable"},
           "interpretation": interpretation}
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
