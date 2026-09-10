#!/usr/bin/env python3
"""End-to-end test of the `tt-frontier` COMMAND LINE, with no GPU.

The golden tests in `test_frontier.py` cover the scoring library. They do not cover the wiring
around it -- argument parsing, the generation lookup, where the ledger is written, whether a
report reaches a file, whether a non-zero exit means what it should. That wiring is what a
trusted evaluator and a contributor both actually run, and a break in it would stay invisible
until somebody spent an hour of GPU time to find out.

So: build a synthetic raw-result file from the FROZEN generation, then drive the real CLI all
the way to a verified receipt in a temporary ledger.

Run: python3 eval/test_cli_pipeline.py
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = [sys.executable, str(ROOT / "tools" / "tt-frontier")]
GENERATION = "TTF-1"

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
        print(f"  FAIL {message}")
    return condition


def run(args, env=None, expect=0):
    result = subprocess.run(CLI + args, capture_output=True, text=True,
                            env={**os.environ, **(env or {})}, cwd=ROOT)
    if expect is not None and result.returncode != expect:
        failures.append(f"`tt-frontier {' '.join(args)}` exited {result.returncode}, "
                        f"expected {expect}\n{result.stderr[-2000:]}")
    return result


def synthetic_raw(path, *, throughput_ratio, latency_ratio, repeats=3, seed=7):
    """A raw result file shaped exactly as the runner writes one, from the frozen bounds.

    Built from `reference.json`'s calibrated controls, so the numbers are in the units and the
    range the generation actually declares -- a synthetic file with plausible-looking but
    out-of-range values would exercise the clipping path instead of the scoring path.
    """
    sys.path.insert(0, str(ROOT / "eval"))
    from frontier import load_generation
    generation = load_generation(ROOT / "eval" / "generations" / GENERATION)
    rng = random.Random(seed)
    rows = []
    for repeat in range(1, repeats + 1):
        for cell in generation.cells:
            bounds = generation.cell_bounds.get(cell, {})
            base_g = bounds.get("goodput_tps", {}).get("measured_control", 100.0)
            base_l = bounds.get("p99_itl_ms", {}).get("measured_control", 20.0)
            for variant, gm, lm in (("main", 1.0, 1.0),
                                    ("candidate", throughput_ratio, latency_ratio)):
                rows.append({
                    "result_schema_version": 1,
                    "generation": GENERATION,
                    "workload_id": cell,
                    "variant": variant,
                    "config_id": "control" if variant == "main" else "persist",
                    "repeat": repeat,
                    "metrics": {
                        "goodput_tps": base_g * gm * (1 + rng.uniform(-3e-4, 3e-4)),
                        "p99_itl_ms": base_l * lm * (1 + rng.uniform(-2e-3, 2e-3)),
                    },
                    "status": "OK",
                    "correctness": "PASS",
                })
    Path(path).write_text(json.dumps({
        "result_schema_version": 1,
        "provenance": {
            "baseline_commit": "a" * 40, "candidate_commit": "b" * 40,
            "mode": "public", "instances": "public_only",
            "hardware": {"gpu": "NVIDIA GeForce RTX 5090", "count": 1},
            "runtime": {"name": "SparkInfer", "commit": "5347b27"},
            "model": {"name": "Qwen3.8-27B"},
            "correctness": {"correctness": "PASS", "compared_against": "baseline build"},
        },
        "results": rows,
    }, indent=1))
    return generation


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ledger = tmp / "ledger"

        print("== generation")
        out = run(["generation", "show", GENERATION]).stdout
        check("checksum      sha256:" in out, "`generation show` prints a checksum")
        check("cells         10:" in out, "...and the ten cells TTF-1 declares")
        run(["generation", "list"])

        print("\n== a clear win")
        raw = tmp / "win.json"
        synthetic_raw(raw, throughput_ratio=1.02, latency_ratio=0.98)
        receipt = tmp / "win-receipt.json"
        result = run(["compute", "--generation", GENERATION, "--results", str(raw),
                      "--output", str(receipt), "--pr", "7"])
        check(receipt.exists(), "`compute` writes a receipt")
        check("FRONTIER_GAIN" in result.stderr, "and prints the status box to stderr")
        doc = json.loads(receipt.read_text())
        check(doc["status"] == "FRONTIER_GAIN", "a 2% throughput win with better latency scores")
        check(doc["frontier"]["verified_gain_percent"] > 1.5,
              f"and credits what it measured (got {doc['frontier']['verified_gain_percent']})")
        check(doc["coverage"]["partial"] is False, "the full matrix is not partial")
        check(doc["correctness_compared_against"] == "baseline build",
              "and the receipt says which build produced the control replays")

        print("\n== every rendering")
        for fmt, marker in (("markdown", "# TensorTransit Frontier Receipt"),
                            ("comment", "## TensorTransit Frontier Evaluation"),
                            ("box", "TENSORTRANSIT FRONTIER"),
                            ("map", "|")):
            target = tmp / f"report.{fmt}"
            run(["report", str(receipt), "--format", fmt, "--output", str(target)])
            check(target.exists() and marker in target.read_text(),
                  f"`report --format {fmt}` renders")

        print("\n== verification")
        run(["receipt", "verify", str(receipt), "--generation", GENERATION])
        tampered = tmp / "tampered.json"
        edited = json.loads(receipt.read_text())
        edited["frontier"]["gain_percent"] = 99.0
        tampered.write_text(json.dumps(edited))
        run(["receipt", "verify", str(tampered)], expect=1)
        check(True, "an edited receipt is refused with a non-zero exit")

        print("\n== the ledger")
        run(["ledger", "append", "--receipt", str(receipt)],
            env={"TT_LEDGER_DIR": str(ledger)})
        out = run(["ledger", "show", GENERATION], env={"TT_LEDGER_DIR": str(ledger)}).stdout
        check("pr-000007" in out and "FRONTIER_GAIN" in out, "`ledger show` lists the receipt")
        audit = run(["ledger", "audit", GENERATION], env={"TT_LEDGER_DIR": str(ledger)}).stdout
        check(json.loads(audit)["ok"], "`ledger audit` passes")
        # And a rewrite is refused rather than accepted.
        rewritten = json.loads(receipt.read_text())
        rewritten["timestamp_utc"] = "2031-01-01T00:00:00+00:00"
        sys.path.insert(0, str(ROOT / "eval"))
        from frontier.receipt import content_digest
        rewritten["content_digest"] = content_digest(rewritten)
        (tmp / "rewrite.json").write_text(json.dumps(rewritten))
        result = run(["ledger", "append", "--receipt", str(tmp / "rewrite.json"),
                      "--id", "pr-000007"],
                     env={"TT_LEDGER_DIR": str(ledger)}, expect=2)
        check("supersede" in result.stderr,
              "rewriting a finalized receipt is refused and points at supersession")

        print("\n== a flat candidate, and a partial matrix")
        flat = tmp / "flat.json"
        synthetic_raw(flat, throughput_ratio=1.0, latency_ratio=1.0, seed=11)
        doc = json.loads(run(["compute", "--generation", GENERATION, "--results", str(flat)],
                             expect=None).stdout)
        check(doc["status"] in ("NO_FRONTIER_GAIN", "INCONCLUSIVE"),
              f"a flat candidate does not score (got {doc['status']})")
        check(doc["frontier"]["verified_gain_percent"] == 0.0, "and credits nothing")

        partial = tmp / "partial.json"
        whole = json.loads(raw.read_text())
        whole["results"] = [r for r in whole["results"] if r["workload_id"] != "ctx128-c32"]
        partial.write_text(json.dumps(whole))
        refused = run(["compute", "--generation", GENERATION, "--results", str(partial)],
                      expect=2)
        check("not run" in refused.stderr,
              "a partial matrix is refused by default and names the missing cell")
        doc = json.loads(run(["compute", "--generation", GENERATION, "--results", str(partial),
                              "--allow-partial"], expect=None).stdout)
        check(doc["coverage"]["partial"] and doc["coverage"]["missing_cells"] == ["ctx128-c32"],
              "--allow-partial scores it and marks it PARTIAL on its face")

    print()
    if failures:
        print(f"{len(failures)} CLI check(s) failed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("tt-frontier CLI pipeline passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
