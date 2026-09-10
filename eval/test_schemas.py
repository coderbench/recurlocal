#!/usr/bin/env python3
"""Every committed artifact validates against its own schema.

A schema nothing is checked against is documentation, and this repository has already
shipped a document whose shape drifted from the tool that wrote it. These run in the same
`ctest` invocation as the planner tests, without a GPU, so the failure arrives in CI rather
than in somebody's benchmark.

The plan schema is checked against a plan produced by the CLI when one has been built;
otherwise that part is skipped rather than silently passing, and says so.
"""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

try:
    import jsonschema
except ImportError:
    print("jsonschema is not installed; skipping the schema tests")
    sys.exit(0)


def load(path):
    with open(path) as handle:
        return json.load(handle)


def check(name, document, schema, failures):
    try:
        jsonschema.validate(document, schema)
        return True
    except jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path)
        failures.append(f"{name}: {exc.message}" + (f" (at {path})" if path else ""))
        return False


def find_cli():
    """The most recently BUILT CLI, and say which one it is.

    This used to return the first path that existed, in a fixed order beginning with
    `build/`. A developer with a stale `build/` from an earlier version therefore validated
    THAT binary's plan output against the CURRENT schema, and the check passed while the
    binary they were actually changing emitted fields the schema rejected. A schema test that
    can pass against a different build is not a schema test.

    Printing the path is half the fix: a check whose subject is ambiguous should say what it
    chose.
    """
    found = []
    for candidate in ("build/tensortransit", "build-cpu/tensortransit",
                      "build/bin/tensortransit", "build-cuda/tensortransit"):
        path = os.path.join(ROOT, candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            found.append(path)
    if not found:
        return None
    newest = max(found, key=os.path.getmtime)
    if len(found) > 1:
        others = [os.path.relpath(p, ROOT) for p in found if p != newest]
        print(f"note: several CLI builds present; using the newest "
              f"({os.path.relpath(newest, ROOT)}), ignoring {', '.join(others)}")
    return newest


def main():
    failures = []
    checked = 0

    trace_schema = load(os.path.join(ROOT, "schemas/trace.schema.json"))
    traces = sorted(glob.glob(os.path.join(ROOT, "tests/golden/*.json")))
    if not traces:
        failures.append("no golden traces found: tests/golden/*.json")
    for path in traces:
        if check(os.path.relpath(path, ROOT), load(path), trace_schema, failures):
            checked += 1

    # Structural invariants a JSON schema cannot state, and both have bitten this project in
    # the equivalent place: an out-of-order stream produces distances that underflow, and a
    # use naming an undeclared tensor silently shrinks the graph.
    for path in traces:
        document = load(path)
        name = os.path.relpath(path, ROOT)
        declared = {t["id"] for t in document["tensors"]}
        last_order = -1
        for event in document["events"]:
            order = event.get("order", 0)
            if order < last_order:
                failures.append(f"{name}: events are not in non-decreasing order at "
                                f"kernel {event['kernel_id']}")
                break
            last_order = order
            for use in event.get("tensors", []):
                if use["id"] not in declared:
                    failures.append(f"{name}: kernel {event['kernel_id']} uses tensor "
                                    f"{use['id']}, which the tensor table does not declare")
                    break

    plan_schema = load(os.path.join(ROOT, "schemas/plan.schema.json"))
    cli = find_cli()
    if cli:
        version = subprocess.run([cli, "version"], capture_output=True, text=True).stdout.strip()
        print(f"plan schema checked against {os.path.relpath(cli, ROOT)} ({version})")
        for preset_args in (["--planner", "budgeted", "--admission", "density"],
                            ["--planner", "budgeted", "--admission", "role_floor"],
                            ["--planner", "recurrent_v0"],
                            ["--planner", "baseline"]):
            trace = os.path.join(ROOT, "tests/golden/trace_recurrent_kv.json")
            result = subprocess.run([cli, "plan", trace, "--json"] + preset_args,
                                    capture_output=True, text=True)
            if result.returncode != 0:
                failures.append(f"cli plan {' '.join(preset_args)}: exit {result.returncode}"
                                f" {result.stderr.strip()}")
                continue
            try:
                document = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                failures.append(f"cli plan {' '.join(preset_args)}: bad JSON: {exc}")
                continue
            if check(f"plan({' '.join(preset_args)})", document, plan_schema, failures):
                checked += 1
            # The provenance marker is not decoration. Anything downstream that prints a
            # predicted figure next to a measured one has to be able to tell them apart, and
            # this is the field that makes that possible.
            if document.get("cost_model", {}).get("basis") != "model":
                failures.append("a plan does not declare cost_model.basis == \"model\"")
    else:
        print("note: no tensortransit CLI built; plan-schema checks skipped")

    # The synthetic bench's OWN output shape, against the schema that pins it.
    #
    # `run_eval.py` validates its result against `result_schema.json`, and that schema closes
    # `additionalProperties`. So a field added to the benchmark's JSON -- which is what happens
    # every time somebody adds an axis -- breaks the evaluator on the first GPU run, with a
    # validation error rather than a helpful one. Checking a representative document here means
    # the break is a CPU-only test failure instead.
    bench_schema = load(os.path.join(ROOT, "eval/result_schema.json"))
    run_schema = bench_schema.get("$defs", {}).get("benchRun")
    if run_schema:
        required = set(run_schema.get("required", []))
        sample = {
            "mode": "persist", "gpu": "NVIDIA GeForce RTX 5090", "compute_capability": "12.0",
            "sm_count": 170, "driver_version": 13030, "runtime_version": 13030,
            "layers": 48, "state_bytes_per_layer": 3145728, "total_state_bytes": 150994944,
            "tokens": 32, "warmup_tokens": 4, "elapsed_ms": 7.5, "ms_per_token": 0.23,
            "layer_updates_per_s": 200000.0, "l2_bytes": 100663296,
            "persisting_l2_max_bytes": 62914560, "access_policy_max_window_bytes": 134217728,
            "actual_l2_set_aside_bytes": 50331648, "checksum": 1991620.175986,
            "windows_applied": 1728, "windows_deferred_to_caller": 0,
            "hot_set_oversubscribed": 0, "pre_touch_launches": 0, "pre_touch_bytes": 0,
            "prefetch_distance": 1, "pre_touch_strategy": "vec4", "sequences": 1,
            "stream_bytes": 0, "hot_set_policy": "proportional", "set_aside_policy": "fixed",
            "hot_set_model": "current_layer", "prefetch_schedule": "uniform",
            "prefetch_impl": "stream", "state_layout": "linear", "qos": False,
            "stream_mode": "reuse",
            # everything 0.2.1 added
            "engine": "transit", "planner": "recurrent_v0", "admission": "density",
            "reuse_metric": "bytes", "window_binding": "per_consumer",
            "window_preference": "widest", "max_windows_per_kernel": 1,
            "budget_fraction": 0.75, "plan_digest": "c806496b6cc08583", "plan_actions": 96,
            "plan_declines": 48, "predicted_saved_bytes": 0, "committed_bytes": 603979776,
            "step_traffic_bytes": 0, "persist_applied": 1728, "prefetch_applied": 0,
            "stream_applied": 0, "clear_applied": 1728, "executor_host_ns": 0,
            "planner_compile_ns": 0, "plan_reuse_rate": 1.0,
        }
        missing = sorted(required - set(sample))
        if missing:
            failures.append(f"the bench-output sample is missing required field(s) {missing}; "
                            f"either the schema gained a requirement the sample does not "
                            f"model, or the bench stopped emitting one")
        elif check("bench output sample", sample, run_schema, failures):
            checked += 1

    # The evaluation schemas that predate this file, against the results committed under
    # them: they are the reproduction path for every published number.
    #
    # Only documents that DECLARE a schema_version are validated. Several files in results/
    # are hand-written roll-ups rather than evaluator output, and holding those to the
    # evaluator's schema would either fail forever or force the schema to be loosened until
    # it stopped catching anything. They are listed instead, because a narrative summary
    # sitting next to an authoritative artifact and looking like one is its own hazard.
    unversioned = []
    for schema_name, pattern in (("eval/real_result_schema.json", "results/*real*.json"),
                                 ("eval/result_schema.json", "results/rtx5090-synthetic.json")):
        schema_path = os.path.join(ROOT, schema_name)
        if not os.path.isfile(schema_path):
            continue
        schema = load(schema_path)
        for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
            document = load(path)
            name = os.path.relpath(path, ROOT)
            if "schema_version" not in document:
                unversioned.append(name)
                continue
            try:
                jsonschema.validate(document, schema)
                checked += 1
            except jsonschema.ValidationError as exc:
                failures.append(f"{name} vs {schema_name}: {exc.message}")
    if unversioned:
        print("note: not evaluator artifacts (no schema_version), not validated: "
              + ", ".join(unversioned))

    quoted, quote_failures = check_documented_numbers()
    checked += quoted
    failures += quote_failures

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"{len(failures)} schema check(s) failed")
        return 1
    print(f"Ran {checked} tests")
    print(f"schema tests passed ({checked} document(s) validated)")
    return 0


def check_documented_numbers():
    """Every benchmark figure in a document is checked against the file it cites.

    "Do not let anyone type benchmark numbers by hand" is the rule this repository turns on,
    and it was enforced for reports -- which are generated -- and not for prose, which is not.
    A number that drifts in a document is worse than one in a report, because prose is what a
    contributor reads before deciding whether to spend a week.

    Deliberately narrow: it checks the tables that carry per-cell measurements against the
    result file those tables name. Adding a table here is cheap; a table nothing checks is how
    a document comes to disagree with its own evidence.
    """
    root = Path(__file__).resolve().parent.parent
    results = root / "results" / "rtx5090-ttf1-first-matrix.json"
    verdict = root / "docs" / "VERDICT.md"
    if not results.exists() or not verdict.exists():
        return 0, []
    doc = json.loads(results.read_text())
    text = verdict.read_text()
    checked, failures = 0, []

    for cell, quoted in re.findall(r"\| `(ctx\d+-c\d+)` \| ([+-\u2212\u00b1]?[\d.]+)% \|", text):
        want = float(quoted.replace("\u2212", "-").replace("\u00b1", ""))
        got = (doc["cells"].get(cell) or {}).get("goodput_delta_pct")
        checked += 1
        if got is None:
            failures.append(f"docs/VERDICT.md quotes {cell} at {want:+.3f}% and "
                            f"{results.name} has no goodput delta for it")
        elif abs(got - want) > 0.002:
            failures.append(f"docs/VERDICT.md quotes {cell} at {want:+.3f}% and "
                            f"{results.name} says {got:+.3f}%")

    # Section 7's reachability table is a THIRD kind of documented number: not measured, not
    # copied from a result file, but computed by the tool from the frozen calibration. Checked
    # against the tool rather than against a stored copy, so the prose, the CLI and the
    # calibration cannot disagree in any pair.
    tool = root / "tools" / "tt-frontier"
    if tool.exists():
        done = subprocess.run([sys.executable, str(tool), "generation", "show", "TTF-1",
                               "--reachable"], capture_output=True, text=True, cwd=root)
        if done.returncode == 0:
            printed = {}
            for line in done.stdout.splitlines():
                found = re.match(r"\s+(ctx\d+-c\d+)\s+([\d.]+)%\s+([\d.]+)%", line)
                if found:
                    printed[found.group(1)] = (float(found.group(2)), float(found.group(3)))
            for cell, quoted_persist, quoted_any in re.findall(
                    r"^\s+(ctx\d+-c\d+)\s+([\d.]+)%\s+([\d.]+)%", text, re.M):
                checked += 1
                got = printed.get(cell)
                if not got:
                    failures.append(f"docs/VERDICT.md quotes a reachability row for {cell} that "
                                    f"`tt-frontier generation show --reachable` does not print")
                elif (abs(got[0] - float(quoted_persist)) > 0.001
                      or abs(got[1] - float(quoted_any)) > 0.001):
                    failures.append(
                        f"docs/VERDICT.md quotes {cell} at persist {quoted_persist}% / any "
                        f"{quoted_any}% and the tool computes {got[0]:.3f}% / {got[1]:.3f}%")

    scaling = (doc.get("main_concurrency_scaling") or {}).get("cells") or {}
    for cell, record in scaling.items():
        found = re.search(rf"`{re.escape(cell)}` \| [\d.]+ \| \*?\*?([\d.]+)x", text)
        if not found:
            continue
        checked += 1
        want = float(found.group(1))
        if abs(want - record["scale"]) > 0.02:
            failures.append(f"docs/VERDICT.md quotes {cell} scaling at {want:.2f}x and "
                            f"{results.name} says {record['scale']:.2f}x")
    return checked, failures


if __name__ == "__main__":
    sys.exit(main())
