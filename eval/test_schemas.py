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
import subprocess
import sys

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
    for candidate in ("build/tensortransit", "build-cpu/tensortransit",
                      "build/bin/tensortransit"):
        path = os.path.join(ROOT, candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


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

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"{len(failures)} schema check(s) failed")
        return 1
    print(f"schema tests passed ({checked} document(s) validated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
