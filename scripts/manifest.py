#!/usr/bin/env python3
"""Regenerate `repo-manifest.json` from the repository instead of from memory.

    scripts/manifest.py                     rewrite the file list and the version
    scripts/manifest.py --check             fail if either is stale (this is what CI runs)
    scripts/manifest.py --with-checks BUILD also re-run the suites and record their counts

Why this exists: the manifest carried a hand-maintained file list and a hand-typed
"ctest 3/3, 102 Python checks, 319 planner checks" summary. Both drifted -- 37 files were
missing by 0.2.1, including every file of the frontier scorer -- and a manifest that is wrong
about what is in the repository is worse than no manifest, because it is quoted.

The file list and the version are mechanical and are checked. The check COUNTS need built
binaries, and a CPU-only CI runner cannot produce the CUDA ones, so they are regenerated on
demand and record which build they came from rather than being asserted every run.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "repo-manifest.json"


def tracked_files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return sorted(line for line in out.splitlines() if line)


def version():
    text = (ROOT / "include/tensortransit/version.h").read_text()
    parts = [re.search(rf"#define TENSORTRANSIT_VERSION_{k}\s+(\d+)", text)
             for k in ("MAJOR", "MINOR", "PATCH")]
    if not all(parts):
        raise SystemExit("!! could not read the version out of include/tensortransit/version.h")
    return ".".join(p.group(1) for p in parts)


def suite_counts(build: Path):
    """Run every built test binary and every Python suite, and read their own totals.

    Each C++ suite prints `<name> tests passed (N checks)`; each Python suite is unittest or
    prints its own line. Nothing here counts anything itself -- a counter maintained in two
    places is a counter that disagrees with itself.
    """
    counts = {}
    for binary in sorted(build.glob("test_*")):
        if not binary.is_file() or not binary.stat().st_mode & 0o111:
            continue
        done = subprocess.run([str(binary)], capture_output=True, text=True, cwd=ROOT)
        found = re.search(r"passed \((\d+) checks\)", done.stdout + done.stderr)
        counts[binary.name] = {"checks": int(found.group(1)) if found else None,
                               "ok": done.returncode == 0}
    for suite in sorted((ROOT / "eval").glob("test_*.py")):
        done = subprocess.run([sys.executable, str(suite)], capture_output=True, text=True,
                              cwd=ROOT)
        blob = done.stdout + done.stderr
        found = re.search(r"^Ran (\d+) tests?", blob, re.M)
        counts[suite.name] = {"checks": int(found.group(1)) if found else None,
                              "ok": done.returncode == 0}
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero; changes nothing")
    ap.add_argument("--with-checks", metavar="BUILD",
                    help="also re-run the suites in BUILD and record their check counts")
    args = ap.parse_args()

    doc = json.loads(MANIFEST.read_text())
    wanted_files, wanted_version = tracked_files(), version()

    if args.check:
        problems = []
        listed = set(doc.get("files", []))
        for path in sorted(set(wanted_files) - listed):
            problems.append(f"tracked but not listed: {path}")
        for path in sorted(listed - set(wanted_files)):
            problems.append(f"listed but not tracked: {path}")
        if doc.get("version") != wanted_version:
            problems.append(f"version says {doc.get('version')}, headers say {wanted_version}")
        if problems:
            print(f"!! repo-manifest.json is stale ({len(problems)} problem(s)):")
            for problem in problems[:60]:
                print(f"     {problem}")
            print("   run scripts/manifest.py")
            return 1
        print(f"repo-manifest.json is current: {len(wanted_files)} files, version {wanted_version}")
        return 0

    doc["version"] = wanted_version
    doc["files"] = wanted_files
    if args.with_checks:
        build = Path(args.with_checks)
        if not build.is_absolute():
            build = ROOT / build
        doc["checks"] = {"build": str(build.relative_to(ROOT)), "suites": suite_counts(build)}
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(wanted_files)} files, version {wanted_version}"
          + (f", {len(doc['checks']['suites'])} suites" if args.with_checks else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
