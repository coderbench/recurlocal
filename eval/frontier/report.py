"""Every human-readable form of a Frontier Receipt, generated -- never typed.

This file exists because of one rule: **do not let anyone type benchmark numbers by hand.**
A number that was retyped into a PR description is a number with no provenance, and this
repository's whole failure history is confident figures whose provenance was not on the page.
So the PR table, the release line and the Markdown report all come from the receipt, and the
receipt comes from the raw results.

Three renderings, for three readers:

    receipt_box()   the terminal summary a contributor sees at the end of a run
    pr_comment()    what the evaluator posts on the PR
    markdown()      the full report: contribution map, per-cell table, statistics, provenance
"""

from __future__ import annotations


def _pct(value, places=4):
    return f"{value:+.{places}f}%"


def receipt_box(receipt: dict) -> str:
    """The framed summary. No S, M or L anywhere in it, deliberately."""
    frontier = receipt["frontier"]
    stats = receipt["statistics"]
    coverage = receipt["coverage"]
    rows = [
        ("Benchmark", receipt["benchmark_generation"]),
        ("Correctness", receipt["correctness"]),
        ("Frontier Gain", _pct(frontier["gain_percent"])),
        ("Improved Cells", f"{coverage['improved_cells']} / {coverage['total_cells']}"),
        ("Regressed Cells", f"{coverage['regressed_cells']} / {coverage['total_cells']}"),
        (f"{stats['confidence_level'] * 100:.0f}% CI",
         f"{stats['lower_gain_percent']:+.2f} ... {stats['upper_gain_percent']:+.2f}%"),
        ("Status", receipt["status"]),
    ]
    if coverage.get("partial"):
        rows.insert(3, ("Coverage", f"PARTIAL -- {len(coverage['missing_cells'])} cell(s) "
                                    f"not run"))
    width = max(len(f"{k}{v}") for k, v in rows) + 6
    width = max(width, 38)
    out = ["+" + "-" * width + "+",
           "|" + "TENSORTRANSIT FRONTIER".center(width) + "|",
           "+" + "-" * width + "+"]
    for key, value in rows:
        pad = width - len(key) - len(str(value)) - 4
        out.append(f"| {key}{' ' * max(pad, 1)}{value} |")
    out.append("+" + "-" * width + "+")
    return "\n".join(out)


def pr_comment(receipt: dict) -> str:
    """The automated PR comment (spec section 54)."""
    frontier = receipt["frontier"]
    stats = receipt["statistics"]
    coverage = receipt["coverage"]
    runtime = receipt.get("runtime") or {}
    model = receipt.get("model") or {}
    hardware = receipt.get("hardware") or {}

    lines = [
        "## TensorTransit Frontier Evaluation",
        "",
        f"Benchmark: {receipt['benchmark_generation']}",
        f"GPU: {hardware.get('gpu', 'unknown')} x{hardware.get('count', 1)}",
        f"Model: {model.get('name', 'unknown')}",
        f"Runtime: {runtime.get('name', 'unknown')} @ {(runtime.get('commit') or '')[:12]}",
        "",
        f"Correctness: {receipt['correctness']}",
        "",
        f"Frontier before: {frontier['before']:.6f}",
        f"Frontier after:  {frontier['after']:.6f}",
        "",
        f"Frontier Gain: **{_pct(frontier['gain_percent'])}**",
        "",
        f"{stats['confidence_level'] * 100:.0f}% CI: "
        f"{stats['lower_gain_percent']:+.2f}% ... {stats['upper_gain_percent']:+.2f}% "
        f"({stats['paired_repeats']} paired repeats)",
        "",
        "Coverage:",
        f"- Improved: {coverage['improved_cells']} / {coverage['total_cells']}",
        f"- Neutral: {coverage['neutral_cells']} / {coverage['total_cells']}",
        f"- Regressed: {coverage['regressed_cells']} / {coverage['total_cells']}",
    ]
    if coverage.get("missing_cells"):
        lines += ["",
                  f"**Partial matrix.** Not run: {', '.join(coverage['missing_cells'])}. "
                  f"A cell that was not run is not averaged in as a zero; this receipt is "
                  f"marked PARTIAL and does not qualify as a full-coverage contribution."]
    aggregation = receipt.get("aggregation") or {}
    if aggregation.get("floor_decided"):
        floored = aggregation.get("cells_at_floor") or {}
        parts = "; ".join(f"{variant}: {', '.join('`' + c + '`' for c in cells)}"
                          for variant, cells in sorted(floored.items()) if cells)
        lines += ["",
                  f"**Scored at the cell floor.** {parts}. Those cells produced no operating "
                  f"point at all for that arm -- an OOM, a timeout, or a fall off the batched "
                  f"decode path -- so their contribution is set by the generation's "
                  f"`cell_floor` ({aggregation.get('cell_floor')}) rather than by a "
                  f"measurement. That is the intended treatment of a lost or newly-created "
                  f"operating region, and it can move dF by a large multiple, so read the "
                  f"per-cell table before quoting the aggregate."]
    violations = receipt.get("regression_guard", {}).get("violations") or []
    if violations:
        lines += ["",
                  "**Protected-workload regression.** "
                  + "; ".join(f"`{v['cell']}` {v['gain'] * 100:+.2f}% against a "
                              f"{v['limit'] * 100:.1f}% limit" for v in violations)]
    if receipt["status"] != "FRONTIER_GAIN":
        lines += ["", _why_not(receipt)]
    lines += ["", f"Status: **{receipt['status']}**", "",
              f"<sub>Receipt `{receipt['content_digest'][:19]}` against generation "
              f"`{receipt['generation_checksum'][:19]}`. Generated by "
              f"`tt-frontier`; no figure in this comment was typed by hand.</sub>"]
    return "\n".join(lines)


def _why_not(receipt: dict) -> str:
    status = receipt["status"]
    stats = receipt["statistics"]
    if status == "INCONCLUSIVE":
        return (f"The observed {_pct(receipt['frontier']['gain_percent'], 2)} is inside its "
                f"own confidence interval, so it is not a verified contribution. Add paired "
                f"repeats up to the generation's maximum; if it is still inconclusive there, "
                f"the effect is smaller than this instrument can resolve on this box.")
    if status == "NO_FRONTIER_GAIN":
        return ("The candidate did not expand the frontier. That is a result, not a failure "
                "of the submission -- most of this repository's own measurements are in that "
                "band, and saying so is the point.")
    if status == "CORRECTNESS_FAIL":
        return ("Correctness precedes performance scoring. A faster run that changed the "
                "model's output scores nothing, whatever it measured.")
    if status == "REGRESSION_GUARD_FAIL":
        return ("A protected workload regressed past this generation's limit. Aggregate "
                "frontier gain does not buy that back.")
    if status == "BUILD_FAIL":
        return "The candidate did not build."
    return (f"Lower bound {stats['lower_gain_percent']:+.2f}% did not clear zero at "
            f"{stats['confidence_level'] * 100:.0f}% confidence.")


def contribution_map(receipt: dict) -> str:
    """The workload map of spec section 55. Explanatory; NOT the canonical score.

    Laid out as context x concurrency when the cell ids follow the `ctx<N>-c<M>` convention,
    and as a flat list otherwise -- a generation is free to name its cells anything, and a
    renderer that assumed a shape would print a confident wrong grid for the ones that do not.
    """
    cells = receipt.get("cells") or {}
    if not cells:
        return ""
    parsed = {}
    for cell in cells:
        ctx, _, conc = cell.partition("-")
        if ctx.startswith("ctx") and conc.startswith("c") and ctx[3:].isdigit() \
                and conc[1:].isdigit():
            parsed[cell] = (int(ctx[3:]), int(conc[1:]))
    if len(parsed) != len(cells):
        rows = ["| workload | gain | weight | protected |", "|---|--:|--:|:--:|"]
        for cell, info in sorted(cells.items()):
            rows.append(f"| `{cell}` | {info['gain'] * 100:+.2f}% | {info['weight']:.3f} | "
                        f"{'yes' if info['protected'] else ''} |")
        return "\n".join(rows)

    contexts = sorted({v[0] for v in parsed.values()})
    concurrencies = sorted({v[1] for v in parsed.values()})
    header = "| | " + " | ".join(f"{c} ctx" for c in contexts) + " |"
    sep = "|---|" + "--:|" * len(contexts)
    rows = [header, sep]
    for conc in concurrencies:
        cells_row = []
        for ctx in contexts:
            match = [c for c, v in parsed.items() if v == (ctx, conc)]
            cells_row.append(f"{cells[match[0]]['gain'] * 100:+.2f}%" if match else "-")
        rows.append(f"| **c{conc}** | " + " | ".join(cells_row) + " |")
    return "\n".join(rows)


def markdown(receipt: dict, *, raw_results_path=None) -> str:
    """The full report. Everything a maintainer needs without opening the JSON."""
    frontier = receipt["frontier"]
    stats = receipt["statistics"]
    coverage = receipt["coverage"]
    agg = receipt["aggregation"]

    out = [
        f"# TensorTransit Frontier Receipt -- {receipt['benchmark_generation']}",
        "",
        f"**Frontier Gain: {_pct(frontier['gain_percent'])}** &nbsp; "
        f"({stats['confidence_level'] * 100:.0f}% CI "
        f"{stats['lower_gain_percent']:+.2f}% ... {stats['upper_gain_percent']:+.2f}%)",
        "",
        f"**Status: {receipt['status']}** &nbsp; | &nbsp; "
        f"Correctness: {receipt['correctness']} &nbsp; | &nbsp; "
        f"Credited: {_pct(frontier['verified_gain_percent'])}",
        "",
        "Frontier Gain is the verified marginal expansion of the real inference serving",
        "frontier beyond current `main`. It is continuous: there are no XS/S/M/L/XL bands,",
        "because a band structure whose lowest paying step sits above what the hardware can",
        "physically deliver tells every contributor the wrong thing about where the room is.",
        "",
        "## Frontier",
        "",
        "| | value |",
        "|---|--:|",
        f"| F(main) | {frontier['before']:.6f} |",
        f"| F(candidate) | {frontier['after']:.6f} |",
        f"| dF | {_pct(frontier['gain_percent'])} |",
        "",
        "## Objectives",
        "",
        "| objective | direction | normalizes 0 at | normalizes 1 at |",
        "|---|:--:|--:|--:|",
    ]
    for objective in receipt["objectives"]:
        unit = f" {objective['unit']}" if objective.get("unit") else ""
        out.append(f"| `{objective['key']}` | {objective['direction']} | "
                   f"{objective['lo']:g}{unit} | {objective['hi']:g}{unit} |")
    out += [
        "",
        f"Reference point {tuple(receipt['reference_point'])}, aggregation "
        f"`{agg['method']}`, cell floor {agg['cell_floor']:g}. Bounds are frozen for the "
        f"lifetime of {receipt['benchmark_generation']}; a receipt whose generation checksum "
        f"has moved does not verify.",
        "",
        "## Contribution map",
        "",
        "Explanatory only. The canonical result is the single dF above.",
        "",
        contribution_map(receipt),
        "",
        "## Per-cell detail",
        "",
        "| workload | H(main) | H(candidate) | gain | weight | protected |",
        "|---|--:|--:|--:|--:|:--:|",
    ]
    for cell, info in sorted((receipt.get("cells") or {}).items()):
        hv = info["hypervolume"]
        out.append(f"| `{cell}` | {hv['main']:.6f} | {hv['candidate']:.6f} | "
                   f"{info['gain'] * 100:+.2f}% | {info['weight']:.3f} | "
                   f"{'yes' if info['protected'] else ''} |")
    out += [
        "",
        f"Improved {coverage['improved_cells']}, neutral {coverage['neutral_cells']}, "
        f"regressed {coverage['regressed_cells']}, of {coverage['total_cells']} declared.",
    ]
    if (receipt.get("aggregation") or {}).get("floor_decided"):
        floored = receipt["aggregation"].get("cells_at_floor") or {}
        out += ["",
                "> **Cells scored at the floor.** "
                + "; ".join(f"{variant}: {', '.join('`' + c + '`' for c in cells)}"
                            for variant, cells in sorted(floored.items()) if cells)
                + f". Those cells produced no operating point for that arm, so their ratio is "
                  f"set by `cell_floor` = {receipt['aggregation'].get('cell_floor')} rather "
                  f"than by a measurement. One such cell can move dF by a large multiple "
                  f"through the geometric mean; it is the intended treatment of a lost or a "
                  f"newly-created operating region, and it is named here so that it cannot "
                  f"do so silently."]
    if coverage.get("missing_cells"):
        out += ["",
                f"> **Partial matrix.** Not run: "
                f"{', '.join('`' + c + '`' for c in coverage['missing_cells'])}. A cell that "
                f"was not run is not averaged in as a zero and is not renormalized away; "
                f"omitting the arm with the most room would otherwise be the cheapest way to "
                f"raise a score."]

    out += [
        "",
        "## Statistics",
        "",
        f"- Method: `{stats['method']}`, {stats['resamples']} resamples, seed "
        f"{stats['seed']} -- so this receipt is reproducible bit for bit from the raw results.",
        f"- Paired repeats: {stats['paired_repeats']}"
        + (f" (unpaired and discarded: {stats['unpaired_repeats']})"
           if stats.get("unpaired_repeats") else ""),
        f"- Confidence is a qualification gate, not a multiplier: dF is never scaled by it.",
        "",
        "Per-repeat frontier scores:",
        "",
        "| repeat | F(main) | F(candidate) |",
        "|--:|--:|--:|",
    ]
    per_repeat = stats.get("per_repeat_frontier") or {}
    for index, (main_v, cand_v) in enumerate(zip(per_repeat.get("main", []),
                                                 per_repeat.get("candidate", [])), start=1):
        out.append(f"| {index} | {main_v:.6f} | {cand_v:.6f} |")

    guard = receipt.get("regression_guard", {})
    out += [
        "",
        "## Regression guard",
        "",
        f"Protected: {', '.join('`' + c + '`' for c in guard.get('protected_cells', [])) or 'none'}; "
        f"limit {guard.get('max_regression', 0) * 100:.1f}%.",
    ]
    if guard.get("violations"):
        out.append("")
        for violation in guard["violations"]:
            out.append(f"- **`{violation['cell']}` {violation['gain'] * 100:+.2f}%** against a "
                       f"{violation['limit'] * 100:.1f}% limit")
    else:
        out.append("")
        out.append("No violation.")

    configurations = receipt.get("configurations") or {}
    out += [
        "",
        "## Portfolio",
        "",
        "The frontier in each cell is formed from every allowed configuration of that variant,",
        "so a candidate does not have to beat every existing strategy everywhere -- it has to",
        "create non-dominated territory that did not exist.",
        "",
        f"- main: {', '.join('`' + c + '`' for c in configurations.get('main', [])) or 'none'}",
        f"- candidate: {', '.join('`' + c + '`' for c in configurations.get('candidate', [])) or 'none'}",
    ]
    failures = receipt.get("failures") or {}
    if failures:
        out += ["",
                "Measurements that produced no operating point (a failure is the absence of a "
                "point, not a slow one):",
                ""]
        for key, count in sorted(failures.items()):
            out.append(f"- `{key}`: {count}")

    out += [
        "",
        "## Provenance",
        "",
        "| | |",
        "|---|---|",
        f"| generation | `{receipt['benchmark_generation']}` |",
        f"| generation checksum | `{receipt['generation_checksum']}` |",
        f"| baseline commit | `{receipt.get('baseline_commit')}` |",
        f"| candidate commit | `{receipt.get('candidate_commit')}` |",
        f"| evaluator commit | `{receipt.get('evaluator_commit')}` |",
        f"| hardware | {receipt.get('hardware')} |",
        f"| runtime | {receipt.get('runtime')} |",
        f"| model | {receipt.get('model')} |",
        f"| environment fingerprint | `{receipt.get('environment_fingerprint')}` |",
        f"| timestamp | {receipt.get('timestamp_utc')} |",
        f"| receipt digest | `{receipt['content_digest']}` |",
    ]
    if raw_results_path:
        out.append(f"| raw results | `{raw_results_path}` |")
    if receipt.get("supersedes"):
        out += ["",
                f"> Supersedes {', '.join('`' + s + '`' for s in receipt['supersedes'])}: "
                f"{receipt.get('supersede_reason')}. Superseded receipts are kept; history is "
                f"never silently erased."]
    out += ["",
            "---",
            "",
            "*Generated by `tt-frontier report`. Every figure above was computed from the raw",
            "results; none was typed.*"]
    return "\n".join(out) + "\n"
