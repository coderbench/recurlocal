"""Raw paired measurements -> Frontier Gain.

The flow, and every step of it is in the specification:

    raw results (per cell, per variant, per configuration, per repeat)
        -> normalize each configuration's metrics against the FROZEN bounds
        -> Pareto frontier within a cell, over the configurations of that variant
        -> fixed-reference hypervolume of that frontier
        -> aggregate cells (geometric mean, published weights)      = F for that repeat
        -> dF = F(candidate)/F(main) - 1, over paired repeats
        -> paired bootstrap                                          = the qualification gate
        -> protected-workload guard                                  = the veto

Two things this file refuses, both because the 0.1 evaluator learned them the hard way:

* **A partial matrix.** A cell that was not run is not averaged in as a zero and is not
  renormalized away either -- it is a MISSING CELL and the computation says so, because
  omitting the arm with the most room is otherwise the cheapest way to raise a score.
* **An unpaired repeat set.** main repeat k and candidate repeat k are one interleaved pair.
  A candidate with more repeats than main is not a better-sampled candidate, it is an
  unpaired comparison, and on a box whose clocks cannot be pinned that is the whole ballgame.
"""

from __future__ import annotations

from collections import defaultdict

from .aggregate import aggregate_cells
from .confidence import paired_bootstrap
from .hypervolume import hypervolume
from .normalize import FAILURE_STATUSES, normalize_point


class ComputeError(ValueError):
    """The raw results cannot be turned into a frontier score."""


class FrontierComputation:
    """Everything the receipt needs, plus the diagnostics that say whether to believe it."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def to_json(self) -> dict:
        return {k: v for k, v in sorted(self.__dict__.items())}


def _require(condition, message):
    if not condition:
        raise ComputeError(message)


def compute_frontier(generation, results, *, allow_partial=False):
    """`results` is a flat list of raw result records (see schemas/frontier_result.schema.json).

    Every record carries: workload_id, variant ("main"|"candidate"), config_id, repeat,
    metrics, status. Nothing here reads a metric the generation did not declare, so a runner
    that reports extra diagnostics cannot influence the score with them -- which is the whole
    point of "hardware counters explain why a PR works, real serving performance decides
    whether it works".
    """
    reference = generation.reference_point

    # variant -> repeat -> cell -> [points], and the same keyed by config so a receipt can say
    # WHICH configuration created the territory. "The candidate gained 4%" and "the candidate's
    # new specialized planner is on the frontier in four cells and its default is on none" are
    # different facts, and only the second tells a contributor where their work earned a place.
    points = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    labelled = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    configs = defaultdict(set)
    failures = defaultdict(int)
    seen_cells = set()
    seen_repeats = defaultdict(set)

    # Cells the candidate lost to a SERVING guard that the evaluator's own probe reproduced
    # with NO policy running. See runner.attribute_serving_losses: the control is unhooked and
    # emits no packing telemetry, so a cell that collapses in both arms can only be seen to
    # collapse in the hooked one. Charging that to the candidate would report a property of the
    # runtime as a locality regression -- at the generation's floor, through a geometric mean.
    unservable_evidence = {}
    candidate_usable = defaultdict(int)
    attributions = defaultdict(set)

    for record in results:
        for key in ("workload_id", "variant", "config_id", "repeat", "metrics"):
            _require(key in record, f"raw result is missing {key!r}: {record!r}")
        variant = record["variant"]
        _require(variant in ("main", "candidate"),
                 f"variant {variant!r} is neither main nor candidate")
        cell = str(record["workload_id"])
        if cell not in generation.weights:
            # A cell the generation does not declare cannot be scored: its weight is
            # undefined and inventing one is how a hidden priority gets in.
            raise ComputeError(f"workload {cell!r} is not declared by {generation.name}. "
                               f"Declared: {sorted(generation.weights)}")
        repeat = int(record["repeat"])
        seen_cells.add(cell)
        seen_repeats[variant].add(repeat)
        configs[variant].add(str(record["config_id"]))
        status = str(record.get("status", "OK"))
        if str(record.get("correctness", "PASS")) != "PASS":
            status = "CORRECTNESS_FAIL"
        if variant == "candidate":
            if status in FAILURE_STATUSES:
                attributions[cell].add(str(record.get("attribution", "unresolved")))
                probe = (record.get("detail") or {}).get("attribution")
                if probe:
                    unservable_evidence[cell] = probe
            else:
                candidate_usable[cell] += 1
        point = normalize_point(record["metrics"], generation.objectives_for(cell), status)
        if point is None:
            failures[f"{variant}:{status}"] += 1
            # Deliberately NOT a zero point. An OOM is the absence of an operating point, and
            # a zero would still be a point on the frontier's axis (spec section 49).
            continue
        points[variant][repeat][cell].append(point)
        labelled[variant][repeat][cell][point] = str(record["config_id"])

    for variant in ("main", "candidate"):
        _require(points[variant], f"no usable {variant} measurements at all")
        _require(len(configs[variant]) <= generation.max_configurations,
                 f"{variant} ran {len(configs[variant])} configurations; "
                 f"{generation.name} allows {generation.max_configurations}. An unbounded "
                 f"configuration search is not a frontier, it is a fit.")

    missing_cells = sorted(set(generation.cells) - seen_cells)
    if missing_cells and not allow_partial:
        raise ComputeError(
            f"{generation.name} declares {len(generation.cells)} workload cells and "
            f"{len(seen_cells)} were measured. Absent: {missing_cells}. A cell that was not "
            f"run is not averaged in as a zero and is not renormalized away -- omitting the "
            f"arm with the most room would otherwise be the cheapest way to raise a score. "
            f"Run the whole matrix, or pass --allow-partial and accept a receipt that says "
            f"PARTIAL on its face.")

    repeats = sorted(seen_repeats["main"] & seen_repeats["candidate"])
    unpaired = sorted((seen_repeats["main"] | seen_repeats["candidate"]) - set(repeats))
    _require(repeats, "no repeat index appears in BOTH arms; the runs are not paired")
    _require(len(repeats) >= generation.min_repeats,
             f"{len(repeats)} paired repeats, {generation.name} requires "
             f"{generation.min_repeats}")

    # A cell is unservable when the candidate produced no operating point there AND every one
    # of its failures was attributed to the runtime by a probe that ran no policy. Anything
    # less -- one usable repeat, one unresolved failure, one probe that came back clean -- and
    # the cell is scored as measured, which is the conservative direction: the candidate keeps
    # the loss.
    unservable = sorted(cell for cell, verdicts in attributions.items()
                        if verdicts == {"runtime"} and not candidate_usable[cell])
    scored_cells = sorted(seen_cells - set(unservable))
    _require(scored_cells,
             f"every cell {generation.name} declares was either missing or unservable; there "
             f"is nothing left to score. Unservable: {unservable}")
    per_repeat = {"main": [], "candidate": []}
    per_cell_detail = defaultdict(dict)
    at_floor = defaultdict(set)
    for variant in ("main", "candidate"):
        for repeat in repeats:
            cells = {}
            for cell in scored_cells:
                cell_points = points[variant][repeat].get(cell, [])
                cells[cell] = hypervolume(cell_points, reference)
            agg = aggregate_cells(cells,
                                  {c: generation.weights[c] for c in scored_cells},
                                  method=generation.aggregation,
                                  cell_floor=generation.cell_floor)
            per_repeat[variant].append(agg["score"])
            for cell in agg["cells_at_floor"]:
                at_floor[variant].add(cell)
            for cell, value in agg["cells"].items():
                per_cell_detail[cell].setdefault(variant, []).append(value)

    # Which configurations were ever non-dominated in a cell. The union over repeats: a
    # configuration that held the frontier in one repeat and lost it to noise in another still
    # created that territory, and a receipt that only reported the last repeat would say
    # otherwise.
    from .pareto import pareto_frontier
    on_frontier = defaultdict(lambda: defaultdict(set))
    for variant in ("main", "candidate"):
        for repeat in repeats:
            for cell in scored_cells:
                cell_points = points[variant][repeat].get(cell, [])
                # Clamped to the reference exactly as hypervolume() clamps, so this diagnostic
                # names the same frontier the score was computed from. With TTF-1's reference
                # at the origin the clamp is a no-op; a generation that moved it would
                # otherwise get a table describing a frontier its own number did not use.
                clamped = {tuple(max(float(v), float(r)) for v, r in zip(p, reference)): p
                           for p in cell_points}
                for winner in pareto_frontier(clamped):
                    original = clamped.get(tuple(winner))
                    name = labelled[variant][repeat][cell].get(original)
                    if name:
                        on_frontier[cell][variant].add(name)

    stats = paired_bootstrap(per_repeat["main"], per_repeat["candidate"],
                             level=generation.confidence_level,
                             resamples=generation.bootstrap_resamples,
                             seed=generation.bootstrap_seed)

    # Per-cell coverage, from the MEDIAN of each cell's paired hypervolumes. The median rather
    # than the mean because a single thermal outlier must not flip a cell's label.
    coverage = {"improved": [], "neutral": [], "regressed": []}
    cell_gain = {}
    for cell in scored_cells:
        main_v = _median(per_cell_detail[cell].get("main", []))
        cand_v = _median(per_cell_detail[cell].get("candidate", []))
        floor = generation.cell_floor
        gain = (max(cand_v, floor) / max(main_v, floor)) - 1.0
        cell_gain[cell] = gain
        if gain > 1e-9:
            coverage["improved"].append(cell)
        elif gain < -1e-9:
            coverage["regressed"].append(cell)
        else:
            coverage["neutral"].append(cell)

    guard_violations = [
        {"cell": cell, "gain": cell_gain[cell], "limit": -generation.protected_max_regression}
        for cell in generation.protected_cells
        if cell in cell_gain and cell_gain[cell] < -generation.protected_max_regression
    ]

    frontier_main = _geomean(per_repeat["main"])
    frontier_candidate = _geomean(per_repeat["candidate"])

    # The run's own noise floor: the peak-to-peak spread of the MAIN arm's frontier score
    # across the repeats that produced it, as a percentage.
    #
    # A paired bootstrap alone is not enough at the repeat counts this generation allows, and
    # that is a property of the estimator rather than a bug. A percentile bootstrap over three
    # paired points has at most 27 distinct resamples; if all three paired ratios happen to
    # fall on the same side of 1.0 -- which pure jitter does one time in four -- every resample
    # does too and the lower bound clears zero however small the effect. A synthetic candidate
    # identical to main to within 0.03% scored FRONTIER_GAIN this way, which is exactly the
    # failure this repository has spent its whole history guarding against.
    #
    # So the confidence gate is joined by the rule the rest of the harness already uses: an
    # axis whose spread sits inside its own run-to-run noise is OPEN, not solved. Both have to
    # pass. This one needs no new generation field because it is measured from the run.
    #
    # The floor is each ARM's own absolute spread, not the spread of the paired ratios, and the
    # objection to that is fair: pairing exists to cancel common-mode drift, so the paired
    # spread is the smaller and in one sense the more correct number. It is not used here for
    # the reason above -- a consistently-signed jitter has a tiny paired spread and would sail
    # through. The absolute spread is deliberately the conservative choice, it is the same rule
    # `eval/decide.py::resolution` has always applied, and the cost of it is a real small effect
    # on a noisy box being reported INCONCLUSIVE rather than credited. The generation allows
    # nine repeats; spending them is the remedy, and it is the honest one.
    noise_floor_pct = _spread_pct(per_repeat["main"])
    candidate_spread_pct = _spread_pct(per_repeat["candidate"])
    floor = max(noise_floor_pct, candidate_spread_pct)
    resolved = abs(stats.point * 100.0) > floor

    return FrontierComputation(
        generation=generation.name,
        generation_checksum=generation.checksum(),
        frontier_before=frontier_main,
        frontier_after=frontier_candidate,
        gain=stats.point,
        gain_percent=stats.point * 100.0,
        statistics=stats.to_json(),
        # BOTH gates. The bootstrap says the difference is unlikely to be zero; the noise
        # floor says it is bigger than the spread of the runs that produced it. A result that
        # passes one and fails the other is not a result.
        qualifies=stats.qualifies() and resolved,
        confidence_qualifies=stats.qualifies(),
        resolved=resolved,
        noise_floor_pct=floor,
        main_spread_pct=noise_floor_pct,
        candidate_spread_pct=candidate_spread_pct,
        per_repeat=dict(per_repeat),
        paired_repeats=repeats,
        unpaired_repeats=unpaired,
        cells_scored=scored_cells,
        cells_missing=missing_cells,
        # Named on the receipt's face rather than renormalized away in silence. Dropping a cell
        # is the cheapest way to raise a score, so the evidence for each drop travels with it.
        cells_unservable=unservable,
        unservable_evidence={c: unservable_evidence[c] for c in unservable
                             if c in unservable_evidence},
        cell_gain=cell_gain,
        cell_hypervolume={c: {"main": _median(per_cell_detail[c].get("main", [])),
                              "candidate": _median(per_cell_detail[c].get("candidate", []))}
                          for c in scored_cells},
        coverage=coverage,
        # Cells whose hypervolume was ZERO for one arm and were scored at the generation's
        # floor instead. This matters out of all proportion to how often it happens: the ratio
        # for such a cell is set by the floor rather than by anything measured, and a single
        # one can move dF by hundreds of percent through the geometric mean. It is the "new
        # capability" case the specification most wants to reward -- main OOMs, the candidate
        # succeeds -- and rewarding it is right; letting it do so SILENTLY is not.
        cells_at_floor={variant: sorted(at_floor[variant]) for variant in ("main", "candidate")},
        floor_decided=bool(at_floor["main"] or at_floor["candidate"]),
        configurations={v: sorted(configs[v]) for v in ("main", "candidate")},
        frontier_configurations={cell: {v: sorted(on_frontier[cell][v])
                                        for v in ("main", "candidate")}
                                 for cell in scored_cells},
        failures=dict(failures),
        guard_violations=guard_violations,
        partial=bool(missing_cells or unservable),
        weights={c: generation.weights[c] for c in scored_cells},
    )


def _spread_pct(values):
    """Peak-to-peak spread as a percentage of the median: this run's own noise floor, measured
    rather than assumed. Fewer than two repeats has no spread to report, and reporting zero
    there would read as "perfectly stable"."""
    if len(values) < 2:
        return float("inf")
    ordered = sorted(values)
    mid = _median(values)
    return (ordered[-1] - ordered[0]) / mid * 100.0 if mid else float("inf")


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _geomean(values):
    import math
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-300)) for v in values) / len(values))
