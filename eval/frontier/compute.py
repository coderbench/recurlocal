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

    # variant -> repeat -> cell -> [points]
    points = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    configs = defaultdict(set)
    failures = defaultdict(int)
    seen_cells = set()
    seen_repeats = defaultdict(set)

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
        point = normalize_point(record["metrics"], generation.objectives_for(cell), status)
        if point is None:
            failures[f"{variant}:{status}"] += 1
            # Deliberately NOT a zero point. An OOM is the absence of an operating point, and
            # a zero would still be a point on the frontier's axis (spec section 49).
            continue
        points[variant][repeat][cell].append(point)

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

    scored_cells = sorted(seen_cells)
    per_repeat = {"main": [], "candidate": []}
    per_cell_detail = defaultdict(dict)
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
            for cell, value in agg["cells"].items():
                per_cell_detail[cell].setdefault(variant, []).append(value)

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

    return FrontierComputation(
        generation=generation.name,
        generation_checksum=generation.checksum(),
        frontier_before=frontier_main,
        frontier_after=frontier_candidate,
        gain=stats.point,
        gain_percent=stats.point * 100.0,
        statistics=stats.to_json(),
        qualifies=stats.qualifies(),
        per_repeat=dict(per_repeat),
        paired_repeats=repeats,
        unpaired_repeats=unpaired,
        cells_scored=scored_cells,
        cells_missing=missing_cells,
        cell_gain=cell_gain,
        cell_hypervolume={c: {"main": _median(per_cell_detail[c].get("main", [])),
                              "candidate": _median(per_cell_detail[c].get("candidate", []))}
                          for c in scored_cells},
        coverage=coverage,
        configurations={v: sorted(configs[v]) for v in ("main", "candidate")},
        failures=dict(failures),
        guard_violations=guard_violations,
        partial=bool(missing_cells),
        weights={c: generation.weights[c] for c in scored_cells},
    )


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
