"""A frozen TTF-N benchmark generation.

A generation freezes everything that decides what a number MEANS: the model and its digest,
the runtime commit, the hardware class, the workload cells and their weights, the objective
definitions and their normalization bounds, the SLOs, the reference point, the aggregation
rule, the repeat policy, the confidence method, the regression guard, and the hidden-seed
distribution.

Never silently change a generation. A receipt stays attached to the generation that produced
it, so if the meaning of the evaluation changes materially, the answer is a new TTF-N and not
an edit -- otherwise every historical receipt in the ledger quietly starts describing
something that was never run.

`checksum()` is what makes that enforceable rather than a promise: it is recorded in every
receipt, and `tt-frontier receipt verify` refuses a receipt whose generation no longer hashes
to what it was scored under.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import Objective, ObjectiveError


class GenerationError(ValueError):
    """The generation definition is missing something a receipt would need."""


@dataclass(frozen=True)
class Generation:
    name: str                       # "TTF-1"
    description: str
    objectives: list                # list[Objective], in reference-point order
    reference_point: tuple          # normalized worst corner, one per objective
    cells: list                     # workload cell ids
    weights: dict                   # cell id -> weight
    cell_floor: float
    aggregation: str
    min_repeats: int
    max_repeats: int
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    protected_cells: list           # cells that may not regress more than the guard allows
    protected_max_regression: float  # e.g. 0.05 for "no more than 5%"
    max_configurations: int
    allowed_planners: list
    slo: dict                       # generation-published acceptance rules for a request
    hardware: dict
    runtime: dict
    model: dict
    # Per-cell frozen normalization bounds, from reference.json. Cells differ in absolute
    # throughput by more than a factor of ten across this matrix, so one global bound would
    # make a geometric mean over cells an implicit weighting BY THROUGHPUT REGIME that nobody
    # chose -- the c32 cell would carry ten times the c1 cell's hypervolume for identical
    # relative behaviour. Per-cell bounds make each cell's score relative to its own
    # calibrated control, which is what "equal workload weighting" is supposed to mean.
    cell_bounds: dict = field(default_factory=dict, repr=False)
    raw: dict = field(default_factory=dict, repr=False)

    def objective_keys(self) -> list:
        return [o.key for o in self.objectives]

    def objectives_for(self, cell: str) -> list:
        """The objectives as frozen FOR THIS CELL, falling back to the generation-wide ones."""
        overrides = self.cell_bounds.get(cell)
        if not overrides:
            return self.objectives
        out = []
        for objective in self.objectives:
            bounds = overrides.get(objective.key)
            if not bounds:
                out.append(objective)
                continue
            out.append(Objective(key=objective.key, direction=objective.direction,
                                 lo=float(bounds["lo"]), hi=float(bounds["hi"]),
                                 unit=objective.unit))
        return out

    def published_spread(self, cell: str, objective_key: str):
        """The control's own run-to-run spread for this cell and objective, as calibrated when
        the generation was frozen, in percent. None where the generation did not publish one.

        This is the noise a contributor was told to beat, and it is frozen alongside the bounds
        so that it cannot be re-estimated from the run being scored. TTF-1 publishes 0.14% for
        `ctx128-c1` goodput and **481%** for `ctx128-c32` p99 -- an axis on which no measurement
        can mean anything, on a cell that is scored anyway.
        """
        bounds = (self.cell_bounds.get(cell) or {}).get(objective_key)
        if not bounds or bounds.get("control_spread_pct") is None:
            return None
        return float(bounds["control_spread_pct"])

    def checksum(self) -> str:
        """SHA-256 of the canonical generation document. Recorded in every receipt."""
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(canonical).hexdigest()

    def to_json(self) -> dict:
        return dict(self.raw)


REQUIRED = ("name", "objectives", "reference_point", "cells", "cell_floor", "aggregation",
            "repeats", "confidence", "regression_guard", "portfolio")


def load_generation(path) -> Generation:
    """Load generation.json and, beside it, the frozen reference.json bounds.

    The two are one definition: `generation.json` says what is measured and `reference.json`
    says what the numbers mean. The CHECKSUM covers both, so re-calibrating bounds after
    receipts exist invalidates every one of them loudly instead of silently redefining them.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "generation.json"
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError:
        raise GenerationError(f"no generation at {path}")
    except json.JSONDecodeError as exc:
        raise GenerationError(f"{path}: {exc}")

    reference_path = path.parent / "reference.json"
    if reference_path.exists():
        try:
            reference = json.loads(reference_path.read_text())
        except json.JSONDecodeError as exc:
            raise GenerationError(f"{reference_path}: {exc}")
        if reference.get("generation") not in (None, doc.get("name")):
            raise GenerationError(f"{reference_path} is calibrated for "
                                  f"{reference.get('generation')}, not {doc.get('name')}")
        doc = dict(doc)
        doc["_reference"] = reference
    return parse_generation(doc, source=str(path))


def parse_generation(doc: dict, source: str = "<memory>") -> Generation:
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        raise GenerationError(f"{source}: generation is missing {', '.join(missing)}. Every "
                              f"one of these decides what a receipt MEANS; a generation that "
                              f"left one implicit would produce receipts nobody can "
                              f"reproduce.")
    try:
        objectives = [Objective.from_json(o) for o in doc["objectives"]]
    except ObjectiveError as exc:
        raise GenerationError(f"{source}: {exc}")
    if not objectives:
        raise GenerationError(f"{source}: no objectives")
    keys = [o.key for o in objectives]
    if len(set(keys)) != len(keys):
        raise GenerationError(f"{source}: duplicate objective keys {keys}")

    reference_point = tuple(float(v) for v in doc["reference_point"])
    if len(reference_point) != len(objectives):
        raise GenerationError(f"{source}: reference_point has {len(reference_point)} components for "
                              f"{len(objectives)} objectives")
    if any(not (0.0 <= v < 1.0) for v in reference_point):
        raise GenerationError(f"{source}: reference_point components must be in [0,1) in "
                              f"NORMALIZED space, where 0 is the worst end of the frozen "
                              f"bounds")

    cells = [str(c) for c in doc["cells"]]
    if not cells:
        raise GenerationError(f"{source}: no workload cells")
    if len(set(cells)) != len(cells):
        raise GenerationError(f"{source}: duplicate workload cells")

    weights_doc = doc.get("weights")
    if weights_doc is None:
        weights = {c: 1.0 for c in cells}          # equal weighting is the default (section 25)
    else:
        weights = {str(k): float(v) for k, v in weights_doc.items()}
        if set(weights) != set(cells):
            raise GenerationError(f"{source}: weights do not cover exactly the declared cells. "
                                  f"Never hide workload weights: a cell with no weight, or a "
                                  f"weight with no cell, is a hidden priority.")

    repeats = doc["repeats"]
    confidence = doc["confidence"]
    guard = doc["regression_guard"]
    portfolio = doc["portfolio"]

    calibration = doc.get("_reference") or {}
    cell_bounds = {}
    for cell, bounds in (calibration.get("cell_bounds") or {}).items():
        if cell not in cells:
            raise GenerationError(f"{source}: reference.json calibrates cell {cell!r}, which "
                                  f"the generation does not declare")
        cell_bounds[str(cell)] = {str(k): dict(v) for k, v in bounds.items()}

    generation = Generation(
        name=str(doc["name"]),
        description=str(doc.get("description", "")),
        objectives=objectives,
        reference_point=reference_point,
        cells=cells,
        weights=weights,
        cell_floor=float(doc["cell_floor"]),
        aggregation=str(doc["aggregation"]),
        min_repeats=int(repeats["minimum"]),
        max_repeats=int(repeats["maximum"]),
        confidence_level=float(confidence["level"]),
        bootstrap_resamples=int(confidence["resamples"]),
        bootstrap_seed=int(confidence["seed"]),
        protected_cells=[str(c) for c in guard.get("protected_cells", [])],
        protected_max_regression=float(guard.get("max_regression", 0.05)),
        max_configurations=int(portfolio["max_configurations"]),
        allowed_planners=[str(p) for p in portfolio.get("allowed_planners", [])],
        slo=dict(doc.get("slo", {})),
        hardware=dict(doc.get("hardware", {})),
        runtime=dict(doc.get("runtime", {})),
        model=dict(doc.get("model", {})),
        cell_bounds=cell_bounds,
        raw=doc,
    )
    # A cell with calibrated bounds must have them for EVERY objective, or its hypervolume
    # would mix one objective measured against its own control with another measured against
    # the matrix-wide default -- a number with no interpretation.
    for cell, bounds in cell_bounds.items():
        absent = [o.key for o in objectives if o.key not in bounds]
        if absent:
            raise GenerationError(f"{source}: cell {cell!r} is calibrated for some objectives "
                                  f"and not others (missing {absent})")
        for key, spec in bounds.items():
            if "lo" not in spec or "hi" not in spec:
                raise GenerationError(f"{source}: cell {cell!r} objective {key!r} needs both "
                                      f"lo and hi")
        # Validate by construction: an unusable per-cell bound must fail at load, not at the
        # moment a receipt is being written.
        generation.objectives_for(cell)
    unknown_protected = sorted(set(generation.protected_cells) - set(cells))
    if unknown_protected:
        raise GenerationError(f"{source}: protected cells that are not declared cells: "
                              f"{unknown_protected}")
    if generation.min_repeats < 1 or generation.max_repeats < generation.min_repeats:
        raise GenerationError(f"{source}: repeats must satisfy 1 <= minimum <= maximum")
    if generation.aggregation != "geometric_mean":
        raise GenerationError(f"{source}: aggregation {generation.aggregation!r} is not "
                              f"implemented; add it to aggregate.py and say so here")
    return generation
