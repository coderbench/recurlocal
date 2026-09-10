"""Per-workload-cell hypervolumes -> one frontier score.

Geometric mean with equal cell weights by default (spec sections 20 and 25). Equal weights
avoid arbitrary hidden priorities, and the geometric mean means one cherry-picked win cannot
carry a candidate that loses elsewhere -- the same reason the 0.1 scorer used one.

The one place this needs care is a ZERO cell. A geometric mean containing a zero is zero, and
if `main` scores zero anywhere the ratio F(candidate)/F(main) is undefined. That is not a
hypothetical: a cell where main OOMs and the candidate succeeds is exactly the "new
capability" case the specification most wants to reward (section 28), and the naive
arithmetic answers it with a division by zero.

So a generation freezes a `cell_floor`: the smallest hypervolume a cell is scored at. It is
part of the frozen definition rather than a constant here, it is recorded in the receipt, and
the receipt says how many cells hit it -- because a run where the floor bound the answer is a
run whose number is about the floor.
"""

from __future__ import annotations

import math


class AggregateError(ValueError):
    """The cells and the weights do not describe an aggregate that can be computed."""


def aggregate_cells(cells: dict, weights: dict = None, *, method: str = "geometric_mean",
                    cell_floor: float = 1e-6) -> dict:
    """Combine per-cell hypervolumes.

    Returns the score and the diagnostics that say whether to believe it: how many cells were
    at the floor, and the effective weight of each.
    """
    if not cells:
        raise AggregateError("no workload cells: an aggregate over nothing is not a score")
    if cell_floor <= 0.0:
        raise AggregateError("cell_floor must be > 0; it exists to keep the ratio defined "
                             "when a cell scores zero")

    if weights is None:
        weights = {cell: 1.0 for cell in cells}
    unknown = sorted(set(weights) - set(cells))
    if unknown:
        raise AggregateError(f"weights name cells that were not scored: {unknown}. A weight "
                             f"for a cell that was not run is how a partial matrix comes to "
                             f"be reported as a full one.")
    missing = sorted(set(cells) - set(weights))
    if missing:
        raise AggregateError(f"cells with no weight: {missing}")
    total_weight = sum(float(weights[c]) for c in cells)
    if total_weight <= 0.0:
        raise AggregateError("weights sum to zero")

    at_floor = []
    log_sum = 0.0
    for cell, value in sorted(cells.items()):
        v = float(value)
        if not math.isfinite(v) or v < 0.0:
            raise AggregateError(f"cell {cell}: hypervolume {value!r} is not a volume")
        if v < cell_floor:
            at_floor.append(cell)
            v = cell_floor
        log_sum += (float(weights[cell]) / total_weight) * math.log(v)

    if method != "geometric_mean":
        raise AggregateError(f"unknown aggregation method {method!r}; a generation that wants "
                             f"another one has to add it here and say so in generation.json")
    return {
        "score": math.exp(log_sum),
        "method": method,
        "cells": {c: float(v) for c, v in sorted(cells.items())},
        "weights": {c: float(weights[c]) / total_weight for c in sorted(cells)},
        "cell_floor": float(cell_floor),
        "cells_at_floor": at_floor,
    }
