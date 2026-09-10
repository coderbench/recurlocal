"""Deterministic Pareto hypervolume against a fixed reference point.

Requirements the specification puts on this file, all of them load-bearing:

    deterministic
    double precision
    unit tested
    fixed reference point
    fixed normalization

The reference point is the WORST corner -- the origin in normalized, higher-is-better space --
so hypervolume is the measure of the region dominated by the frontier and bounded below by
the reference. It is monotone: adding a non-dominated point can only increase it, and losing
one can only decrease it. That is exactly the property that makes it a fair single number for
a portfolio of operating points, and the reason no regression penalty is needed (spec section
26): a lost region reduces the volume by its own size, automatically.

The 2D case is closed-form and is what TTF-1 uses. The general case is exact
inclusion-exclusion, which is O(2^n) in the number of POINTS -- fine for the tens of
configurations a generation's portfolio budget allows, and exact rather than approximate,
which a receipt requires.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

from .pareto import pareto_frontier


def hypervolume(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """Volume dominated by `points` and bounded by `reference`.

    `points` need not be a frontier; dominated points are removed first, which cannot change
    the answer and does bound the exponential term below.
    """
    if not points:
        return 0.0
    dim = len(reference)
    for p in points:
        if len(p) != dim:
            raise ValueError(f"point of dimension {len(p)} against a {dim}-dimensional "
                             f"reference point")
    # Clamp to the reference: a point worse than the reference in any objective contributes
    # nothing, and letting it through would produce a negative box volume.
    clamped = [tuple(max(float(v), float(r)) for v, r in zip(p, reference)) for p in points]
    frontier = pareto_frontier(clamped)
    frontier = [p for p in frontier if all(v > r for v, r in zip(p, reference))]
    if not frontier:
        return 0.0
    if dim == 1:
        return max(p[0] for p in frontier) - float(reference[0])
    if dim == 2:
        return _hypervolume_2d(frontier, reference)
    return _hypervolume_inclusion_exclusion(frontier, reference)


def _hypervolume_2d(frontier, reference) -> float:
    """Sweep: sort by the first objective descending and add each point's exclusive strip."""
    rx, ry = float(reference[0]), float(reference[1])
    ordered = sorted(frontier, key=lambda p: (-p[0], -p[1]))
    total = 0.0
    previous_y = ry
    for x, y in ordered:
        if y <= previous_y:
            continue
        total += (x - rx) * (y - previous_y)
        previous_y = y
    return total


def _hypervolume_inclusion_exclusion(frontier, reference) -> float:
    """Exact union of the boxes [reference, p], by inclusion-exclusion.

    The intersection of a set of such boxes is the box to the componentwise MINIMUM of their
    corners, which makes each term a product and the whole sum exact in double precision.
    """
    ref = [float(r) for r in reference]
    total = 0.0
    n = len(frontier)
    for size in range(1, n + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for subset in combinations(frontier, size):
            volume = 1.0
            for axis in range(len(ref)):
                edge = min(p[axis] for p in subset) - ref[axis]
                if edge <= 0.0:
                    volume = 0.0
                    break
                volume *= edge
            total += sign * volume
    return total if total > 0.0 else 0.0
