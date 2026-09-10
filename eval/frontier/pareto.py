"""Pareto dominance over normalized, higher-is-better points.

A point dominates another when it is at least as good in every objective and strictly better
in at least one. Dominated points do not define the serving frontier, and that is the whole
reason the frontier is the right object to score: a candidate does not have to beat every
existing configuration everywhere. It has to create operating territory that did not exist.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True when `a` dominates `b`. Both are normalized, higher-is-better tuples."""
    if len(a) != len(b):
        raise ValueError(f"points of different dimension: {len(a)} vs {len(b)}")
    at_least_as_good = all(x >= y for x, y in zip(a, b))
    strictly_better = any(x > y for x, y in zip(a, b))
    return at_least_as_good and strictly_better


def pareto_frontier(points: Iterable[Sequence[float]]) -> list:
    """The non-dominated subset, deduplicated, in a deterministic order.

    Deterministic order matters more than it looks: the frontier feeds a hypervolume whose
    value must be reproducible bit for bit from the same raw results, or the receipt is not
    verifiable. Sorting rather than preserving input order also makes the result independent
    of the order the runner happened to execute configurations in.
    """
    unique = sorted({tuple(float(v) for v in p) for p in points})
    keep = []
    for candidate in unique:
        if any(dominates(other, candidate) for other in unique if other != candidate):
            continue
        keep.append(candidate)
    return keep
