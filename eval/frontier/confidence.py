"""Paired bootstrap over interleaved main/candidate repeats.

Confidence is a QUALIFICATION GATE, not a score multiplier (spec section 29). Multiplying dF
by a confidence would turn two different questions -- how much was created, and how sure are
we -- into one number that answers neither. So:

    if the lower confidence bound of dF is above 0  -> the measured dF is verified
    otherwise                                       -> INCONCLUSIVE

The resampling is PAIRED over repeat index, because the runs are paired: repeat k of main and
repeat k of the candidate ran adjacently on the same box under the same thermal state. Any
estimator that resampled the two arms independently would throw away exactly the pairing that
makes a same-box delta trustworthy on hardware whose clocks cannot be pinned.

The receipt has to be reproducible from the raw results, so the seed, the resample count and
the confidence level are all inputs and all recorded.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BootstrapResult:
    point: float          # dF computed from all repeats
    lower: float          # `level` two-sided lower bound
    upper: float
    level: float
    resamples: int
    seed: int
    repeats: int
    method: str = "paired_bootstrap_percentile"

    def to_json(self) -> dict:
        return asdict(self)

    def qualifies(self) -> bool:
        """The initial positive qualification of spec section 29: lower bound above zero."""
        return self.lower > 0.0


def _delta(main_scores, candidate_scores) -> float:
    """dF from a set of paired per-repeat frontier scores.

    The paired repeats are combined by GEOMETRIC mean before the ratio, matching how cells are
    combined: dF is a ratio, and the mean of ratios is not the ratio of arithmetic means.
    """
    if not main_scores:
        raise ValueError("no repeats")
    log_main = sum(math.log(max(v, 1e-300)) for v in main_scores) / len(main_scores)
    log_cand = sum(math.log(max(v, 1e-300)) for v in candidate_scores) / len(candidate_scores)
    return math.exp(log_cand - log_main) - 1.0


def paired_bootstrap(main_scores, candidate_scores, *, level: float = 0.99,
                     resamples: int = 20000, seed: int = 20260910) -> BootstrapResult:
    """Percentile bootstrap of dF over paired repeats.

    `main_scores[k]` and `candidate_scores[k]` are the frontier scores of repeat k. The
    resample draws repeat INDICES with replacement and applies the same index to both arms,
    which is what makes it paired.
    """
    main_scores = [float(v) for v in main_scores]
    candidate_scores = [float(v) for v in candidate_scores]
    if len(main_scores) != len(candidate_scores):
        raise ValueError(f"unpaired repeats: {len(main_scores)} main, "
                         f"{len(candidate_scores)} candidate. Interleaved pairing is the whole "
                         f"basis of a same-box delta; an unpaired set cannot be bootstrapped "
                         f"this way.")
    n = len(main_scores)
    if n == 0:
        raise ValueError("no repeats")
    if not (0.5 < level < 1.0):
        raise ValueError("confidence level must be in (0.5, 1.0)")
    if resamples < 1000:
        raise ValueError("resample count below 1000 cannot resolve a 99% interval")

    point = _delta(main_scores, candidate_scores)
    if n == 1:
        # One pair carries no information about spread. Saying so is the honest answer;
        # returning a zero-width interval would let a single run qualify.
        return BootstrapResult(point=point, lower=float("-inf"), upper=float("inf"),
                               level=level, resamples=0, seed=seed, repeats=1,
                               method="insufficient_repeats")

    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(_delta([main_scores[i] for i in idx],
                            [candidate_scores[i] for i in idx]))
    draws.sort()
    tail = (1.0 - level) / 2.0
    lower = draws[_percentile_index(len(draws), tail)]
    upper = draws[_percentile_index(len(draws), 1.0 - tail)]
    return BootstrapResult(point=point, lower=lower, upper=upper, level=level,
                           resamples=resamples, seed=seed, repeats=n)


def _percentile_index(count: int, q: float) -> int:
    """Nearest-rank index, clamped. Deterministic and free of interpolation choices."""
    idx = int(math.floor(q * count))
    return 0 if idx < 0 else (count - 1 if idx >= count else idx)
