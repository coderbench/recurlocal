"""Raw objective units -> a normalized [0,1] score where higher is always better.

Two rules, both of which this repository has broken before in other places:

1. **The bounds are frozen per generation.** Normalizing against "today's fastest PR" makes
   every historical receipt mean something different the moment a new PR lands, and a ledger
   whose past entries silently change is not a ledger. `frontier/TTF-N/reference.json` freezes
   them for the lifetime of the generation.

2. **A failure is not a bad number.** An OOM, a timeout or an SLO violation is the absence of
   an operating point, not a slow one, and it must not be normalized into a small positive
   score that still contributes hypervolume. `normalize_point` returns None for those, and
   every consumer treats None as "this configuration produced no point here".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class ObjectiveError(ValueError):
    """A generation's objective definition is unusable, or a measurement is not a number."""


# A measurement carrying one of these is a serving failure, not a slow point (spec section
# 49). They are values of `status` in a raw result record.
FAILURE_STATUSES = frozenset({"OOM", "TIMEOUT", "CRASH", "SLO_FAIL", "CORRECTNESS_FAIL",
                              "NOT_RUN",
                              # A run that fell off the runtime's batched decode path measured
                              # the single-sequence path once per row; its throughput is not
                              # this cell's number and must not become a point on this cell's
                              # frontier. It used to be excluded only by accident -- such a
                              # record carries no metrics, so it fell into the missing-
                              # objective branch below and was counted as a harness fault.
                              "UNBATCHED",
                              # The configuration failures. The runner aborts on these rather
                              # than recording them, so they reach here only from a raw file
                              # written by something else; naming them is what keeps such a
                              # file from being scored as if the arm had merely been slow.
                              "UNHOOKED", "NULL_POLICY", "EVAL_ERROR", "UNMEASURABLE"})


@dataclass(frozen=True)
class Objective:
    """One dimension of the serving frontier.

    `lo`/`hi` are in RAW units and are the generation's frozen normalization bounds. They are
    not the observed range: they are calibrated on the target hardware before the generation
    launches, precisely so that an observed range cannot move them.
    """

    key: str
    direction: str        # "max" (goodput) or "min" (latency, energy)
    lo: float             # raw value that normalizes to 0.0
    hi: float             # raw value that normalizes to 1.0
    unit: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("max", "min"):
            raise ObjectiveError(f"{self.key}: direction must be 'max' or 'min', "
                                 f"not {self.direction!r}")
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ObjectiveError(f"{self.key}: bounds must be finite")
        if self.lo == self.hi:
            raise ObjectiveError(f"{self.key}: lo == hi ({self.lo}); every measurement would "
                                 f"normalize to the same score and the objective would carry "
                                 f"no information")
        if self.direction == "max" and self.hi <= self.lo:
            raise ObjectiveError(f"{self.key}: a maximized objective needs hi > lo")
        if self.direction == "min" and self.hi >= self.lo:
            raise ObjectiveError(f"{self.key}: a minimized objective needs hi < lo, so that "
                                 f"hi is the GOOD end -- state the bounds in the direction "
                                 f"the objective actually improves")

    def normalize(self, raw: float) -> float:
        """Raw -> [0,1], higher better, clipped at both ends.

        Clipping is deliberate and generation-defined: a measurement outside the frozen bounds
        is real, but letting it extend the hypervolume past 1.0 would let one extraordinary
        cell dominate an aggregate that is supposed to be a portfolio.
        """
        if raw is None or not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ObjectiveError(f"{self.key}: measurement {raw!r} is not a number")
        value = float(raw)
        if not math.isfinite(value):
            raise ObjectiveError(f"{self.key}: measurement is {value}, which is not a "
                                 f"measurement -- a failed run must carry a failure status, "
                                 f"not a non-finite number")
        score = (value - self.lo) / (self.hi - self.lo)
        return 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)

    def to_json(self) -> dict:
        return {"key": self.key, "direction": self.direction, "lo": self.lo, "hi": self.hi,
                "unit": self.unit}

    @staticmethod
    def from_json(doc: dict) -> "Objective":
        missing = [k for k in ("key", "direction", "lo", "hi") if k not in doc]
        if missing:
            raise ObjectiveError(f"objective is missing {', '.join(missing)}")
        return Objective(key=str(doc["key"]), direction=str(doc["direction"]),
                         lo=float(doc["lo"]), hi=float(doc["hi"]),
                         unit=str(doc.get("unit", "")))


def normalize_point(metrics: dict, objectives: list, status: str = "OK"):
    """One measured configuration -> a normalized point, or None when it produced no point.

    Returns None for a failure status and for a missing objective. A missing objective is a
    failure of the harness rather than of the candidate, but treating it as a zero would
    silently reward a runner that stopped reporting the dimension a candidate is worst in --
    so it is absent, and `compute` reports how many points were absent and why.
    """
    if status in FAILURE_STATUSES:
        return None
    out = []
    for objective in objectives:
        if objective.key not in metrics or metrics[objective.key] is None:
            return None
        out.append(objective.normalize(metrics[objective.key]))
    return tuple(out)
