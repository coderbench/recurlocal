"""The Transit Frontier Ledger: continuous, verified marginal frontier expansion.

A TensorTransit performance PR earns credit for one thing only -- how much new useful
inference serving capability it creates beyond current `main`. That quantity is continuous:

    dF = F(candidate) / F(main) - 1

and it replaces the S/M/L/XL impact bands this repository used to score with. The bands were
not merely coarse; on the pinned model and device they were *unreachable*. The weighted
ceiling for the whole persist family is 0.52% on Qwen3.8-27B and 1.94% on the best model
found, against a floor of 2.0% -- so a submission could remove every recoverable byte of
recurrent traffic and still score `none`. A scoring regime whose lowest paying band is above
the physical ceiling is not a strict regime, it is a broken instrument, and it tells every
contributor the wrong thing about where the room is.

dF has no bands. It reports what was actually created, and `frontier/TTF-N/reference.json`
records what the generation's own calibration says is reachable, so a contributor can see
the size of the prize before spending a week.

Layout follows the evaluation specification:

    normalize    raw objective units -> [0,1], higher-is-better, frozen bounds
    pareto       dominance and non-dominated frontier extraction
    hypervolume  deterministic fixed-reference hypervolume
    aggregate    per-cell hypervolume -> one frontier score
    confidence   paired bootstrap over interleaved repeats
    receipt      the permanent machine-readable evidence artifact
    report       Markdown, the human receipt box, and the PR comment
    generations  a frozen TTF-N definition, loaded and validated
    ledger       append-only receipt history

Nothing in this package measures anything. It turns raw paired measurements into a verified
number, and every guard in it exists because a confident number once hid a null result.
"""

from .normalize import Objective, normalize_point, ObjectiveError
from .pareto import dominates, pareto_frontier
from .hypervolume import hypervolume
from .aggregate import aggregate_cells, AggregateError
from .confidence import paired_bootstrap, BootstrapResult
from .generations import Generation, load_generation, GenerationError
from .receipt import build_receipt, verify_receipt, ReceiptError, RECEIPT_SCHEMA_VERSION
from .compute import compute_frontier, FrontierComputation, ComputeError

__all__ = [
    "Objective", "normalize_point", "ObjectiveError",
    "dominates", "pareto_frontier", "hypervolume",
    "aggregate_cells", "AggregateError",
    "paired_bootstrap", "BootstrapResult",
    "Generation", "load_generation", "GenerationError",
    "build_receipt", "verify_receipt", "ReceiptError", "RECEIPT_SCHEMA_VERSION",
    "compute_frontier", "FrontierComputation", "ComputeError",
]
