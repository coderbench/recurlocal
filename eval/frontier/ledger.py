"""The Transit Frontier Ledger: append-only receipt history, on disk.

    frontier/
    |-- TTF-1/
    |   |-- generation.json        the frozen definition
    |   |-- reference.json         calibrated bounds and what is REACHABLE on this hardware
    |   |-- current-frontier.json  the running frontier and every receipt that moved it
    |   `-- receipts/pr-000184.json
    `-- README.md

Two rules, and both are enforced here rather than described:

* **A finalized receipt is never silently rewritten.** Writing over an existing receipt id
  with different content is refused; a correction is a NEW receipt that names what it
  supersedes and why.
* **A receipt stays attached to the generation that produced it.** When TTF-2 launches, TTF-1
  receipts are not re-scored and not migrated. Historical performance achievements remain
  auditable exactly as they were earned.
"""

from __future__ import annotations

import json
from pathlib import Path

from .receipt import content_digest, verify_receipt, ReceiptError


class LedgerError(ValueError):
    """The ledger would have to lose or rewrite history to accept this."""


def receipt_path(root, generation_name: str, receipt_id: str) -> Path:
    return Path(root) / generation_name / "receipts" / f"{receipt_id}.json"


def default_receipt_id(receipt: dict) -> str:
    pr = receipt.get("pr")
    if pr:
        return f"pr-{int(pr):06d}"
    candidate = (receipt.get("candidate_commit") or "unknown")[:12]
    return f"commit-{candidate}"


def append_receipt(root, receipt: dict, *, receipt_id: str = None, generation=None) -> Path:
    """Write a receipt into the ledger and update the generation's running frontier."""
    verify_receipt(receipt, generation)
    generation_name = receipt["benchmark_generation"]
    receipt_id = receipt_id or default_receipt_id(receipt)
    path = receipt_path(root, generation_name, receipt_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("content_digest") == receipt.get("content_digest"):
            return path                       # idempotent: the same receipt, written again
        raise LedgerError(
            f"{path} already holds a different receipt "
            f"({existing.get('content_digest')}). A finalized receipt is not rewritten. If an "
            f"evaluator bug requires a correction, write a NEW receipt whose `supersedes` "
            f"names this one and whose `supersede_reason` says why.")

    path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    _update_current_frontier(root, generation_name)
    return path


def load_receipts(root, generation_name: str) -> list:
    directory = Path(root) / generation_name / "receipts"
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append((path.stem, json.loads(path.read_text())))
        except json.JSONDecodeError as exc:
            raise LedgerError(f"{path}: {exc}")
    return out


def _superseded_ids(receipts) -> set:
    superseded = set()
    for _, receipt in receipts:
        for old in receipt.get("supersedes") or []:
            superseded.add(old)
    return superseded


def _update_current_frontier(root, generation_name: str) -> dict:
    """Recompute the generation's running frontier from every canonical receipt.

    Compounding is multiplicative, because each dF is measured against the `main` of its own
    moment: a miner earns the marginal expansion relative to main AT EVALUATION TIME, so two
    +5% contributions are +10.25% together and not +10%.
    """
    receipts = load_receipts(root, generation_name)
    superseded = _superseded_ids(receipts)
    canonical = [(rid, r) for rid, r in receipts if rid not in superseded]

    compounded = 1.0
    entries = []
    for receipt_id, receipt in sorted(canonical, key=lambda item: item[1].get("timestamp_utc") or ""):
        credited = receipt.get("frontier", {}).get("verified_gain_percent", 0.0) / 100.0
        if receipt.get("status") == "FRONTIER_GAIN" and credited > 0:
            compounded *= (1.0 + credited)
        entries.append({
            "receipt": receipt_id,
            "pr": receipt.get("pr"),
            "timestamp_utc": receipt.get("timestamp_utc"),
            "candidate_commit": receipt.get("candidate_commit"),
            "status": receipt.get("status"),
            "gain_percent": receipt.get("frontier", {}).get("gain_percent"),
            "verified_gain_percent": receipt.get("frontier", {}).get("verified_gain_percent"),
            "content_digest": receipt.get("content_digest"),
        })

    document = {
        "generation": generation_name,
        "canonical_receipts": len(canonical),
        "superseded_receipts": sorted(superseded),
        "verified_frontier_receipts": sum(1 for e in entries if e["status"] == "FRONTIER_GAIN"),
        "compounded_expansion_percent": (compounded - 1.0) * 100.0,
        "note": ("Compounded multiplicatively: each dF is measured against the `main` of its "
                 "own moment, so two +5% contributions are +10.25% together, not +10%. "
                 "Receipts with any status other than FRONTIER_GAIN credit nothing and are "
                 "listed for audit."),
        "history": entries,
    }
    path = Path(root) / generation_name / "current-frontier.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return document


def show(root, generation_name: str) -> dict:
    path = Path(root) / generation_name / "current-frontier.json"
    if path.exists():
        return json.loads(path.read_text())
    return _update_current_frontier(root, generation_name)


def audit(root, generation_name: str, generation=None) -> dict:
    """Verify every receipt in a generation. Returns the failures rather than raising."""
    problems = []
    for receipt_id, receipt in load_receipts(root, generation_name):
        try:
            verify_receipt(receipt, generation)
        except ReceiptError as exc:
            problems.append({"receipt": receipt_id, "problem": str(exc)})
        else:
            stated = receipt.get("content_digest")
            if stated != content_digest(receipt):
                problems.append({"receipt": receipt_id, "problem": "digest mismatch"})
    return {"generation": generation_name, "checked": len(load_receipts(root, generation_name)),
            "problems": problems, "ok": not problems}
