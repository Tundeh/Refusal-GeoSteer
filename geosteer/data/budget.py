"""Data budgets: use only N prompts of a role (e.g. 100 for direction estimation or repair).

Selection is balanced across strata (remainder goes to strata in Stratum.PRIMARY order),
deterministic, and nested: the budget-50 set is a subset of the budget-100 set, so budget
sweeps form learning curves. Within a stratum, prompts are ranked by a seeded hash of
their *group*, so topic-matched pairs (which share a group across strata) tend to be
selected together; ``budget_summary`` reports how many pairs survive intact.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from geosteer.data.schema import PromptRecord, Stratum
from geosteer.utils import stable_hash


def budget_quota(n: int, strata: Sequence[str]) -> dict[str, int]:
    base, extra = divmod(n, len(strata))
    return {s: base + (i < extra) for i, s in enumerate(strata)}


def select_budget(records: Sequence[PromptRecord], n: int | None, seed: int, name: str = "budget",
                  strata: Sequence[str] = Stratum.PRIMARY) -> list[PromptRecord]:
    """Balanced nested subset of ``n`` records (``None`` = all records, unchanged)."""
    if n is None:
        return list(records)
    quota = budget_quota(n, strata)
    out = []
    for s in strata:
        pool = sorted((r for r in records if r.stratum == s),
                      key=lambda r: (stable_hash(seed, name, r.group_id), r.uid))
        if len(pool) < quota[s]:
            raise ValueError(f"budget {n}: needs {quota[s]} {s} prompts, only {len(pool)} available")
        out += pool[:quota[s]]
    return sorted(out, key=lambda r: r.uid)


def budget_summary(records: Sequence[PromptRecord]) -> dict:
    pairs = Counter(r.pair_id for r in records if r.pair_id)
    return {"n": len(records), "by_stratum": dict(Counter(r.stratum for r in records)),
            "n_pair_ids": len(pairs), "n_complete_pairs": sum(c >= 2 for c in pairs.values())}
