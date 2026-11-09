"""Data budgets: use only N prompts of a role (e.g. 100 for direction estimation or repair).

Selection is balanced across strata by default (remainder goes to strata in Stratum.PRIMARY
order). Optional fixed ``weights`` (e.g. harmful 1 : alarming_benign 2 : benign 1) shift the
allocation; they must be declared before analysis and kept identical across models and
budgets, since the stratum mix changes what r and d measure. Selection is deterministic and
nested for a fixed allocation: the budget-50 set is a subset of the budget-100 set, so budget
sweeps form learning curves. Within a stratum, prompts are ranked by a seeded hash of
their *group*, so topic-matched pairs (which share a group across strata) tend to be
selected together; ``budget_summary`` reports how many pairs survive intact.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence

from geosteer.data.schema import PromptRecord, Stratum
from geosteer.utils import stable_hash


def budget_quota(n: int, strata: Sequence[str], weights: Mapping[str, float] | None = None) -> dict[str, int]:
    """Split ``n`` over strata proportionally to ``weights`` (default equal), largest-remainder rounding."""
    w = {s: float((weights or {}).get(s, 1.0 if weights is None else 0.0)) for s in strata}
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError(f"invalid budget weights {weights}")
    unknown = set(weights or {}) - set(strata)
    if unknown:
        raise ValueError(f"budget weights for unknown strata {sorted(unknown)}")
    total = sum(w.values())
    exact = {s: n * v / total for s, v in w.items()}
    quota = {s: int(math.floor(x)) for s, x in exact.items()}
    # ties broken by stratum order, so the default reproduces base + 1 for the first strata
    for s in sorted(strata, key=lambda s: (-(exact[s] - quota[s]), list(strata).index(s)))[: n - sum(quota.values())]:
        quota[s] += 1
    return quota


def weights_tag(weights: Mapping[str, float] | None, strata: Sequence[str] = Stratum.PRIMARY) -> str:
    """'' for balanced, else e.g. 'w1-2-1' (weights in stratum order) for file names."""
    if weights is None:
        return ""
    return "w" + "-".join(f"{float(weights.get(s, 0)):g}" for s in strata)


def parse_weights(text: str | None, strata: Sequence[str] = Stratum.PRIMARY) -> dict[str, float] | None:
    """'1:2:1' (stratum order) or 'balanced'/None -> weights mapping."""
    if text is None or text.lower() in ("", "balanced", "none"):
        return None
    parts = [float(x) for x in text.split(":")]
    if len(parts) != len(strata):
        raise ValueError(f"expected {len(strata)} weights ({':'.join(strata)}), got {text!r}")
    return dict(zip(strata, parts))


def select_budget(records: Sequence[PromptRecord], n: int | None, seed: int, name: str = "budget",
                  strata: Sequence[str] = Stratum.PRIMARY,
                  weights: Mapping[str, float] | None = None) -> list[PromptRecord]:
    """Nested subset of ``n`` records, balanced or by fixed ``weights`` (``n=None`` = all records)."""
    if n is None:
        return list(records)
    quota = budget_quota(n, strata, weights)
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
