"""Cross-dataset duplicate audit and group merging.

Near-duplicates (e.g. JBB behaviours sourced from AdvBench, AdvBench's many
paraphrased bomb/hacking requests) are merged into one behaviour group with
union-find, so a request and its paraphrase can never fall in different splits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from geosteer.data.schema import PromptRecord
from geosteer.utils import normalize_for_dedup


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Deterministic representative: lexicographically smallest root.
        lo, hi = sorted((ra, rb))
        self.parent[hi] = lo
        return True


@dataclass
class DedupReport:
    threshold: float
    n_records: int
    n_groups_before: int
    n_groups_after: int
    exact_pairs: list[tuple[str, str]] = field(default_factory=list)
    near_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    largest_groups: list[tuple[str, int]] = field(default_factory=list)
    cross_source_pairs: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "threshold": self.threshold,
            "n_records": self.n_records,
            "n_groups_before": self.n_groups_before,
            "n_groups_after": self.n_groups_after,
            "n_exact_pairs": len(self.exact_pairs),
            "n_near_pairs": len(self.near_pairs),
            "cross_source_pairs": self.cross_source_pairs,
            "largest_groups": self.largest_groups,
            "near_pair_examples": self.near_pairs[:50],
        }


def _similar_pairs(texts: list[str], threshold: float, chunk: int = 2048) -> list[tuple[int, int, float]]:
    """All (i<j) pairs with TF-IDF word-1/2-gram cosine >= threshold, via chunked sparse products."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit_transform(texts)  # L2-normalised
    XT = X.T.tocsc()
    pairs = []
    for start in range(0, X.shape[0], chunk):
        S = (X[start:start + chunk] @ XT).tocoo()
        rows, cols, vals = S.row + start, S.col, S.data
        keep = (vals >= threshold) & (rows < cols)
        pairs.extend(zip(rows[keep].tolist(), cols[keep].tolist(), np.round(vals[keep], 4).tolist()))
    return pairs


def merge_duplicate_groups(records: list[PromptRecord], threshold: float = 0.8) -> DedupReport:
    """Merge groups of exact (normalised) and near-duplicate records **in place**.

    Each record's original group is kept in ``meta['orig_group_id']``.
    """
    groups_before = {r.group_id for r in records}
    uf = _UnionFind(sorted(groups_before))
    report = DedupReport(threshold=threshold, n_records=len(records), n_groups_before=len(groups_before),
                         n_groups_after=0)

    norm = [normalize_for_dedup(r.text) for r in records]
    by_norm: dict[str, int] = {}
    for i, t in enumerate(norm):
        if t in by_norm:
            j = by_norm[t]
            report.exact_pairs.append((records[j].uid, records[i].uid))
            uf.union(records[j].group_id, records[i].group_id)
        else:
            by_norm[t] = i

    cross = Counter()
    for i, j, s in _similar_pairs(norm, threshold):
        a, b = records[i], records[j]
        report.near_pairs.append((a.uid, b.uid, s))
        if a.source != b.source:
            cross["|".join(sorted((a.source, b.source)))] += 1
        uf.union(a.group_id, b.group_id)
    report.near_pairs.sort(key=lambda p: -p[2])
    report.cross_source_pairs = dict(cross.most_common())

    members = defaultdict(int)
    for r in records:
        new = uf.find(r.group_id)
        if new != r.group_id:
            r.meta.setdefault("orig_group_id", r.group_id)
            r.group_id = new
        members[new] += 1
    report.n_groups_after = len(members)
    report.largest_groups = sorted(members.items(), key=lambda kv: -kv[1])[:10]
    return report
