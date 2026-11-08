"""Group-aware split allocation (plan §4).

Allocation unit is the behaviour group (after duplicate merging), never the
single prompt, so paraphrases, topic-matched pairs and XSTest contrast pairs
stay together. Steps:

1. ``reserved``: any group containing a record from a reserved source goes
   wholly to that source's role (keeps final test sets uncontaminated).
2. ``roles`` (in config order): walk unassigned groups in seeded-hash order and
   take a group when every record's source is allowed and every record's
   stratum still has quota. A quota of ``null`` admits that stratum without a
   cap. Groups containing a ``prefer_sources`` source are tried first.
3. ``remainder``: leftover groups whose sources are all listed go to a role.
4. Anything else is ``unassigned``.

Nested subsets (e.g. tuning sets of 24/48/96/192) are carved from a role with a
fixed per-stratum hash order, so smaller sets are prefixes of larger ones.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from geosteer.data.schema import PromptRecord
from geosteer.utils import get_logger, normalize_for_dedup, stable_hash

log = get_logger(__name__)
UNASSIGNED = "unassigned"


@dataclass
class RoleSpec:
    name: str
    quota: dict[str, int | None]
    sources: list[str] | None = None
    prefer_sources: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> "RoleSpec":
        return cls(name=name, quota=dict(cfg["quota"]), sources=cfg.get("sources"),
                   prefer_sources=cfg.get("prefer_sources", []))


@dataclass
class SplitResult:
    assignment: dict[str, str]                    # group_id -> role
    roles: dict[str, list[PromptRecord]]          # role -> records (uid-sorted)
    shortfalls: dict[str, dict[str, int]]         # role -> stratum -> missing count
    reserved_conflicts: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, dict]:
        out = {}
        for role, recs in self.roles.items():
            out[role] = {"n": len(recs), "n_groups": len({r.group_id for r in recs}),
                         "by_stratum": dict(Counter(r.stratum for r in recs)),
                         "by_source": dict(Counter(r.source for r in recs))}
        return out


def _group(records: list[PromptRecord]) -> dict[str, list[PromptRecord]]:
    groups: dict[str, list[PromptRecord]] = defaultdict(list)
    for r in records:
        groups[r.group_id].append(r)
    return groups


def allocate_splits(records: list[PromptRecord], roles: list[RoleSpec], reserved: dict[str, str],
                    remainder: list[dict[str, Any]], seed: int) -> SplitResult:
    groups = _group(records)
    order = sorted(groups, key=lambda g: stable_hash(seed, "split", g))
    assignment: dict[str, str] = {}
    conflicts = []

    # 1. reserved sources (config order of `reserved` decides conflicts)
    for gid in order:
        hits = [reserved[r.source] for r in groups[gid] if r.source in reserved]
        if hits:
            role_order = list(dict.fromkeys(reserved.values()))
            chosen = min(hits, key=role_order.index)
            if len(set(hits)) > 1:
                conflicts.append(f"{gid}: {sorted(set(hits))} -> {chosen}")
            assignment[gid] = chosen

    # 2. quota roles
    shortfalls = {}
    for spec in roles:
        remaining = dict(spec.quota)
        allowed = set(spec.sources) if spec.sources else None
        prefer = set(spec.prefer_sources)
        candidates = [g for g in order if g not in assignment]
        if prefer:
            candidates.sort(key=lambda g: not any(r.source in prefer for r in groups[g]))  # stable sort
        capped = [s for s, v in remaining.items() if v is not None]
        if not capped:
            raise ValueError(f"role {spec.name}: at least one stratum needs a numeric quota")
        for gid in candidates:
            if all(remaining[s] <= 0 for s in capped):
                break
            recs = groups[gid]
            if allowed is not None and any(r.source not in allowed for r in recs):
                continue
            need = Counter(r.stratum for r in recs)
            if any(s not in remaining or (remaining[s] is not None and remaining[s] < n) for s, n in need.items()):
                continue
            if not any(s in capped for s in need):  # uncapped strata ride along, never on their own
                continue
            for s, n in need.items():
                if remaining[s] is not None:
                    remaining[s] -= n
            assignment[gid] = spec.name
        short = {s: v for s, v in remaining.items() if v is not None and v > 0}
        if short:
            shortfalls[spec.name] = short
            log.warning("Role %s short of quota: %s", spec.name, short)

    # 3. remainder rules
    for rule in remainder:
        allowed = set(rule["sources"])
        for gid in order:
            if gid not in assignment and all(r.source in allowed for r in groups[gid]):
                assignment[gid] = rule["role"]

    # 4. leftovers
    for gid in order:
        assignment.setdefault(gid, UNASSIGNED)

    by_role: dict[str, list[PromptRecord]] = defaultdict(list)
    for gid, role in assignment.items():
        by_role[role].extend(groups[gid])
    roles_out = {role: sorted(recs, key=lambda r: r.uid) for role, recs in by_role.items()}
    for r_list in roles_out.values():
        for r in r_list:
            r.meta["role"] = assignment[r.group_id]
    return SplitResult(assignment=assignment, roles=roles_out, shortfalls=shortfalls, reserved_conflicts=conflicts)


def nested_subsets(records: list[PromptRecord], sizes: list[int], strata: list[str], seed: int,
                   name: str = "subset") -> dict[int, list[str]]:
    """Balanced nested subsets: size n takes the first n/len(strata) of each stratum's hash order."""
    per = {}
    for s in strata:
        pool = [r for r in records if r.stratum == s]
        per[s] = sorted(pool, key=lambda r: stable_hash(seed, name, r.uid))
    out = {}
    for n in sorted(sizes):
        if n % len(strata):
            raise ValueError(f"subset size {n} not divisible by {len(strata)} strata")
        k = n // len(strata)
        short = {s: len(p) for s, p in per.items() if len(p) < k}
        if short:
            raise ValueError(f"{name}: size {n} needs {k}/stratum, have {short}")
        out[n] = sorted(r.uid for s in strata for r in per[s][:k])
    return out


def validate_splits(result: SplitResult) -> list[str]:
    """Return a list of leakage problems (empty if clean)."""
    problems = []
    group_roles = defaultdict(set)
    text_roles = defaultdict(set)
    for role, recs in result.roles.items():
        for r in recs:
            group_roles[r.group_id].add(role)
            text_roles[normalize_for_dedup(r.text)].add(role)
    problems += [f"group {g} in roles {sorted(rs)}" for g, rs in group_roles.items() if len(rs) > 1]
    problems += [f"text {t[:60]!r} in roles {sorted(rs)}" for t, rs in text_roles.items() if len(rs) > 1]
    return problems
