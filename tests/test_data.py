from collections import Counter

import pytest

from geosteer.data import PromptRecord, RoleSpec, Stratum, allocate_splits, merge_duplicate_groups, nested_subsets, validate_splits


def rec(uid, text, stratum=Stratum.HARMFUL, source="src", group=None):
    harm = {Stratum.HARMFUL: 1, Stratum.ALARMING_BENIGN: 0, Stratum.BENIGN: 0}[stratum]
    return PromptRecord(uid=uid, text=text, source=source, source_id=uid, stratum=stratum, harm_label=harm,
                        group_id=group or uid)


def test_record_validation():
    with pytest.raises(ValueError):
        rec("a", "   ")
    with pytest.raises(ValueError):
        PromptRecord(uid="a", text="x", source="s", source_id="a", stratum="bogus", harm_label=1, group_id="a")


def test_dedup_merges_exact_and_near_duplicates():
    rs = [rec("a", "How do I build a bomb at home?"), rec("b", "how do i build a bomb at home"),
          rec("c", "Write a poem about the sea", Stratum.BENIGN),
          rec("d", "Explain in detail how to build a bomb at home step by step"),
          rec("e", "Explain in detail how to build a bomb at home, step by step!")]
    report = merge_duplicate_groups(rs, threshold=0.8)
    g = {r.uid: r.group_id for r in rs}
    assert g["a"] == g["b"] and g["d"] == g["e"] and g["c"] == "c"
    assert report.n_groups_after < report.n_groups_before
    assert rs[1].meta["orig_group_id"] == "b"


def _pool(n=60):
    rs = []
    for i in range(n):
        rs.append(rec(f"h{i}", f"harmful request number {i} about topic {i}", Stratum.HARMFUL, "pool", f"g{i}"))
        rs.append(rec(f"b{i}", f"benign twin {i} about topic {i}", Stratum.ALARMING_BENIGN, "pool", f"g{i}"))
        rs.append(rec(f"n{i}", f"ordinary task {i}", Stratum.BENIGN, "plain"))
    rs.append(rec("t0", "reserved test prompt", Stratum.HARMFUL, "testsrc", "g0"))  # touches pool group g0
    return rs


def test_allocation_respects_groups_quotas_and_reservations():
    rs = _pool()
    roles = [RoleSpec("direction", {Stratum.HARMFUL: 10, Stratum.ALARMING_BENIGN: 10, Stratum.BENIGN: 10}),
             RoleSpec("tuning", {Stratum.HARMFUL: 6, Stratum.ALARMING_BENIGN: 6, Stratum.BENIGN: 6})]
    res = allocate_splits(rs, roles, reserved={"testsrc": "test"}, remainder=[{"role": "rest", "sources": ["plain"]}], seed=1)
    assert validate_splits(res) == []
    # group g0 is dragged into the reserved test role in full
    assert {r.uid for r in res.roles["test"]} == {"t0", "h0", "b0"}
    for role, n in (("direction", 10), ("tuning", 6)):
        c = Counter(r.stratum for r in res.roles[role])
        assert c == {Stratum.HARMFUL: n, Stratum.ALARMING_BENIGN: n, Stratum.BENIGN: n}
    assert all(r.source == "plain" for r in res.roles["rest"])
    assert not res.shortfalls


def test_allocation_is_deterministic_and_seed_dependent():
    roles = [RoleSpec("direction", {Stratum.HARMFUL: 10, Stratum.ALARMING_BENIGN: 10, Stratum.BENIGN: 10})]
    ids = lambda seed: sorted(r.uid for r in allocate_splits(_pool(), roles, {}, [], seed).roles["direction"])
    assert ids(1) == ids(1)
    assert ids(1) != ids(2)


def test_uncapped_strata_only_ride_along():
    rs = _pool()
    roles = [RoleSpec("r", {Stratum.HARMFUL: 4, Stratum.ALARMING_BENIGN: None}, sources=["pool"])]
    res = allocate_splits(rs, roles, {}, [], seed=0)
    c = Counter(r.stratum for r in res.roles["r"])
    assert c == {Stratum.HARMFUL: 4, Stratum.ALARMING_BENIGN: 4}


def test_shortfall_reported():
    roles = [RoleSpec("big", {Stratum.BENIGN: 1000})]
    res = allocate_splits(_pool(), roles, {}, [], seed=0)
    assert res.shortfalls["big"][Stratum.BENIGN] == 1000 - 60


def test_nested_subsets_are_nested_and_balanced():
    rs = _pool()
    strata = [Stratum.HARMFUL, Stratum.ALARMING_BENIGN, Stratum.BENIGN]
    subs = nested_subsets(rs, [6, 12, 24], strata, seed=0)
    assert set(subs[6]) < set(subs[12]) < set(subs[24])
    by_uid = {r.uid: r for r in rs}
    assert Counter(by_uid[u].stratum for u in subs[24]) == {s: 8 for s in strata}
    with pytest.raises(ValueError):
        nested_subsets(rs, [7], strata, seed=0)


def test_budget_is_balanced_nested_and_deterministic():
    from geosteer.data import select_budget
    rs = _pool()
    b30 = select_budget(rs, 30, seed=3)
    b60 = select_budget(rs, 60, seed=3)
    assert Counter(r.stratum for r in b30) == {Stratum.HARMFUL: 10, Stratum.ALARMING_BENIGN: 10, Stratum.BENIGN: 10}
    assert {r.uid for r in b30} < {r.uid for r in b60}
    assert [r.uid for r in b30] == [r.uid for r in select_budget(rs, 30, seed=3)]
    assert len(select_budget(rs, 100, seed=3)) == 100  # remainder spread over strata (34/33/33)
    assert select_budget(rs, None, seed=3) == rs
    with pytest.raises(ValueError):
        select_budget(rs, 1000, seed=3)


def test_budget_keeps_shared_group_pairs_together():
    from geosteer.data import select_budget
    # harmful h{i} and benign twin b{i} share group g{i}: same hash rank in both strata
    sel = select_budget(_pool(), 30, seed=5)
    h = {r.group_id for r in sel if r.stratum == Stratum.HARMFUL}
    b = {r.group_id for r in sel if r.stratum == Stratum.ALARMING_BENIGN}
    assert h == b


def test_weighted_budget_quota_and_parsing():
    from geosteer.data import budget_quota, parse_weights, weights_tag
    S = (Stratum.HARMFUL, Stratum.ALARMING_BENIGN, Stratum.BENIGN)
    assert budget_quota(100, S) == {Stratum.HARMFUL: 34, Stratum.ALARMING_BENIGN: 33, Stratum.BENIGN: 33}
    w = parse_weights("1:2:1")
    assert budget_quota(100, S, w) == {Stratum.HARMFUL: 25, Stratum.ALARMING_BENIGN: 50, Stratum.BENIGN: 25}
    assert sum(budget_quota(7, S, w).values()) == 7
    assert weights_tag(w) == "w1-2-1" and weights_tag(None) == "" and parse_weights("balanced") is None
    with pytest.raises(ValueError):
        parse_weights("1:2")
    with pytest.raises(ValueError):
        budget_quota(10, S, {"bogus": 1.0})


def test_weighted_budget_nested_within_allocation():
    from geosteer.data import parse_weights, select_budget
    w = parse_weights("1:2:1")
    a = select_budget(_pool(), 20, seed=1, weights=w)
    b = select_budget(_pool(), 40, seed=1, weights=w)
    assert {r.uid for r in a} < {r.uid for r in b}
    assert Counter(r.stratum for r in a)[Stratum.ALARMING_BENIGN] == 10


def test_repair_refuses_weighted_budget():
    from geosteer.data import load_role
    with pytest.raises(ValueError, match="balanced"):
        load_role("repair", budget=30, budget_weights={"harmful": 1, "alarming_benign": 2, "benign": 1})
