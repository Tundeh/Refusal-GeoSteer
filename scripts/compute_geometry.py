"""Layerwise refusal/harmfulness geometry from saved activations + observed refusal labels.

    python scripts/compute_geometry.py --model qwen2.5-1.5b --role direction
    python scripts/compute_geometry.py --model qwen2.5-1.5b --role direction --budget 100
    python scripts/compute_geometry.py --model qwen2.5-1.5b --role direction --budget 100 --budget-weights 1:2:1

Activations and labels are computed once for the whole role; ``--budget N`` estimates the
directions from a nested subset of N prompts (data/budget.py), balanced across strata unless
fixed ``--budget-weights`` (harmful:alarming_benign:benign) are given. Output name:
<role>[_b<N>][_w<weights>].csv. A contrast whose cells fall below ``--min-count`` is reported
as unavailable: at small budgets that is itself a result, not something to work around.

Reads  artifacts/activations/<model>/<variant>/<role>.pt, artifacts/labels/<model>/<variant>/<role>.jsonl
Writes artifacts/geometry/<model>/<variant>/<tag>.csv (one row per layer x contrast pair x position pair)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geosteer.data import Stratum, budget_summary, load_role, parse_weights, weights_tag
from geosteer.processing import ActivationSet, Labels, layerwise_geometry
from geosteer.utils import (artifact_path, get_logger, limit_cpu_threads, load_config, project_path, read_jsonl,
                            write_json)

log = get_logger("geometry")

# (r contrast, d contrast): primary first, then the plan's sensitivity analyses. ("r", "d_matched") at
# t_post/t_post is the refusal-vs-safety-signal [1] construction (safe-only r, topic-matched d).
# r_alarming holds alarming wording fixed as well as harm.
CONTRAST_PAIRS = [("r", "d"), ("r", "d_matched"), ("r_alarming", "d"), ("r_alarming", "d_matched"),
                  ("r", "d_answered"), ("r", "d_refused"), ("r_harmful", "d")]


def geometry_tag(role: str, budget: int | None, weights: dict | None) -> str:
    wt = weights_tag(weights)
    return role + ("" if budget is None else f"_b{budget}") + (f"_{wt}" if wt else "")


def build_labels(selected, acts: ActivationSet, labels: dict) -> Labels:
    recs = {r.uid: r for r in selected}
    # Topic-matched = both members of a harmful/benign pair are in the selected set.
    pair_harm: dict[str, set] = {}
    for r in selected:
        if r.pair_id:
            pair_harm.setdefault(r.pair_id, set()).add(r.harm_label)
    complete = {p for p, hs in pair_harm.items() if hs == {0, 1}}
    return Labels(harm=np.array([recs[u].harm_label for u in acts.uids]),
                  refused=np.array([np.nan if labels[u]["refused"] is None else labels[u]["refused"]
                                    for u in acts.uids], dtype=float),
                  groups=np.array([recs[u].group_id for u in acts.uids]),
                  matched=np.array([recs[u].pair_id in complete for u in acts.uids]),
                  alarming=np.array([recs[u].stratum == Stratum.ALARMING_BENIGN for u in acts.uids]))


def cell_sizes(lab: Labels) -> dict[str, int]:
    cells = {f"harm={h},refused={r}": int(((lab.harm == h) & (lab.refused == r)).sum())
             for h in (0, 1) for r in (0, 1)}
    cells.update({f"alarming,refused={r}": int((lab.alarming & (lab.refused == r)).sum()) for r in (0, 1)})
    cells["refused=NaN"] = int(np.isnan(lab.refused).sum())
    cells["matched"] = int(lab.matched.sum())
    return cells


def run_geometry(model: str, variant: str = "base", role: str = "direction", budget: int | None = None,
                 weights: dict | None = None, contrast_pairs=CONTRAST_PAIRS, layers=None, min_count: int = 20,
                 n_boot: int = 1000, min_valid_frac: float = 0.95, seed: int = 0, budget_seed: int | None = None,
                 acts: ActivationSet | None = None, labels: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Geometry rows for one model variant at one data budget. ``acts``/``labels`` may be passed in to
    avoid reloading in sweeps; ``layers`` restricts to a subset of layers (0-indexed); ``budget_seed``
    draws a different budget subset (default: the split seed)."""
    acts = acts or ActivationSet.load(artifact_path("activations", model, variant, f"{role}.pt"))
    labels = labels or {r["uid"]: r for r in read_jsonl(artifact_path("labels", model, variant, f"{role}.jsonl"))}
    selected = load_role(role, budget=budget, budget_weights=weights, budget_seed=budget_seed)
    keep = {r.uid for r in selected}
    acts = acts.subset([u for u in acts.uids if u in keep])
    if len(acts.uids) != len(keep):
        raise SystemExit(f"{len(keep) - len(acts.uids)} selected prompts have no saved activations")
    absent = [u for u in acts.uids if u not in labels]
    if absent:
        raise SystemExit(f"{len(absent)} activation uids have no refusal label, e.g. {absent[:3]}")
    if layers is not None:
        idx = [acts.layers.index(l) for l in layers]
        acts = ActivationSet(acts.acts[:, :, idx], acts.uids, acts.positions, list(layers), acts.meta)

    lab = build_labels(selected, acts, labels)
    rows = []
    for r_name, d_name in contrast_pairs:
        rows += layerwise_geometry(acts, lab, r_name=r_name, d_name=d_name, min_count=min_count,
                                   n_boot=n_boot, seed=seed)
    df = pd.DataFrame(rows)
    # theta_dc (noise-corrected) is the primary angle; plain theta is inflated toward 90 deg by estimation
    # noise in small cells. A row is unreliable when many bootstrap replicates have less signal than noise.
    if "dc_valid_frac" not in df:
        df["dc_valid_frac"] = np.nan
    df["reliable"] = df.available & (df.dc_valid_frac.fillna(0 if n_boot else 1) >= min_valid_frac)
    # refusal-vs-safety-signal [1] numbers layers 1..n as hidden_states[L] = output of block L-1. Their last
    # layer is post-final-norm; ours is pre-norm, so only layer_1idx < n is directly comparable.
    df["layer_1idx"] = df.layer + 1
    df["budget"] = budget if budget is not None else len(selected)
    df["weights"] = weights_tag(weights) or "balanced"
    meta = {"budget": budget, "weights": weights, "budget_seed": budget_seed, "usage": budget_summary(selected), "cells": cell_sizes(lab),
            "min_count": min_count, "min_valid_frac": min_valid_frac, "n_boot": n_boot, "seed": seed,
            "contrast_pairs": contrast_pairs,
            "activations_meta": {k: v for k, v in acts.meta.items() if k != "seq_lens"}}
    return df, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--role", default="direction")
    ap.add_argument("--budget", type=int, default=None,
                    help="prompts used for direction estimation (default: budgets.<role> in configs/data.yaml)")
    ap.add_argument("--budget-weights", default=None,
                    help='harmful:alarming_benign:benign, e.g. "1:2:1" (default: budget_weights.<role> in config)')
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-valid-frac", type=float, default=0.95,
                    help="min fraction of bootstrap replicates with a defined noise-corrected cosine")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    limit_cpu_threads()

    cfg = load_config(project_path("configs", "data.yaml"))
    budget = args.budget if args.budget is not None else (cfg.get("budgets") or {}).get(args.role)
    weights = parse_weights(args.budget_weights if args.budget_weights is not None
                            else (cfg.get("budget_weights") or {}).get(args.role))

    df, meta = run_geometry(args.model, args.variant, args.role, budget, weights, min_count=args.min_count,
                            n_boot=args.n_boot, min_valid_frac=args.min_valid_frac, seed=args.seed)
    log.info("using %s prompts: %s", "all" if budget is None else budget, meta["usage"])
    log.info("cell sizes: %s", meta["cells"])
    out = artifact_path("geometry", args.model, args.variant, f"{geometry_tag(args.role, budget, weights)}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    write_json(meta, out.with_suffix(".meta.json"))

    for r_name in ("r", "r_alarming"):
        sub = df[(df.r_name == r_name) & (df.d_name == "d")]
        if sub.available.any():
            view = sub[sub.available].pivot(index="layer", columns=["r_pos", "d_pos"], values="theta_dc").round(1)
            log.info("noise-corrected theta(%s, d) in degrees by layer:\n%s", r_name, view.to_string())
            n_unrel = int((~sub[sub.available].reliable).sum())
            if n_unrel:
                log.warning("%d/%d %s rows unreliable (dc_valid_frac < %.2f)", n_unrel, int(sub.available.sum()),
                            r_name, args.min_valid_frac)
        else:
            log.warning("%s vs d unavailable: %s", r_name, sub.reason.iloc[0])
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
