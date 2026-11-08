"""Difference-of-means directions and layerwise refusal/harmfulness geometry (plan §5).

Contrasts (all at one layer and token position):
  d            harmful            - safe                     (unconditioned harmfulness)
  r            safe & refused     - safe & answered          (safe-only refusal, [1])
  r_harmful    harmful & refused  - harmful & answered       (sensitivity analysis)
  d_refused    harmful & refused  - safe & refused           (harmfulness within a refusal stratum)
  d_answered   harmful & answered - safe & answered

``refused`` is the *observed* refusal label for this model variant. A contrast
whose cells have fewer than ``min_count`` examples is reported as unavailable,
never substituted with another model's direction.

Bootstrap: behaviour groups are resampled jointly (one multinomial draw per
replicate shared by all contrasts), so r and d keep their dependence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

CONTRASTS: dict[str, tuple[dict, dict]] = {
    # name: (positive cell, negative cell); cells filter on harm / refused (None = any)
    "d": ({"harm": 1}, {"harm": 0}),
    "r": ({"harm": 0, "refused": 1}, {"harm": 0, "refused": 0}),
    "r_harmful": ({"harm": 1, "refused": 1}, {"harm": 1, "refused": 0}),
    "d_refused": ({"harm": 1, "refused": 1}, {"harm": 0, "refused": 1}),
    "d_answered": ({"harm": 1, "refused": 0}, {"harm": 0, "refused": 0}),
}


def unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(unit(a.double()) @ unit(b.double()))


def angle_deg(cos_value: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))


@dataclass
class Labels:
    harm: np.ndarray                  # [N] in {0,1}
    refused: np.ndarray | None        # [N] in {0,1} or NaN (unlabelled); None if not yet available
    groups: np.ndarray                # [N] behaviour-group ids

    def cell(self, spec: dict) -> np.ndarray:
        m = self.harm == spec["harm"]
        if "refused" in spec:
            if self.refused is None:
                return np.zeros_like(m)
            m &= self.refused == spec["refused"]   # NaN never matches
        return m


@dataclass
class Direction:
    name: str
    vector: torch.Tensor | None       # unit vector, None if unavailable
    raw_norm: float                   # ||mu_pos - mu_neg|| before normalisation
    n_pos: int
    n_neg: int
    available: bool
    reason: str = ""


def contrast_direction(X: torch.Tensor, labels: Labels, name: str, min_count: int = 20) -> Direction:
    pos_spec, neg_spec = CONTRASTS[name]
    pos, neg = labels.cell(pos_spec), labels.cell(neg_spec)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if min(n_pos, n_neg) < min_count:
        return Direction(name, None, float("nan"), n_pos, n_neg, False,
                         f"cell sizes ({n_pos}, {n_neg}) below min_count={min_count}")
    Xd = X.double()
    diff = Xd[torch.from_numpy(pos)].mean(0) - Xd[torch.from_numpy(neg)].mean(0)
    return Direction(name, unit(diff).float(), float(diff.norm()), n_pos, n_neg, True)


class GroupBootstrap:
    """Fast joint group bootstrap of difference-of-means directions via per-group cell sums."""

    def __init__(self, X: torch.Tensor, labels: Labels, n_boot: int = 1000, seed: int = 0):
        self.labels = labels
        _, g_idx = np.unique(labels.groups, return_inverse=True)
        self.n_groups = int(g_idx.max()) + 1
        self.g_idx = torch.from_numpy(g_idx)
        self.X = X.double()
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.n_groups, size=(n_boot, self.n_groups))
        W = np.zeros((n_boot, self.n_groups))
        np.add.at(W, (np.arange(n_boot)[:, None], draws), 1.0)
        self.W = torch.from_numpy(W)  # [B, G] resample multiplicities
        self._cells: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _cell_sums(self, spec: dict) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(sorted(spec.items()))
        if key not in self._cells:
            m = torch.from_numpy(self.labels.cell(spec))
            S = torch.zeros(self.n_groups, self.X.shape[1], dtype=torch.float64)
            S.index_add_(0, self.g_idx[m], self.X[m])
            n = torch.bincount(self.g_idx[m], minlength=self.n_groups).double()
            self._cells[key] = (S, n)
        return self._cells[key]

    def directions(self, name: str) -> torch.Tensor:
        """[B, d] unit directions for one contrast (NaN rows where a cell is empty in a replicate)."""
        pos_spec, neg_spec = CONTRASTS[name]
        (Sp, np_), (Sn, nn_) = self._cell_sums(pos_spec), self._cell_sums(neg_spec)
        cp, cn = self.W @ np_, self.W @ nn_
        diff = (self.W @ Sp) / cp[:, None] - (self.W @ Sn) / cn[:, None]
        out = unit(diff)
        out[(cp == 0) | (cn == 0)] = float("nan")
        return out


def bootstrap_angle(boot_a: torch.Tensor, boot_b: torch.Tensor, alpha: float = 0.05) -> dict:
    cos = (boot_a * boot_b).sum(-1)
    cos = cos[~torch.isnan(cos)].numpy()
    if cos.size == 0:
        return {"cos_lo": float("nan"), "cos_hi": float("nan"), "theta_lo": float("nan"),
                "theta_hi": float("nan"), "n_valid": 0}
    lo, hi = np.quantile(cos, [alpha / 2, 1 - alpha / 2])
    return {"cos_lo": float(lo), "cos_hi": float(hi), "theta_lo": angle_deg(hi), "theta_hi": angle_deg(lo),
            "cos_sd": float(cos.std(ddof=1)) if cos.size > 1 else float("nan"), "n_valid": int(cos.size)}


def direction_stability(boot: torch.Tensor, point: torch.Tensor) -> float:
    """Median cosine between bootstrap directions and the point estimate (1 = perfectly stable)."""
    c = boot @ point.double()
    c = c[~torch.isnan(c)]
    return float(c.median()) if c.numel() else float("nan")


def geometry_row(X_r: torch.Tensor, X_d: torch.Tensor, labels: Labels, r_name: str = "r", d_name: str = "d",
                 min_count: int = 20, n_boot: int = 1000, seed: int = 0) -> dict:
    """Angle between refusal contrast (from X_r) and harmfulness contrast (from X_d).

    Pass the same tensor twice for same-token geometry; different positions for
    the cross-token r(t_post) vs d(t_inst) comparison.
    """
    r = contrast_direction(X_r, labels, r_name, min_count)
    d = contrast_direction(X_d, labels, d_name, min_count)
    row = {"r_name": r_name, "d_name": d_name, "n_r_pos": r.n_pos, "n_r_neg": r.n_neg, "n_d_pos": d.n_pos,
           "n_d_neg": d.n_neg, "r_raw_norm": r.raw_norm, "d_raw_norm": d.raw_norm,
           "available": r.available and d.available, "reason": "; ".join(x for x in (r.reason, d.reason) if x)}
    if not row["available"]:
        return row
    c = cosine(r.vector, d.vector)
    row.update(cos=c, theta=angle_deg(c))
    if n_boot:
        bs_r = GroupBootstrap(X_r, labels, n_boot, seed)
        bs_d = bs_r if X_d is X_r else GroupBootstrap(X_d, labels, n_boot, seed)  # same seed -> same group draws
        br, bd = bs_r.directions(r_name), bs_d.directions(d_name)
        row.update(bootstrap_angle(br, bd))
        row.update(r_stability=direction_stability(br, r.vector), d_stability=direction_stability(bd, d.vector))
    return row


def layerwise_geometry(actset, labels: Labels, position_pairs: Sequence[tuple[str, str]] = (("t_inst", "t_inst"), ("t_post", "t_post"), ("t_post", "t_inst")),
                       r_name: str = "r", d_name: str = "d", min_count: int = 20, n_boot: int = 1000,
                       seed: int = 0) -> list[dict]:
    """One row per (layer, r-position, d-position). ``actset`` is an extraction.ActivationSet
    whose uid order matches ``labels``."""
    n_total = actset.meta.get("n_layers_total", max(actset.layers) + 1)
    rows = []
    for layer in actset.layers:
        for r_pos, d_pos in position_pairs:
            row = geometry_row(actset.get(r_pos, layer), actset.get(d_pos, layer), labels, r_name, d_name,
                               min_count, n_boot, seed)
            row.update(layer=layer, rel_depth=layer / max(n_total - 1, 1), r_pos=r_pos, d_pos=d_pos,
                       same_token=r_pos == d_pos)
            rows.append(row)
    return rows
