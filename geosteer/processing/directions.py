"""Difference-of-means directions and layerwise refusal/harmfulness geometry (plan §5).

Contrasts (all at one layer and token position):
  d            harmful            - safe                     (unconditioned harmfulness)
  r            safe & refused     - safe & answered          (safe-only refusal, [1])
  r_harmful    harmful & refused  - harmful & answered       (sensitivity analysis)
  r_alarming   alarming-benign & refused - alarming-benign & answered (harm *and* alarming wording
               held fixed; removes the OR-Bench-vs-Alpaca style confound in r)
  d_refused    harmful & refused  - safe & refused           (harmfulness within a refusal stratum)
  d_answered   harmful & answered - safe & answered
  d_matched    harmful - safe, topic-matched pairs only (refusal-vs-safety-signal [1] style d)

``refused`` is the *observed* refusal label for this model variant. A contrast
whose cells have fewer than ``min_count`` examples is reported as unavailable,
never substituted with another model's direction.

Bootstrap: behaviour groups are resampled jointly (one multinomial draw per
replicate shared by all contrasts), so r and d keep their dependence.
"""

from __future__ import annotations

import itertools
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
    "r_alarming": ({"harm": 0, "alarming": True, "refused": 1}, {"harm": 0, "alarming": True, "refused": 0}),
    "d_refused": ({"harm": 1, "refused": 1}, {"harm": 0, "refused": 1}),
    "d_answered": ({"harm": 1, "refused": 0}, {"harm": 0, "refused": 0}),
    "d_matched": ({"harm": 1, "matched": True}, {"harm": 0, "matched": True}),
}


FLAG_KEYS = ("matched", "alarming")  # boolean per-example flags a contrast cell may condition on


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
    matched: np.ndarray | None = None   # [N] bool: member of a complete topic-matched harmful/benign pair
    alarming: np.ndarray | None = None  # [N] bool: alarming-but-benign stratum

    def flag(self, key: str) -> np.ndarray:
        v = getattr(self, key)
        return v if v is not None else np.zeros(len(self.harm), dtype=bool)

    def cell(self, spec: dict) -> np.ndarray:
        m = self.harm == spec["harm"]
        for key in FLAG_KEYS:
            if key in spec:
                m &= self.flag(key) == spec[key]
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


def bootstrap_angle(boot_a: torch.Tensor, boot_b: torch.Tensor, alpha: float = 0.05,
                    theta_point: float | None = None) -> dict:
    """Bootstrap summary of the angle between paired replicate directions.

    Estimation noise in difference-of-means vectors pushes angles toward 90 deg, and
    replicates carry *more* noise than the point estimate, so the percentile interval
    (``theta_lo/hi``) is shifted. With ``theta_point`` we also report the bootstrap
    bias estimate (mean replicate - point), a bias-corrected angle, and the basic
    bootstrap interval ``2*theta - q`` (``theta_basic_lo/hi``), which is the one to use
    for small contrast cells.
    """
    cos = (boot_a * boot_b).sum(-1)
    cos = cos[~torch.isnan(cos)].numpy()
    nan = float("nan")
    if cos.size == 0:
        return {"cos_lo": nan, "cos_hi": nan, "theta_lo": nan, "theta_hi": nan, "n_valid": 0}
    lo, hi = np.quantile(cos, [alpha / 2, 1 - alpha / 2])
    out = {"cos_lo": float(lo), "cos_hi": float(hi), "theta_lo": angle_deg(hi), "theta_hi": angle_deg(lo),
           "cos_sd": float(cos.std(ddof=1)) if cos.size > 1 else nan, "n_valid": int(cos.size)}
    if theta_point is not None:
        thetas = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        q_lo, q_hi = np.quantile(thetas, [alpha / 2, 1 - alpha / 2])
        bias = float(thetas.mean() - theta_point)
        out.update(theta_sd=float(thetas.std(ddof=1)) if thetas.size > 1 else nan, theta_bias=bias,
                   theta_bc=float(np.clip(theta_point - bias, 0, 180)),
                   theta_basic_lo=float(np.clip(2 * theta_point - q_hi, 0, 180)),
                   theta_basic_hi=float(np.clip(2 * theta_point - q_lo, 0, 180)))
    return out


def direction_stability(boot: torch.Tensor, point: torch.Tensor) -> float:
    """Median cosine between bootstrap directions and the point estimate (1 = perfectly stable)."""
    c = boot @ point.double()
    c = c[~torch.isnan(c)]
    return float(c.median()) if c.numel() else float("nan")


class DebiasedCosine:
    """Noise-corrected cosine between two difference-of-means contrasts.

    Write each example as x_i = m_c(i) + e_i, with c the fine cell (harm, refused in
    {0, 1, NaN}, and each flag in FLAG_KEYS). A contrast is r_hat = sum_i a_i x_i (a_i = +1/n_pos, -1/n_neg, 0), so
    E<r_hat, d_hat> = <r, d> + sum_c n_c a_c b_c tr Cov_c(x_r, x_d): estimation noise
    inflates both norms (and, where contrasts share examples, the inner product),
    pulling the plain cosine toward 0 / the angle toward 90 deg. We subtract the
    within-cell (cross-)covariance traces from the inner product and squared norms.
    For cross-token geometry the cross-covariance between positions is used.

    Group bootstrap replicates reuse the same per-group sufficient statistics.
    Within-group correlation (e.g. paired JBB prompts) is ignored in the bias term.
    """

    def __init__(self, X_r: torch.Tensor, X_d: torch.Tensor, labels: Labels, r_name: str, d_name: str):
        self.r_name, self.d_name = r_name, d_name
        _, g_idx = np.unique(labels.groups, return_inverse=True)
        G = int(g_idx.max()) + 1
        g_idx = torch.from_numpy(g_idx)
        Xr, Xd = X_r.double(), X_d.double()
        ref = labels.refused if labels.refused is not None else np.full(len(labels.harm), np.nan)
        ref_cat = np.where(np.isnan(ref), -1, np.nan_to_num(ref, nan=-1)).astype(int)
        (rp, rn), (dp, dn) = CONTRASTS[r_name], CONTRASTS[d_name]
        flags = [labels.flag(k) for k in FLAG_KEYS]
        self.cells = []
        for h, rc, *fv in itertools.product((0, 1), (-1, 0, 1), *[(False, True)] * len(FLAG_KEYS)):
            m = (labels.harm == h) & (ref_cat == rc)
            for f, v in zip(flags, fv):
                m &= f == v
            if not m.any():
                continue
            rep = {"harm": h, "refused": rc if rc >= 0 else math.nan, **dict(zip(FLAG_KEYS, fv))}
            a = 1 if self._match(rp, rep) else (-1 if self._match(rn, rep) else 0)
            b = 1 if self._match(dp, rep) else (-1 if self._match(dn, rep) else 0)
            if a == 0 and b == 0:
                continue
            mt = torch.from_numpy(m)
            gi = g_idx[mt]
            def gsum(v):
                out = torch.zeros((G,) + v.shape[1:], dtype=torch.float64)
                return out.index_add_(0, gi, v)
            self.cells.append({
                "a": a, "b": b, "n": torch.bincount(gi, minlength=G).double(),
                "Sr": gsum(Xr[mt]), "Sd": gsum(Xd[mt]),
                "Qrr": gsum((Xr[mt] ** 2).sum(-1)), "Qdd": gsum((Xd[mt] ** 2).sum(-1)),
                "Qrd": gsum((Xr[mt] * Xd[mt]).sum(-1))})
        self.G = G

    @staticmethod
    def _match(spec: dict, rep: dict) -> bool:
        if rep["harm"] != spec["harm"]:
            return False
        if any(k in spec and rep[k] != spec[k] for k in FLAG_KEYS):
            return False
        return "refused" not in spec or rep["refused"] == spec["refused"]

    def __call__(self, W: torch.Tensor | None = None) -> torch.Tensor:
        """Debiased cosines, one per row of W ([B, G] group multiplicities; None = original sample)."""
        W = torch.ones(1, self.G, dtype=torch.float64) if W is None else W.double()
        B = W.shape[0]
        stats = []
        for c in self.cells:
            # Replicate weights act as frequency weights w_i (shared within a group). For iid x_i,
            # E||sum w x / sum w - m||^2 = v * sum w^2 / (sum w)^2, and the unbiased variance uses
            # denominator sum w - sum w^2 / sum w. Both reduce to the usual n and n-1 when W = 1.
            n = W @ c["n"]
            n2 = (W * W) @ c["n"]
            Sr, Sd = W @ c["Sr"], W @ c["Sd"]
            den = (n - n2 / n.clamp_min(1)).clamp_min(1e-12)
            v = {k: (W @ c["Q" + k] - (S1 * S2).sum(-1) / n.clamp_min(1)) / den
                 for k, S1, S2 in (("rr", Sr, Sr), ("dd", Sd, Sd), ("rd", Sr, Sd))}
            stats.append((c, n2, Sr, Sd, v, n))
        def contrast(key, S_key):
            pos_n = sum(st[5] for st in stats if st[0][key] == 1)
            neg_n = sum(st[5] for st in stats if st[0][key] == -1)
            S_idx = 2 if S_key == "Sr" else 3
            mu_p = sum(st[S_idx] for st in stats if st[0][key] == 1) / pos_n[:, None]
            mu_n = sum(st[S_idx] for st in stats if st[0][key] == -1) / neg_n[:, None]
            w = [(c[key] / pos_n if c[key] == 1 else (-1 / neg_n if c[key] == -1 else torch.zeros(B, dtype=torch.float64)))
                 for c, *_ in stats]
            return mu_p - mu_n, w, pos_n, neg_n
        r, wa, rpn, rnn = contrast("a", "Sr")
        d, wb, dpn, dnn = contrast("b", "Sd")
        bias = {k: torch.zeros(B, dtype=torch.float64) for k in ("rr", "dd", "rd")}
        for (c, n2, _, _, v, _), a, b in zip(stats, wa, wb):
            bias["rr"] += n2 * a * a * v["rr"]
            bias["dd"] += n2 * b * b * v["dd"]
            bias["rd"] += n2 * a * b * v["rd"]
        nr = (r * r).sum(-1) - bias["rr"]
        nd = (d * d).sum(-1) - bias["dd"]
        cos = ((r * d).sum(-1) - bias["rd"]) / (nr.clamp_min(0) * nd.clamp_min(0)).sqrt()
        bad = (nr <= 0) | (nd <= 0) | (rpn == 0) | (rnn == 0) | (dpn == 0) | (dnn == 0)
        cos[bad] = float("nan")
        return cos.clamp(-1, 1)


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
        row.update(bootstrap_angle(br, bd, theta_point=row["theta"]))
        row.update(r_stability=direction_stability(br, r.vector), d_stability=direction_stability(bd, d.vector))
    dc = DebiasedCosine(X_r, X_d, labels, r_name, d_name)
    c_dc = float(dc()[0])
    row.update(cos_dc=c_dc, theta_dc=angle_deg(c_dc) if not math.isnan(c_dc) else float("nan"))
    if n_boot:
        reps = dc(bs_r.W).numpy()
        ok = reps[~np.isnan(reps)]
        row["dc_valid_frac"] = float(ok.size / reps.size)
        if ok.size:
            lo, hi = np.quantile(ok, [0.025, 0.975])
            row.update(theta_dc_lo=angle_deg(hi), theta_dc_hi=angle_deg(lo))
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
