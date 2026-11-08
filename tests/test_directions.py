import math

import numpy as np
import torch

from geosteer.processing.directions import GroupBootstrap, Labels, angle_deg, contrast_direction, geometry_row


def synthetic(n=400, d=32, angle=60.0, seed=0):
    """harm shifts along e0; refusal (any harm label) shifts along a unit vector at `angle` degrees from e0."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    harm = rng.integers(0, 2, n)
    refused = rng.integers(0, 2, n).astype(float)
    e0, e1 = torch.zeros(d), torch.zeros(d)
    e0[0], e1[1] = 1, 1
    a = math.radians(angle)
    rdir = math.cos(a) * e0 + math.sin(a) * e1
    X = 0.05 * torch.randn(n, d, generator=g)
    X += torch.from_numpy(harm).float()[:, None] * 3 * e0
    X += torch.from_numpy(refused).float()[:, None] * 3 * rdir
    return X, Labels(harm=harm, refused=refused, groups=np.arange(n) // 2)


def test_recovers_known_angle():
    X, lab = synthetic(angle=60)
    row = geometry_row(X, X, lab, n_boot=200)
    assert row["available"]
    assert abs(row["theta"] - 60) < 3
    assert row["theta_lo"] <= row["theta"] <= row["theta_hi"]
    assert row["r_stability"] > 0.99


def _noisy(seed):
    torch.manual_seed(seed)
    X, lab = synthetic(n=80, d=512, angle=30, seed=seed)  # true angle 30 deg, few examples, 512 dims
    return X + 0.6 * torch.randn_like(X), lab


def test_small_cells_inflate_plain_angle_and_debiased_cosine_corrects():
    rows = [geometry_row(X, X, lab, min_count=5, n_boot=0) for X, lab in map(_noisy, range(10))]
    plain = np.mean([r["theta"] for r in rows])
    dc = np.mean([r["theta_dc"] for r in rows])
    assert plain > 60                 # noise pulls the plain angle toward 90 deg
    assert abs(dc - 30) < 6           # noise-corrected estimate is roughly unbiased


def test_bias_diagnostics_and_debiased_interval():
    X, lab = _noisy(0)
    row = geometry_row(X, X, lab, min_count=5, n_boot=500)
    assert row["theta_bias"] > 0 and row["theta_bc"] < row["theta"]
    assert row["theta_basic_lo"] < row["theta_lo"]
    assert row["theta_dc_lo"] <= 30 <= row["theta_dc_hi"]
    assert row["dc_valid_frac"] > 0.9


def test_debiased_cosine_matches_plain_when_noise_free():
    X, lab = synthetic(angle=45)
    row = geometry_row(X, X, lab, n_boot=0)
    assert abs(row["theta_dc"] - row["theta"]) < 1


def test_unconditioned_d_absorbs_refusal_imbalance():
    """If every harmful prompt is refused, d = harm axis + refusal axis (the confound §5 warns about)."""
    n, dim = 400, 32
    harm = np.arange(n) % 2
    X = 0.05 * torch.randn(n, dim, generator=torch.Generator().manual_seed(0))
    X[:, 0] += torch.from_numpy(harm).float() * 3  # harm axis e0
    X[:, 1] += torch.from_numpy(harm).float() * 3  # refusal axis e1, perfectly correlated with harm
    lab = Labels(harm=harm, refused=harm.astype(float), groups=np.arange(n))
    d = contrast_direction(X, lab, "d", min_count=5).vector
    assert abs(float(d[1]) - math.sqrt(0.5)) < 0.05


def test_conditioned_d_recovers_harm_axis():
    X, lab = synthetic(angle=60)
    row = geometry_row(X, X, lab, d_name="d_answered", n_boot=0)
    assert abs(row["theta"] - 60) < 3


def test_min_count_marks_unavailable():
    X, lab = synthetic(n=40)
    d = contrast_direction(X, lab, "r", min_count=50)
    assert not d.available and d.vector is None and "min_count" in d.reason
    lab.refused = None
    assert not geometry_row(X, X, lab, n_boot=0)["available"]


def test_nan_refusal_labels_are_excluded():
    X, lab = synthetic()
    lab.refused[:100] = np.nan
    d = contrast_direction(X, lab, "r", min_count=5)
    assert d.n_pos + d.n_neg == int(((lab.harm == 0) & ~np.isnan(lab.refused)).sum())


def test_bootstrap_identity_weights_reproduce_point_estimate():
    X, lab = synthetic()
    bs = GroupBootstrap(X, lab, n_boot=1, seed=0)
    bs.W = torch.ones_like(bs.W)  # every group once == original sample
    point = contrast_direction(X, lab, "d", min_count=1).vector.double()
    assert torch.allclose(bs.directions("d")[0], point, atol=1e-6)


def test_angle_deg_clamps():
    assert angle_deg(1.0000001) == 0.0 and angle_deg(-1.5) == 180.0


def test_d_matched_uses_only_matched_pairs():
    X, lab = synthetic()
    lab.matched = np.zeros(len(lab.harm), dtype=bool)
    assert not contrast_direction(X, lab, "d_matched", min_count=5).available
    lab.matched[:200] = True
    d = contrast_direction(X, lab, "d_matched", min_count=5)
    assert d.available and d.n_pos + d.n_neg == 200
    row = geometry_row(X, X, lab, d_name="d_matched", n_boot=50)
    assert row["available"] and not np.isnan(row["theta_dc"])
