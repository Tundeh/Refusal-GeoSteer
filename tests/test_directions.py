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
