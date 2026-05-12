"""Statistical and dχ-privacy tests for the Laplace sampler.

Most tests here are *statistical* — they run thousands of samples and assert
that empirical moments match the closed-form values within tolerance. The
seeded RNG keeps them deterministic.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from nanobot.privacy.metric_dp import (
    laplace_noise,
    noise_norm_mean,
    noise_norm_variance,
)

_SAMPLES = 5000


def _rng(seed: int = 0xC0FFEE) -> random.Random:
    """Deterministic RNG so every assertion is reproducible."""
    return random.Random(seed)


# --- input validation -----------------------------------------------------------------


@pytest.mark.parametrize("bad_dim", [0, -1, 1.5, "10", None])
def test_rejects_invalid_dim(bad_dim):
    with pytest.raises(ValueError):
        laplace_noise(bad_dim, epsilon=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_eps", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_epsilon(bad_eps):
    with pytest.raises(ValueError):
        laplace_noise(8, epsilon=bad_eps)


# --- shape ----------------------------------------------------------------------------


def test_returns_correct_dimension():
    v = laplace_noise(768, epsilon=1.0, rng=_rng())
    assert len(v) == 768
    assert all(isinstance(x, float) for x in v)


# --- statistical moments --------------------------------------------------------------


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


@pytest.mark.parametrize("dim, eps", [(2, 1.0), (8, 2.0), (32, 0.5)])
def test_radius_matches_gamma_moments(dim, eps):
    """E[‖η‖] = d/ε  and  Var[‖η‖] = d/ε² for a Gamma(d, 1/ε) radius."""
    rng = _rng()
    norms = [_norm(laplace_noise(dim, eps, rng=rng)) for _ in range(_SAMPLES)]

    mean_emp = statistics.mean(norms)
    var_emp = statistics.variance(norms)

    mean_theo = noise_norm_mean(dim, eps)
    var_theo = noise_norm_variance(dim, eps)

    # Central limit theorem: standard error of the mean ≈ √(var/N).
    se_mean = math.sqrt(var_theo / _SAMPLES)
    # Allow ~5 SE — extremely generous, catches a real bias without flakiness.
    assert abs(mean_emp - mean_theo) < 5 * se_mean, (
        f"empirical mean {mean_emp:.4f} too far from theory {mean_theo:.4f}"
    )
    # Variance has a sloppier finite-sample distribution; allow ±15 %.
    assert abs(var_emp - var_theo) / var_theo < 0.15, (
        f"empirical var {var_emp:.4f} too far from theory {var_theo:.4f}"
    )


def test_components_are_zero_mean_and_isotropic():
    """Each coordinate of η has mean ≈ 0 and equal variance (isotropy)."""
    dim, eps = 4, 1.0
    rng = _rng()
    samples = [laplace_noise(dim, eps, rng=rng) for _ in range(_SAMPLES)]

    per_dim_mean = [statistics.mean(s[i] for s in samples) for i in range(dim)]
    per_dim_var = [statistics.variance(s[i] for s in samples) for i in range(dim)]

    # Zero mean
    for m in per_dim_mean:
        assert abs(m) < 0.15, f"per-dim mean {m:.4f} not centred"

    # Isotropy: all per-dim variances within ~15 % of each other
    v_avg = sum(per_dim_var) / dim
    for v in per_dim_var:
        assert abs(v - v_avg) / v_avg < 0.15, (
            f"per-dim variance {v:.4f} too far from average {v_avg:.4f} (anisotropy)"
        )


def test_off_diagonal_covariance_near_zero():
    """The covariance matrix of η should be (close to) a scalar multiple of I."""
    dim, eps = 4, 1.0
    rng = _rng()
    samples = [laplace_noise(dim, eps, rng=rng) for _ in range(_SAMPLES)]

    # Subtract per-dim mean
    means = [statistics.mean(s[i] for s in samples) for i in range(dim)]
    centred = [[s[i] - means[i] for i in range(dim)] for s in samples]

    diag_var = statistics.mean(
        sum(s[i] * s[i] for s in centred) / _SAMPLES for i in range(dim)
    )
    # Average off-diagonal covariance — should be small compared to diagonal.
    off_diag_sum = 0.0
    pairs = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            cov_ij = sum(s[i] * s[j] for s in centred) / _SAMPLES
            off_diag_sum += abs(cov_ij)
            pairs += 1
    off_diag_avg = off_diag_sum / pairs

    assert off_diag_avg < 0.1 * diag_var, (
        f"off-diag covariance {off_diag_avg:.4f} too large relative to "
        f"diagonal {diag_var:.4f}"
    )


# --- ε controls noise size -----------------------------------------------------------


def test_larger_epsilon_shrinks_noise():
    """ε is the privacy budget: larger ε → less noise → smaller ‖η‖."""
    rng = _rng()
    small = [_norm(laplace_noise(8, epsilon=0.5, rng=rng)) for _ in range(2000)]
    large = [_norm(laplace_noise(8, epsilon=8.0, rng=rng)) for _ in range(2000)]
    assert statistics.mean(small) > 8 * statistics.mean(large), (
        "ε=8 should produce ~16× smaller norms than ε=0.5, observed "
        f"mean={statistics.mean(small):.2f} vs {statistics.mean(large):.2f}"
    )


# --- empirical dχ-privacy verification ------------------------------------------------


def test_empirical_dchi_privacy_bound():
    """Pr[M(x) ∈ B] / Pr[M(x') ∈ B] ≤ exp(ε · ‖x − x'‖) for any bucket B.

    We test in 1-D for tractability: x = 0, x' = Δ. The mechanism is
    M(z) = z + η. We bin the outputs and check the density ratio against
    the theoretical bound.
    """
    dim, eps, delta = 1, 1.0, 0.5
    rng = _rng()
    bound = math.exp(eps * delta)

    x_samples = [laplace_noise(dim, eps, rng=rng)[0] for _ in range(20_000)]
    xp_samples = [
        laplace_noise(dim, eps, rng=rng)[0] + delta for _ in range(20_000)
    ]

    # Bucketize in [-3, 3] with 0.25-wide bins; ignore tail buckets with <50 samples
    # (their empirical ratio is noisy and not informative).
    lo, hi, step = -3.0, 3.0, 0.25
    n_bins = int((hi - lo) / step)
    cnt_x = [0] * n_bins
    cnt_xp = [0] * n_bins
    for v in x_samples:
        i = int((v - lo) / step)
        if 0 <= i < n_bins:
            cnt_x[i] += 1
    for v in xp_samples:
        i = int((v - lo) / step)
        if 0 <= i < n_bins:
            cnt_xp[i] += 1

    violations = 0
    inspected = 0
    for i in range(n_bins):
        if min(cnt_x[i], cnt_xp[i]) < 50:
            continue
        inspected += 1
        ratio = max(cnt_x[i], cnt_xp[i]) / min(cnt_x[i], cnt_xp[i])
        # Use a 1.5× slack to absorb finite-sample noise.
        if ratio > 1.5 * bound:
            violations += 1

    assert inspected >= 4, "not enough well-populated bins to verify"
    assert violations == 0, (
        f"empirical density ratio exceeded {1.5 * bound:.2f} in "
        f"{violations}/{inspected} well-populated bins"
    )


# --- determinism ---------------------------------------------------------------------


def test_seeded_rng_produces_identical_samples():
    a = laplace_noise(16, epsilon=1.0, rng=_rng(42))
    b = laplace_noise(16, epsilon=1.0, rng=_rng(42))
    assert a == b
