"""Multivariate Laplace noise sampler for the dχ-privacy (Metric-DP) path.

The privacy guarantee that M3 actually delivers is **(ε)-dχ-privacy**
(Andrés et al. 2013 "Geo-indistinguishability"; Feyisetan et al. WSDM 2020
"Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate
Perturbations"). Informally:

    Pr[M(x) ∈ S] ≤ exp(ε · d(x, x')) · Pr[M(x') ∈ S]    for any inputs x, x'.

For the mechanism M(x) = x + η with η drawn from this module's
:func:`laplace_noise`, the guarantee is provable from the density:

    f(η)  ∝  exp(-ε · ‖η‖₂)

A naive implementation that samples each coordinate from a 1-D Laplace would
give density ∝ exp(-ε · ‖η‖₁) — i.e. L1, not L2 — which yields a *different*
privacy notion and breaks the standard proof. We use the **polar
decomposition** instead:

    1. Sample a direction  u  uniformly on the unit sphere  S^(d-1).
    2. Sample a radius     r ~ Gamma(shape=d, scale=1/ε).
    3. Return              η = r · u.

The Jacobian of the change to spherical coordinates is r^(d-1) dr dΩ, so the
joint density factors:

      f(η)  ∝  exp(-ε r)
                                 ← spherical density on direction (uniform)
      ──────────────────────────────────────────
      density on (r, u) ∝ r^(d-1) · exp(-ε r) · 1

The radius marginal is exactly the Gamma(d, 1/ε) kernel above; the direction
marginal is uniform. Multiplying back to Cartesian recovers
f(η) ∝ exp(-ε ‖η‖). ∎

Why this matters for the rest of M3
-----------------------------------

The token-level transform (M3 step 3) embeds each sensitive token into ℝ^d,
adds an :func:`laplace_noise` sample, and projects to the nearest vocabulary
embedding. By post-processing the noisy point survives the projection: the
final discrete token is still ε-dχ-private. Compose enough of those (privacy
accountant, M3 step 4) and you get a session-level budget you can spend.
"""

from __future__ import annotations

import math
import random as _random
from typing import Optional

__all__ = ["laplace_noise", "noise_norm_mean", "noise_norm_variance"]


def laplace_noise(
    dim: int,
    epsilon: float,
    *,
    rng: Optional[_random.Random] = None,
) -> list[float]:
    """Sample a d-dimensional multivariate Laplace vector.

    Density: ``f(η) ∝ exp(-ε · ‖η‖₂)``.

    Parameters
    ----------
    dim:
        Embedding dimension. Must be a positive integer.
    epsilon:
        Privacy budget. Strictly positive; larger ε = less noise = weaker
        privacy. Typical M3 ranges: 1 (strong) … 16 (utility-leaning).
    rng:
        Optional :class:`random.Random` instance. Pass one in tests for
        determinism; in production use the module-level default RNG.

    Returns
    -------
    list[float]
        A length-``dim`` vector. Norms follow Gamma(d, 1/ε); directions are
        uniform on the unit sphere.

    Raises
    ------
    ValueError
        On non-positive ``dim`` or ``epsilon``.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"dim must be a positive int, got {dim!r}")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError(f"epsilon must be positive and finite, got {epsilon!r}")

    r = rng or _random
    # Direction: normalize a d-dim standard normal (Marsaglia, Box-Muller flavour).
    while True:
        z = [r.gauss(0.0, 1.0) for _ in range(dim)]
        norm_z = math.sqrt(sum(x * x for x in z))
        if norm_z > 0:
            break
        # Astronomically unlikely (P ≈ 0 in continuous case) — retry to be safe.
    direction = [x / norm_z for x in z]

    # Radius: Gamma(shape=d, scale=1/ε). Python's gammavariate uses
    # (alpha=shape, beta=scale) convention with density ∝ x^(α-1) exp(-x/β).
    radius = r.gammavariate(dim, 1.0 / epsilon)

    return [radius * x for x in direction]


# --- closed-form moments, exposed so tests can assert against them ---


def noise_norm_mean(dim: int, epsilon: float) -> float:
    """Theoretical E[‖η‖] for a Gamma(dim, 1/ε) radius."""
    return dim / epsilon


def noise_norm_variance(dim: int, epsilon: float) -> float:
    """Theoretical Var[‖η‖] for a Gamma(dim, 1/ε) radius."""
    return dim / (epsilon * epsilon)
