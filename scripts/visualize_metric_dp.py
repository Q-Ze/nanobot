"""Visualize the multivariate Laplace sampler — both numerically and graphically.

Run:
    python scripts/visualize_metric_dp.py
    python scripts/visualize_metric_dp.py --dim 2 --epsilons 0.5,1,5
    python scripts/visualize_metric_dp.py --png out.png    # only if matplotlib installed

What you'll see:
  • Per-ε numeric summary (mean / std of ‖η‖ vs theory) — proves the
    sampler matches Gamma(d, 1/ε).
  • An ASCII histogram of ‖η‖ next to a unicode "scatter" of 2-D points
    so you can eyeball isotropy and ε's effect on spread.
  • Optional PNG dump if matplotlib is available — same plots, prettier.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys

from nanobot.privacy.metric_dp import (
    laplace_noise,
    noise_norm_mean,
    noise_norm_variance,
)

_DEFAULT_EPSILONS = (0.5, 1.0, 5.0)
_SAMPLES_DEFAULT = 1500
_HIST_BINS = 25
_HIST_WIDTH = 50  # characters
_SCATTER_W, _SCATTER_H = 60, 21


def _ascii_histogram(values: list[float], bins: int, width: int) -> str:
    if not values:
        return "(no samples)"
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1e-9
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        i = min(bins - 1, int((v - lo) / step))
        counts[i] += 1
    max_c = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * int(width * c / max_c)
        lo_edge = lo + i * step
        lines.append(f"  [{lo_edge:6.3f}, {lo_edge + step:6.3f})  {bar} {c}")
    return "\n".join(lines)


def _ascii_scatter(points: list[tuple[float, float]], width: int, height: int) -> str:
    if not points:
        return "(no points)"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    if xhi == xlo:
        xhi = xlo + 1e-9
    if yhi == ylo:
        yhi = ylo + 1e-9
    # density grid
    grid = [[0] * width for _ in range(height)]
    for x, y in points:
        col = min(width - 1, int((x - xlo) / (xhi - xlo) * (width - 1)))
        row = height - 1 - min(
            height - 1, int((y - ylo) / (yhi - ylo) * (height - 1))
        )
        grid[row][col] += 1
    chars = " .:-=+*#%@"
    max_d = max(c for row in grid for c in row) or 1
    rows = []
    for row in grid:
        rows.append("".join(chars[min(len(chars) - 1, int(c / max_d * (len(chars) - 1)))] for c in row))
    legend = f"  x ∈ [{xlo:+.2f}, {xhi:+.2f}]   y ∈ [{ylo:+.2f}, {yhi:+.2f}]   density: '{chars}'"
    return "\n".join("  " + r for r in rows) + "\n" + legend


def _summary(values: list[float], dim: int, eps: float) -> str:
    mean_emp = statistics.mean(values)
    std_emp = statistics.stdev(values)
    mean_theo = noise_norm_mean(dim, eps)
    std_theo = math.sqrt(noise_norm_variance(dim, eps))
    return (
        f"  ε = {eps:<6g} dim = {dim}\n"
        f"    empirical:  E[‖η‖] = {mean_emp:7.4f}   SD = {std_emp:7.4f}\n"
        f"    theoretical: E[‖η‖] = {mean_theo:7.4f}   SD = {std_theo:7.4f}\n"
        f"    rel. error:  mean {abs(mean_emp - mean_theo) / mean_theo * 100:5.2f}%   "
        f"sd {abs(std_emp - std_theo) / std_theo * 100:5.2f}%"
    )


def _maybe_save_png(path: str, dim: int, epsilons: list[float], samples: int) -> None:
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not installed — skipping {path})")
        return
    rng = random.Random(0xC0FFEE)
    fig, axes = plt.subplots(2, len(epsilons), figsize=(4 * len(epsilons), 7))
    if len(epsilons) == 1:
        axes = axes.reshape(2, 1)
    for col, eps in enumerate(epsilons):
        pts = [laplace_noise(dim, eps, rng=rng) for _ in range(samples)]
        norms = [math.sqrt(sum(x * x for x in p)) for p in pts]
        if dim >= 2:
            axes[0, col].scatter(
                [p[0] for p in pts], [p[1] for p in pts],
                s=4, alpha=0.4,
            )
            axes[0, col].set_title(f"ε = {eps}  (dim 0 vs dim 1)")
            axes[0, col].axhline(0, color="gray", lw=0.5)
            axes[0, col].axvline(0, color="gray", lw=0.5)
            axes[0, col].set_aspect("equal")
        axes[1, col].hist(norms, bins=40, density=True, alpha=0.7)
        # Overlay theoretical Gamma density
        import math as _m
        xs = [i * (max(norms) / 100) for i in range(1, 101)]
        gamma_pdf = [
            (eps ** dim) * (x ** (dim - 1)) * _m.exp(-eps * x) / _m.gamma(dim)
            for x in xs
        ]
        axes[1, col].plot(xs, gamma_pdf, "r--", lw=1, label=f"Gamma({dim}, 1/{eps})")
        axes[1, col].set_title(f"‖η‖ histogram  (ε = {eps})")
        axes[1, col].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"  ✓ saved {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2)
    p.add_argument(
        "--epsilons",
        default=",".join(str(e) for e in _DEFAULT_EPSILONS),
        help="comma-separated ε values to sweep",
    )
    p.add_argument("--samples", type=int, default=_SAMPLES_DEFAULT)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--png", default=None, help="optional output PNG path")
    args = p.parse_args()

    eps_list = [float(s) for s in args.epsilons.split(",")]
    rng = random.Random(args.seed)
    line = "─" * 72

    print(line)
    print(f"  Multivariate Laplace noise — dim={args.dim}, samples={args.samples}")
    print(line)

    for eps in eps_list:
        pts = [laplace_noise(args.dim, eps, rng=rng) for _ in range(args.samples)]
        norms = [math.sqrt(sum(x * x for x in p)) for p in pts]
        print()
        print(_summary(norms, args.dim, eps))
        print()
        print(f"  ‖η‖ histogram  (ε = {eps})")
        print(_ascii_histogram(norms, bins=_HIST_BINS, width=_HIST_WIDTH))
        if args.dim >= 2:
            print()
            print(f"  2-D scatter (dim 0 vs dim 1, ε = {eps})")
            print(_ascii_scatter(
                [(p[0], p[1]) for p in pts], _SCATTER_W, _SCATTER_H
            ))
        print()
        print(line)

    if args.png:
        print(f"  Writing PNG plot to {args.png}…")
        _maybe_save_png(args.png, args.dim, eps_list, args.samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
