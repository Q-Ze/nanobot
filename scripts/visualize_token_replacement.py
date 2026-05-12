"""Visualize how MetricDPTransform's replacement distribution depends on ε.

Pick one sensitive value, a small candidate pool with hand-crafted
distances, then sweep ε across {0.2, 1, 5, 50} and run 200 trials each.
Print which candidate gets chosen and how often. Lets you eyeball the
core trade-off:

    ε small  → noise huge → choice almost uniform over the pool       (strong privacy)
    ε large  → noise tiny → choice concentrates on the nearest candidate (weak privacy)

No external service required — uses a fake backend with canned embeddings.

Run:
    python scripts/visualize_token_replacement.py
"""

from __future__ import annotations

import asyncio
import random

from nanobot.privacy.transform import MetricDPTransform
from nanobot.privacy.types import DetectedEntity, EntityType, Linkability, RiskClass

# Carefully chosen embeddings so distances are visible:
# - "alice@x.com" is the user's secret
# - candidates ordered by distance from "alice@x.com"
_TABLE = {
    "alice@x.com":              [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "alex.morgan@example.com":  [0.95, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # nearest
    "robin.chen@example.com":   [0.80, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "jordan.lee@example.org":   [0.50, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "taylor.kim@example.net":   [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "casey.nguyen@example.io":  [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "morgan.davis@example.org": [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
    "sam.patel@example.co":     [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
    "jamie.singh@example.io":   [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
}


class _FakeBackend:
    def __init__(self, table): self._t = table
    name = "fake"
    def is_available(self) -> bool: return True
    async def generate(self, prompt, **_): return ""
    async def embed(self, text, **_): return list(self._t.get(text, [1.0] + [0.0] * 7))


def _entity():
    return DetectedEntity(
        type=EntityType.EMAIL,
        span=(0, 11),
        value="alice@x.com",
        risk_class=RiskClass.MEDIUM,
        linkability=Linkability.SINGLE_USE,
    )


def _bar(count: int, total: int, width: int = 40) -> str:
    if total == 0:
        return ""
    fill = int(width * count / total)
    return "█" * fill + "·" * (width - fill)


async def _run_one(epsilon: float, trials: int):
    counter: dict[str, int] = {c: 0 for c in _TABLE if c != "alice@x.com"}
    for seed in range(trials):
        t = MetricDPTransform(
            _FakeBackend(_TABLE),
            epsilon=epsilon,
            rng=random.Random(seed),
        )
        result = await t.transform("alice@x.com", entities=[_entity()])
        choice = result.anonymized_text
        counter[choice] = counter.get(choice, 0) + 1
    return counter


def _print_distribution(label: str, counter: dict[str, int], trials: int):
    print(label)
    total = sum(counter.values())
    # Sort by frequency descending so the dominant candidate is on top.
    ordered = sorted(counter.items(), key=lambda kv: -kv[1])
    for cand, c in ordered:
        pct = 100 * c / total if total else 0.0
        # Pad candidate label so bars align
        print(f"  {cand:<28} {_bar(c, trials)} {c:>3}  ({pct:5.1f}%)")
    print()


async def main() -> int:
    trials = 200
    line = "─" * 76
    print(line)
    print(f"  Replacement distribution for 'alice@x.com'  ({trials} trials per ε)")
    print("  Candidates ordered by L2 distance from the original embedding.")
    print(line)
    print()

    for eps in [0.2, 1.0, 5.0, 50.0]:
        counter = await _run_one(eps, trials)
        unique = len([c for c in counter.values() if c > 0])
        top_pct = max(counter.values()) / trials * 100
        header = (
            f"ε = {eps:<6g}  (noise scale: {1/eps:.2f})   "
            f"unique={unique}/{len(counter)}   "
            f"top candidate dominance: {top_pct:.1f}%"
        )
        _print_distribution(header, counter, trials)

    print(line)
    print("  Reading the picture:")
    print("    • ε = 0.2  → distribution close to uniform → strong privacy, weak utility")
    print("    • ε = 50   → almost always the nearest neighbour → weak privacy, high utility")
    print("    • The sweet spot lives in between and is the dial you expose to users.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
