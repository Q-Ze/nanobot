"""Token-level dχ-privacy transform (M3 step 3).

Apply the Andrés-2013 / Feyisetan-2020 mechanism to each detected sensitive
entity:

    sensitive value  v_text
            │
            ▼  (backend.embed)
        v ∈ ℝ^d
            │
            ▼  (+ multivariate Laplace η ~ exp(-ε‖η‖))
        v'
            │
            ▼  (nearest neighbour in a per-type candidate pool, L2)
        chosen candidate c*
            │
            ▼  (substitute c* in place of v_text in the message)

The chosen candidate is what the cloud LLM sees; the (c*, v_text) pair is
written to a local mapping the Restorer (M3 step 5) consults when the cloud
response comes back. By the post-processing property, nearest-neighbour
projection preserves the underlying (ε)-dχ-privacy of v'.

What this module deliberately does NOT do
-----------------------------------------

* No vocabulary-wide search. We project onto a finite, typed candidate pool
  (email / name / phone / generic). A true full-vocab projection is
  unrealistic without local embeddings for an entire tokenizer's vocabulary
  table; the typed pool also lets us preserve the *kind* of token rather
  than swapping "Alice" for "正则表达式".
* No caching. Candidate embeddings are recomputed on every call. M3 step 3
  ships correctness; a per-pool LRU cache lands when we benchmark step 6.
* No accountant integration. The ε passed in is per-call. The session
  budget book-keeping lives in M3 step 4.

Hard guarantees this module DOES uphold
---------------------------------------

* If the backend is unavailable / cannot embed a value, the entity is
  replaced with a generic ``[<TYPE>_REDACTED]`` placeholder and added to
  ``failures``. The original value never falls back to the cloud.
* The original value is removed from the candidate pool before projection,
  so trivial "identity" replacements (which would leak everything) cannot
  happen at high ε.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from nanobot.privacy.local_model import LocalModelBackend
from nanobot.privacy.metric_dp import laplace_noise
from nanobot.privacy.types import DetectedEntity, EntityType

# --- default candidate pools ----------------------------------------------------------
#
# Conservative defaults. Override per-instance via the `candidate_pools`
# constructor argument. The pools below are small on purpose — they're
# enough to demonstrate the mechanism and to keep tests deterministic.
# Production deployments should override with larger, domain-appropriate
# pools (often pulled from public datasets) and pre-embed them.

_DEFAULT_POOLS: dict[EntityType, list[str]] = {
    EntityType.EMAIL: [
        "alex.morgan@example.com",
        "jordan.lee@example.org",
        "taylor.kim@example.net",
        "casey.nguyen@example.io",
        "sam.patel@example.co",
        "robin.chen@example.com",
        "morgan.davis@example.org",
        "jamie.singh@example.io",
    ],
    EntityType.NAME: [
        # Mixed Western + Mandarin given names, kept short to embed cheaply.
        "Alex", "Jordan", "Taylor", "Casey", "Sam", "Robin",
        "Morgan", "Jamie", "Riley", "Quinn",
        "李伟", "王芳", "张敏", "刘洋", "陈静", "杨光",
        "黄磊", "周林", "吴超", "徐丽",
    ],
    EntityType.PHONE: [
        "+1 415 555 0101", "+1 415 555 0102", "+1 415 555 0103",
        "+44 20 7946 0958", "+44 20 7946 0959",
        "13800000001", "13800000002", "13900000003",
    ],
    EntityType.ADDRESS: [
        "742 Evergreen Terrace, Springfield",
        "12 Park Avenue, Boston",
        "221B Baker Street, London",
        "1 Infinite Loop, Cupertino",
        "海淀区中关村大街 1 号",
        "浦东新区世纪大道 100 号",
    ],
}


# --- result types ---------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDPResult:
    """Output of :meth:`MetricDPTransform.transform`."""

    anonymized_text: str
    mapping: dict[str, str] = field(default_factory=dict)
    """``{anonymized_value: original_value}`` — consumed by the Restorer."""

    failures: list[DetectedEntity] = field(default_factory=list)
    """Entities that could not be processed (backend down, empty pool, …).
    Each is replaced in *anonymized_text* with a generic placeholder."""

    eps_consumed: float = 0.0


# --- transform -----------------------------------------------------------------------


class MetricDPTransform:
    """Apply token-level dχ-privacy to detected entities."""

    def __init__(
        self,
        backend: LocalModelBackend,
        epsilon: float,
        *,
        candidate_pools: dict[EntityType, list[str]] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError(f"epsilon must be positive and finite, got {epsilon!r}")
        self._backend = backend
        self._epsilon = float(epsilon)
        self._pools = dict(_DEFAULT_POOLS)
        if candidate_pools:
            self._pools.update(candidate_pools)
        self._rng = rng or random.Random()

    async def transform(
        self,
        raw_message: str,
        entities: list[DetectedEntity],
    ) -> MetricDPResult:
        """Replace each entity in ``entities`` with a metric-DP-chosen alternative.

        Entities are processed right-to-left so earlier spans stay valid as
        we splice replacements in. Spans must be non-overlapping; the
        detector already enforces this via ``_merge_overlaps``.
        """
        if not entities:
            return MetricDPResult(anonymized_text=raw_message)
        if not self._backend.is_available():
            # Every entity becomes a placeholder. Original values stay local.
            return self._all_placeholders(raw_message, entities)

        # Sort right-to-left for safe splicing.
        ordered = sorted(entities, key=lambda e: e.span[0], reverse=True)
        text = raw_message
        mapping: dict[str, str] = {}
        failures: list[DetectedEntity] = []
        eps_consumed = 0.0

        for entity in ordered:
            replacement, ok = await self._replace_one(entity)
            start, end = entity.span
            text = text[:start] + replacement + text[end:]
            if ok:
                # Reverse map: anonymized → original. If two entities collide
                # to the same replacement (rare but possible), the later one
                # wins; the prior collision is silently merged.
                mapping[replacement] = entity.value
                eps_consumed += self._epsilon
            else:
                failures.append(entity)

        return MetricDPResult(
            anonymized_text=text,
            mapping=mapping,
            failures=failures,
            eps_consumed=eps_consumed,
        )

    # --- internals --------------------------------------------------------------------

    async def _replace_one(self, entity: DetectedEntity) -> tuple[str, bool]:
        """Pick a replacement for one entity. Returns (replacement, success).

        On any embedding failure or empty pool, returns the placeholder and
        success=False so the caller can record it as a failure.
        """
        pool = [c for c in self._pools.get(entity.type, []) if c != entity.value]
        if not pool:
            return _placeholder(entity.type), False

        try:
            v = await self._backend.embed(entity.value)
        except Exception:
            v = []
        if not v:
            return _placeholder(entity.type), False

        # Embed every candidate. Production code would cache these; M3 step 3
        # prioritises correctness over throughput.
        candidate_vecs: list[tuple[str, list[float]]] = []
        for c in pool:
            try:
                e = await self._backend.embed(c)
            except Exception:
                e = []
            if e and len(e) == len(v):
                candidate_vecs.append((c, e))
        if not candidate_vecs:
            return _placeholder(entity.type), False

        # Add Laplace noise of the right dimension for *this* embedding model.
        noise = laplace_noise(len(v), self._epsilon, rng=self._rng)
        v_noised = [a + b for a, b in zip(v, noise)]

        # Nearest neighbour in L2.
        best_dist = math.inf
        best_text = candidate_vecs[0][0]
        for text, vec in candidate_vecs:
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(v_noised, vec)))
            if d < best_dist:
                best_dist = d
                best_text = text
        return best_text, True

    def _all_placeholders(
        self,
        raw_message: str,
        entities: list[DetectedEntity],
    ) -> MetricDPResult:
        ordered = sorted(entities, key=lambda e: e.span[0], reverse=True)
        text = raw_message
        for entity in ordered:
            start, end = entity.span
            text = text[:start] + _placeholder(entity.type) + text[end:]
        return MetricDPResult(
            anonymized_text=text,
            mapping={},
            failures=list(entities),
            eps_consumed=0.0,
        )


# --- helpers --------------------------------------------------------------------------


def _placeholder(t: EntityType) -> str:
    return f"[{t.value.upper()}_REDACTED]"


__all__ = ["MetricDPTransform", "MetricDPResult"]
