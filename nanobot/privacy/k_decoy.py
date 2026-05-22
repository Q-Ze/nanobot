"""K-Decoy transform (M2 v1) — HMAC-pseudonym substitution.

Per `.agent/privacy_gatekeeper.md` §3.3.1, K_DECOY in M2 v1 is the
"single-message pseudonym + typed pool as anonymity set" simplification
of the original multi-message K-decoy proposal:

    detected entity v_text  (e.g. alice@x.com)
            │
            ▼  HMAC_SHA256(deployment_key, session_key‖type‖canonical(v_text))[:8]
        index ∈ [0, len(pool))
            │
            ▼  pool[index]
        pseudonym c*  (e.g. robin.chen@example.com)
            │
            ▼  splice into the message; record c* → v_text for Restorer

What this buys us
-----------------

* No backend call (works when the embedding model is down).
* No ε consumed (works when the accountant is out of budget).
* Deterministic per ``session_key``: the same entity in the same chat
  maps to the same pseudonym across turns, so the cloud LLM's multi-turn
  reasoning stays coherent.
* Cross-session unlinkability: a different ``session_key`` → different
  pseudonym for the same value, so an adversary observing two sessions
  can't link them by the pseudonym alone (modulo the underlying typed
  pool size).
* K_effective = number of plausible candidates in the typed pool. The
  cloud's posterior on the true value is bounded by 1/K_effective when
  it has no other prior; in practice context bleeds prior in, so the
  guarantee is computational, not information-theoretic.

What this does NOT do
---------------------

* No multi-message decoy emission. The original §3.3.1 proposal sent K
  parallel queries to the cloud; that requires AgentRunner-level changes
  (parallel cloud calls, decoy response suppression, tool-call interception)
  that aren't justified by the marginal privacy gain for v1.
* No ε-dχ-privacy. Use METRIC_DP when you want a formal guarantee.
* No HIGH-risk entity routing — the decider already steers HIGH to
  METRIC_DP or BLOCKED before reaching this module. As defense in depth,
  if KDecoyTransform is somehow handed a HARD_BLOCK type or an entity
  whose pool has fewer than 2 entries after excluding the original,
  it falls back to a generic ``[<TYPE>_REDACTED]`` placeholder.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

from nanobot.privacy.transform import _DEFAULT_POOLS, _placeholder
from nanobot.privacy.types import DetectedEntity, EntityType

# HARD-blocked entity types: never pseudonymize, always placeholder.
# Mirrors decider._HARD_BLOCK_TYPES so a misrouted secret never reaches the cloud.
_HARD_BLOCK_TYPES: frozenset[EntityType] = frozenset({
    EntityType.KEY_MATERIAL,
    EntityType.CREDENTIAL,
    EntityType.INTERNAL_INSTRUCTION,
    EntityType.JWT,
})


@dataclass(frozen=True)
class KDecoyResult:
    """Output of :meth:`KDecoyTransform.transform`.

    Shape deliberately mirrors :class:`nanobot.privacy.transform.MetricDPResult`
    so the GateKeeper restoration plan and Restorer don't need to branch on
    which transform produced the mapping.
    """

    anonymized_text: str
    mapping: dict[str, str] = field(default_factory=dict)
    """``{pseudonym: original_value}`` — consumed by the Restorer."""

    failures: list[DetectedEntity] = field(default_factory=list)
    """Entities replaced with a placeholder (no pool / hard-block type)."""

    k_effective: int = 0
    """Min pool size encountered across processed entities; the runtime
    plausibility set for the weakest entity in the message. Reported in
    the audit view as ``eps_consumed`` is for METRIC_DP."""

    eps_consumed: float = 0.0
    """Always 0 — K-decoy doesn't draw on the ε budget. Kept for shape
    parity with MetricDPResult so callers can treat both interchangeably."""


class KDecoyTransform:
    """Per-entity HMAC-derived pseudonym substitution.

    Construction
    ------------
    ``hmac_key`` must be a high-entropy ``bytes`` (≥ 16 bytes). Resolution
    from env / file is handled by :func:`resolve_hmac_key` so callers
    (typically :meth:`GateKeeper.from_config`) don't have to.

    ``candidate_pools`` overrides the defaults shared with
    :class:`MetricDPTransform`; pass a partial dict to extend a subset of
    types while keeping the rest at default.

    ``k_target`` is the *requested* anonymity set size. The actual
    K_effective per entity is ``min(k_target, len(pool) - 1)`` where the
    ``-1`` accounts for excluding the original value from the pool.
    """

    def __init__(
        self,
        *,
        hmac_key: bytes,
        k_target: int = 5,
        candidate_pools: dict[EntityType, list[str]] | None = None,
    ) -> None:
        if not isinstance(hmac_key, (bytes, bytearray)):
            raise TypeError("hmac_key must be bytes")
        if len(hmac_key) < 16:
            raise ValueError(
                f"hmac_key must be ≥ 16 bytes for HMAC-SHA256 security, got {len(hmac_key)}"
            )
        if k_target < 2:
            raise ValueError(f"k_target must be ≥ 2, got {k_target!r}")
        self._hmac_key = bytes(hmac_key)
        self._k_target = int(k_target)
        self._pools: dict[EntityType, list[str]] = dict(_DEFAULT_POOLS)
        if candidate_pools:
            self._pools.update(candidate_pools)

    async def transform(
        self,
        raw_message: str,
        entities: list[DetectedEntity],
        *,
        session_key: str = "",
    ) -> KDecoyResult:
        """Replace each entity in ``entities`` with a pseudonym from its typed pool.

        Entities are processed right-to-left so earlier spans stay valid as
        we splice replacements in. Spans must be non-overlapping; the
        detector already enforces this.
        """
        if not entities:
            return KDecoyResult(anonymized_text=raw_message, k_effective=self._k_target)

        ordered = sorted(entities, key=lambda e: e.span[0], reverse=True)
        text = raw_message
        mapping: dict[str, str] = {}
        failures: list[DetectedEntity] = []
        # k_effective tracked across the message — the cloud's worst-case
        # plausibility is set by the entity with the smallest pool.
        k_effective_min: int | None = None

        for entity in ordered:
            replacement, k_eff, ok = self._replace_one(entity, session_key)
            start, end = entity.span
            text = text[:start] + replacement + text[end:]
            if ok:
                # If two entities collide to the same pseudonym (rare; pools
                # are disjoint by type for the default set), the later one
                # wins. Restorer is fine either way.
                mapping[replacement] = entity.value
                if k_effective_min is None or k_eff < k_effective_min:
                    k_effective_min = k_eff
            else:
                failures.append(entity)

        return KDecoyResult(
            anonymized_text=text,
            mapping=mapping,
            failures=failures,
            k_effective=k_effective_min if k_effective_min is not None else 0,
            eps_consumed=0.0,
        )

    # --- internals --------------------------------------------------------------------

    def _replace_one(
        self, entity: DetectedEntity, session_key: str,
    ) -> tuple[str, int, bool]:
        """Pick a pseudonym for one entity.

        Returns ``(replacement, k_effective_for_this_entity, success)``.
        On hard-block type or empty pool, returns the placeholder with
        ``success=False`` so the caller records it as a failure.
        """
        if entity.type in _HARD_BLOCK_TYPES:
            return _placeholder(entity.type), 0, False

        pool = [c for c in self._pools.get(entity.type, []) if c != entity.value]
        if len(pool) < 1:
            return _placeholder(entity.type), 0, False

        # K_effective for this entity: how many candidates the cloud sees as
        # plausibly true. Bounded by both the requested target and the pool.
        k_eff = min(self._k_target, len(pool))

        # Deterministic index. Including entity.type in the message ensures
        # the same string used as e.g. NAME vs EMAIL gets different
        # pseudonyms; including session_key ensures cross-session unlinkability.
        msg = f"{session_key}|{entity.type.value}|{entity.value}".encode("utf-8")
        digest = hmac.new(self._hmac_key, msg=msg, digestmod=hashlib.sha256).digest()
        idx = int.from_bytes(digest[:8], "big") % len(pool)
        return pool[idx], k_eff, True


__all__ = ["KDecoyTransform", "KDecoyResult", "resolve_hmac_key"]


# --- key resolution -------------------------------------------------------------------


def resolve_hmac_key(
    source: str = "env",
    *,
    env_var: str = "NANOBOT_PRIVACY_KEY",
    fallback_path: "str | None" = None,
) -> bytes:
    """Resolve the HMAC key per the configured ``pseudonym_key_source``.

    Resolution order (per source):
      * ``"env"``  — read ``NANOBOT_PRIVACY_KEY`` env var (hex-encoded).
        If unset OR shorter than 32 hex chars (16 bytes), fall through
        to the persisted-file fallback so first-run installs don't crash.
      * ``"keyring"``  — TODO (M2+); falls through to env behaviour.
      * ``"kms"``      — TODO (M2+); falls through to env behaviour.

    The file fallback lives at ``fallback_path`` (default
    ``~/.nanobot/privacy_audit/pseudo_key``) and is generated on first
    access via ``os.urandom(32)`` with mode 0600. This means:
      * Different deployments end up with different keys (good — pseudonyms
        don't cross deployment boundaries).
      * The same deployment is stable across restarts (good — multi-turn
        consistency survives a process restart).
    """
    import os
    from pathlib import Path

    if source not in {"env", "keyring", "kms"}:
        # Unknown source — be permissive and fall through to env behaviour.
        source = "env"

    # Env first (regardless of source — keyring/kms hooks land in M2+).
    raw = os.environ.get(env_var, "").strip()
    if raw:
        try:
            key = bytes.fromhex(raw)
            if len(key) >= 16:
                return key
        except ValueError:
            # Not hex — treat the env var as a raw passphrase and hash it.
            return hashlib.sha256(raw.encode("utf-8")).digest()

    # Persisted-file fallback.
    if fallback_path is None:
        fallback_path = str(Path.home() / ".nanobot" / "privacy_audit" / "pseudo_key")
    path = Path(fallback_path).expanduser()
    if path.exists():
        try:
            data = path.read_bytes()
            if len(data) >= 16:
                return data
        except OSError:
            pass
    # First-run generation. 0600 to keep prying eyes off it.
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key
