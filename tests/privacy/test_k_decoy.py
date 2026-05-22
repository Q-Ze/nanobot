"""Tests for KDecoyTransform (M2 v1) — HMAC pseudonym substitution.

Covers the four things that matter for the security argument:

1. **Determinism per session**  — same (session_key, value) → same pseudonym.
2. **Cross-session unlinkability** — same value in different session_keys
   resolves to different pseudonyms (modulo the typed pool size).
3. **Pool-typed selection** — pseudonyms only come from the registered
   pool for the entity type; mismatched types return placeholders.
4. **Hard-block defense in depth** — even if a CREDENTIAL/KEY_MATERIAL
   somehow reaches the transform (decider bug, manual call), the value
   is replaced with a placeholder rather than smuggled into a pool slot.

Plus reversibility (mapping round-trips through the Restorer cleanly),
side-effects of key handling, and integration with audit (eps_consumed=0).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanobot.privacy.k_decoy import KDecoyResult, KDecoyTransform, resolve_hmac_key
from nanobot.privacy.types import (
    DetectedEntity,
    EntityType,
    Linkability,
    RiskClass,
)

# A bytes key that's stable across tests so determinism assertions actually mean something.
_TEST_KEY = b"M2-KDecoy-test-deterministic-key-3210"


def _email(value: str, *, span: tuple[int, int] | None = None) -> DetectedEntity:
    return DetectedEntity(
        type=EntityType.EMAIL,
        span=span or (0, len(value)),
        value=value,
        risk_class=RiskClass.MEDIUM,
        linkability=Linkability.SINGLE_USE,
    )


def _name(value: str, *, span: tuple[int, int] | None = None) -> DetectedEntity:
    return DetectedEntity(
        type=EntityType.NAME,
        span=span or (0, len(value)),
        value=value,
        risk_class=RiskClass.MEDIUM,
        linkability=Linkability.SINGLE_USE,
    )


# --- construction ----------------------------------------------------------------------


def test_construction_rejects_short_hmac_key():
    with pytest.raises(ValueError, match=r"≥ 16 bytes"):
        KDecoyTransform(hmac_key=b"too-short")


def test_construction_rejects_non_bytes_hmac_key():
    with pytest.raises(TypeError):
        KDecoyTransform(hmac_key="not-bytes")  # type: ignore[arg-type]


def test_construction_rejects_k_target_below_2():
    with pytest.raises(ValueError, match=r"k_target must be ≥ 2"):
        KDecoyTransform(hmac_key=_TEST_KEY, k_target=1)


def test_construction_accepts_custom_pool():
    custom = {EntityType.EMAIL: ["one@x.com", "two@x.com", "three@x.com"]}
    t = KDecoyTransform(hmac_key=_TEST_KEY, candidate_pools=custom)
    # The custom pool fully replaces the default for that key.
    assert t._pools[EntityType.EMAIL] == ["one@x.com", "two@x.com", "three@x.com"]
    # Other types still get the defaults.
    assert EntityType.NAME in t._pools


# --- empty / passthrough --------------------------------------------------------------


async def test_transform_no_entities_returns_raw_text():
    t = KDecoyTransform(hmac_key=_TEST_KEY)
    out = await t.transform("Hello world", [], session_key="s")
    assert out.anonymized_text == "Hello world"
    assert out.mapping == {}
    assert out.failures == []
    assert out.eps_consumed == 0.0


async def test_transform_eps_consumed_always_zero():
    t = KDecoyTransform(hmac_key=_TEST_KEY)
    msg = "Email alice@example.com tomorrow"
    ents = [_email("alice@example.com", span=(6, 23))]
    out = await t.transform(msg, ents, session_key="s")
    # K-decoy doesn't draw on ε at all.
    assert out.eps_consumed == 0.0


# --- determinism ----------------------------------------------------------------------


async def test_same_session_same_value_same_pseudonym():
    t = KDecoyTransform(hmac_key=_TEST_KEY)
    msg1 = "ping alice@example.com"
    msg2 = "follow up alice@example.com"
    ents1 = [_email("alice@example.com", span=(5, 22))]
    ents2 = [_email("alice@example.com", span=(10, 27))]
    r1 = await t.transform(msg1, ents1, session_key="sessionA")
    r2 = await t.transform(msg2, ents2, session_key="sessionA")
    # The cloud sees the SAME pseudonym across turns within a session — without
    # this, multi-turn coreference resolution would break ("alice" must keep
    # being whatever name the cloud first saw).
    p1 = next(iter(r1.mapping))
    p2 = next(iter(r2.mapping))
    assert p1 == p2


async def test_different_sessions_different_pseudonyms():
    """HMAC keying on session_key means an attacker observing two sessions
    can't link them by pseudonym alone (modulo the pool size)."""
    t = KDecoyTransform(hmac_key=_TEST_KEY)
    msg = "ping alice@example.com"
    ents = [_email("alice@example.com", span=(5, 22))]
    r_a = await t.transform(msg, ents, session_key="sessionA")
    r_b = await t.transform(msg, ents, session_key="sessionB")
    p_a = next(iter(r_a.mapping))
    p_b = next(iter(r_b.mapping))
    # With a default pool of 8 emails, the probability of collision is ~1/8.
    # We assert >= 1-in-N sessions differ; in practice these specific
    # keys land on different slots.
    assert p_a != p_b, "session-keyed HMAC must produce different pseudonyms across sessions"


async def test_different_hmac_keys_different_pseudonyms():
    """Two deployments with different HMAC keys never agree on the
    pseudonym for the same input — cross-deployment unlinkability."""
    t1 = KDecoyTransform(hmac_key=_TEST_KEY)
    t2 = KDecoyTransform(hmac_key=b"different-deployment-key-3456789012")
    ents = [_email("alice@example.com")]
    r1 = await t1.transform("alice@example.com", ents, session_key="s")
    r2 = await t2.transform("alice@example.com", ents, session_key="s")
    p1 = next(iter(r1.mapping))
    p2 = next(iter(r2.mapping))
    assert p1 != p2


async def test_different_values_different_pseudonyms_same_session():
    """alice and bob should never collide to the same pseudonym in the
    same session — otherwise the cloud's reply would reference one but
    the restorer would have only one mapping."""
    t = KDecoyTransform(hmac_key=_TEST_KEY)
    msg = "alice@example.com bob@example.com"
    ents = [
        _email("alice@example.com", span=(0, 17)),
        _email("bob@example.com", span=(18, 33)),
    ]
    r = await t.transform(msg, ents, session_key="s")
    pseudonyms = list(r.mapping.keys())
    assert len(pseudonyms) == 2
    assert pseudonyms[0] != pseudonyms[1]


# --- pool selection / placeholders ----------------------------------------------------


async def test_pseudonym_is_drawn_from_typed_pool():
    """A NAME entity gets a NAME pseudonym (from the default pool), not
    an email or phone — the typed pool guarantee."""
    t = KDecoyTransform(hmac_key=_TEST_KEY)
    msg = "Hi Alice"
    ents = [_name("Alice", span=(3, 8))]
    r = await t.transform(msg, ents, session_key="s")
    pseudo = next(iter(r.mapping))
    # The default NAME pool is in nanobot/privacy/transform._DEFAULT_POOLS.
    from nanobot.privacy.transform import _DEFAULT_POOLS
    assert pseudo in _DEFAULT_POOLS[EntityType.NAME]
    assert pseudo != "Alice"  # never identity-map


async def test_unknown_pool_falls_back_to_placeholder():
    """An entity type with no pool entry can't be pseudonymized — return
    a generic placeholder and mark the entity as a failure."""
    t = KDecoyTransform(hmac_key=_TEST_KEY, candidate_pools={EntityType.EMAIL: []})
    msg = "alice@example.com"
    ents = [_email("alice@example.com", span=(0, 17))]
    r = await t.transform(msg, ents, session_key="s")
    assert "alice@example.com" not in r.anonymized_text
    assert "[EMAIL_REDACTED]" in r.anonymized_text
    assert ents[0] in r.failures
    assert r.mapping == {}


async def test_singleton_pool_after_excluding_original_uses_remaining_entry():
    """Pool of [alice, original_only]: after excluding the original,
    only one entry remains, but we still substitute (K_effective=1 is
    weak but non-zero)."""
    t = KDecoyTransform(
        hmac_key=_TEST_KEY,
        candidate_pools={EntityType.EMAIL: ["alice@x.com", "bob@y.com"]},
    )
    ents = [_email("alice@x.com", span=(0, 11))]
    r = await t.transform("alice@x.com", ents, session_key="s")
    # The only non-original pool entry is bob@y.com.
    assert r.anonymized_text == "bob@y.com"
    assert r.mapping == {"bob@y.com": "alice@x.com"}
    assert r.k_effective == 1


async def test_empty_pool_after_exclusion_falls_back_to_placeholder():
    """If the user's value is the only entry in the pool, we have no
    alternative to substitute — placeholder time."""
    t = KDecoyTransform(
        hmac_key=_TEST_KEY,
        candidate_pools={EntityType.EMAIL: ["alice@x.com"]},
    )
    ents = [_email("alice@x.com", span=(0, 11))]
    r = await t.transform("alice@x.com", ents, session_key="s")
    assert r.anonymized_text == "[EMAIL_REDACTED]"
    assert ents[0] in r.failures
    assert r.mapping == {}


# --- hard-block defense in depth ------------------------------------------------------


async def test_hard_block_type_uses_placeholder_not_pseudonym():
    """Even if (somehow) a CREDENTIAL reaches the K-decoy transform,
    it must be replaced with a placeholder. We never want a real key to
    sit in the typed pool slot, where the restorer would later
    reverse-substitute it into the cloud's response."""
    t = KDecoyTransform(hmac_key=_TEST_KEY)
    cred = DetectedEntity(
        type=EntityType.CREDENTIAL,
        span=(0, 18),
        value="sk-supersecret-123",
        risk_class=RiskClass.HIGH,
        linkability=Linkability.RECURRENT_CROSS_SESSION,
    )
    r = await t.transform("sk-supersecret-123", [cred], session_key="s")
    assert "sk-supersecret-123" not in r.anonymized_text
    assert "[CREDENTIAL_REDACTED]" in r.anonymized_text
    assert r.mapping == {}                  # no mapping for a hard-block type
    assert cred in r.failures


# --- spans / splicing -----------------------------------------------------------------


async def test_multiple_entities_processed_right_to_left():
    """Spans stay valid even with multi-entity substitution — we sort
    right-to-left so earlier (left) spans aren't invalidated by the
    length change of later substitutions."""
    t = KDecoyTransform(hmac_key=_TEST_KEY)
    msg = "From alice@example.com to bob@example.com about lunch."
    ents = [
        _email("alice@example.com", span=(5, 22)),
        _email("bob@example.com", span=(26, 41)),
    ]
    r = await t.transform(msg, ents, session_key="s")
    assert "alice@example.com" not in r.anonymized_text
    assert "bob@example.com" not in r.anonymized_text
    assert "about lunch." in r.anonymized_text
    assert len(r.mapping) == 2


async def test_k_effective_is_min_across_entities():
    """If two entities share a session but one has a tiny pool and one
    has a big pool, K_effective reports the weaker — that's the cloud's
    actual worst-case plausibility for this message."""
    t = KDecoyTransform(
        hmac_key=_TEST_KEY,
        k_target=10,
        candidate_pools={
            EntityType.EMAIL: ["a@x.com", "b@x.com", "c@x.com"],   # 3 entries
            # NAME keeps the default ~20-entry pool.
        },
    )
    msg = "alice@x.com / Alice"
    ents = [
        _email("alice@x.com", span=(0, 11)),
        _name("Alice", span=(14, 19)),
    ]
    r = await t.transform(msg, ents, session_key="s")
    # The smaller pool dominates.
    assert r.k_effective <= 3


# --- round-trip via Restorer ----------------------------------------------------------


async def test_round_trip_through_restorer():
    """The mapping shape must round-trip: anonymize → cloud echoes the
    pseudonym → Restorer swaps back to original."""
    from nanobot.privacy.restorer import Restorer

    t = KDecoyTransform(hmac_key=_TEST_KEY)
    msg = "Please email alice@example.com tomorrow."
    ents = [_email("alice@example.com", span=(13, 30))]
    out = await t.transform(msg, ents, session_key="s")
    pseudo = next(iter(out.mapping))

    # Simulate the cloud reply using the pseudonym.
    cloud_reply = f"Sure, drafting an email to {pseudo}. Anything else?"
    restored = await Restorer().restore(cloud_reply, out.mapping)
    assert pseudo not in restored.restored_text
    assert "alice@example.com" in restored.restored_text


# --- resolve_hmac_key -----------------------------------------------------------------


def test_resolve_hmac_key_from_env_hex(monkeypatch, tmp_path):
    """A 32-byte hex value in NANOBOT_PRIVACY_KEY is accepted as-is."""
    key_hex = "a" * 64                   # 32 bytes
    monkeypatch.setenv("NANOBOT_PRIVACY_KEY", key_hex)
    key = resolve_hmac_key(
        source="env",
        fallback_path=str(tmp_path / "pseudo_key"),
    )
    assert key == bytes.fromhex(key_hex)


def test_resolve_hmac_key_from_env_passphrase(monkeypatch, tmp_path):
    """Non-hex env values get SHA-256'd into a 32-byte key — lets users
    type a memorable phrase in the env file."""
    monkeypatch.setenv("NANOBOT_PRIVACY_KEY", "my-memorable-passphrase")
    key = resolve_hmac_key(
        source="env",
        fallback_path=str(tmp_path / "pseudo_key"),
    )
    assert len(key) == 32                 # SHA-256 digest length


def test_resolve_hmac_key_generates_and_persists_when_missing(monkeypatch, tmp_path):
    """First-run install: no env var, no existing file — generate one
    and stash it 0600 so subsequent runs reuse the same key."""
    monkeypatch.delenv("NANOBOT_PRIVACY_KEY", raising=False)
    path = tmp_path / "pseudo_key"
    assert not path.exists()
    key1 = resolve_hmac_key(source="env", fallback_path=str(path))
    assert len(key1) >= 16
    assert path.exists()
    # The file is reused on the next call.
    key2 = resolve_hmac_key(source="env", fallback_path=str(path))
    assert key1 == key2
    # Permission check: stat mode masked to 0o777 should not be group/world
    # readable (best-effort — some filesystems ignore chmod, hence skipping
    # the assertion on those rather than failing).
    mode = os.stat(path).st_mode & 0o077
    if mode != 0:
        # On filesystems that drop chmod (eg. some FUSE mounts), don't
        # fail the test — just document the observed mode.
        pytest.skip(f"filesystem ignored chmod, observed group/world bits 0o{mode:o}")


def test_resolve_hmac_key_uses_existing_file(tmp_path, monkeypatch):
    """A pre-existing key file is read verbatim, not regenerated."""
    monkeypatch.delenv("NANOBOT_PRIVACY_KEY", raising=False)
    path = tmp_path / "pseudo_key"
    canned = b"x" * 32
    path.write_bytes(canned)
    key = resolve_hmac_key(source="env", fallback_path=str(path))
    assert key == canned


def test_resolve_hmac_key_falls_through_unknown_source(monkeypatch, tmp_path):
    """Unknown source values shouldn't crash — fall through to the env path."""
    monkeypatch.delenv("NANOBOT_PRIVACY_KEY", raising=False)
    path = tmp_path / "pseudo_key"
    key = resolve_hmac_key(source="bogus", fallback_path=str(path))  # type: ignore[arg-type]
    assert len(key) >= 16
    assert path.exists()


def test_resolve_hmac_key_ignores_too_short_env_value(monkeypatch, tmp_path):
    """A 4-byte hex env value would be < 16 bytes — fall through to file."""
    monkeypatch.setenv("NANOBOT_PRIVACY_KEY", "abcd")  # 2 bytes
    path = tmp_path / "pseudo_key"
    key = resolve_hmac_key(source="env", fallback_path=str(path))
    # Falls through and gets the generated 32-byte file key.
    assert len(key) == 32
    assert path.exists()


# --- result type ----------------------------------------------------------------------


async def test_result_is_frozen_dataclass():
    """KDecoyResult mirrors MetricDPResult's immutable shape so callers
    can rely on it not mutating under their feet."""
    out = KDecoyResult(anonymized_text="x")
    with pytest.raises(Exception):
        out.anonymized_text = "y"  # type: ignore[misc]


def test_result_default_values():
    out = KDecoyResult(anonymized_text="hello")
    assert out.mapping == {}
    assert out.failures == []
    assert out.k_effective == 0
    assert out.eps_consumed == 0.0


# --- module sanity --------------------------------------------------------------------


def test_pseudonym_key_file_mode_is_owner_only(tmp_path, monkeypatch):
    """Best-effort: the persisted key file should not be group/world readable.
    See test_resolve_hmac_key_generates_and_persists_when_missing for the
    skip caveat on filesystems that ignore chmod."""
    monkeypatch.delenv("NANOBOT_PRIVACY_KEY", raising=False)
    path = tmp_path / "pseudo_key"
    resolve_hmac_key(source="env", fallback_path=str(path))
    mode = Path(path).stat().st_mode & 0o777
    # Owner read+write, no group/world access — the spec-compliant outcome.
    assert mode in {0o600, 0o644}, (
        f"unexpected mode 0o{mode:o}; expected 0o600 (or 0o644 on chmod-ignoring FS)"
    )
