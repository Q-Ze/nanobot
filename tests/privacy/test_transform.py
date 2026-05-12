"""Tests for MetricDPTransform — the token-level dχ-privacy mechanism.

We inject a deterministic fake backend so every test is reproducible and
no real LLM is consulted. The fake assigns each value a *fixed*
high-dimensional embedding from a small table; the closer two values are
in that table, the more likely they are to swap under noise.
"""

from __future__ import annotations

import random

import pytest

from nanobot.privacy.transform import MetricDPResult, MetricDPTransform, _placeholder
from nanobot.privacy.types import DetectedEntity, EntityType, Linkability, RiskClass

# --- fake backend ---------------------------------------------------------------------


def _entity(value: str, t: EntityType, span: tuple[int, int]) -> DetectedEntity:
    return DetectedEntity(
        type=t,
        span=span,
        value=value,
        risk_class=RiskClass.MEDIUM,
        linkability=Linkability.SINGLE_USE,
    )


class _FakeBackend:
    """Deterministic backend with a hand-crafted embedding table.

    Values not in the table map to (1,0,0,0,...) — far from every "real"
    embedding so the noise can still drive a replacement.
    """

    def __init__(self, table: dict[str, list[float]], available: bool = True):
        self._table = table
        self._available = available

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return self._available

    async def generate(self, prompt: str, **_) -> str:
        return ""

    async def embed(self, text: str, **_) -> list[float]:
        if text in self._table:
            return list(self._table[text])
        # Distant generic vector for unknown strings
        return [1.0] + [0.0] * 7


# --- happy path -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_input_when_no_entities():
    t = MetricDPTransform(_FakeBackend({}), epsilon=1.0)
    result = await t.transform("hello world", entities=[])
    assert result.anonymized_text == "hello world"
    assert result.mapping == {}
    assert result.failures == []


@pytest.mark.asyncio
async def test_replaces_email_with_candidate_at_high_epsilon():
    """At high ε, noise is tiny → replacement should be the nearest candidate."""
    table = {
        "alice@x.com":                   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Closest candidate (cosine-ish neighbour):
        "alex.morgan@example.com":       [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Far candidates:
        "jordan.lee@example.org":        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "taylor.kim@example.net":        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "casey.nguyen@example.io":       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "sam.patel@example.co":          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "robin.chen@example.com":        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "morgan.davis@example.org":      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "jamie.singh@example.io":        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    t = MetricDPTransform(
        _FakeBackend(table), epsilon=50.0, rng=random.Random(0xC0FFEE)
    )

    e = _entity("alice@x.com", EntityType.EMAIL, span=(0, 11))
    result = await t.transform("alice@x.com please help", entities=[e])

    # The replacement must be one of the candidates other than the original.
    assert "alice@x.com" not in result.anonymized_text
    assert " please help" in result.anonymized_text
    # The nearest neighbour wins at high ε:
    assert "alex.morgan@example.com" in result.anonymized_text
    # Mapping is reversible.
    assert result.mapping["alex.morgan@example.com"] == "alice@x.com"
    assert result.failures == []
    assert result.eps_consumed == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_low_epsilon_spreads_choices():
    """At low ε, noise is large → replacement distribution should cover more
    than just the nearest candidate over many runs."""
    table = {
        "alice@x.com": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "alex.morgan@example.com":  [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "jordan.lee@example.org":   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "taylor.kim@example.net":   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "casey.nguyen@example.io":  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "sam.patel@example.co":     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "robin.chen@example.com":   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "morgan.davis@example.org": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "jamie.singh@example.io":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    chosen: set[str] = set()
    for seed in range(40):
        t = MetricDPTransform(
            _FakeBackend(table), epsilon=0.2, rng=random.Random(seed)
        )
        e = _entity("alice@x.com", EntityType.EMAIL, span=(0, 11))
        result = await t.transform("alice@x.com", entities=[e])
        replacement = result.anonymized_text
        chosen.add(replacement)
    # With ε=0.2 the chooser should hit several different candidates,
    # not just the nearest.
    assert len(chosen) >= 3, f"only {len(chosen)} unique choices: {chosen}"


# --- robustness -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uses_placeholder_when_backend_unavailable():
    t = MetricDPTransform(_FakeBackend({}, available=False), epsilon=1.0)
    e = _entity("alice@x.com", EntityType.EMAIL, span=(0, 11))
    result = await t.transform("alice@x.com please", entities=[e])
    assert "alice@x.com" not in result.anonymized_text
    assert _placeholder(EntityType.EMAIL) in result.anonymized_text
    assert result.failures == [e]
    assert result.mapping == {}
    assert result.eps_consumed == 0.0


@pytest.mark.asyncio
async def test_uses_placeholder_when_pool_is_empty():
    t = MetricDPTransform(
        _FakeBackend({"x": [0.0] * 8}),
        epsilon=1.0,
        # Provide an explicit empty pool for OTHER, overriding the default
        candidate_pools={EntityType.OTHER: []},
    )
    e = _entity("the-secret", EntityType.OTHER, span=(0, 10))
    result = await t.transform("the-secret leaked", entities=[e])
    assert "the-secret" not in result.anonymized_text
    assert _placeholder(EntityType.OTHER) in result.anonymized_text
    assert result.failures == [e]


@pytest.mark.asyncio
async def test_uses_placeholder_when_backend_embed_returns_empty():
    """Simulate a real-world failure: embed model rejects (e.g. chat model
    used for embeddings → HTTP 400 → empty list)."""
    table_with_empty = {"alice@x.com": []}
    t = MetricDPTransform(_FakeBackend(table_with_empty), epsilon=1.0)
    e = _entity("alice@x.com", EntityType.EMAIL, span=(0, 11))
    result = await t.transform("alice@x.com", entities=[e])
    assert _placeholder(EntityType.EMAIL) in result.anonymized_text
    assert result.failures == [e]


@pytest.mark.asyncio
async def test_original_value_excluded_from_pool():
    """If the original value happens to be in the candidate pool, it must be
    removed before projection — otherwise high ε would just return the input."""
    table = {
        "alex.morgan@example.com": [1.0, 0.0] + [0.0] * 6,
        "jordan.lee@example.org":  [0.99, 0.01] + [0.0] * 6,
        # Identical embedding to the original; if the filter is broken,
        # we'd see the input bounced back.
    }
    pool_with_original = ["alex.morgan@example.com", "jordan.lee@example.org"]
    t = MetricDPTransform(
        _FakeBackend(table),
        epsilon=50.0,
        candidate_pools={EntityType.EMAIL: pool_with_original},
        rng=random.Random(0),
    )
    e = _entity("alex.morgan@example.com", EntityType.EMAIL, span=(0, 24))
    result = await t.transform("alex.morgan@example.com", entities=[e])
    # The original is filtered from the pool, so the only valid choice is
    # the other candidate.
    assert result.anonymized_text == "jordan.lee@example.org"


# --- multiple entities and spans ------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_entities_preserve_intervening_text():
    table = {
        "alice@x.com":              [1.0] + [0.0] * 7,
        "alex.morgan@example.com":  [0.95, 0.05] + [0.0] * 6,
        "+44 20 7946 0958":         [0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.05, 0.0],
        # Make the *user's* phone embed close to the candidate above
        "+1 415 555 0000": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }
    t = MetricDPTransform(_FakeBackend(table), epsilon=50.0,
                          rng=random.Random(0))
    msg = "email alice@x.com, phone +1 415 555 0000."
    entities = [
        _entity("alice@x.com",      EntityType.EMAIL, span=(6, 17)),
        _entity("+1 415 555 0000",  EntityType.PHONE, span=(25, 40)),
    ]
    result = await t.transform(msg, entities=entities)
    assert "alice@x.com" not in result.anonymized_text
    assert "+1 415 555 0000" not in result.anonymized_text
    assert result.anonymized_text.startswith("email ")
    assert ", phone " in result.anonymized_text
    assert result.anonymized_text.endswith(".")
    assert len(result.mapping) == 2


@pytest.mark.asyncio
async def test_utf8_spans_are_handled_correctly():
    table = {
        "张三": [1.0, 0.0] + [0.0] * 6,
        "李伟": [0.9, 0.1] + [0.0] * 6,
    }
    t = MetricDPTransform(
        _FakeBackend(table), epsilon=50.0,
        candidate_pools={EntityType.NAME: ["李伟", "Alex"]},
        rng=random.Random(0),
    )
    msg = "我是张三，住北京"
    span = (msg.find("张三"), msg.find("张三") + len("张三"))
    e = _entity("张三", EntityType.NAME, span=span)
    result = await t.transform(msg, entities=[e])
    assert "张三" not in result.anonymized_text
    assert result.anonymized_text.startswith("我是")
    assert result.anonymized_text.endswith("，住北京")


# --- constructor validation -----------------------------------------------------------


def test_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        MetricDPTransform(_FakeBackend({}), epsilon=0.0)
    with pytest.raises(ValueError):
        MetricDPTransform(_FakeBackend({}), epsilon=-1.0)
    with pytest.raises(ValueError):
        MetricDPTransform(_FakeBackend({}), epsilon=float("inf"))


# --- result type --------------------------------------------------------------------


def test_result_is_frozen_dataclass():
    r = MetricDPResult(anonymized_text="x")
    with pytest.raises(Exception):  # FrozenInstanceError
        r.anonymized_text = "y"  # type: ignore[misc]
