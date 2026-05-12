"""Tests for the Restorer — deterministic anti-substitution of cloud responses.

The MetricDPTransform→Restorer roundtrip is the integration test that
matters most: a real-looking message goes through the transform and
then back, and we verify the user sees what they typed (modulo the
transform's intended anonymization happening in between).
"""

from __future__ import annotations

import random

import pytest

from nanobot.privacy.restorer import Restorer
from nanobot.privacy.transform import MetricDPTransform
from nanobot.privacy.types import DetectedEntity, EntityType, Linkability, RiskClass


def _entity(value: str, t: EntityType, span: tuple[int, int]) -> DetectedEntity:
    return DetectedEntity(
        type=t, span=span, value=value,
        risk_class=RiskClass.MEDIUM,
        linkability=Linkability.SINGLE_USE,
    )


# --- empty / degenerate -------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_input_when_mapping_empty():
    r = Restorer()
    out = await r.restore("hello", mapping={})
    assert out.restored_text == "hello"
    assert out.replacements_applied == 0
    assert out.unmatched_keys == []


@pytest.mark.asyncio
async def test_returns_input_when_text_empty():
    r = Restorer()
    out = await r.restore("", mapping={"bob": "alice"})
    assert out.restored_text == ""
    assert out.replacements_applied == 0


@pytest.mark.asyncio
async def test_drops_invalid_mapping_entries():
    r = Restorer()
    out = await r.restore("hello world", mapping={"": "x", "foo": ""})
    # Empty key is dropped; empty-value entries are kept as the value is allowed
    # to be empty (rare but possible). Verify no crash and no false positives.
    assert "hello world" in out.restored_text


# --- basic substitution ------------------------------------------------------------


@pytest.mark.asyncio
async def test_replaces_one_entity():
    r = Restorer()
    out = await r.restore(
        "Reply to bob@gmail.com about lunch.",
        mapping={"bob@gmail.com": "alice@example.com"},
    )
    assert out.restored_text == "Reply to alice@example.com about lunch."
    assert out.replacements_applied == 1
    assert out.unmatched_keys == []


@pytest.mark.asyncio
async def test_replaces_multiple_entities():
    r = Restorer()
    out = await r.restore(
        "Hi Alex, please call +1 415 555 0101.",
        mapping={"Alex": "Zhang San", "+1 415 555 0101": "13800138000"},
    )
    assert out.restored_text == "Hi Zhang San, please call 13800138000."
    assert out.replacements_applied == 2


@pytest.mark.asyncio
async def test_replaces_multiple_occurrences_of_same_key():
    r = Restorer()
    out = await r.restore(
        "Alex left. Alex's keys are on the desk. Tell Alex.",
        mapping={"Alex": "Bob"},
    )
    assert out.restored_text == "Bob left. Bob's keys are on the desk. Tell Bob."
    assert out.replacements_applied == 3


# --- chain-replacement protection (sentinels) -------------------------------------


@pytest.mark.asyncio
async def test_no_chain_replacement_when_keys_appear_in_other_values():
    """Mapping is { Bob → Alice, Alice → Eve }. Naive str.replace would
    rewrite Bob→Alice and *then* that Alice→Eve. We must not."""
    r = Restorer()
    out = await r.restore(
        "Bob met Alice yesterday.",
        mapping={"Bob": "Alice", "Alice": "Eve"},
    )
    assert out.restored_text == "Alice met Eve yesterday."
    assert out.replacements_applied == 2


# --- length-first ordering --------------------------------------------------------


@pytest.mark.asyncio
async def test_longer_key_matches_first():
    """If two keys overlap, the longer one should win."""
    r = Restorer()
    out = await r.restore(
        "Email alex.morgan@example.com — yes, Alex is the one.",
        mapping={
            "Alex": "USER_NAME",
            "alex.morgan@example.com": "USER_EMAIL",
        },
    )
    # The email must be matched as a whole — not partially via "Alex" first.
    assert "USER_EMAIL" in out.restored_text
    assert "USER_NAME" in out.restored_text
    assert "alex.morgan@example.com" not in out.restored_text
    assert out.replacements_applied == 2


# --- ASCII word boundaries --------------------------------------------------------


@pytest.mark.asyncio
async def test_word_boundary_prevents_partial_match():
    """The mapping for 'Bob' should not clobber the prefix of 'Bobby'."""
    r = Restorer()
    out = await r.restore(
        "Bobby is not Bob.",
        mapping={"Bob": "Alice"},
    )
    assert out.restored_text == "Bobby is not Alice."
    assert out.replacements_applied == 1


@pytest.mark.asyncio
async def test_word_boundary_allows_punctuation_adjacency():
    r = Restorer()
    out = await r.restore(
        "Bob's email is fine. (Bob), did you reply?",
        mapping={"Bob": "Alice"},
    )
    assert out.restored_text == "Alice's email is fine. (Alice), did you reply?"
    assert out.replacements_applied == 2


@pytest.mark.asyncio
async def test_email_replacement_works_with_boundary():
    """The dot and @ are \\W so \\bemail\\b still matches a full email span."""
    r = Restorer()
    out = await r.restore(
        "Forward to bob@gmail.com, then archive.",
        mapping={"bob@gmail.com": "alice@example.com"},
    )
    assert out.restored_text == "Forward to alice@example.com, then archive."


# --- non-ASCII (Chinese) ----------------------------------------------------------


@pytest.mark.asyncio
async def test_chinese_replacement_no_word_boundary():
    """Chinese has no \\b semantics — must fall back to plain substring."""
    r = Restorer()
    out = await r.restore(
        "我是李四的朋友，李四住在北京。",
        mapping={"李四": "张三"},
    )
    assert out.restored_text == "我是张三的朋友，张三住在北京。"
    assert out.replacements_applied == 2


@pytest.mark.asyncio
async def test_mixed_chinese_and_ascii_mapping():
    r = Restorer()
    out = await r.restore(
        "Hi 李四, email me at bob@gmail.com.",
        mapping={"李四": "张三", "bob@gmail.com": "zhangsan@x.com"},
    )
    assert out.restored_text == "Hi 张三, email me at zhangsan@x.com."
    assert out.replacements_applied == 2


# --- case sensitivity -------------------------------------------------------------


@pytest.mark.asyncio
async def test_case_sensitive_by_default():
    """Cloud almost always echoes the candidate verbatim; case-insensitive
    matching invites silent over-replacement."""
    r = Restorer()
    out = await r.restore(
        "BOB is not the same as Bob.",
        mapping={"Bob": "Alice"},
    )
    assert out.restored_text == "BOB is not the same as Alice."
    assert out.replacements_applied == 1


# --- unmatched key reporting ------------------------------------------------------


@pytest.mark.asyncio
async def test_unmatched_keys_are_reported():
    r = Restorer()
    out = await r.restore(
        "Bob met someone yesterday.",
        mapping={"Bob": "Alice", "Carol": "Dave"},
    )
    assert out.restored_text == "Alice met someone yesterday."
    assert out.unmatched_keys == ["Carol"]
    assert out.replacements_applied == 1


# --- counts -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_aggregates_across_keys():
    r = Restorer()
    out = await r.restore(
        "Alex, Alex, Bob, Carol",
        mapping={"Alex": "X", "Bob": "Y"},
    )
    assert out.replacements_applied == 3   # 2 × Alex + 1 × Bob
    assert out.unmatched_keys == []
    assert out.restored_text == "X, X, Y, Carol"


# --- sentinel safety against unusual response text --------------------------------


@pytest.mark.asyncio
async def test_response_with_null_byte_does_not_collide_with_sentinel():
    """\\x00 doesn't appear in normal LM output, but be paranoid: even if the
    cloud emitted a literal \\x00 the sentinel format includes SENTINEL_N\\x00."""
    r = Restorer()
    out = await r.restore(
        "Bob said \x00boo\x00 to the room.",
        mapping={"Bob": "Alice"},
    )
    # The \x00s in the cloud response survive; only the Bob → Alice swap happens.
    assert out.restored_text == "Alice said \x00boo\x00 to the room."


# --- result dataclass shape -------------------------------------------------------


def test_result_is_frozen():
    from nanobot.privacy.restorer import RestoreResult

    r = RestoreResult("x", 0, [])
    with pytest.raises(Exception):
        r.restored_text = "y"  # type: ignore[misc]


# --- end-to-end roundtrip with MetricDPTransform ----------------------------------


class _DeterministicBackend:
    """Tiny embedding table so the transform's behaviour is reproducible."""

    def __init__(self, table: dict[str, list[float]]):
        self._table = table

    name = "fake"

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, **_) -> str:
        return ""

    async def embed(self, text: str, **_) -> list[float]:
        return list(self._table.get(text, [1.0] + [0.0] * 7))


@pytest.mark.asyncio
async def test_roundtrip_transform_then_restore_recovers_original():
    """Send a message through the transform, then pretend the cloud echoed
    the anonymized text verbatim; Restorer should yield the original message.
    """
    table = {
        "alice@x.com":             [1.00, 0.00] + [0.0] * 6,
        "alex.morgan@example.com": [0.95, 0.05] + [0.0] * 6,  # nearest
    }
    transform = MetricDPTransform(
        _DeterministicBackend(table),
        epsilon=50.0,
        candidate_pools={EntityType.EMAIL: ["alex.morgan@example.com"]},
        rng=random.Random(0),
    )

    msg = "please write to alice@x.com about the demo"
    entity = _entity("alice@x.com", EntityType.EMAIL, span=(16, 27))
    transformed = await transform.transform(msg, entities=[entity])
    assert "alice@x.com" not in transformed.anonymized_text
    assert "alex.morgan@example.com" in transformed.anonymized_text

    # Simulate the cloud: it just echoes the message back verbatim.
    cloud_response = transformed.anonymized_text

    restored = await Restorer().restore(
        cloud_response, mapping=transformed.mapping
    )
    assert restored.restored_text == msg
    assert restored.replacements_applied == 1
    assert restored.unmatched_keys == []


@pytest.mark.asyncio
async def test_roundtrip_when_cloud_paraphrases_around_anonymized_value():
    """Realistic case: cloud writes a paragraph that mentions the candidate
    several times; the Restorer hits each occurrence."""
    table = {
        "alice@x.com":             [1.00, 0.00] + [0.0] * 6,
        "alex.morgan@example.com": [0.95, 0.05] + [0.0] * 6,
    }
    transform = MetricDPTransform(
        _DeterministicBackend(table),
        epsilon=50.0,
        candidate_pools={EntityType.EMAIL: ["alex.morgan@example.com"]},
        rng=random.Random(0),
    )
    entity = _entity("alice@x.com", EntityType.EMAIL, span=(0, 11))
    transformed = await transform.transform("alice@x.com please", entities=[entity])

    cloud_response = (
        "Sure, I'll send the report to alex.morgan@example.com. "
        "If alex.morgan@example.com bounces, retry tomorrow."
    )
    restored = await Restorer().restore(cloud_response, mapping=transformed.mapping)
    assert restored.replacements_applied == 2
    assert "alex.morgan@example.com" not in restored.restored_text
    assert restored.restored_text.count("alice@x.com") == 2
