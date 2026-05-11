"""Detector unit tests — regex layer coverage + override / semantic hook."""

from __future__ import annotations

import pytest

from nanobot.privacy.detector import NoopSemanticDetector, PrivacyEntityDetector
from nanobot.privacy.types import DetectedEntity, EntityType, RiskClass


@pytest.fixture
def detector() -> PrivacyEntityDetector:
    return PrivacyEntityDetector()


async def _types(det: PrivacyEntityDetector, text: str) -> list[EntityType]:
    return [e.type for e in await det.detect(text)]


async def test_detects_email(detector):
    types = await _types(detector, "Reach me at alice@example.com please.")
    assert EntityType.EMAIL in types


async def test_detects_cn_mobile(detector):
    types = await _types(detector, "电话 13800138000 联系我")
    assert EntityType.PHONE in types


async def test_detects_e164_phone(detector):
    types = await _types(detector, "Call +1 415 555 0123 or +44 20 7946 0958")
    assert types.count(EntityType.PHONE) == 2


async def test_detects_aws_access_key(detector):
    ents = await detector.detect("key=AKIAIOSFODNN7EXAMPLE")
    assert any(e.type == EntityType.CREDENTIAL and e.risk_class == RiskClass.CATASTROPHIC for e in ents)


async def test_detects_openai_key(detector):
    ents = await detector.detect("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz12345")
    assert any(e.type == EntityType.CREDENTIAL for e in ents)


async def test_detects_jwt(detector):
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.SflKxwRJSMeKKF0_signature"
    ents = await detector.detect(f"token={jwt}")
    assert any(e.type == EntityType.JWT and e.risk_class == RiskClass.CATASTROPHIC for e in ents)


async def test_detects_pem_block(detector):
    text = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXkt..."
    ents = await detector.detect(text)
    assert any(e.type == EntityType.KEY_MATERIAL for e in ents)


async def test_detects_chinese_id_only_with_valid_checksum(detector):
    # Valid 18-digit ID with correct checksum
    valid = "110101199003079577"
    invalid = "110101199003079570"  # wrong last digit
    ents_valid = await detector.detect(valid)
    ents_invalid = await detector.detect(invalid)
    assert any(e.type == EntityType.ID_NUMBER for e in ents_valid)
    assert not any(e.type == EntityType.ID_NUMBER for e in ents_invalid)


async def test_bank_card_requires_luhn(detector):
    # Real test Visa number (passes Luhn)
    luhn_ok = "4242424242424242"
    luhn_bad = "1234567812345678"
    ents_ok = await detector.detect(luhn_ok)
    ents_bad = await detector.detect(luhn_bad)
    assert any(e.type == EntityType.BANK_CARD for e in ents_ok)
    # Plain 16-digit non-Luhn should not flag as bank card (may be entropy)
    assert not any(e.type == EntityType.BANK_CARD for e in ents_bad)


async def test_high_entropy_string():
    det = PrivacyEntityDetector()
    # 32-char random base64-ish string with entropy ≥ 4
    s = "Z9k3Hq2pXyV8nM4tQ7wL1bE6oR0aJsCd"
    ents = await det.detect(f"secret={s}")
    assert any(e.type == EntityType.HIGH_ENTROPY_STRING for e in ents)


async def test_risk_class_override():
    # User downgrades EMAIL risk class from MEDIUM to LOW
    det = PrivacyEntityDetector(risk_class_overrides={"email": "low"})
    ents = await det.detect("contact alice@example.com")
    email = next(e for e in ents if e.type == EntityType.EMAIL)
    assert email.risk_class == RiskClass.LOW


async def test_user_regex_extension():
    det = PrivacyEntityDetector(regex_extensions=[r"PROJECT-\d{4}"])
    ents = await det.detect("Internal: PROJECT-1234 ships Friday.")
    assert any(e.type == EntityType.OTHER and "PROJECT-1234" in e.value for e in ents)


async def test_overlap_merge_prefers_higher_risk(detector):
    # An email's local part may overlap with high-entropy detection;
    # detector should pick the structured-type hit over the entropy hit.
    ents = await detector.detect("user.long.unique.identity@corp.example.com")
    types = [e.type for e in ents]
    assert EntityType.EMAIL in types
    assert EntityType.HIGH_ENTROPY_STRING not in types


class _FakeSemantic:
    async def detect(self, raw_message: str, regex_hits):
        return [
            DetectedEntity(
                type=EntityType.NAME,
                span=(0, 5),
                value="Alice",
                risk_class=RiskClass.MEDIUM,
                detector="semantic:fake",
            )
        ]


async def test_semantic_layer_is_merged():
    det = PrivacyEntityDetector(semantic=_FakeSemantic())
    ents = await det.detect("Alice works at the lab.")
    assert any(e.type == EntityType.NAME and e.detector == "semantic:fake" for e in ents)


async def test_noop_semantic_returns_nothing(detector):
    sem = NoopSemanticDetector()
    out = await sem.detect("anything", [])
    assert out == []
