"""Tests for the LLM-backed SemanticDetector."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from nanobot.privacy.local_model import NullLocalModel
from nanobot.privacy.semantic_detector import (
    LLMSemanticDetector,
    _find_span,
    _parse_json_array,
    is_available_backend,
)
from nanobot.privacy.types import EntityType, Linkability, RiskClass

# --- low-level parser helpers ---------------------------------------------------------


def test_parse_json_array_strips_markdown_fence():
    raw = '```json\n[{"type": "name", "value": "Alice"}]\n```'
    assert _parse_json_array(raw) == [{"type": "name", "value": "Alice"}]


def test_parse_json_array_extracts_from_prose():
    raw = 'Sure, here it is: [{"type": "email", "value": "x@y.z"}] hope this helps!'
    assert _parse_json_array(raw) == [{"type": "email", "value": "x@y.z"}]


def test_parse_json_array_handles_empty_array():
    assert _parse_json_array("[]") == []


def test_parse_json_array_returns_empty_on_garbage():
    assert _parse_json_array("not json at all") == []
    assert _parse_json_array("") == []
    assert _parse_json_array('{"not": "an array"}') == []


def test_find_span_locates_first_match():
    raw = "Hello Alice, this is Alice again."
    assert _find_span(raw, "Alice") == (6, 11)


def test_find_span_returns_none_for_hallucination():
    assert _find_span("Hello world", "Bob") is None


# --- backend-availability helper ------------------------------------------------------


def test_is_available_backend_handles_null_and_none():
    assert not is_available_backend(None)
    assert not is_available_backend(NullLocalModel())


def test_is_available_backend_swallows_exceptions():
    class _Broken:
        name = "broken"

        def is_available(self):
            raise RuntimeError("oops")

        async def generate(self, prompt, **_):
            return ""

        async def embed(self, text):
            return []

    assert not is_available_backend(_Broken())


# --- LLMSemanticDetector --------------------------------------------------------------


def _backend(reply: str):
    class _Stub:
        name = "stub"

        def is_available(self):
            return True

        async def generate(self, prompt, **_):
            return reply

        async def embed(self, text):
            return []

    return _Stub()


@pytest.mark.asyncio
async def test_detects_name_missed_by_regex():
    detector = LLMSemanticDetector(
        backend=_backend('[{"type": "name", "value": "Alice"}]')
    )
    out = await detector.detect("Alice works at the lab.", regex_hits=[])
    assert len(out) == 1
    e = out[0]
    assert e.type == EntityType.NAME
    assert e.value == "Alice"
    assert e.span == (0, 5)
    assert e.risk_class == RiskClass.MEDIUM   # default risk for NAME
    assert e.linkability == Linkability.SINGLE_USE
    assert e.detector == "semantic:llm"
    assert 0.0 < e.confidence < 1.0


@pytest.mark.asyncio
async def test_drops_hallucinated_value():
    detector = LLMSemanticDetector(
        backend=_backend('[{"type": "name", "value": "Bob"}]')   # Bob isn't in the text
    )
    out = await detector.detect("Alice works at the lab.", regex_hits=[])
    assert out == []


@pytest.mark.asyncio
async def test_unknown_type_maps_to_other():
    detector = LLMSemanticDetector(
        backend=_backend('[{"type": "favourite-color", "value": "Alice"}]')
    )
    out = await detector.detect("Alice", regex_hits=[])
    assert len(out) == 1
    assert out[0].type == EntityType.OTHER


@pytest.mark.asyncio
async def test_returns_empty_when_backend_unavailable():
    detector = LLMSemanticDetector(backend=NullLocalModel())
    out = await detector.detect("Alice works at the lab.", regex_hits=[])
    assert out == []


@pytest.mark.asyncio
async def test_returns_empty_on_garbage_response():
    detector = LLMSemanticDetector(backend=_backend("definitely not json"))
    out = await detector.detect("Alice works at the lab.", regex_hits=[])
    assert out == []


@pytest.mark.asyncio
async def test_returns_empty_on_short_text():
    detector = LLMSemanticDetector(backend=_backend('[{"type":"name","value":"x"}]'))
    out = await detector.detect("hi", regex_hits=[])
    assert out == []


@pytest.mark.asyncio
async def test_returns_empty_on_timeout():
    class _Slow:
        name = "slow"

        def is_available(self):
            return True

        async def generate(self, prompt, **_):
            await asyncio.sleep(2.0)
            return "[]"

        async def embed(self, text):
            return []

    detector = LLMSemanticDetector(backend=_Slow(), timeout_seconds=0.05)
    out = await detector.detect("Alice works at the lab.", regex_hits=[])
    assert out == []


@pytest.mark.asyncio
async def test_swallows_backend_exceptions():
    class _Boom:
        name = "boom"

        def is_available(self):
            return True

        async def generate(self, prompt, **_):
            raise RuntimeError("upstream gone")

        async def embed(self, text):
            return []

    detector = LLMSemanticDetector(backend=_Boom())
    out = await detector.detect("Alice works at the lab.", regex_hits=[])
    assert out == []


@pytest.mark.asyncio
async def test_risk_class_overrides_apply():
    detector = LLMSemanticDetector(
        backend=_backend('[{"type": "name", "value": "Alice"}]'),
        risk_class_overrides={"name": "low"},
    )
    out = await detector.detect("Alice works at the lab.", regex_hits=[])
    assert len(out) == 1
    assert out[0].risk_class == RiskClass.LOW


@pytest.mark.asyncio
async def test_skips_malformed_items_but_keeps_others():
    detector = LLMSemanticDetector(
        backend=_backend(
            '[{"type": "name", "value": "Alice"},'
            ' {"value": "missing type"},'
            ' "bare string",'
            ' {"type": "email", "value": "x@y.z"}]'
        )
    )
    out = await detector.detect("Alice at x@y.z", regex_hits=[])
    types = sorted(e.type.value for e in out)
    assert types == ["email", "name"]


# --- auto-wiring through GateKeeper.from_config ---------------------------------------


@pytest.mark.asyncio
async def test_gate_auto_wires_semantic_when_backend_available(tmp_path):
    """GateKeeper.from_config plugs LLMSemanticDetector in automatically."""
    from nanobot.config.schema import PrivacyConfig
    from nanobot.privacy import GateKeeper

    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    backend = _backend('[{"type": "name", "value": "Alice"}]')
    gate = GateKeeper.from_config(cfg, local_model=backend)

    rec = await gate.detect_and_recommend("Alice works at the lab.")
    types = [e.type for e in rec.entities]
    assert EntityType.NAME in types


@pytest.mark.asyncio
async def test_gate_skips_auto_wire_when_backend_null(tmp_path):
    from nanobot.config.schema import PrivacyConfig
    from nanobot.privacy import GateKeeper

    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    gate = GateKeeper.from_config(cfg)  # no backend → no semantic detector

    rec = await gate.detect_and_recommend("Alice works at the lab.")
    # Regex doesn't know names — without LM, nothing detected.
    assert all(e.type != EntityType.NAME for e in rec.entities)


@pytest.mark.asyncio
async def test_explicit_semantic_detector_overrides_auto_wire(tmp_path):
    """Caller-supplied semantic detector is honoured even when a backend exists."""
    from nanobot.config.schema import PrivacyConfig
    from nanobot.privacy import GateKeeper

    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    custom = AsyncMock()
    custom.detect = AsyncMock(return_value=[])

    gate = GateKeeper.from_config(
        cfg,
        local_model=_backend('[{"type": "name", "value": "Alice"}]'),
        semantic_detector=custom,
    )
    await gate.detect_and_recommend("Alice works at the lab.")
    custom.detect.assert_awaited()


# --- integration with regex layer (dedup) ---------------------------------------------


@pytest.mark.asyncio
async def test_regex_email_takes_priority_over_semantic_duplicate(tmp_path):
    """If both regex and LM flag the same span, the regex (higher-confidence)
    hit must win so we don't show 'two emails' in the confirmation prompt.
    """
    from nanobot.config.schema import PrivacyConfig
    from nanobot.privacy import GateKeeper

    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    backend = _backend('[{"type": "email", "value": "alice@example.com"}]')
    gate = GateKeeper.from_config(cfg, local_model=backend)
    rec = await gate.detect_and_recommend("Email me at alice@example.com")
    emails = [e for e in rec.entities if e.type == EntityType.EMAIL]
    assert len(emails) == 1
    # Regex layer's confidence is higher (1.0) than the semantic default (0.7),
    # so the kept entity is the regex one.
    assert emails[0].detector.startswith("regex")
