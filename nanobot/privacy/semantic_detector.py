"""LLM-backed semantic privacy entity detector.

Second-pass detector for entities the regex layer cannot catch — primarily
names, freeform addresses, medical conditions, and similar context-bound
PII. Plugs in via :class:`nanobot.privacy.detector.SemanticDetector`.

Design notes
------------

* **Fail-closed at the layer boundary, fail-open at the entity boundary.**
  If the backend is unavailable, times out, or returns un-parseable
  garbage, the detector returns ``[]`` and the regex hits stand on their
  own. We never raise — privacy decisions are downstream and depend on
  this method completing.
* **Hallucinations get dropped.** Small models sometimes report values that
  don't actually appear in the text. We locate every reported ``value`` in
  the raw message via ``str.find``; misses are discarded.
* **Type mapping shares the regex layer's risk table.** The LM only needs
  to emit ``type``; risk class is resolved through the same
  ``_resolve_risk`` helper detector.py already uses, so user overrides
  apply uniformly.
* **Confidence is moderate.** Default 0.7 — high enough that the
  decider routes through it, low enough that downstream "uncertain → escalate"
  rules can distinguish from a regex hit.

The prompt is deliberately small for ≤1B parameter local models. Larger
backends will tolerate more guidance, but we trade prompt length for
latency here. Adjust :data:`_PROMPT_TEMPLATE` if you wire a larger
backend and want richer few-shot examples.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from nanobot.privacy.detector import _resolve_risk
from nanobot.privacy.local_model import LocalModelBackend, NullLocalModel
from nanobot.privacy.types import DetectedEntity, EntityType, Linkability

# Categories we ask the LM to consider. We map unknown returns to OTHER so
# typos / made-up categories don't crash; the type mapping then drives risk.
_KNOWN_TYPES: dict[str, EntityType] = {t.value: t for t in EntityType}

_PROMPT_TEMPLATE = """\
You are a strict privacy entity detector. Find personal/sensitive information \
in the text. Reply with JSON only — an array of objects, each:
  {{"type": "<category>", "value": "<exact substring from the text>"}}

Categories: email, phone, id_number, bank_card, address, name, ip, geo, \
medical, credential, jwt, other.

Rules:
- Only flag actual sensitive content; never flag generic words.
- The "value" must be an exact substring that appears in the text.
- If the text has no sensitive content, return [].
- Output JSON only, no prose.

Text:
\"\"\"
{text}
\"\"\"

JSON:"""

# Generation knobs — small models drift past ~256 tokens of pure JSON.
_MAX_TOKENS = 256
_TEMPERATURE = 0.0

# Don't bother calling the LM on tiny inputs (also avoids prompt-cost noise).
_MIN_TEXT_LEN = 4

# Per-call timeout. Local models on CPU sometimes stall; we don't want the
# whole agent turn to hang waiting on the privacy gate.
_DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass
class LLMSemanticDetector:
    """Use a :class:`LocalModelBackend` to do a second-pass entity scan.

    Parameters
    ----------
    backend:
        The model that performs the scan. When ``backend.is_available()``
        is False (e.g. :class:`NullLocalModel`), :meth:`detect` short-circuits
        to ``[]`` without touching the network.
    risk_class_overrides:
        Same shape as :class:`PrivacyEntityDetector` accepts. Lets users
        promote/demote categories the LM emits.
    confidence:
        Score attached to every LM-derived entity. Default 0.7.
    timeout_seconds:
        Per-detect-call wall clock cap. Hitting the timeout returns ``[]``.
    """

    backend: LocalModelBackend
    risk_class_overrides: dict[str, str] | None = None
    confidence: float = 0.7
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    async def detect(
        self,
        raw_message: str,
        regex_hits: list[DetectedEntity],
    ) -> list[DetectedEntity]:
        if not self.backend.is_available():
            return []
        if not raw_message or len(raw_message.strip()) < _MIN_TEXT_LEN:
            return []

        prompt = _PROMPT_TEMPLATE.format(text=raw_message)
        try:
            response = await asyncio.wait_for(
                self.backend.generate(
                    prompt,
                    max_tokens=_MAX_TOKENS,
                    temperature=_TEMPERATURE,
                ),
                timeout=self.timeout_seconds,
            )
        except (asyncio.TimeoutError, Exception):
            return []

        parsed = _parse_json_array(response)
        if not parsed:
            return []

        overrides = self.risk_class_overrides or {}
        entities: list[DetectedEntity] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            type_raw = item.get("type")
            value = item.get("value")
            if not isinstance(type_raw, str) or not isinstance(value, str) or not value:
                continue
            entity_type = _KNOWN_TYPES.get(type_raw.strip().lower(), EntityType.OTHER)
            span = _find_span(raw_message, value)
            if span is None:
                continue  # hallucination — value not in raw text
            entities.append(
                DetectedEntity(
                    type=entity_type,
                    span=span,
                    value=value,
                    risk_class=_resolve_risk(entity_type, overrides),
                    linkability=Linkability.SINGLE_USE,
                    confidence=self.confidence,
                    detector="semantic:llm",
                )
            )
        return entities


# --- helpers ----------------------------------------------------------------


def _parse_json_array(text: str) -> list[object]:
    """Tolerant JSON extractor.

    Small models occasionally wrap their JSON in markdown fences or add
    a sentence before/after. We locate the first ``[`` and the matching
    last ``]`` and try ``json.loads`` on that slice.
    """
    if not text:
        return []
    stripped = text.strip()
    # Strip common ```json ... ``` fences first.
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", stripped, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        candidate = stripped[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _find_span(raw: str, value: str) -> tuple[int, int] | None:
    """Locate *value* in *raw*; return the first match or None."""
    idx = raw.find(value)
    if idx == -1:
        return None
    return (idx, idx + len(value))


def is_available_backend(backend: LocalModelBackend | None) -> bool:
    """Cheap check used by the auto-wire path in :mod:`nanobot.privacy.gate`."""
    if backend is None:
        return False
    if isinstance(backend, NullLocalModel):
        return False
    try:
        return bool(backend.is_available())
    except Exception:
        return False
