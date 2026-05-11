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

from loguru import logger

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

# Per-call timeout. Local fast models on a workstation are typically <2 s; the
# default budget is sized for slower / cloud-backed local roles. Overridable
# via PrivacyConfig.semantic_timeout_seconds (see schema.py).
_DEFAULT_TIMEOUT_SECONDS = 15.0


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
        Per-detect-call wall clock cap. Hitting the timeout returns ``[]``
        and emits a warning log so users can spot the misconfiguration.
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
        except asyncio.TimeoutError:
            logger.warning(
                "privacy.semantic: LM call exceeded timeout={}s — returning no entities. "
                "Increase privacy.semantic_timeout_seconds in config if your backend is slow.",
                self.timeout_seconds,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "privacy.semantic: backend.generate raised {!r} — returning no entities.",
                exc,
            )
            return []

        parsed = _parse_json_array(response)
        if not parsed:
            if response:
                logger.debug(
                    "privacy.semantic: LM returned non-JSON {!r:.120} — no entities.", response
                )
            return []

        overrides = self.risk_class_overrides or {}
        entities: list[DetectedEntity] = []
        dropped_hallucinations = 0
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
                dropped_hallucinations += 1
                continue
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
        if dropped_hallucinations:
            logger.debug(
                "privacy.semantic: dropped {} hallucinated entit{} (value not found in source)",
                dropped_hallucinations,
                "y" if dropped_hallucinations == 1 else "ies",
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
