"""PrivacyEntityDetector — regex layer + semantic-LM hook.

M1: regex layer fully implemented; semantic layer is a no-op stub the rest of
the system can replace later (tests inject a fake to exercise multi-detector
merging). See `.agent/privacy_gatekeeper.md` §3.1.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from nanobot.privacy.types import (
    DetectedEntity,
    EntityType,
    Linkability,
    RiskClass,
)

# --- Risk class default mapping (overridable via PrivacyConfig.risk_class_overrides) ----

_DEFAULT_RISK: dict[EntityType, RiskClass] = {
    EntityType.KEY_MATERIAL: RiskClass.CATASTROPHIC,
    EntityType.CREDENTIAL: RiskClass.CATASTROPHIC,
    EntityType.INTERNAL_INSTRUCTION: RiskClass.CATASTROPHIC,
    EntityType.JWT: RiskClass.CATASTROPHIC,
    EntityType.BANK_CARD: RiskClass.HIGH,
    EntityType.ID_NUMBER: RiskClass.HIGH,
    EntityType.MEDICAL: RiskClass.HIGH,
    EntityType.HIGH_ENTROPY_STRING: RiskClass.HIGH,
    EntityType.EMAIL: RiskClass.MEDIUM,
    EntityType.PHONE: RiskClass.MEDIUM,
    EntityType.NAME: RiskClass.MEDIUM,
    EntityType.ADDRESS: RiskClass.MEDIUM,
    EntityType.GEO: RiskClass.MEDIUM,
    EntityType.IP: RiskClass.LOW,
    EntityType.OTHER: RiskClass.LOW,
}


def _resolve_risk(t: EntityType, overrides: dict[str, str] | None) -> RiskClass:
    if overrides and (key := t.value) in overrides:
        return RiskClass(overrides[key])
    return _DEFAULT_RISK[t]


# --- Regex catalogue ------------------------------------------------------------------
#
# Patterns are conservative: precision matters more than recall here, because false
# positives become user-visible confirmation noise. The semantic layer (M1: noop;
# later: small LM) exists to catch what regex misses.

_RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_RE_PHONE_E164 = re.compile(r"\+\d{1,3}(?:[\s\-]?\d{2,4}){2,5}\b")
# Mainland China mobile (1[3-9]xxxxxxxxx); avoid matching longer numeric IDs by
# requiring word boundaries on both sides.
_RE_PHONE_CN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_RE_ID_CN = re.compile(
    r"(?<!\d)\d{17}[\dXx](?!\d)"
)  # 18-digit Chinese ID; checksum verified separately
_RE_BANK = re.compile(r"(?<!\d)\d{13,19}(?!\d)")  # bank-card length range; Luhn-validated below
_RE_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)
_RE_IPV6 = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b")

# Cloud / SaaS credential prefixes — high precision, chosen to avoid catching prose.
# The `sk-` family (OpenAI, DeepSeek, Moonshot, …) accepts ≥8 chars after the prefix
# so test/fake keys also trigger; real keys are far longer.
_CRED_PREFIXES = [
    ("aws_access_key_id", r"\bAKIA[0-9A-Z]{16}\b"),
    ("aws_secret", r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),  # context-checked below
    ("google_api", r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    ("anthropic", r"\bsk-ant-[A-Za-z0-9\-_]{8,}\b"),
    ("sk_prefixed", r"\bsk-(?!ant-)[A-Za-z0-9_\-]{8,}\b"),  # OpenAI/DeepSeek/Moonshot etc.
    ("github_pat", r"\bghp_[A-Za-z0-9]{20,}\b"),
    ("github_oauth", r"\bgho_[A-Za-z0-9]{20,}\b"),
    ("github_user", r"\bghu_[A-Za-z0-9]{20,}\b"),
    ("github_server", r"\bghs_[A-Za-z0-9]{20,}\b"),
    ("github_refresh", r"\bghr_[A-Za-z0-9]{20,}\b"),
    ("slack_bot", r"\bxox[abp]-[A-Za-z0-9\-]{10,}\b"),
    ("stripe_live", r"\bsk_live_[A-Za-z0-9]{16,}\b"),
    ("stripe_test", r"\bsk_test_[A-Za-z0-9]{16,}\b"),
    ("hf_token", r"\bhf_[A-Za-z0-9]{20,}\b"),
]
_RE_CREDENTIALS = [(name, re.compile(p)) for name, p in _CRED_PREFIXES]

_RE_SSH_PRIVATE = re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP) PRIVATE KEY-----")
_RE_PEM_BLOCK = re.compile(r"-----BEGIN [A-Z ]+-----")  # broader fallback
_RE_JWT = re.compile(r"\beyJ[A-Za-z0-9\-_]{8,}\.[A-Za-z0-9\-_]{8,}\.[A-Za-z0-9\-_]{8,}\b")

# High-entropy string: configurable per design. Length ≥ 20, Shannon entropy ≥ 4.0 bit/char.
_HIGH_ENTROPY_LEN = 20
_HIGH_ENTROPY_BITS = 4.0


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _luhn_ok(num: str) -> bool:
    digits = [int(d) for d in num if d.isdigit()]
    if not digits:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _id_cn_checksum_ok(s: str) -> bool:
    """Validate Chinese 18-digit ID checksum (GB 11643)."""
    if len(s) != 18:
        return False
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    table = "10X98765432"
    try:
        total = sum(int(s[i]) * weights[i] for i in range(17))
    except ValueError:
        return False
    return table[total % 11].lower() == s[17].lower()


# --- Semantic-detector hook ----------------------------------------------------------


class SemanticDetector(Protocol):
    """Pluggable second-pass detector. M1 default is NoopSemanticDetector."""

    async def detect(
        self, raw_message: str, regex_hits: list[DetectedEntity]
    ) -> list[DetectedEntity]: ...


class NoopSemanticDetector:
    async def detect(
        self, raw_message: str, regex_hits: list[DetectedEntity]
    ) -> list[DetectedEntity]:
        return []


# --- Detector --------------------------------------------------------------------------


@dataclass
class _RawHit:
    type: EntityType
    span: tuple[int, int]
    value: str
    detector: str = "regex"
    confidence: float = 1.0


class PrivacyEntityDetector:
    """Two-pass detector. Regex first; semantic LM (optional) second."""

    def __init__(
        self,
        risk_class_overrides: dict[str, str] | None = None,
        regex_extensions: list[str] | None = None,
        semantic: SemanticDetector | None = None,
    ) -> None:
        self._overrides = risk_class_overrides or {}
        self._user_patterns = [re.compile(p) for p in (regex_extensions or [])]
        self._semantic: SemanticDetector = semantic or NoopSemanticDetector()

    async def detect(self, raw_message: str) -> list[DetectedEntity]:
        regex_hits = self._run_regex_layer(raw_message)
        semantic_hits = await self._semantic.detect(raw_message, regex_hits)
        merged = _merge_overlaps(regex_hits + semantic_hits)
        return merged

    # --- internals ---------------------------------------------------------------------

    def _run_regex_layer(self, text: str) -> list[DetectedEntity]:
        hits: list[_RawHit] = []

        # CATASTROPHIC patterns first so they win in overlap merge.
        for m in _RE_SSH_PRIVATE.finditer(text):
            hits.append(_RawHit(EntityType.KEY_MATERIAL, m.span(), m.group(), "regex"))
        for m in _RE_PEM_BLOCK.finditer(text):
            # Only count PEM blocks not already covered by SSH detection.
            if not any(h.type == EntityType.KEY_MATERIAL and _spans_overlap(h.span, m.span()) for h in hits):
                hits.append(_RawHit(EntityType.KEY_MATERIAL, m.span(), m.group(), "regex", 0.7))
        for m in _RE_JWT.finditer(text):
            hits.append(_RawHit(EntityType.JWT, m.span(), m.group(), "regex"))
        for name, pat in _RE_CREDENTIALS:
            for m in pat.finditer(text):
                # AWS secret regex is broad — only count when an access-key id appears nearby.
                if name == "aws_secret":
                    window = text[max(0, m.start() - 200) : m.end() + 200]
                    if not re.search(r"AKIA[0-9A-Z]{16}", window):
                        continue
                hits.append(_RawHit(EntityType.CREDENTIAL, m.span(), m.group(), f"regex:{name}"))

        # HIGH
        for m in _RE_ID_CN.finditer(text):
            v = m.group()
            if _id_cn_checksum_ok(v):
                hits.append(_RawHit(EntityType.ID_NUMBER, m.span(), v, "regex:cn_id"))
        for m in _RE_BANK.finditer(text):
            v = m.group()
            if _luhn_ok(v) and not _looks_like_id_or_phone(v):
                hits.append(_RawHit(EntityType.BANK_CARD, m.span(), v, "regex:luhn"))

        # MEDIUM
        for m in _RE_EMAIL.finditer(text):
            hits.append(_RawHit(EntityType.EMAIL, m.span(), m.group(), "regex"))
        for m in _RE_PHONE_E164.finditer(text):
            hits.append(_RawHit(EntityType.PHONE, m.span(), m.group(), "regex:e164"))
        for m in _RE_PHONE_CN.finditer(text):
            hits.append(_RawHit(EntityType.PHONE, m.span(), m.group(), "regex:cn_mobile"))

        # LOW
        for m in _RE_IPV4.finditer(text):
            hits.append(_RawHit(EntityType.IP, m.span(), m.group(), "regex:ipv4"))
        for m in _RE_IPV6.finditer(text):
            hits.append(_RawHit(EntityType.IP, m.span(), m.group(), "regex:ipv6"))

        # User extensions
        for pat in self._user_patterns:
            for m in pat.finditer(text):
                hits.append(_RawHit(EntityType.OTHER, m.span(), m.group(), "regex:user"))

        # High-entropy fallback (catches arbitrary unknown secrets, last so other rules win)
        for token in _word_spans(text):
            substr = text[token[0] : token[1]]
            if (
                len(substr) >= _HIGH_ENTROPY_LEN
                and _shannon_entropy(substr) >= _HIGH_ENTROPY_BITS
                and not any(_spans_overlap(token, h.span) for h in hits)
            ):
                hits.append(
                    _RawHit(
                        EntityType.HIGH_ENTROPY_STRING,
                        token,
                        substr,
                        "regex:entropy",
                        confidence=0.6,
                    )
                )

        return [
            DetectedEntity(
                type=h.type,
                span=h.span,
                value=h.value,
                risk_class=_resolve_risk(h.type, self._overrides),
                linkability=Linkability.SINGLE_USE,
                confidence=h.confidence,
                detector=h.detector,
            )
            for h in hits
        ]


# --- helpers --------------------------------------------------------------------------


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _looks_like_id_or_phone(num: str) -> bool:
    """Distinguish phone/ID-number false positives from real bank cards."""
    return len(num) in (11, 17, 18)


def _word_spans(text: str) -> Iterable[tuple[int, int]]:
    for m in re.finditer(r"\S+", text):
        yield m.span()


def _merge_overlaps(entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Drop strictly-contained or duplicate spans, keeping the higher-risk hit."""
    if not entities:
        return entities
    risk_rank = {RiskClass.CATASTROPHIC: 3, RiskClass.HIGH: 2, RiskClass.MEDIUM: 1, RiskClass.LOW: 0}
    sorted_ents = sorted(
        entities, key=lambda e: (-risk_rank[e.risk_class], e.span[0], -(e.span[1] - e.span[0]))
    )
    kept: list[DetectedEntity] = []
    for e in sorted_ents:
        if any(_spans_overlap(e.span, k.span) for k in kept):
            continue
        kept.append(e)
    return sorted(kept, key=lambda e: e.span[0])
