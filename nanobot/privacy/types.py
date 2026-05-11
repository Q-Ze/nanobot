"""Shared types for the privacy GateKeeper.

Definitions track `.agent/privacy_gatekeeper.md` §2/§3/§4. Keep this module free of
runtime dependencies so detector/decider/confirmation/audit/gate can all import
from here without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, FrozenSet, Optional


class ExecutionPath(str, Enum):
    """The five routing paths defined in §3.2 of the design spec."""

    BLOCKED = "blocked"
    SIMPLE = "simple"
    METRIC_DP = "metric_dp"
    K_DECOY = "k_decoy"
    NORMAL = "normal"


_STRICTNESS_ORDER: dict[ExecutionPath, int] = {
    ExecutionPath.NORMAL: 0,
    ExecutionPath.K_DECOY: 1,
    ExecutionPath.METRIC_DP: 2,
    ExecutionPath.SIMPLE: 3,
    ExecutionPath.BLOCKED: 4,
}


def is_at_least_as_strict(a: ExecutionPath, b: ExecutionPath) -> bool:
    """True if path *a* offers at least as much protection as *b* (per §3.2 偏序)."""
    return _STRICTNESS_ORDER[a] >= _STRICTNESS_ORDER[b]


class RiskClass(str, Enum):
    """Risk tiers from §4.1. Ordered LOW < MEDIUM < HIGH < CATASTROPHIC."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CATASTROPHIC = "catastrophic"


_RISK_ORDER: dict[RiskClass, int] = {
    RiskClass.LOW: 0,
    RiskClass.MEDIUM: 1,
    RiskClass.HIGH: 2,
    RiskClass.CATASTROPHIC: 3,
}


def risk_at_least(a: RiskClass, b: RiskClass) -> bool:
    return _RISK_ORDER[a] >= _RISK_ORDER[b]


def max_risk(risks: list[RiskClass]) -> RiskClass:
    """Return the highest risk class in *risks*; assumes non-empty input."""
    return max(risks, key=lambda r: _RISK_ORDER[r])


class EntityType(str, Enum):
    """Detected entity categories. Maps to RiskClass via §4.1 (with config override)."""

    EMAIL = "email"
    PHONE = "phone"
    ID_NUMBER = "id_number"
    BANK_CARD = "bank_card"
    ADDRESS = "address"
    NAME = "name"
    IP = "ip"
    GEO = "geo"
    MEDICAL = "medical"
    CREDENTIAL = "credential"
    KEY_MATERIAL = "key_material"
    INTERNAL_INSTRUCTION = "internal_instruction"
    JWT = "jwt"
    HIGH_ENTROPY_STRING = "high_entropy_string"
    OTHER = "other"


class Linkability(str, Enum):
    SINGLE_USE = "single_use"
    RECURRENT_IN_SESSION = "recurrent_in_session"
    RECURRENT_CROSS_SESSION = "recurrent_cross_session"


class PathSource(str, Enum):
    """How the final path was selected (used in audit)."""

    SYSTEM_AUTO = "system_auto"
    USER_PRESELECTED = "user_preselected"
    USER_CONFIRMED = "user_confirmed"
    FALLBACK_TIMEOUT = "fallback_timeout"
    FALLBACK_NO_CONFIRM = "fallback_no_confirm"


class ConfirmationMode(str, Enum):
    ALWAYS = "always"
    RISK_THRESHOLD = "risk_threshold"
    NEVER = "never"


class ChannelFallback(str, Enum):
    FORCED_CONSERVATIVE = "forced_conservative"
    USE_RECOMMENDED = "use_recommended"
    REJECT = "reject"


@dataclass(frozen=True)
class DetectedEntity:
    """A single privacy entity discovered by the detector."""

    type: EntityType
    span: tuple[int, int]                  # (start, end) byte offsets in raw_message
    value: str                             # the matched substring (kept local; never logged plain)
    risk_class: RiskClass
    linkability: Linkability = Linkability.SINGLE_USE
    confidence: float = 1.0                # 0..1
    domain_entropy_bits: float | None = None
    detector: str = "regex"                # "regex" | "semantic" | "user_pattern"


# AllowedPathSet is a frozenset for cheap subset checks; alias for clarity.
AllowedPathSet = FrozenSet[ExecutionPath]


@dataclass(frozen=True)
class Recommendation:
    """Output of ExecutionDecider — proposed path plus the user-overridable set."""

    path: ExecutionPath
    allowed: AllowedPathSet
    reason: str
    entities: tuple[DetectedEntity, ...] = ()

    def floor(self) -> ExecutionPath:
        """The weakest (most permissive) path the user may choose."""
        if not self.allowed:
            return self.path
        return min(self.allowed, key=lambda p: _STRICTNESS_ORDER[p])


@dataclass(frozen=True)
class Decision:
    """Final decision after ConfirmationGate. Consumed by transformer + audit."""

    path: ExecutionPath
    source: PathSource
    recommendation: Recommendation
    user_choice_latency_ms: int | None = None
    violation_attempt: ExecutionPath | None = None  # user predfered a sub-floor path
    refusal_message: str = ""                       # populated when path == BLOCKED


@dataclass(frozen=True)
class AuditView:
    """Lightweight, JSON-serializable view of a single GateKeeper turn for downstream metadata."""

    path: ExecutionPath
    recommended_path: ExecutionPath
    source: PathSource
    entity_counts: dict[str, int]
    risk_counts: dict[str, int]
    reason: str
    fidelity: str = "EXACT"                # EXACT | RESTORED_LOSSY (M3+)
    eps_consumed: float = 0.0              # populated by accountant in M2/M3


@dataclass(frozen=True)
class TransformOutcome:
    """What `GateKeeper.transform` produces. M1 == passthrough for SIMPLE/NORMAL."""

    decision: Decision
    privacy_message: str | list[str]       # str for single-shot paths; list for K_DECOY (M2+)
    restoration_plan: dict[str, Any] = field(default_factory=dict)
    audit_view: AuditView | None = None


@dataclass
class ChannelCapabilities:
    """Declared by each channel (or supplied by a stub for tests/SDK).

    `send_confirmation` and `await_confirmation_reply` are only invoked when
    `supports_interactive_confirm` is True. They are awaited from inside
    AgentLoop while the turn is suspended.
    """

    supports_interactive_confirm: bool = False
    confirmation_max_latency_seconds: int = 60
    send_confirmation: Optional[
        Callable[[str, "ConfirmationPrompt"], Awaitable[None]]
    ] = None
    await_confirmation_reply: Optional[
        Callable[[str, float], Awaitable["ConfirmationReply | None"]]
    ] = None


@dataclass(frozen=True)
class ConfirmationPrompt:
    """Payload sent to the user when ConfirmationGate decides to ask."""

    confirmation_id: str
    recommendation: Recommendation
    chat_id: str
    eps_remaining_session: float | None = None


@dataclass(frozen=True)
class ConfirmationReply:
    confirmation_id: str
    chosen_path: ExecutionPath | None       # None == "cancel/withdraw"
