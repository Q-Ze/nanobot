"""ExecutionDecider — produces a Recommendation per `.agent/privacy_gatekeeper.md` §3.2.

M1 only routes to BLOCKED / NORMAL (and SIMPLE if `local_model` is configured).
K_DECOY and METRIC_DP collapse to BLOCKED with reason "not_implemented_yet"
to keep the safety floor even when the doc tree later asks for them.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanobot.privacy.types import (
    AllowedPathSet,
    DetectedEntity,
    EntityType,
    ExecutionPath,
    Recommendation,
    RiskClass,
    is_at_least_as_strict,
    max_risk,
)

# Hard-blocked entity types (cannot be downgraded by user override; see §4.5).
_HARD_BLOCK_TYPES = frozenset(
    {EntityType.KEY_MATERIAL, EntityType.CREDENTIAL, EntityType.INTERNAL_INSTRUCTION, EntityType.JWT}
)


@dataclass(frozen=True)
class DeciderInputs:
    """Inputs collected by the GateKeeper before invoking the decider."""

    entities: tuple[DetectedEntity, ...]
    task_difficulty: float = 0.0       # 0..1; M1 ignores
    local_model_available: bool = False
    k_decoy_supported: bool = False    # M2+
    metric_dp_supported: bool = False  # M3+


class ExecutionDecider:
    """Pure function over (entities, capabilities) -> Recommendation."""

    def decide(self, inputs: DeciderInputs) -> Recommendation:
        ents = inputs.entities

        # Hard floor: catastrophic risk or hard-blocked type → BLOCKED, no overrides.
        for e in ents:
            if e.risk_class == RiskClass.CATASTROPHIC or e.type in _HARD_BLOCK_TYPES:
                return Recommendation(
                    path=ExecutionPath.BLOCKED,
                    allowed=frozenset({ExecutionPath.BLOCKED}),
                    reason=_blocked_reason(e),
                    entities=ents,
                )

        # Recommendation
        rec = self._recommend(inputs)

        # Allowed set: every path strictly more protective than rec, plus rec itself,
        # gated by whichever capabilities the runtime currently supports.
        allowed = self._allowed_set(rec, inputs)
        return Recommendation(path=rec, allowed=allowed, reason=_recommend_reason(rec, ents), entities=ents)

    # --- internals ---------------------------------------------------------------------

    def _recommend(self, inputs: DeciderInputs) -> ExecutionPath:
        ents = inputs.entities
        if not ents:
            return ExecutionPath.NORMAL

        worst = max_risk([e.risk_class for e in ents])
        if worst == RiskClass.HIGH:
            if inputs.metric_dp_supported:
                return ExecutionPath.METRIC_DP
            return ExecutionPath.BLOCKED  # fail-closed when no metric-DP path available
        if worst == RiskClass.MEDIUM:
            # MEDIUM: prefer K_DECOY when available, else METRIC_DP, else BLOCKED.
            if inputs.k_decoy_supported:
                return ExecutionPath.K_DECOY
            if inputs.metric_dp_supported:
                return ExecutionPath.METRIC_DP
            return ExecutionPath.BLOCKED  # M1 reaches here for medium entities
        # LOW: fine to send as-is (per §4.5).
        return ExecutionPath.NORMAL

    def _allowed_set(self, rec: ExecutionPath, inputs: DeciderInputs) -> AllowedPathSet:
        # Always include rec.
        allowed: set[ExecutionPath] = {rec}
        # User can always escalate to BLOCKED (zero-cost rejection).
        allowed.add(ExecutionPath.BLOCKED)
        # SIMPLE only if local model is wired up (M3+ surfacing).
        if inputs.local_model_available:
            allowed.add(ExecutionPath.SIMPLE)
        # METRIC_DP escalation when user wants stronger formal guarantee than K_DECOY.
        if inputs.metric_dp_supported and is_at_least_as_strict(ExecutionPath.METRIC_DP, rec):
            allowed.add(ExecutionPath.METRIC_DP)
        # K_DECOY when user prefers it over NORMAL.
        if inputs.k_decoy_supported and is_at_least_as_strict(ExecutionPath.K_DECOY, rec):
            allowed.add(ExecutionPath.K_DECOY)
        return frozenset(allowed)


# --- reason strings ---


def _blocked_reason(e: DetectedEntity) -> str:
    if e.type in _HARD_BLOCK_TYPES:
        return f"hard_secret:{e.type.value}"
    return f"catastrophic_risk:{e.type.value}"


def _recommend_reason(rec: ExecutionPath, ents: tuple[DetectedEntity, ...]) -> str:
    if not ents:
        return "no_privacy_entity_detected"
    if rec == ExecutionPath.BLOCKED:
        return "no_anonymization_path_available_yet"
    return f"recommended_{rec.value}"
