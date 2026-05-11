"""Decider unit tests — recommendation table + AllowedPathSet floor."""

from __future__ import annotations

import pytest

from nanobot.privacy.decider import DeciderInputs, ExecutionDecider
from nanobot.privacy.types import (
    DetectedEntity,
    EntityType,
    ExecutionPath,
    Linkability,
    RiskClass,
)


def _e(t: EntityType, risk: RiskClass) -> DetectedEntity:
    return DetectedEntity(
        type=t,
        span=(0, 0),
        value="",
        risk_class=risk,
        linkability=Linkability.SINGLE_USE,
        confidence=1.0,
    )


@pytest.fixture
def decider() -> ExecutionDecider:
    return ExecutionDecider()


def test_no_entities_recommends_normal(decider):
    rec = decider.decide(DeciderInputs(entities=()))
    assert rec.path == ExecutionPath.NORMAL
    assert ExecutionPath.BLOCKED in rec.allowed


def test_catastrophic_forces_blocked_floor(decider):
    rec = decider.decide(
        DeciderInputs(
            entities=(_e(EntityType.BANK_CARD, RiskClass.CATASTROPHIC),),
            metric_dp_supported=True,
            k_decoy_supported=True,
            local_model_available=True,
        )
    )
    # Even with every capability enabled, CATASTROPHIC stays blocked-only.
    assert rec.path == ExecutionPath.BLOCKED
    assert rec.allowed == frozenset({ExecutionPath.BLOCKED})


def test_credential_type_forces_blocked_floor(decider):
    rec = decider.decide(
        DeciderInputs(
            entities=(_e(EntityType.CREDENTIAL, RiskClass.MEDIUM),),  # misclassified but type wins
            metric_dp_supported=True,
        )
    )
    assert rec.path == ExecutionPath.BLOCKED
    assert rec.allowed == frozenset({ExecutionPath.BLOCKED})


def test_high_risk_routes_to_metric_dp_when_available(decider):
    rec = decider.decide(
        DeciderInputs(
            entities=(_e(EntityType.ID_NUMBER, RiskClass.HIGH),),
            metric_dp_supported=True,
        )
    )
    assert rec.path == ExecutionPath.METRIC_DP


def test_high_risk_blocks_when_metric_dp_unavailable(decider):
    rec = decider.decide(
        DeciderInputs(entities=(_e(EntityType.ID_NUMBER, RiskClass.HIGH),))
    )
    assert rec.path == ExecutionPath.BLOCKED


def test_medium_blocked_in_m1(decider):
    """M1 has neither K_DECOY nor METRIC_DP wired — medium-risk PII must block."""
    rec = decider.decide(
        DeciderInputs(entities=(_e(EntityType.EMAIL, RiskClass.MEDIUM),))
    )
    assert rec.path == ExecutionPath.BLOCKED


def test_medium_prefers_k_decoy_when_supported(decider):
    rec = decider.decide(
        DeciderInputs(
            entities=(_e(EntityType.EMAIL, RiskClass.MEDIUM),),
            k_decoy_supported=True,
        )
    )
    assert rec.path == ExecutionPath.K_DECOY


def test_low_entities_route_to_normal(decider):
    rec = decider.decide(
        DeciderInputs(entities=(_e(EntityType.IP, RiskClass.LOW),))
    )
    assert rec.path == ExecutionPath.NORMAL


def test_allowed_set_includes_blocked_escalation(decider):
    rec = decider.decide(DeciderInputs(entities=()))
    # User can always choose to NOT send the message.
    assert ExecutionPath.BLOCKED in rec.allowed


def test_floor_helper_picks_least_strict(decider):
    rec = decider.decide(DeciderInputs(entities=()))
    assert rec.floor() == ExecutionPath.NORMAL
