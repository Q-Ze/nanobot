"""Privacy GateKeeper — message-level privacy detection, routing, and confirmation.

See `.agent/privacy_gatekeeper.md` for the full design spec.
M1 scope: detector + decider + confirmation + audit (BLOCKED/SIMPLE/NORMAL paths only).
"""

from nanobot.privacy.gate import GateKeeper
from nanobot.privacy.types import (
    AllowedPathSet,
    AuditView,
    ChannelCapabilities,
    Decision,
    DetectedEntity,
    EntityType,
    ExecutionPath,
    PathSource,
    Recommendation,
    RiskClass,
)

__all__ = [
    "AllowedPathSet",
    "AuditView",
    "ChannelCapabilities",
    "Decision",
    "DetectedEntity",
    "EntityType",
    "ExecutionPath",
    "GateKeeper",
    "PathSource",
    "Recommendation",
    "RiskClass",
]
