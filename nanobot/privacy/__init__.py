"""Privacy GateKeeper — message-level privacy detection, routing, and confirmation.

See `.agent/privacy_gatekeeper.md` for the full design spec.
M1.5 scope: detector + decider + confirmation + audit (BLOCKED/NORMAL paths only).
Local-model abstraction is defined; no backend is wired yet.
"""

from nanobot.privacy import local_model
from nanobot.privacy.gate import GateKeeper
from nanobot.privacy.local_model import LocalModelBackend, NullLocalModel
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
    "LocalModelBackend",
    "NullLocalModel",
    "PathSource",
    "Recommendation",
    "RiskClass",
    "local_model",
]
