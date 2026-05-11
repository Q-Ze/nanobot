"""AuditLogger — append-only privacy decision log.

M1 writes JSONL to a directory outside the agent workspace (so the agent's
filesystem tools cannot accidentally read it). Envelope encryption arrives
in M4; this module deliberately stores no raw entity values.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from nanobot.privacy.types import (
    AuditView,
    Decision,
    DetectedEntity,
    Recommendation,
)


class AuditLogger:
    def __init__(self, log_dir: str | os.PathLike[str], enabled: bool = True) -> None:
        self._enabled = enabled
        self._dir = Path(os.path.expanduser(str(log_dir)))
        self._lock = threading.Lock()
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def build_view(
        self,
        decision: Decision,
        entities: tuple[DetectedEntity, ...],
    ) -> AuditView:
        return AuditView(
            path=decision.path,
            recommended_path=decision.recommendation.path,
            source=decision.source,
            entity_counts=_counts(e.type.value for e in entities),
            risk_counts=_counts(e.risk_class.value for e in entities),
            reason=decision.recommendation.reason,
        )

    def record(
        self,
        *,
        session_key: str,
        decision: Decision,
        entities: tuple[DetectedEntity, ...],
        view: AuditView | None = None,
    ) -> None:
        if not self._enabled:
            return
        view = view or self.build_view(decision, entities)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id_hash": _hash_session(session_key),
            "path": view.path.value,
            "recommended_path": view.recommended_path.value,
            "path_source": view.source.value,
            "user_choice_latency_ms": decision.user_choice_latency_ms,
            "violation_attempt": decision.violation_attempt.value if decision.violation_attempt else None,
            "user_downgrade": _is_downgrade(decision),
            "entity_counts": view.entity_counts,
            "risk_counts": view.risk_counts,
            "decision_reason": view.reason,
            "fidelity": view.fidelity,
            "eps_consumed": view.eps_consumed,
        }
        path = self._dir / f"audit-{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (OSError, AttributeError):
                    pass


def _counts(values) -> dict[str, int]:
    return dict(Counter(values))


def _hash_session(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _is_downgrade(decision: Decision) -> bool:
    """User choice was strictly weaker than the system recommendation."""
    from nanobot.privacy.types import is_at_least_as_strict

    return decision.path != decision.recommendation.path and not is_at_least_as_strict(
        decision.path, decision.recommendation.path
    )


# expose Recommendation for callers that import audit only
__all__ = ["AuditLogger", "Recommendation"]
