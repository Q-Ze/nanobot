"""local_model abstraction tests — protocol shape, NullLocalModel, registry helpers."""

from __future__ import annotations

import pytest

from nanobot.privacy import local_model
from nanobot.privacy.local_model import LocalModelBackend, NullLocalModel


async def test_null_backend_advertises_unavailable():
    b = NullLocalModel()
    assert b.is_available() is False
    assert b.name == "null"
    assert await b.generate("anything") == ""
    assert await b.embed("anything") == []


def test_null_backend_satisfies_protocol():
    assert isinstance(NullLocalModel(), LocalModelBackend)


def test_default_registry_starts_with_null():
    local_model.reset_default()
    assert isinstance(local_model.get_default(), NullLocalModel)


def test_set_default_replaces_registry():
    class _Fake:
        name = "fake"

        def is_available(self) -> bool:
            return True

        async def generate(self, prompt, **_):
            return f"[fake:{prompt}]"

        async def embed(self, text):
            return [0.1, 0.2, 0.3]

    local_model.reset_default()
    try:
        local_model.set_default(_Fake())
        assert local_model.get_default().name == "fake"
    finally:
        local_model.reset_default()


def test_set_default_rejects_non_conforming_backend():
    class _Bogus:
        pass

    with pytest.raises(TypeError):
        local_model.set_default(_Bogus())


async def test_gatekeeper_accepts_injected_backend(tmp_path):
    from nanobot.config.schema import PrivacyConfig
    from nanobot.privacy import GateKeeper
    from nanobot.privacy.types import ExecutionPath

    class _AvailableBackend:
        name = "test"

        def is_available(self) -> bool:
            return True

        async def generate(self, prompt, **_):
            return ""

        async def embed(self, text):
            return []

    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    # Even with an available backend injected, M1.5 still hard-disables SIMPLE.
    gate = GateKeeper.from_config(cfg, local_model=_AvailableBackend())
    rec = await gate.detect_and_recommend("Hello world")
    assert ExecutionPath.SIMPLE not in rec.allowed
