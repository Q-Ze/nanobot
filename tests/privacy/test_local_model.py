"""local_model abstraction tests — protocol shape, NullLocalModel, registry helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.privacy import local_model
from nanobot.privacy.local_model import (
    LLMProviderBackend,
    LocalModelBackend,
    NullLocalModel,
    build_from_root_config,
)


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


# --- LLMProviderBackend adapter -------------------------------------------------------


def _fake_response(content):
    r = MagicMock()
    r.content = content
    return r


async def test_llm_provider_backend_generates_text_via_chat():
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=_fake_response("decoy answer"))
    backend = LLMProviderBackend(provider=provider, model="ollama/qwen2.5:0.5b")

    assert backend.is_available()
    assert backend.name == "llm:ollama/qwen2.5:0.5b"

    result = await backend.generate("say hi", max_tokens=64, temperature=0.0)
    assert result == "decoy answer"
    # Provider received the model override + a properly-formed message.
    args, kwargs = provider.chat.await_args
    assert kwargs["model"] == "ollama/qwen2.5:0.5b"
    assert kwargs["messages"] == [{"role": "user", "content": "say hi"}]
    assert kwargs["max_tokens"] == 64


async def test_llm_provider_backend_swallows_exceptions():
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=RuntimeError("upstream timeout"))
    backend = LLMProviderBackend(provider=provider, model="ollama/x")
    # Privacy pipeline depends on fail-closed semantics; no raise allowed.
    assert await backend.generate("ping") == ""


async def test_llm_provider_backend_coerces_block_content():
    provider = MagicMock()
    provider.chat = AsyncMock(
        return_value=_fake_response([{"type": "text", "text": "a"}, "b"])
    )
    backend = LLMProviderBackend(provider=provider, model="anthropic/claude-haiku")
    assert await backend.generate("any") == "ab"


async def test_llm_provider_backend_embed_returns_empty_when_text_empty():
    backend = LLMProviderBackend(provider=MagicMock(), model="x")
    assert await backend.embed("") == []


async def test_llm_provider_backend_embed_calls_provider_with_chat_model_by_default():
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    backend = LLMProviderBackend(provider=provider, model="ollama/qwen2.5:0.5b")

    result = await backend.embed("hello world")

    assert result == [0.1, 0.2, 0.3]
    provider.embed.assert_awaited_once_with("hello world", model="ollama/qwen2.5:0.5b")


async def test_llm_provider_backend_embed_honours_embedding_model_override():
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[0.5, 0.6])
    backend = LLMProviderBackend(
        provider=provider,
        model="ollama/qwen2.5:0.5b",
        embedding_model="ollama/nomic-embed-text",
    )

    await backend.embed("text")

    provider.embed.assert_awaited_once_with("text", model="ollama/nomic-embed-text")


async def test_llm_provider_backend_embed_swallows_exceptions():
    provider = MagicMock()
    provider.embed = AsyncMock(side_effect=RuntimeError("backend down"))
    backend = LLMProviderBackend(provider=provider, model="x")
    assert await backend.embed("text") == []


# --- build_from_root_config -----------------------------------------------------------


def test_build_returns_none_when_privacy_disabled():
    root = MagicMock()
    root.privacy.enabled = False
    root.privacy.local_model = "ollama/x"
    assert build_from_root_config(root) is None


def test_build_returns_none_when_local_model_missing():
    root = MagicMock()
    root.privacy.enabled = True
    root.privacy.local_model = None
    assert build_from_root_config(root) is None


def test_build_returns_backend_when_configured(monkeypatch):
    root = MagicMock()
    root.privacy.enabled = True
    root.privacy.local_model = "ollama/qwen2.5:0.5b"
    root.privacy.embedding_model = None

    fake_provider = MagicMock()

    def fake_make_provider(config, *, model_override=None):
        assert model_override == "ollama/qwen2.5:0.5b"
        return fake_provider

    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider", fake_make_provider
    )
    backend = build_from_root_config(root)
    assert isinstance(backend, LLMProviderBackend)
    assert backend.name == "llm:ollama/qwen2.5:0.5b"
    assert backend._provider is fake_provider


def test_build_returns_none_when_provider_factory_raises(monkeypatch):
    root = MagicMock()
    root.privacy.enabled = True
    root.privacy.local_model = "bogus/x"
    root.privacy.embedding_model = None

    def boom(config, *, model_override=None):
        raise ValueError("no api key")

    monkeypatch.setattr("nanobot.providers.factory.make_provider", boom)
    # Misconfiguration must not crash the agent loop — fall back to None.
    assert build_from_root_config(root) is None


async def test_build_uses_separate_embedding_provider(monkeypatch):
    root = MagicMock()
    root.privacy.enabled = True
    root.privacy.local_model = "ollama/qwen2.5:0.5b"
    root.privacy.embedding_model = "ollama/nomic-embed-text"

    chat = MagicMock(name="chat_provider")
    embed = MagicMock(name="embed_provider")
    embed.embed = AsyncMock(return_value=[0.1, 0.2])
    calls: list[str] = []

    def fake_make_provider(config, *, model_override=None):
        calls.append(model_override)
        return chat if model_override == "ollama/qwen2.5:0.5b" else embed

    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider", fake_make_provider
    )
    backend = build_from_root_config(root)
    assert backend is not None
    # Both providers were resolved.
    assert "ollama/qwen2.5:0.5b" in calls
    assert "ollama/nomic-embed-text" in calls

    out = await backend.embed("hello")
    assert out == [0.1, 0.2]
    embed.embed.assert_awaited_once_with("hello", model="ollama/nomic-embed-text")


async def test_build_falls_back_to_no_embeddings_when_embedding_provider_fails(monkeypatch):
    root = MagicMock()
    root.privacy.enabled = True
    root.privacy.local_model = "ollama/qwen2.5:0.5b"
    root.privacy.embedding_model = "ollama/nomic-embed-text"

    chat = MagicMock(name="chat_provider")
    chat.embed = AsyncMock(return_value=[])

    def fake_make_provider(config, *, model_override=None):
        if model_override == "ollama/qwen2.5:0.5b":
            return chat
        raise RuntimeError("embedding provider misconfigured")

    monkeypatch.setattr("nanobot.providers.factory.make_provider", fake_make_provider)
    backend = build_from_root_config(root)
    assert backend is not None
    # Chat path still works; embed path silently returns [].
    out = await backend.embed("hello")
    assert out == []
