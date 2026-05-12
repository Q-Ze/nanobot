"""Tests for OpenAICompatProvider.embed and the LLMProvider.embed default."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.providers.openai_compat_provider import OpenAICompatProvider


def _provider_with_fake_client(embeddings_create):
    """Build an OpenAICompatProvider but swap in a fake AsyncOpenAI client."""
    provider = OpenAICompatProvider(
        api_key="test-key",
        api_base="http://localhost:11434/v1",
        default_model="qwen2.5:0.5b",
    )
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = embeddings_create
    provider._client = fake_client
    return provider


@pytest.mark.asyncio
async def test_embed_returns_vector_from_openai_compat_endpoint():
    payload = MagicMock()
    payload.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    provider = _provider_with_fake_client(AsyncMock(return_value=payload))

    out = await provider.embed("hello", model="nomic-embed-text")

    assert out == [0.1, 0.2, 0.3]
    provider._client.embeddings.create.assert_awaited_once_with(
        input="hello", model="nomic-embed-text"
    )


@pytest.mark.asyncio
async def test_embed_falls_back_to_default_model_when_not_specified():
    payload = MagicMock()
    payload.data = [MagicMock(embedding=[0.0])]
    provider = _provider_with_fake_client(AsyncMock(return_value=payload))

    await provider.embed("anything")

    provider._client.embeddings.create.assert_awaited_once_with(
        input="anything", model="qwen2.5:0.5b"
    )


@pytest.mark.asyncio
async def test_embed_returns_empty_on_empty_input():
    provider = _provider_with_fake_client(AsyncMock())
    assert await provider.embed("") == []
    provider._client.embeddings.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_swallows_transport_errors():
    provider = _provider_with_fake_client(
        AsyncMock(side_effect=RuntimeError("network down"))
    )
    assert await provider.embed("x") == []


@pytest.mark.asyncio
async def test_embed_handles_malformed_response():
    bogus = MagicMock()
    bogus.data = []   # Empty data list
    provider = _provider_with_fake_client(AsyncMock(return_value=bogus))
    assert await provider.embed("x") == []


@pytest.mark.asyncio
async def test_base_provider_embed_default_returns_empty():
    """The default LLMProvider.embed in base.py returns []."""
    # Use a concrete subclass that doesn't override embed (Anthropic does not).
    from nanobot.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="x", default_model="claude-3")
    assert await provider.embed("hello") == []


# --- OpenRouter / gateway whitespace-prefix fallback ----------------------------


@pytest.mark.asyncio
async def test_embed_falls_back_to_httpx_when_sdk_rejects_whitespace_prefix(
    monkeypatch,
):
    """OpenRouter free-tier prepends `\\n         \\n` to the JSON body.

    The OpenAI SDK raises ``ValueError: No embedding data received`` and we
    must fall back to a direct HTTP call that ``lstrip``s the body before
    parsing. Verified by simulating both the SDK rejection and a synthetic
    OpenRouter-style response.
    """
    import httpx

    provider = _provider_with_fake_client(
        AsyncMock(side_effect=ValueError("No embedding data received"))
    )
    # Make sure the fallback target is reachable looking
    provider._effective_base = "https://openrouter.ai/api/v1"
    provider.api_key = "sk-or-fake"

    # Synthetic body shaped like the broken OpenRouter response.
    body = (
        b'\n         \n'
        b'{"object":"list","data":[{"object":"embedding",'
        b'"embedding":[0.1, 0.2, 0.3]}]}'
    )

    class _FakeAsyncClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, **kw):
            req = httpx.Request("POST", url)
            return httpx.Response(200, content=body, request=req)

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)

    out = await provider.embed("hello", model="x")
    assert out == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embed_fallback_returns_empty_on_http_error(monkeypatch):
    import httpx

    provider = _provider_with_fake_client(
        AsyncMock(side_effect=ValueError("No embedding data received"))
    )
    provider._effective_base = "https://example.com"
    provider.api_key = "k"

    class _FakeAsyncClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, **kw):
            req = httpx.Request("POST", url)
            return httpx.Response(429, content=b'{"error":"rate"}', request=req)

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)
    assert await provider.embed("hello") == []


@pytest.mark.asyncio
async def test_embed_unrelated_value_error_still_returns_empty():
    """ValueError that *isn't* the whitespace-prefix one is not retried."""
    provider = _provider_with_fake_client(
        AsyncMock(side_effect=ValueError("invalid input format"))
    )
    assert await provider.embed("hello") == []
