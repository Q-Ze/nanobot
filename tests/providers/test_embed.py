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
