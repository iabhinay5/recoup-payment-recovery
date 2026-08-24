"""Concrete LLM providers.

This is the only module in Recoup that knows how to talk to a specific vendor. Everything
else depends on the ``LLMProvider`` protocol in ``base``.

Three implementations:

- ``GroqProvider``   — hosted, used during development
- ``OllamaProvider`` — local, the on-premise path from ADR-004
- ``StubProvider``   — deterministic, for tests that must not touch a network

The Groq and Ollama providers speak the same OpenAI-compatible chat-completions shape, so
they share a request builder. That is a convenience of those two APIs, not an assumption
baked into the architecture — a provider with a different wire format simply implements
``complete`` differently.
"""

from __future__ import annotations

import os
from typing import Any

from recoup.llm.base import LLMRequest

__all__ = ["GroqProvider", "OllamaProvider", "StubProvider"]

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


def _chat_payload(request: LLMRequest, model: str) -> dict[str, Any]:
    """Build an OpenAI-compatible chat-completions body."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }


def _extract_text(body: dict[str, Any]) -> str:
    try:
        return str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response shape: {body!r}") from exc


class GroqProvider:
    """Hosted inference via Groq's OpenAI-compatible endpoint.

    The free tier is rate limited on tokens per minute rather than requests, which is why
    the cache in ``base`` matters even during development: a warm cache keeps interactive
    work well clear of the ceiling.
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GROQ_MODEL,
        timeout: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in; "
                "never commit the real key."
            )
        self._key = key
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> str:
        import httpx

        response = httpx.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            json=_chat_payload(request, self._model),
            timeout=self._timeout,
        )
        response.raise_for_status()
        return _extract_text(response.json())


class OllamaProvider:
    """Local inference via Ollama.

    The point of this provider is not cost. It is that the identical system can run with
    no transaction data leaving the merchant's infrastructure — which for a payment
    processor is an architectural property worth having, not a fallback.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._url = f"{host.rstrip('/')}/v1/chat/completions"
        # Local CPU inference is far slower than a hosted call, so the default timeout is
        # generous. A 7B model on a laptop CPU can take tens of seconds per completion.
        self._timeout = timeout

    @property
    def model(self) -> str:
        return f"ollama/{self._model}"

    def complete(self, request: LLMRequest) -> str:
        import httpx

        response = httpx.post(
            self._url,
            json=_chat_payload(request, self._model),
            timeout=self._timeout,
        )
        response.raise_for_status()
        return _extract_text(response.json())


class StubProvider:
    """Deterministic provider for tests.

    Returns a canned response, or echoes a digest of the request when no canned response
    is configured. Never touches a network, so unit tests stay fast and hermetic.
    """

    def __init__(self, responses: dict[str, str] | None = None, model: str = "stub") -> None:
        self._responses = responses or {}
        self._model = model
        self.calls: list[LLMRequest] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        if request.user in self._responses:
            return self._responses[request.user]
        return f"stub:{request.cache_key(self._model)[:12]}"
