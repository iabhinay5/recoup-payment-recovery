"""Provider-abstracted LLM interface.

Recoup never imports a vendor SDK outside this package. Everything upstream talks to the
``LLMProvider`` protocol, so the same system runs against a hosted API or a local model
with a configuration change and no code change.

That is a data-residency decision before it is a portability one. Transaction decline data
is among the most sensitive data a merchant holds, and a recovery system that *requires*
shipping it to a third party to decide whether to retry imposes a cost some processors
will not accept. See docs/DECISIONS.md ADR-004.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "CacheMiss",
    "CacheMode",
    "CachedProvider",
    "LLMProvider",
    "LLMRequest",
]


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """A single completion request.

    Deliberately minimal. Anything vendor-specific belongs inside a provider
    implementation, not in the shape every caller has to know about.
    """

    system: str
    user: str
    max_tokens: int = 1024
    temperature: float = 0.0

    def cache_key(self, model: str) -> str:
        """Content-addressed key covering everything that can change the response.

        The model name is part of the key: a cached answer from one model is not a valid
        answer for another, and silently reusing it across a model swap would corrupt an
        evaluation run in a way that is very hard to notice afterwards.
        """
        payload = json.dumps(
            {
                "model": model,
                "system": self.system,
                "user": self.user,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn a request into text."""

    @property
    def model(self) -> str:
        """Identifier of the underlying model, used in cache keys."""
        ...

    def complete(self, request: LLMRequest) -> str:
        """Return the completion text for ``request``."""
        ...


class CacheMode(Enum):
    """How a :class:`CachedProvider` treats a cache miss."""

    LIVE = "live"
    """Bypass the cache entirely. Development and interactive use only."""

    RECORD = "record"
    """Serve from cache when possible; on a miss, call the provider and store the result."""

    REPLAY = "replay"
    """Serve from cache only. A miss raises :class:`CacheMiss`.

    This is the mode the evaluation harness runs in, and the raise is the point. An
    evaluation that silently falls back to a live call stops being reproducible: results
    would drift between runs for reasons unrelated to the policy under test, and the run
    would depend on network availability and rate limits. Failing loudly on a miss makes
    that class of error impossible rather than merely unlikely. See ADR-005.
    """


class CacheMiss(KeyError):
    """Raised when REPLAY mode encounters an uncached request."""


class CachedProvider:
    """Content-addressed disk cache in front of any provider.

    Two jobs, in order of importance. First, it makes evaluation runs deterministic and
    reproducible by removing live model calls from the loop. Second, it makes them free
    and fast, which matters because the harness executes on the order of 10^5 episodes.

    The cache is effective here because the decline taxonomy is finite: there are roughly
    two dozen canonical decline reasons, so normalisation requests collapse onto a small
    set of distinct inputs and hit the cache almost always after a short warm-up.
    """

    def __init__(
        self,
        inner: LLMProvider,
        cache_dir: Path | str = Path("data/cache/llm"),
        mode: CacheMode = CacheMode.RECORD,
    ) -> None:
        self._inner = inner
        self._dir = Path(cache_dir)
        self._mode = mode
        self.hits = 0
        self.misses = 0
        if mode is not CacheMode.LIVE:
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def mode(self) -> CacheMode:
        return self._mode

    def _path(self, key: str) -> Path:
        # Shard by the first two hex characters to avoid a single directory with tens of
        # thousands of entries, which some filesystems handle poorly.
        return self._dir / key[:2] / f"{key}.json"

    def complete(self, request: LLMRequest) -> str:
        if self._mode is CacheMode.LIVE:
            return self._inner.complete(request)

        key = request.cache_key(self._inner.model)
        path = self._path(key)

        if path.exists():
            self.hits += 1
            return str(json.loads(path.read_text(encoding="utf-8"))["response"])

        self.misses += 1

        if self._mode is CacheMode.REPLAY:
            raise CacheMiss(
                f"No cached response for {key[:12]} (model={self._inner.model}). "
                f"REPLAY mode does not make live calls — see ADR-005. Warm the cache "
                f"with a RECORD-mode run before evaluating."
            )

        response = self._inner.complete(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": self._inner.model,
                    "system": request.system,
                    "user": request.user,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "response": response,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return response

    @property
    def hit_rate(self) -> float:
        """Fraction of requests served from cache. Zero when nothing has been requested."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
