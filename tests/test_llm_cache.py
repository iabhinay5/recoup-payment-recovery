"""Tests for the provider abstraction and its cache.

The cache is what makes evaluation runs reproducible (DECISIONS.md ADR-005), so the tests
that matter most here are the ones proving REPLAY mode cannot quietly make a live call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recoup.llm.base import CachedProvider, CacheMiss, CacheMode, LLMProvider, LLMRequest
from recoup.llm.providers import StubProvider


@pytest.fixture
def request_() -> LLMRequest:
    return LLMRequest(system="classify decline reasons", user="card has expired")


class TestCacheKey:
    def test_same_request_same_key(self, request_: LLMRequest) -> None:
        assert request_.cache_key("m") == request_.cache_key("m")

    def test_different_model_different_key(self, request_: LLMRequest) -> None:
        """A cached answer from one model is not a valid answer for another."""
        assert request_.cache_key("model-a") != request_.cache_key("model-b")

    def test_temperature_changes_key(self) -> None:
        a = LLMRequest(system="s", user="u", temperature=0.0)
        b = LLMRequest(system="s", user="u", temperature=0.7)
        assert a.cache_key("m") != b.cache_key("m")

    def test_system_prompt_changes_key(self) -> None:
        a = LLMRequest(system="prompt one", user="u")
        b = LLMRequest(system="prompt two", user="u")
        assert a.cache_key("m") != b.cache_key("m")


class TestRecordMode:
    def test_first_call_hits_provider_second_hits_cache(
        self, tmp_path: Path, request_: LLMRequest
    ) -> None:
        stub = StubProvider({"card has expired": "card_expired"})
        cached = CachedProvider(stub, cache_dir=tmp_path, mode=CacheMode.RECORD)

        assert cached.complete(request_) == "card_expired"
        assert len(stub.calls) == 1

        assert cached.complete(request_) == "card_expired"
        assert len(stub.calls) == 1, "second identical request should not reach the provider"
        assert cached.hits == 1
        assert cached.misses == 1

    def test_cache_survives_a_new_provider_instance(
        self, tmp_path: Path, request_: LLMRequest
    ) -> None:
        """The cache is on disk, so a fresh process reuses it."""
        first = StubProvider({"card has expired": "card_expired"})
        CachedProvider(first, cache_dir=tmp_path, mode=CacheMode.RECORD).complete(request_)

        second = StubProvider({"card has expired": "SHOULD NOT BE CALLED"})
        cached = CachedProvider(second, cache_dir=tmp_path, mode=CacheMode.RECORD)

        assert cached.complete(request_) == "card_expired"
        assert second.calls == []


class TestReplayMode:
    """The guarantee behind ADR-005."""

    def test_miss_raises_instead_of_calling_the_provider(
        self, tmp_path: Path, request_: LLMRequest
    ) -> None:
        stub = StubProvider()
        cached = CachedProvider(stub, cache_dir=tmp_path, mode=CacheMode.REPLAY)

        with pytest.raises(CacheMiss, match="does not make live calls"):
            cached.complete(request_)

        assert stub.calls == [], "REPLAY must never reach the underlying provider"

    def test_hit_is_served_without_calling_the_provider(
        self, tmp_path: Path, request_: LLMRequest
    ) -> None:
        warm = StubProvider({"card has expired": "card_expired"})
        CachedProvider(warm, cache_dir=tmp_path, mode=CacheMode.RECORD).complete(request_)

        stub = StubProvider()
        cached = CachedProvider(stub, cache_dir=tmp_path, mode=CacheMode.REPLAY)

        assert cached.complete(request_) == "card_expired"
        assert stub.calls == []

    def test_model_swap_invalidates_replay(self, tmp_path: Path, request_: LLMRequest) -> None:
        """Warming the cache with one model must not silently satisfy another.

        Without this, swapping models mid-project would reuse stale answers and corrupt an
        evaluation run in a way that is very hard to spot after the fact.
        """
        warm = StubProvider({"card has expired": "card_expired"}, model="model-a")
        CachedProvider(warm, cache_dir=tmp_path, mode=CacheMode.RECORD).complete(request_)

        other = StubProvider(model="model-b")
        cached = CachedProvider(other, cache_dir=tmp_path, mode=CacheMode.REPLAY)

        with pytest.raises(CacheMiss):
            cached.complete(request_)


class TestLiveMode:
    def test_live_mode_always_calls_through(self, tmp_path: Path, request_: LLMRequest) -> None:
        stub = StubProvider({"card has expired": "card_expired"})
        cached = CachedProvider(stub, cache_dir=tmp_path, mode=CacheMode.LIVE)

        cached.complete(request_)
        cached.complete(request_)

        assert len(stub.calls) == 2


class TestProtocolConformance:
    def test_stub_satisfies_the_protocol(self) -> None:
        assert isinstance(StubProvider(), LLMProvider)

    def test_cached_provider_satisfies_the_protocol(self, tmp_path: Path) -> None:
        """A CachedProvider must be substitutable for a provider, so wrapping composes."""
        assert isinstance(CachedProvider(StubProvider(), cache_dir=tmp_path), LLMProvider)

    def test_hit_rate_starts_at_zero(self, tmp_path: Path) -> None:
        assert CachedProvider(StubProvider(), cache_dir=tmp_path).hit_rate == 0.0

    def test_hit_rate_reflects_usage(self, tmp_path: Path, request_: LLMRequest) -> None:
        cached = CachedProvider(StubProvider(), cache_dir=tmp_path, mode=CacheMode.RECORD)
        cached.complete(request_)
        cached.complete(request_)
        assert cached.hit_rate == 0.5
