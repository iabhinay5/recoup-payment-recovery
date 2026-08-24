"""Tests for configuration loading and secret redaction.

The redaction tests exist because the cheapest way to leak a payment credential is a log
line written by someone who was not thinking about it. Making redaction a tested property
rather than a convention is the difference between "we try to" and "we do".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recoup.config import Settings, load_dotenv, redact


class TestRedact:
    def test_unset_value(self) -> None:
        assert redact(None) == "<unset>"
        assert redact("") == "<unset>"

    def test_keeps_a_prefix_and_masks_the_rest(self) -> None:
        assert redact("abcdefghij", keep=4) == "abcd******"

    def test_never_leaks_the_tail(self) -> None:
        secret = "rzp_test_SUPERSECRETVALUE"
        out = redact(secret, keep=9)
        assert "SUPERSECRETVALUE" not in out
        assert out.startswith("rzp_test_")

    def test_short_values_are_fully_masked(self) -> None:
        """A value shorter than the prefix must not be echoed verbatim."""
        assert redact("abc", keep=4) == "***"

    def test_length_is_preserved(self) -> None:
        assert len(redact("0123456789")) == 10


class TestLoadDotenv:
    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        """In CI and containers the values arrive as real environment variables."""
        assert load_dotenv(tmp_path / "nope.env") == {}

    def test_parses_pairs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")
        monkeypatch.delenv("FOO", raising=False)
        monkeypatch.delenv("BAZ", raising=False)

        assert load_dotenv(env) == {"FOO": "bar", "BAZ": "qux"}

    def test_ignores_comments_and_blanks(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("# a comment\n\nFOO=bar\n   \n", encoding="utf-8")
        assert load_dotenv(env) == {"FOO": "bar"}

    def test_strips_surrounding_quotes(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=\"bar\"\nBAZ='qux'\n", encoding="utf-8")
        assert load_dotenv(env) == {"FOO": "bar", "BAZ": "qux"}

    def test_values_containing_equals_survive(self, tmp_path: Path) -> None:
        """Base64 secrets routinely contain '=' padding."""
        env = tmp_path / ".env"
        env.write_text("TOKEN=abc==\n", encoding="utf-8")
        assert load_dotenv(env) == {"TOKEN": "abc=="}

    def test_real_environment_wins_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stray .env must not silently override a deployed configuration."""
        env = tmp_path / ".env"
        env.write_text("FOO=from_file\n", encoding="utf-8")
        monkeypatch.setenv("FOO", "from_environment")

        load_dotenv(env)

        import os

        assert os.environ["FOO"] == "from_environment"

    def test_override_is_opt_in(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=from_file\n", encoding="utf-8")
        monkeypatch.setenv("FOO", "from_environment")

        load_dotenv(env, override=True)

        import os

        assert os.environ["FOO"] == "from_file"


class TestSettings:
    def test_missing_credentials_report_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "GROQ_API_KEY",
            "RAZORPAY_KEY_ID",
            "RAZORPAY_KEY_SECRET",
            "RAZORPAY_WEBHOOK_SECRET",
        ):
            monkeypatch.delenv(key, raising=False)

        settings = Settings.from_env(load_file=False)
        assert not settings.has_groq
        assert not settings.has_razorpay

    def test_test_key_is_recognised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_abc123")
        monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
        settings = Settings.from_env(load_file=False)
        assert settings.has_razorpay
        assert settings.razorpay_is_test_mode

    def test_live_key_is_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Recoup schedules real payment retries. A live key must never read as safe."""
        monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_abc123")
        monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
        settings = Settings.from_env(load_file=False)
        assert settings.has_razorpay
        assert not settings.razorpay_is_test_mode

    def test_summary_never_contains_a_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point of summary() is that it is safe to paste anywhere."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_THISMUSTNOTAPPEAR")
        monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_IDMUSTNOTAPPEARINFULL")
        monkeypatch.setenv("RAZORPAY_KEY_SECRET", "SECRETMUSTNOTAPPEAR")

        summary = Settings.from_env(load_file=False).summary()

        assert "THISMUSTNOTAPPEAR" not in summary
        assert "SECRETMUSTNOTAPPEAR" not in summary
        assert "IDMUSTNOTAPPEARINFULL" not in summary
        assert "gsk_" in summary, "a redacted value should still be identifiable"
