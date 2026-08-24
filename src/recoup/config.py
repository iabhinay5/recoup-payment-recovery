"""Configuration loading.

Reads ``.env`` without a third-party dependency. The file format we need is a handful of
``KEY=value`` lines, and pulling in a package to parse that would be the more surprising
choice.

Every accessor here is built so that a credential can be checked for *presence* without
being printed. Recoup handles payment credentials, and the cheapest way to leak one is a
debug log or a stack trace written by someone who was not thinking about it at the time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Settings", "load_dotenv", "redact"]

DEFAULT_ENV_PATH = Path(".env")


def redact(value: str | None, keep: int = 4) -> str:
    """Render a secret safely for logs and error messages.

    Shows enough leading characters to tell two keys apart, never enough to use one.
    """
    if not value:
        return "<unset>"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - keep)}"


def load_dotenv(path: Path | str = DEFAULT_ENV_PATH, *, override: bool = False) -> dict[str, str]:
    """Load ``KEY=value`` pairs from ``path`` into ``os.environ``.

    Returns the parsed pairs. A missing file is not an error: in CI and in container
    deployments the values arrive as real environment variables instead, and requiring a
    file there would mean shipping a fake one.

    Existing environment variables win unless ``override`` is set, so a real deployment
    cannot be silently overridden by a stray ``.env`` left in the working directory.
    """
    path = Path(path)
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return parsed


@dataclass(frozen=True, slots=True)
class Settings:
    """Credentials and endpoints, resolved from the environment."""

    groq_api_key: str | None = None
    razorpay_key_id: str | None = None
    razorpay_key_secret: str | None = None
    razorpay_webhook_secret: str | None = None

    @classmethod
    def from_env(cls, *, load_file: bool = True) -> Settings:
        if load_file:
            load_dotenv()
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY") or None,
            razorpay_key_id=os.environ.get("RAZORPAY_KEY_ID") or None,
            razorpay_key_secret=os.environ.get("RAZORPAY_KEY_SECRET") or None,
            razorpay_webhook_secret=os.environ.get("RAZORPAY_WEBHOOK_SECRET") or None,
        )

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_razorpay(self) -> bool:
        return bool(self.razorpay_key_id and self.razorpay_key_secret)

    @property
    def razorpay_is_test_mode(self) -> bool:
        """Whether the configured Razorpay key is a test key.

        Recoup schedules payment retries. Running it against a live key would move real
        money, so this is checked at the boundary rather than trusted — see
        ``scripts/verify_razorpay.py``.
        """
        return bool(self.razorpay_key_id and self.razorpay_key_id.startswith("rzp_test_"))

    def summary(self) -> str:
        """Human-readable status with every secret redacted. Safe to print or log."""
        return "\n".join(
            (
                f"  GROQ_API_KEY            {redact(self.groq_api_key, keep=7)}",
                f"  RAZORPAY_KEY_ID         {redact(self.razorpay_key_id, keep=12)}",
                f"  RAZORPAY_KEY_SECRET     {redact(self.razorpay_key_secret)}",
                f"  RAZORPAY_WEBHOOK_SECRET {redact(self.razorpay_webhook_secret)}",
            )
        )
