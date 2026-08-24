"""Razorpay API client, test mode only.

Every method here goes through ``_request``, and ``_request`` refuses to run against a key
that does not begin with ``rzp_test_``. That check is not defensive programming for its own
sake: Recoup's entire purpose is issuing payment retries, so a live key would move real
money on behalf of real people. The guard sits at the boundary rather than in a comment.

Only the endpoints the demo needs are implemented. A thin client that does four things
correctly is more useful here than a wrapper around an API surface we do not use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from recoup.config import Settings

__all__ = ["LiveKeyRefused", "RazorpayClient", "RazorpayError"]

API_BASE = "https://api.razorpay.com/v1"
TEST_KEY_PREFIX = "rzp_test_"


class RazorpayError(RuntimeError):
    """The API returned an error response."""


class LiveKeyRefused(RuntimeError):
    """A live-mode key was supplied. Recoup will not run against real money."""


@dataclass
class RazorpayClient:
    """Minimal test-mode client."""

    key_id: str
    key_secret: str
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.key_id.startswith(TEST_KEY_PREFIX):
            raise LiveKeyRefused(
                f"key {self.key_id[:12]}... is not a test key. Recoup schedules payment "
                f"retries; running it against live credentials would move real money. "
                f"Generate a Test Mode key in the Razorpay dashboard."
            )

    @classmethod
    def from_env(cls) -> RazorpayClient:
        settings = Settings.from_env()
        if not settings.has_razorpay:
            raise RuntimeError(
                "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set. "
                "Copy .env.example to .env and fill them in."
            )
        assert settings.razorpay_key_id is not None
        assert settings.razorpay_key_secret is not None
        return cls(settings.razorpay_key_id, settings.razorpay_key_secret)

    # --- transport --------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = httpx.request(
                method,
                f"{API_BASE}{path}",
                auth=(self.key_id, self.key_secret),
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise RazorpayError(f"could not reach the Razorpay API: {exc!r}") from exc

        if response.status_code >= 400:
            # Razorpay error bodies describe the problem and never echo credentials.
            raise RazorpayError(f"HTTP {response.status_code}: {response.text[:400]}")

        body: dict[str, Any] = response.json()
        return body

    # --- endpoints --------------------------------------------------------------------

    def create_order(
        self,
        amount_paise: int,
        receipt: str,
        currency: str = "INR",
        notes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create an order for checkout to pay against.

        ``receipt`` is the merchant's own reference. Recoup puts the episode id here so a
        sandbox payment can be traced back to the recovery run that produced it.
        """
        if amount_paise < 100:
            raise ValueError("Razorpay rejects amounts below 100 paise (Rs 1)")
        payload: dict[str, Any] = {
            "amount": amount_paise,
            "currency": currency,
            "receipt": receipt,
        }
        if notes:
            payload["notes"] = notes
        return self._request("POST", "/orders", json=payload)

    def fetch_payment(self, payment_id: str) -> dict[str, Any]:
        """Fetch one payment, including its error fields when it failed."""
        return self._request("GET", f"/payments/{payment_id}")

    def list_payments(self, count: int = 10) -> list[dict[str, Any]]:
        """Most recent payments on the account, newest first."""
        body = self._request("GET", "/payments", params={"count": count})
        items: list[dict[str, Any]] = body.get("items", [])
        return items

    def payments_for_order(self, order_id: str) -> list[dict[str, Any]]:
        body = self._request("GET", f"/orders/{order_id}/payments")
        items: list[dict[str, Any]] = body.get("items", [])
        return items

    def ping(self) -> int:
        """Confirm the credentials authenticate. Returns the visible payment count."""
        body = self._request("GET", "/payments", params={"count": 1})
        return int(body.get("count", 0))
