"""Verify Razorpay test credentials without ever printing them.

Run:  python scripts/verify_razorpay.py

Makes one harmless authenticated GET against the Razorpay API and reports whether the
credentials work. Secrets are redacted in all output, so the result of this script is safe
to paste anywhere — including into a chat window.

The live-key check is not decoration. Recoup's entire purpose is scheduling payment
retries; pointed at a live key it would move real money. That check belongs at the
boundary, before anything else runs.
"""

from __future__ import annotations

import sys

import httpx

from recoup.config import Settings

# Listing payments is read-only and succeeds on a brand-new account with an empty result.
VERIFY_URL = "https://api.razorpay.com/v1/payments?count=1"


def main() -> int:
    settings = Settings.from_env()

    print("Credentials found:")
    print(settings.summary())
    print()

    if not settings.has_razorpay:
        print("FAIL  RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must both be set in .env")
        print("      Copy .env.example to .env and fill them in.")
        return 1

    if not settings.razorpay_is_test_mode:
        print("FAIL  Key does not start with 'rzp_test_'.")
        print()
        print("      This looks like a LIVE key. Recoup schedules payment retries, so")
        print("      running it against live credentials would move real money.")
        print("      Refusing to continue. Generate a Test Mode key instead:")
        print("      Dashboard -> Test Mode -> Account & Settings -> API Keys")
        return 1

    assert settings.razorpay_key_id is not None
    assert settings.razorpay_key_secret is not None

    try:
        response = httpx.get(
            VERIFY_URL,
            auth=(settings.razorpay_key_id, settings.razorpay_key_secret),
            timeout=30.0,
        )
    except httpx.RequestError as exc:
        print(f"FAIL  Could not reach the Razorpay API: {exc.__class__.__name__}")
        print("      Check your network connection and try again.")
        return 1

    if response.status_code == 200:
        body = response.json()
        count = body.get("count", 0)
        print("OK    Test credentials authenticate against the Razorpay API.")
        print(f"      Endpoint reachable, {count} payment(s) visible on this account.")
        print("      Ready for the sandbox integration on day 7.")
        return 0

    if response.status_code == 401:
        print("FAIL  Razorpay rejected the credentials (401 Unauthorized).")
        print("      The Key ID and Key Secret must be from the SAME generated pair,")
        print("      and both from Test Mode. Regenerate and try again if unsure.")
        return 1

    print(f"FAIL  Unexpected response: HTTP {response.status_code}")
    # Razorpay error bodies describe the problem without echoing credentials.
    print(f"      {response.text[:300]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
