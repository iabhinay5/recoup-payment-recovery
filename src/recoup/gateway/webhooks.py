"""Razorpay webhook verification and normalisation.

Two jobs, and the first one is security rather than parsing.

**Verification.** A webhook endpoint is a public URL that causes your system to act on
payment events. Anyone who finds it can post to it. Razorpay signs each delivery with
HMAC-SHA256 over the raw request body, and the signature must be checked against the
**bytes as received** — parsing the JSON first and re-serialising it changes whitespace and
key order, and the signature will never match.

The comparison uses ``hmac.compare_digest``. A plain ``==`` on a signature leaks timing
information that can be used to forge one byte at a time.

**Normalisation.** Razorpay reports the machine-readable failure reason in
``error_reason``, which is the same vocabulary as ``recoup.taxonomy``. That is not a
coincidence — the taxonomy was built from Razorpay's published error documentation
precisely so that live payloads land on it without translation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

from recoup.taxonomy import DeclineReason, PaymentMethod, is_known, lookup

__all__ = [
    "SIGNATURE_HEADER",
    "FailedPaymentEvent",
    "InvalidSignature",
    "parse_failed_payment",
    "verify_signature",
]

SIGNATURE_HEADER = "X-Razorpay-Signature"


class InvalidSignature(Exception):
    """The webhook did not carry a valid signature and must not be acted upon."""


def verify_signature(raw_body: bytes, signature: str, secret: str) -> None:
    """Verify a Razorpay webhook signature, or raise.

    ``raw_body`` must be the exact bytes received. Handing this a re-serialised dict is the
    most common way to get a permanently failing endpoint that looks correct in review.

    Raises:
        InvalidSignature: if the signature does not match.
    """
    if not secret:
        raise InvalidSignature(
            "no webhook secret configured; refusing to accept unverified webhooks"
        )

    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

    # Constant-time comparison. A byte-by-byte "==" returns faster on an earlier mismatch,
    # which is enough to recover a valid signature one character at a time.
    if not hmac.compare_digest(expected, signature or ""):
        raise InvalidSignature("signature mismatch")


@dataclass(frozen=True, slots=True)
class FailedPaymentEvent:
    """A ``payment.failed`` webhook, normalised onto Recoup's vocabulary."""

    payment_id: str
    order_id: str | None
    amount_paise: int
    method: str
    error_reason: str
    error_code: str
    error_description: str
    error_source: str
    error_step: str
    created_at: int

    @property
    def decline_code(self) -> str:
        """The canonical taxonomy code for this failure.

        Razorpay's ``error_reason`` already uses the taxonomy vocabulary, so this is
        usually a pass-through. It is expressed as a property anyway because the
        pass-through is a fact about today's payloads, not a guarantee — a gateway is free
        to add a reason we have never seen.
        """
        return self.error_reason

    @property
    def reason(self) -> DeclineReason:
        """Taxonomy entry for this failure, falling back conservatively if unrecognised."""
        return lookup(self.decline_code)

    @property
    def is_recognised(self) -> bool:
        """Whether the taxonomy knows this reason.

        An unrecognised reason is an operational event worth surfacing, not an error: it
        means Razorpay shipped a code we have not classified, and the policy is currently
        treating it with the conservative unknown default.
        """
        return is_known(self.decline_code)

    @property
    def payment_method(self) -> PaymentMethod | None:
        """The rail, where it maps onto one the taxonomy models."""
        try:
            return PaymentMethod(self.method)
        except ValueError:
            return None

    @property
    def amount_rupees(self) -> float:
        return self.amount_paise / 100.0


def parse_failed_payment(payload: dict[str, Any] | bytes | str) -> FailedPaymentEvent:
    """Extract a ``payment.failed`` event from a webhook body.

    Raises:
        ValueError: if the body is not a ``payment.failed`` event, or is missing the
            payment entity. Both indicate the endpoint was called with something it should
            not act on, and both should be visible rather than silently skipped.
    """
    if isinstance(payload, bytes | str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("webhook body must be a JSON object")

    event = payload.get("event")
    if event != "payment.failed":
        raise ValueError(f"expected event 'payment.failed', got {event!r}")

    try:
        entity = payload["payload"]["payment"]["entity"]
    except (KeyError, TypeError) as exc:
        raise ValueError("webhook is missing payload.payment.entity") from exc

    return FailedPaymentEvent(
        payment_id=str(entity.get("id", "")),
        order_id=entity.get("order_id"),
        amount_paise=int(entity.get("amount", 0)),
        method=str(entity.get("method", "")),
        error_reason=str(entity.get("error_reason") or "__unknown__"),
        error_code=str(entity.get("error_code") or ""),
        error_description=str(entity.get("error_description") or ""),
        error_source=str(entity.get("error_source") or ""),
        error_step=str(entity.get("error_step") or ""),
        created_at=int(entity.get("created_at", 0)),
    )
