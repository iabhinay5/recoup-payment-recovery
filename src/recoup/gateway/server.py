"""Webhook receiver.

Razorpay posts payment events here. The endpoint does three things in a deliberate order:
verify the signature, normalise the payload, then hand it to the policy. Verification comes
first and is unconditional — an endpoint that acts on an unverified payload is an endpoint
anyone on the internet can use to drive your retry logic.

The handler returns 200 for anything it has *received and understood*, including events it
chooses not to act on. Razorpay retries non-2xx deliveries, so returning an error for
"this is an event type I don't handle" turns a normal condition into a redelivery loop.
The two things that do return an error are a bad signature (401 — never accept it) and an
unparseable body (400 — retrying will not help).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request

from recoup.config import Settings
from recoup.gateway.webhooks import (
    FailedPaymentEvent,
    InvalidSignature,
    parse_failed_payment,
    verify_signature,
)

__all__ = ["EventLog", "create_app"]


@dataclass
class EventLog:
    """In-memory record of what the endpoint received.

    Exists so the demo and the dashboard can show live traffic without a database. A
    production deployment would persist these, because the audit question — "what did you
    receive, and what did you do about it?" — outlives the process.
    """

    events: list[FailedPaymentEvent] = field(default_factory=list)
    rejected: int = 0
    ignored: int = 0

    def record(self, event: FailedPaymentEvent) -> None:
        self.events.append(event)

    @property
    def count(self) -> int:
        return len(self.events)

    def latest(self, n: int = 10) -> list[FailedPaymentEvent]:
        return self.events[-n:][::-1]


def create_app(
    secret: str | None = None,
    on_failure: Callable[[FailedPaymentEvent], None] | None = None,
    log: EventLog | None = None,
) -> FastAPI:
    """Build the webhook application.

    Args:
        secret: webhook secret. Falls back to ``RAZORPAY_WEBHOOK_SECRET``.
        on_failure: called with each verified ``payment.failed`` event.
        log: event log to record into. One is created if not supplied.
    """
    resolved_secret = (
        secret if secret is not None else (Settings.from_env().razorpay_webhook_secret or "")
    )
    event_log = log if log is not None else EventLog()

    app = FastAPI(title="Recoup webhook receiver", docs_url=None, redoc_url=None)
    app.state.log = event_log

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "events_received": event_log.count,
            "rejected_signatures": event_log.rejected,
            "signature_verification": "enabled" if resolved_secret else "NOT CONFIGURED",
        }

    @app.post("/webhook/razorpay")
    async def webhook(
        request: Request,
        x_razorpay_signature: str = Header(default=""),
    ) -> dict[str, Any]:
        # The raw bytes, before any parsing. Re-serialising the JSON would change
        # whitespace and key order and the signature would never match again.
        raw = await request.body()

        try:
            verify_signature(raw, x_razorpay_signature, resolved_secret)
        except InvalidSignature as exc:
            event_log.rejected += 1
            # 401 rather than 400: this is an authentication failure, and Razorpay should
            # not keep redelivering something we will never accept.
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        try:
            event = parse_failed_payment(raw)
        except ValueError as exc:
            message = str(exc)
            if "expected event" in message:
                # A verified event we simply do not handle. Acknowledge it, or Razorpay
                # will redeliver it indefinitely.
                event_log.ignored += 1
                return {"status": "ignored", "reason": message}
            raise HTTPException(status_code=400, detail=message) from exc

        event_log.record(event)
        if on_failure is not None:
            on_failure(event)

        return {
            "status": "accepted",
            "payment_id": event.payment_id,
            "decline_code": event.decline_code,
            "recognised": event.is_recognised,
            "remedy": event.reason.remedy.value,
            "max_attempts": event.reason.max_attempts,
        }

    @app.get("/events")
    def events(limit: int = 10) -> dict[str, Any]:
        return {
            "count": event_log.count,
            "events": [
                {
                    "payment_id": e.payment_id,
                    "amount_rupees": e.amount_rupees,
                    "method": e.method,
                    "decline_code": e.decline_code,
                    "decline_class": e.reason.decline_class.value,
                    "remedy": e.reason.remedy.value,
                    "recognised": e.is_recognised,
                }
                for e in event_log.latest(limit)
            ],
        }

    return app
