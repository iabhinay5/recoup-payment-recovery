"""The demo dashboard: one screen, showing what the engine actually did.

**This module contains no decision logic, and that is the point.** Every number and every
verdict on the screen comes from ``recoup.trace.explain``, which runs the real policy
against the real guardrails. A dashboard that described what the policy would probably do
would be right on the day it was written and quietly wrong afterwards, and the failure
would be invisible — a screen that looks correct is not the same as a screen that is.

Three streams feed it, and they are badged differently because they are not equally
strong evidence:

- **webhook** — a signed ``payment.failed`` delivery from Razorpay. The deployed path.
- **sandbox** — a real failed payment pulled back from the Razorpay API. Same events,
  fetched rather than pushed, because a public tunnel is one more thing to fail on camera.
- **injected** — a synthetic decline, used to show classes the sandbox will not produce on
  demand. Labelled as synthetic everywhere it appears.

Conflating those three would be the easiest and most dishonest way to make the demo look
better than it is.

The guardrail instance is shared across every trace on purpose. Its refusal ledger is the
audit panel, and a per-request instance would reset the double-charge memory that makes
the idempotency guarantee mean anything.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from recoup.config import Settings
from recoup.gateway.server import EventLog, create_app
from recoup.gateway.webhooks import FailedPaymentEvent
from recoup.guardrails import Guardrails, idempotency_key
from recoup.policies.taxonomy_aware import TaxonomyAware
from recoup.sim.episode import Policy
from recoup.taxonomy import all_reasons, lookup
from recoup.trace import DecisionTrace, explain

__all__ = ["DashboardState", "FeedEntry", "create_dashboard"]

STATIC = Path(__file__).parent / "static"
RESULTS_PATH = Path("data/results/eval.json")
FEED_LIMIT = 50
"""How many traces the feed keeps. A demo, not a datastore — see ``EventLog``."""


@dataclass(frozen=True, slots=True)
class FeedEntry:
    """One trace, plus where the event came from."""

    trace: DecisionTrace
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, **self.trace.to_dict()}


@dataclass
class DashboardState:
    """Everything the screen shows. In memory, for the life of the process."""

    policy: Policy = field(default_factory=TaxonomyAware)
    guardrails: Guardrails = field(default_factory=Guardrails)
    events: EventLog = field(default_factory=EventLog)
    entries: list[FeedEntry] = field(default_factory=list)
    seen_payment_ids: set[str] = field(default_factory=set)
    results_path: Path = RESULTS_PATH

    def record(self, event: FailedPaymentEvent, source: str) -> FeedEntry:
        """Run one event through the engine and keep the trace."""
        entry = FeedEntry(
            trace=explain(event, policy=self.policy, guardrails=self.guardrails),
            source=source,
        )
        self.entries.append(entry)
        self.seen_payment_ids.add(event.payment_id)
        del self.entries[:-FEED_LIMIT]
        return entry

    def latest(self, limit: int) -> list[FeedEntry]:
        return self.entries[-limit:][::-1]

    def results(self) -> dict[str, Any]:
        """The measured evaluation, or an honest absence.

        Never a fallback set of numbers. A dashboard that shows plausible figures when the
        real ones are missing is the single worst thing this file could do.
        """
        if not self.results_path.exists():
            return {
                "available": False,
                "reason": f"{self.results_path} not found. Run: python scripts/run_eval.py",
            }
        try:
            document = json.loads(self.results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"available": False, "reason": f"could not read results: {exc}"}
        return {"available": True, **document}


class InjectRequest(BaseModel):
    """A synthetic decline, for showing a class the sandbox will not produce on demand."""

    decline_code: str = Field(default="card_expired")
    amount_paise: int = Field(default=125_000, gt=0)
    method: str = Field(default="card")


def _synthetic_event(request: InjectRequest) -> FailedPaymentEvent:
    """Build an event that is obviously synthetic, including in its payment id."""
    reason = lookup(request.decline_code)
    return FailedPaymentEvent(
        payment_id=f"pay_SIMULATED_{secrets.token_hex(6)}",
        order_id=None,
        amount_paise=request.amount_paise,
        method=request.method,
        error_reason=request.decline_code,
        error_code="BAD_REQUEST_ERROR",
        error_description=reason.description,
        error_source="simulated",
        error_step="payment_authorization",
        created_at=int(time.time()),
    )


def create_dashboard(
    secret: str | None = None,
    state: DashboardState | None = None,
) -> FastAPI:
    """Build the demo server: webhook receiver and dashboard on one port.

    The webhook routes come from ``recoup.gateway.server`` unchanged rather than being
    re-declared here, so the endpoint being demonstrated is the endpoint that is tested.
    """
    dashboard = state if state is not None else DashboardState()

    def on_failure(event: FailedPaymentEvent) -> None:
        dashboard.record(event, "webhook")

    app = create_app(secret=secret, on_failure=on_failure, log=dashboard.events)
    app.state.dashboard = dashboard

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        # Read per request so the page can be edited while the demo is running.
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/state")
    def api_state(limit: int = 20) -> dict[str, Any]:
        settings = Settings.from_env()
        return {
            "policy": dashboard.policy.name,
            "webhook_events": dashboard.events.count,
            "rejected_signatures": dashboard.events.rejected,
            "ignored_events": dashboard.events.ignored,
            "signature_verification": bool(settings.razorpay_webhook_secret),
            "razorpay_configured": settings.has_razorpay,
            "charges_recorded": dashboard.guardrails.charge_count,
            "refusals": dashboard.guardrails.refusal_counts(),
            "traces": [entry.to_dict() for entry in dashboard.latest(limit)],
        }

    @app.get("/api/results")
    def api_results() -> dict[str, Any]:
        return dashboard.results()

    @app.get("/api/decline-codes")
    def api_decline_codes() -> dict[str, Any]:
        """The taxonomy, for the injection menu. One source of truth, not a hardcoded list."""
        return {
            "codes": [
                {
                    "code": reason.code,
                    "class": reason.decline_class.value,
                    "remedy": reason.remedy.value,
                    "max_attempts": reason.max_attempts,
                }
                for reason in sorted(all_reasons(), key=lambda r: (r.decline_class.value, r.code))
            ]
        }

    @app.post("/api/inject")
    def api_inject(request: InjectRequest) -> dict[str, Any]:
        return dashboard.record(_synthetic_event(request), "injected").to_dict()

    @app.post("/api/double-charge")
    def api_double_charge() -> JSONResponse:
        """Try to charge the most recent payment again. It must be refused.

        Not a mock. It rebuilds the same idempotency key from the same inputs and submits
        it to the same guardrail instance that recorded the original, which is exactly what
        a replayed webhook or a crashed-and-restarted worker would do.
        """
        retries = [e for e in dashboard.entries if e.trace.action == "retry" and e.trace.allowed]
        if not retries:
            return JSONResponse(
                {
                    "attempted": False,
                    "reason": "no allowed charge to replay yet — send or inject a failure first",
                },
                status_code=409,
            )

        trace = retries[-1].trace
        key = idempotency_key(trace.payment_id, 0, f"{trace.payment_id}:instrument")
        verdict = dashboard.guardrails.check_retry(
            key=key,
            attempts_made=0,
            decline_code=trace.decline_code,
            instrument_expired=False,
        )
        return JSONResponse(
            {
                "attempted": True,
                "payment_id": trace.payment_id,
                "idempotency_key": key[:16],
                "allowed": verdict.allowed,
                "rule": verdict.refusal.rule.value if verdict.refusal else None,
                "detail": verdict.refusal.detail if verdict.refusal else "",
            }
        )

    @app.post("/api/sandbox/pull")
    def api_sandbox_pull(count: int = 20) -> dict[str, Any]:
        """Pull real failed payments from the Razorpay sandbox and trace them.

        Imported here rather than at module scope so the dashboard runs with no Razorpay
        credentials and no ``httpx`` installed — the simulator half of this project must
        never depend on the gateway half being configured.
        """
        from recoup.gateway.client import RazorpayClient
        from recoup.gateway.webhooks import event_from_entity

        try:
            client = RazorpayClient.from_env()
            payments = client.list_payments(count=count)
        except Exception as exc:  # noqa: BLE001 — any credential or network problem, shown as-is
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}", "added": 0}

        added = 0
        for payment in payments:
            if payment.get("status") != "failed":
                continue
            if str(payment.get("id")) in dashboard.seen_payment_ids:
                continue
            dashboard.record(event_from_entity(payment), "sandbox")
            added += 1

        return {"ok": True, "added": added, "inspected": len(payments)}

    return app
