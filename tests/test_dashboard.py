"""Tests for the demo surface.

Two things are worth testing about a dashboard and only two. First, that it reports the
engine rather than a story about the engine — a screen that agrees with the code today and
diverges silently tomorrow is worse than no screen. Second, that it never invents a number:
when the measured results are missing it must say so, because a plausible placeholder on a
results panel is indistinguishable from a real result to everyone except the person who
wrote it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recoup.dashboard import DashboardState, create_dashboard
from recoup.guardrails import Rule

SECRET = "whsec_dashboard_test"


def payment_failed_body(reason: str = "insufficient_funds", payment_id: str = "pay_dash") -> bytes:
    return json.dumps(
        {
            "entity": "event",
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "id": payment_id,
                        "amount": 125_000,
                        "method": "card",
                        "status": "failed",
                        "order_id": "order_dash",
                        "error_reason": reason,
                        "error_code": "BAD_REQUEST_ERROR",
                        "error_description": "test",
                        "error_source": "customer",
                        "error_step": "payment_authorization",
                        "created_at": 1_756_100_000,
                    }
                }
            },
        }
    ).encode()


def signed(body: bytes) -> dict[str, str]:
    return {
        "X-Razorpay-Signature": hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest(),
        "content-type": "application/json",
    }


@pytest.fixture
def state(tmp_path: Path) -> DashboardState:
    return DashboardState(results_path=tmp_path / "absent.json")


@pytest.fixture
def client(state: DashboardState) -> TestClient:
    return TestClient(create_dashboard(secret=SECRET, state=state))


class TestThePageItself:
    def test_it_serves(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "Recoup" in response.text

    def test_it_pulls_nothing_from_the_network_except_fonts(self, client: TestClient) -> None:
        """The demo machine may have no internet, and a CDN outage must not blank the screen.

        Fonts are allowed to degrade to the fallback stack. Everything that carries meaning
        — the script, the styles — is inline.
        """
        import re

        page = client.get("/").text
        external = [
            url
            for url in re.findall(r'(?:src|href)="(https?://[^"]+)"', page)
            if "fonts.googleapis.com" not in url and "fonts.gstatic.com" not in url
        ]
        assert external == []
        assert "<script>" in page
        assert "/api/state" in page


class TestTheFeedIsDrivenByRealEvents:
    def test_a_signed_webhook_produces_a_trace(self, client: TestClient) -> None:
        body = payment_failed_body()
        assert (
            client.post("/webhook/razorpay", content=body, headers=signed(body)).status_code == 200
        )

        state = client.get("/api/state").json()
        assert state["webhook_events"] == 1
        assert len(state["traces"]) == 1
        assert state["traces"][0]["source"] == "webhook"
        assert state["traces"][0]["decline_code"] == "insufficient_funds"

    def test_an_unsigned_webhook_produces_nothing(self, client: TestClient) -> None:
        """A forged delivery must not reach the feed, let alone the policy."""
        body = payment_failed_body()
        response = client.post(
            "/webhook/razorpay", content=body, headers={"X-Razorpay-Signature": "forged"}
        )
        assert response.status_code == 401

        state = client.get("/api/state").json()
        assert state["traces"] == []
        assert state["rejected_signatures"] == 1

    def test_injected_events_are_labelled_synthetic(self, client: TestClient) -> None:
        """The single most important honesty property of the demo."""
        trace = client.post("/api/inject", json={"decline_code": "card_expired"}).json()
        assert trace["source"] == "injected"
        assert "SIMULATED" in trace["payment_id"]

    def test_the_injection_menu_comes_from_the_taxonomy(self, client: TestClient) -> None:
        """A hardcoded menu is a second source of truth waiting to disagree."""
        from recoup.taxonomy import all_reasons

        codes = client.get("/api/decline-codes").json()["codes"]
        assert {c["code"] for c in codes} == {r.code for r in all_reasons()}


class TestGuardrailsAreLiveNotIllustrated:
    def test_the_double_charge_button_is_actually_refused(self, client: TestClient) -> None:
        client.post("/api/inject", json={"decline_code": "insufficient_funds"})

        result = client.post("/api/double-charge").json()
        assert result["attempted"] is True
        assert result["allowed"] is False
        assert result["rule"] == Rule.DUPLICATE_CHARGE.value

    def test_it_refuses_to_pretend_when_there_is_nothing_to_replay(
        self, client: TestClient
    ) -> None:
        """Better a 409 than a theatrical refusal of a charge that never happened."""
        response = client.post("/api/double-charge")
        assert response.status_code == 409
        assert response.json()["attempted"] is False

    def test_the_ledger_accumulates_across_events(self, client: TestClient) -> None:
        client.post("/api/inject", json={"decline_code": "insufficient_funds"})
        client.post("/api/double-charge")

        refusals = client.get("/api/state").json()["refusals"]
        assert refusals[Rule.DUPLICATE_CHARGE.value] >= 1

    def test_the_guardrail_instance_is_shared_not_per_request(self, client: TestClient) -> None:
        """A fresh instance per request would reset the memory the guarantee depends on."""
        first = client.post("/api/inject", json={"decline_code": "insufficient_funds"}).json()
        assert first["allowed"] is True
        assert client.get("/api/state").json()["charges_recorded"] == 1


class TestResultsAreReadNeverInvented:
    def test_missing_results_are_reported_as_missing(self, client: TestClient) -> None:
        payload = client.get("/api/results").json()
        assert payload["available"] is False
        assert "run_eval" in payload["reason"]

    def test_unreadable_results_are_reported_rather_than_swallowed(self, tmp_path: Path) -> None:
        broken = tmp_path / "eval.json"
        broken.write_text("{not json", encoding="utf-8")
        client = TestClient(
            create_dashboard(secret=SECRET, state=DashboardState(results_path=broken))
        )
        payload = client.get("/api/results").json()
        assert payload["available"] is False

    def test_real_results_are_passed_through_unchanged(self, tmp_path: Path) -> None:
        document = {
            "baseline": "fixed_1_3_5_7",
            "n_episodes": 20_000,
            "policies": [{"policy": "fixed_1_3_5_7", "recovery_rate": 0.584}],
        }
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        client = TestClient(
            create_dashboard(secret=SECRET, state=DashboardState(results_path=path))
        )

        payload = client.get("/api/results").json()
        assert payload["available"] is True
        assert payload["policies"] == document["policies"]


class TestSandboxPull:
    def test_it_reports_a_credential_problem_instead_of_crashing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The demo machine may have no keys loaded. That is a message, not a 500."""
        monkeypatch.setenv("RAZORPAY_KEY_ID", "")
        monkeypatch.setenv("RAZORPAY_KEY_SECRET", "")

        response = client.post("/api/sandbox/pull")
        assert response.status_code == 200
        body = response.json()
        assert body["added"] == 0
        if not body["ok"]:
            assert body["error"]
