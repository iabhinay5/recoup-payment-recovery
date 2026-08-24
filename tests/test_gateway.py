"""Tests for the Razorpay integration.

The signature tests carry the weight. A webhook endpoint is a public URL that makes your
system act on payment events; if verification is wrong, anyone who finds the URL can drive
your retry logic. These check that forgeries, tampering and misconfiguration are all
rejected — and that the endpoint still works when the signature is genuine.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from recoup.gateway import (
    InvalidSignature,
    LiveKeyRefused,
    RazorpayClient,
    parse_failed_payment,
    verify_signature,
)
from recoup.gateway.server import EventLog, create_app

SECRET = "whsec_test_abc123"


def payment_failed_body(
    reason: str = "insufficient_funds",
    amount: int = 125_000,
    method: str = "upi",
) -> bytes:
    """A payload shaped exactly like Razorpay's documented payment.failed event."""
    return json.dumps(
        {
            "entity": "event",
            "account_id": "acc_TEST",
            "event": "payment.failed",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_TEST123",
                        "entity": "payment",
                        "amount": amount,
                        "currency": "INR",
                        "status": "failed",
                        "order_id": "order_TEST123",
                        "method": method,
                        "error_code": "BAD_REQUEST_ERROR",
                        "error_description": "Your payment did not go through",
                        "error_reason": reason,
                        "error_source": "bank",
                        "error_step": "payment_authorization",
                        "created_at": 1767610214,
                    }
                }
            },
            "created_at": 1767610215,
        }
    ).encode("utf-8")


def sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class TestSignatureVerification:
    def test_a_genuine_signature_passes(self) -> None:
        body = payment_failed_body()
        verify_signature(body, sign(body), SECRET)  # must not raise

    def test_a_forged_signature_is_rejected(self) -> None:
        body = payment_failed_body()
        with pytest.raises(InvalidSignature, match="mismatch"):
            verify_signature(body, "f" * 64, SECRET)

    def test_tampering_with_the_body_invalidates_the_signature(self) -> None:
        """The attack the signature exists to stop: change the amount in transit."""
        original = payment_failed_body(amount=125_000)
        signature = sign(original)
        tampered = payment_failed_body(amount=1)

        with pytest.raises(InvalidSignature):
            verify_signature(tampered, signature, SECRET)

    def test_the_wrong_secret_is_rejected(self) -> None:
        body = payment_failed_body()
        with pytest.raises(InvalidSignature):
            verify_signature(body, sign(body, "some_other_secret"), SECRET)

    def test_a_missing_secret_refuses_rather_than_accepting_everything(self) -> None:
        """An unconfigured endpoint must fail closed.

        Treating "no secret" as "skip verification" is how a staging shortcut becomes a
        production hole.
        """
        body = payment_failed_body()
        with pytest.raises(InvalidSignature, match="no webhook secret"):
            verify_signature(body, sign(body), "")

    def test_an_empty_signature_is_rejected(self) -> None:
        body = payment_failed_body()
        with pytest.raises(InvalidSignature):
            verify_signature(body, "", SECRET)

    def test_reserialised_json_does_not_verify(self) -> None:
        """Documents the most common way this goes wrong.

        Signing a parsed-then-re-dumped body changes whitespace and key order, so the
        signature never matches. Verification must use the bytes as received.
        """
        body = payment_failed_body()
        reserialised = json.dumps(json.loads(body), indent=2).encode()
        assert reserialised != body
        with pytest.raises(InvalidSignature):
            verify_signature(reserialised, sign(body), SECRET)


class TestPayloadParsing:
    def test_extracts_the_documented_fields(self) -> None:
        event = parse_failed_payment(payment_failed_body())
        assert event.payment_id == "pay_TEST123"
        assert event.order_id == "order_TEST123"
        assert event.amount_paise == 125_000
        assert event.amount_rupees == 1250.0
        assert event.method == "upi"
        assert event.error_source == "bank"
        assert event.error_step == "payment_authorization"

    def test_error_reason_lands_on_the_taxonomy_without_translation(self) -> None:
        """Razorpay's error_reason uses the same vocabulary the taxonomy was built from."""
        event = parse_failed_payment(payment_failed_body("card_expired"))
        assert event.is_recognised
        assert event.reason.max_attempts == 0, "an expired card is never retried"

    def test_an_unknown_reason_degrades_conservatively(self) -> None:
        """Razorpay is free to ship a code we have not classified."""
        event = parse_failed_payment(payment_failed_body("some_code_from_2027"))
        assert not event.is_recognised
        assert event.reason.max_attempts == 1
        assert event.reason.min_backoff_hours >= 24.0

    def test_other_events_are_refused(self) -> None:
        body = json.dumps({"event": "payment.captured", "payload": {}}).encode()
        with pytest.raises(ValueError, match="expected event"):
            parse_failed_payment(body)

    def test_a_missing_payment_entity_is_refused(self) -> None:
        body = json.dumps({"event": "payment.failed", "payload": {}}).encode()
        with pytest.raises(ValueError, match="payload.payment.entity"):
            parse_failed_payment(body)

    def test_accepts_bytes_str_and_dict(self) -> None:
        body = payment_failed_body()
        for form in (body, body.decode(), json.loads(body)):
            assert parse_failed_payment(form).payment_id == "pay_TEST123"


class TestWebhookEndpoint:
    def _client(self, log: EventLog | None = None) -> TestClient:
        return TestClient(create_app(secret=SECRET, log=log))

    def test_a_signed_failure_is_accepted_and_classified(self) -> None:
        body = payment_failed_body("insufficient_funds")
        response = self._client().post(
            "/webhook/razorpay",
            content=body,
            headers={"X-Razorpay-Signature": sign(body)},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["decline_code"] == "insufficient_funds"
        assert data["remedy"] == "retry_later"

    def test_an_unsigned_request_is_rejected_with_401(self) -> None:
        body = payment_failed_body()
        response = self._client().post("/webhook/razorpay", content=body)
        assert response.status_code == 401

    def test_a_forged_request_never_reaches_the_log(self) -> None:
        log = EventLog()
        client = self._client(log)
        body = payment_failed_body()

        client.post(
            "/webhook/razorpay",
            content=body,
            headers={"X-Razorpay-Signature": "0" * 64},
        )

        assert log.count == 0, "a forged event must not be recorded or acted upon"
        assert log.rejected == 1

    def test_unhandled_event_types_are_acknowledged_not_errored(self) -> None:
        """Razorpay redelivers non-2xx responses.

        Returning an error for an event we simply do not handle would turn a normal
        condition into an infinite redelivery loop.
        """
        body = json.dumps(
            {"event": "payment.captured", "payload": {"payment": {"entity": {}}}}
        ).encode()
        response = self._client().post(
            "/webhook/razorpay",
            content=body,
            headers={"X-Razorpay-Signature": sign(body)},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ignored"

    def test_the_callback_receives_verified_events_only(self) -> None:
        seen = []
        app = create_app(secret=SECRET, on_failure=seen.append)
        client = TestClient(app)

        good = payment_failed_body()
        client.post("/webhook/razorpay", content=good, headers={"X-Razorpay-Signature": sign(good)})
        client.post("/webhook/razorpay", content=good, headers={"X-Razorpay-Signature": "bad"})

        assert len(seen) == 1

    def test_health_reports_whether_verification_is_configured(self) -> None:
        assert self._client().get("/health").json()["signature_verification"] == "enabled"
        unconfigured = TestClient(create_app(secret=""))
        assert unconfigured.get("/health").json()["signature_verification"] == "NOT CONFIGURED"

    def test_events_endpoint_lists_classified_failures(self) -> None:
        client = self._client()
        body = payment_failed_body("payment_cancelled")
        client.post("/webhook/razorpay", content=body, headers={"X-Razorpay-Signature": sign(body)})

        data = client.get("/events").json()
        assert data["count"] == 1
        assert data["events"][0]["decline_class"] == "session_conditional"


class TestLiveKeyRefusal:
    def test_a_live_key_is_refused_at_construction(self) -> None:
        """Recoup issues payment retries. A live key would move real money."""
        with pytest.raises(LiveKeyRefused, match="not a test key"):
            RazorpayClient(key_id="rzp_live_abc123", key_secret="secret")

    def test_a_test_key_is_accepted(self) -> None:
        assert RazorpayClient(key_id="rzp_test_abc123", key_secret="s").key_id.startswith(
            "rzp_test_"
        )

    def test_tiny_amounts_are_rejected_before_the_api_sees_them(self) -> None:
        client = RazorpayClient(key_id="rzp_test_abc", key_secret="s")
        with pytest.raises(ValueError, match="below 100 paise"):
            client.create_order(amount_paise=50, receipt="r")
