"""Tests for the decline taxonomy.

These are mostly invariant tests rather than behaviour tests. The taxonomy is a table of
facts, and the thing worth guarding is that the table cannot drift into a state where the
policy's structural guarantees stop holding.
"""

from __future__ import annotations

import pytest

from recoup.taxonomy import (
    CARD_ERRORS_SOURCE,
    UNKNOWN_REASON,
    UPI_ERRORS_SOURCE,
    DeclineClass,
    DeclineReason,
    PaymentMethod,
    Remedy,
    all_reasons,
    by_class,
    by_method,
    is_known,
    is_retryable,
    lookup,
    max_attempts_for,
)


class TestRegistryIntegrity:
    def test_no_duplicate_codes(self) -> None:
        codes = [r.code for r in all_reasons()]
        assert len(codes) == len(set(codes))

    def test_every_reason_cites_a_source(self) -> None:
        for reason in all_reasons():
            assert CARD_ERRORS_SOURCE in reason.source or UPI_ERRORS_SOURCE in reason.source, (
                f"{reason.code} has no Razorpay documentation source. Every entry must be "
                f"traceable — see docs/CALIBRATION.md."
            )

    def test_every_reason_has_at_least_one_method(self) -> None:
        for reason in all_reasons():
            assert reason.methods, f"{reason.code} is not attached to any payment method"

    def test_card_coverage_matches_documented_count(self) -> None:
        """Razorpay's card error page documents 16 codes; all 16 are encoded."""
        assert len(by_method(PaymentMethod.CARD)) == 16

    def test_upi_coverage_matches_verified_count(self) -> None:
        """10 of the 11 UPI reasons are encoded.

        The eleventh is listed in UNVERIFIED_REASONS because its machine-readable code
        string could not be confirmed verbatim.
        """
        assert len(by_method(PaymentMethod.UPI)) == 10


class TestStructuralInvariant:
    """The never-retry guarantee must hold by construction, not by convention.

    This is the test that backs DECISIONS.md ADR-007. If it can be made to fail, the
    guardrail is advisory and the whole claim about structural safety is hollow.
    """

    def test_never_retryable_codes_have_zero_attempts(self) -> None:
        for reason in by_class(DeclineClass.NEVER_RETRYABLE):
            assert reason.max_attempts == 0, (
                f"{reason.code} is NEVER_RETRYABLE but permits {reason.max_attempts} attempts"
            )
            assert not reason.is_retryable

    def test_constructing_a_violating_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="must have max_attempts == 0"):
            DeclineReason(
                code="bogus",
                methods=frozenset({PaymentMethod.CARD}),
                description="A never-retryable code that claims to be retryable.",
                decline_class=DeclineClass.NEVER_RETRYABLE,
                remedy=Remedy.NEW_INSTRUMENT,
                max_attempts=3,
                min_backoff_hours=1.0,
                source=CARD_ERRORS_SOURCE,
            )

    def test_negative_attempts_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_attempts must be >= 0"):
            DeclineReason(
                code="bogus",
                methods=frozenset({PaymentMethod.CARD}),
                description="Negative attempts.",
                decline_class=DeclineClass.HARD_DECLINED,
                remedy=Remedy.ESCALATE,
                max_attempts=-1,
                min_backoff_hours=1.0,
                source=CARD_ERRORS_SOURCE,
            )

    def test_reasons_are_immutable(self) -> None:
        reason = lookup("card_expired")
        with pytest.raises(AttributeError):
            reason.max_attempts = 5  # type: ignore[misc]


class TestHeadlineClaims:
    """The specific claims made in the README and the pitch video."""

    def test_expired_card_is_never_retried(self) -> None:
        assert not is_retryable("card_expired")
        assert max_attempts_for("card_expired") == 0

    def test_insufficient_funds_is_time_conditional(self) -> None:
        reason = lookup("insufficient_funds")
        assert reason.decline_class is DeclineClass.TIME_CONDITIONAL
        assert reason.remedy is Remedy.RETRY_LATER
        assert reason.is_retryable

    def test_bank_downtime_waits_for_rail_recovery(self) -> None:
        for code in ("bank_technical_error", "gateway_technical_error"):
            reason = lookup(code)
            assert reason.decline_class is DeclineClass.RAIL_CONDITIONAL
            assert reason.remedy is Remedy.AWAIT_RAIL_RECOVERY

    def test_session_conditional_codes_need_customer_contact(self) -> None:
        for reason in by_class(DeclineClass.SESSION_CONDITIONAL):
            assert reason.needs_customer_contact, (
                f"{reason.code} needs a live customer session but is not flagged as "
                f"requiring contact — a silent retry cannot resolve it"
            )

    def test_risk_flagged_payments_are_not_auto_retried(self) -> None:
        """Repeated attempts against a fraud flag can harden the issuer's stance."""
        assert max_attempts_for("payment_risk_check_failed") == 0

    def test_ambiguous_credit_leg_is_not_retried(self) -> None:
        """A blind retry when the debit state is unknown risks a double charge."""
        assert max_attempts_for("credit_failed") == 0


class TestUnknownCodes:
    """New decline codes are a routine operational event, not an exception."""

    def test_unknown_code_does_not_raise(self) -> None:
        assert lookup("some_code_razorpay_ships_next_year") is UNKNOWN_REASON

    def test_unknown_code_degrades_conservatively(self) -> None:
        reason = lookup("totally_unseen_code")
        assert reason.remedy is Remedy.ESCALATE
        assert reason.max_attempts == 1
        assert reason.min_backoff_hours >= 24.0

    def test_is_known_distinguishes_real_codes(self) -> None:
        assert is_known("card_expired")
        assert not is_known("card_exp1red")

    def test_typo_in_code_does_not_silently_become_retryable(self) -> None:
        """A near-miss on a never-retryable code must not fall through to a permissive
        default — that would turn a typo into a retry storm."""
        assert max_attempts_for("card_expired_") <= 1


class TestClassConsistency:
    def test_rail_conditional_codes_all_await_recovery(self) -> None:
        for reason in by_class(DeclineClass.RAIL_CONDITIONAL):
            assert reason.remedy is Remedy.AWAIT_RAIL_RECOVERY

    def test_retryable_codes_have_a_positive_backoff(self) -> None:
        """Any code we will retry must specify how long to wait first. A zero backoff on
        a retryable code is how retry storms start."""
        for reason in all_reasons():
            if reason.is_retryable:
                assert reason.min_backoff_hours > 0, (
                    f"{reason.code} is retryable with no minimum backoff"
                )

    def test_every_class_is_represented(self) -> None:
        for decline_class in DeclineClass:
            assert by_class(decline_class), f"{decline_class} has no codes"
