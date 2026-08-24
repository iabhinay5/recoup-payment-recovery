"""Adversarial tests: try to make the system harm a customer, and watch it refuse.

Ordinary tests check that correct input produces correct output. These do the opposite —
each one plays the part of something going wrong (a buggy policy, a replayed webhook, a
crashed worker, a policy that has learned to be greedy) and asserts that the guardrail
layer stops it.

The distinction matters because a payment system's worst failures do not look like errors.
Charging someone twice returns HTTP 200 twice. Nothing raises. The only thing standing
between a duplicate request and a duplicate debit is a check that was written on purpose,
so these tests exist to prove that check cannot be bypassed.
"""

from __future__ import annotations

import numpy as np
import pytest

from recoup.guardrails import Guardrails, Rule, idempotency_key
from recoup.sim import (
    Action,
    ContactChannel,
    Customer,
    FailedPayment,
    Instrument,
    OutcomeModel,
    RailHealth,
    SimParams,
    run_episode,
)
from recoup.sim.episode import EpisodeState
from recoup.taxonomy import PaymentMethod


@pytest.fixture
def params() -> SimParams:
    return SimParams(n_customers=10, seed=1)


@pytest.fixture
def customer() -> Customer:
    return Customer(
        id="c1",
        instruments=(Instrument("i1", PaymentMethod.CARD, "sbi"),),
        monthly_income_paise=5_000_000,
        salary_day_of_month=1,
    )


@pytest.fixture
def payment() -> FailedPayment:
    return FailedPayment(
        id="pay_1",
        customer_id="c1",
        instrument_id="i1",
        amount_paise=250_000,
        initial_decline_code="insufficient_funds",
        reference_day_of_month=10,
        reference_hour_of_day=11.0,
    )


class _Relentless:
    """A policy that will not stop asking. Stands in for a bug or a bad objective."""

    def __init__(self, action: Action) -> None:
        self._action = action

    @property
    def name(self) -> str:
        return "relentless"

    def decide(self, state: EpisodeState) -> Action:
        return self._action


# =====================================================================================
# The double charge
# =====================================================================================


class TestDoubleCharge:
    """The worst thing this system could do."""

    def test_the_same_logical_attempt_cannot_be_charged_twice(self) -> None:
        """The core guarantee, stated as plainly as it can be.

        A replayed webhook, a retried HTTP request after a timeout, or a worker that died
        holding in-flight state all produce the same thing: an instruction to charge that
        has already been carried out. The ledger refuses it.
        """
        guards = Guardrails()
        key = idempotency_key("pay_1", 0, "i1")

        first = guards.check_retry(
            key, attempts_made=0, decline_code="insufficient_funds", instrument_expired=False
        )
        assert first.allowed
        guards.record_charge(key)

        replay = guards.check_retry(
            key, attempts_made=0, decline_code="insufficient_funds", instrument_expired=False
        )
        assert not replay.allowed
        assert replay.refusal is not None
        assert replay.refusal.rule is Rule.DUPLICATE_CHARGE

    def test_the_key_is_deterministic_not_random(self) -> None:
        """A random key would defeat the mechanism entirely while looking correct.

        The point is that the same logical attempt derives the same key on a different
        machine, in a different process, after a restart.
        """
        assert idempotency_key("pay_1", 0, "i1") == idempotency_key("pay_1", 0, "i1")

    def test_different_attempts_are_different_charges(self) -> None:
        """Idempotency must not block *legitimate* sequential retries.

        A guard that blocked every retry after the first would be safe and useless.
        """
        keys = {idempotency_key("pay_1", n, "i1") for n in range(4)}
        assert len(keys) == 4

    def test_switching_instrument_is_a_different_charge(self) -> None:
        assert idempotency_key("pay_1", 0, "i1") != idempotency_key("pay_1", 0, "i2")

    def test_different_payments_never_collide(self) -> None:
        assert idempotency_key("pay_1", 0, "i1") != idempotency_key("pay_2", 0, "i1")

    def test_a_ledger_shared_across_restarts_still_refuses(self, params, customer, payment) -> None:  # type: ignore[no-untyped-def]
        """Simulates a crash: the episode runs, the process restarts, the same work is
        re-issued against a durable ledger. The charges do not double."""
        guards = Guardrails()
        rails = RailHealth({"sbi": 0.0})
        model = OutcomeModel(params, rails)

        run_episode(
            payment,
            customer,
            _Relentless(Action.retry(6.0)),
            model,
            rails,
            params,
            np.random.default_rng(0),
            guardrails=guards,
        )
        charges_after_first_run = guards.charge_count

        # Same ledger, same payment, replayed from the beginning.
        run_episode(
            payment,
            customer,
            _Relentless(Action.retry(6.0)),
            model,
            rails,
            params,
            np.random.default_rng(0),
            guardrails=guards,
        )

        assert guards.charge_count == charges_after_first_run, (
            "replaying the episode against a shared ledger must not add charges"
        )
        assert guards.refusal_counts().get(Rule.DUPLICATE_CHARGE.value, 0) > 0


# =====================================================================================
# A policy that will not take no for an answer
# =====================================================================================


class TestGreedyPolicyIsContained:
    def test_infinite_retry_demand_is_capped(self, params, customer) -> None:  # type: ignore[no-untyped-def]
        """card_expired permits zero attempts. The policy asks forever; none execute."""
        payment = FailedPayment("p", "c1", "i1", 1000, "card_expired", 10)
        guards = Guardrails()
        rails = RailHealth({"sbi": 0.0})

        result = run_episode(
            payment,
            customer,
            _Relentless(Action.retry(1.0)),
            OutcomeModel(params, rails),
            rails,
            params,
            np.random.default_rng(0),
            guardrails=guards,
        )

        assert result.attempt_count == 0
        assert guards.charge_count == 0
        assert result.refused_actions > 0

    def test_a_dead_instrument_is_never_charged(self) -> None:
        """Charging an expired card cannot succeed, so the attempt is pure waste."""
        guards = Guardrails()
        verdict = guards.check_retry(
            idempotency_key("p", 0, "i1"),
            attempts_made=0,
            decline_code="insufficient_funds",
            instrument_expired=True,
        )
        assert not verdict.allowed
        assert verdict.refusal is not None
        assert verdict.refusal.rule is Rule.DEAD_INSTRUMENT

    def test_attempt_cap_holds_at_the_boundary(self) -> None:
        guards = Guardrails()
        cap = 4  # insufficient_funds
        for made in range(cap):
            v = guards.check_retry(
                idempotency_key("p", made, "i1"), made, "insufficient_funds", False
            )
            assert v.allowed, f"attempt {made} should be permitted"
            guards.record_charge(idempotency_key("p", made, "i1"))

        over = guards.check_retry(idempotency_key("p", cap, "i1"), cap, "insufficient_funds", False)
        assert not over.allowed
        assert over.refusal is not None
        assert over.refusal.rule is Rule.ATTEMPT_CAP


# =====================================================================================
# Harassing the customer
# =====================================================================================


class TestCustomerProtection:
    def test_an_opted_out_customer_is_never_contacted(self, params, payment) -> None:  # type: ignore[no-untyped-def]
        opted_out = Customer(
            "c1", (Instrument("i1", PaymentMethod.CARD, "sbi"),), 5_000_000, 1, opted_out=True
        )
        guards = Guardrails()
        rails = RailHealth({"sbi": 0.0})

        result = run_episode(
            payment,
            opted_out,
            _Relentless(Action.outreach(1.0, ContactChannel.SMS)),
            OutcomeModel(params, rails),
            rails,
            params,
            np.random.default_rng(0),
            guardrails=guards,
        )

        assert result.contact_count == 0
        assert guards.refusal_counts().get(Rule.OPTED_OUT.value, 0) > 0

    def test_contact_rate_limit_stops_a_flood(self) -> None:
        guards = Guardrails(max_contacts_per_window=3)
        for n in range(3):
            assert guards.check_outreach(
                hour_of_day=12.0, opted_out=False, contacts_in_window=n
            ).allowed
        blocked = guards.check_outreach(hour_of_day=12.0, opted_out=False, contacts_in_window=3)
        assert not blocked.allowed
        assert blocked.refusal is not None
        assert blocked.refusal.rule is Rule.CONTACT_RATE

    def test_nobody_is_messaged_at_three_in_the_morning(self) -> None:
        guards = Guardrails()
        verdict = guards.check_outreach(hour_of_day=3.0, opted_out=False, contacts_in_window=0)
        assert not verdict.allowed
        assert verdict.deferred, "quiet hours defers; it does not discard the message"
        assert verdict.defer_hours == pytest.approx(5.0)

    def test_quiet_hours_wrap_past_midnight(self) -> None:
        """The case a naive start <= h < end comparison gets silently wrong."""
        guards = Guardrails(quiet_start_hour=22.0, quiet_end_hour=8.0)
        for hour in (22.0, 23.5, 0.0, 3.0, 7.9):
            assert guards.in_quiet_hours(hour), f"{hour}h should be quiet"
        for hour in (8.0, 12.0, 18.0, 21.9):
            assert not guards.in_quiet_hours(hour), f"{hour}h should be permitted"

    def test_deferral_lands_outside_quiet_hours(self) -> None:
        guards = Guardrails()
        for hour in (22.5, 1.0, 4.0, 7.0):
            woken = (hour + guards.hours_until_wake(hour)) % 24
            assert not guards.in_quiet_hours(woken + 0.01)

    def test_a_message_at_noon_is_not_delayed(self) -> None:
        assert Guardrails().hours_until_wake(12.0) == 0.0


# =====================================================================================
# The audit trail
# =====================================================================================


class TestAuditability:
    def test_every_refusal_names_its_rule_and_reason(self) -> None:
        """A merchant asking "why wasn't my customer retried?" gets an answer, not a shrug."""
        guards = Guardrails()
        guards.check_outreach(hour_of_day=12.0, opted_out=True, contacts_in_window=0)

        assert len(guards.refusals) == 1
        refusal = guards.refusals[0]
        assert refusal.rule is Rule.OPTED_OUT
        assert refusal.detail
        assert "opted_out" in str(refusal)

    def test_refusal_counts_summarise_by_rule(self) -> None:
        guards = Guardrails()
        for _ in range(3):
            guards.check_outreach(hour_of_day=12.0, opted_out=True, contacts_in_window=0)
        guards.check_outreach(hour_of_day=3.0, opted_out=False, contacts_in_window=0)

        counts = guards.refusal_counts()
        assert counts[Rule.OPTED_OUT.value] == 3
        assert counts[Rule.QUIET_HOURS.value] == 1

    def test_allowed_actions_are_not_recorded_as_refusals(self) -> None:
        guards = Guardrails()
        guards.check_outreach(hour_of_day=12.0, opted_out=False, contacts_in_window=0)
        assert guards.refusals == []
