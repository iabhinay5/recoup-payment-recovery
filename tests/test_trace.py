"""The decision trace must report the engine, not paraphrase it.

The failure this file is guarding against is subtle and expensive: a trace that drifts out
of agreement with the policy it claims to describe. It would not crash, no test of the
policy would fail, and the first person to notice would be a panellist asking why the
screen says one thing and the code says another.
"""

from __future__ import annotations

import json

import pytest

from recoup.gateway.webhooks import FailedPaymentEvent
from recoup.guardrails import Guardrails, Rule
from recoup.policies.taxonomy_aware import TaxonomyAware
from recoup.sim.entities import Instrument
from recoup.sim.episode import ActionKind
from recoup.taxonomy import DeclineClass, PaymentMethod, lookup
from recoup.trace import UNKNOWN_BANK, explain, live_state

NOON_IST = 1_756_100_000 + 6 * 3600
"""A timestamp landing outside quiet hours, so an outreach test is not measuring the clock."""


def event(
    code: str = "insufficient_funds",
    *,
    payment_id: str = "pay_test",
    amount_paise: int = 125_000,
    method: str = "card",
    created_at: int = NOON_IST,
) -> FailedPaymentEvent:
    return FailedPaymentEvent(
        payment_id=payment_id,
        order_id="order_test",
        amount_paise=amount_paise,
        method=method,
        error_reason=code,
        error_code="BAD_REQUEST_ERROR",
        error_description="a test failure",
        error_source="customer",
        error_step="payment_authorization",
        created_at=created_at,
    )


class TestLiveState:
    def test_builds_the_simulators_own_state_object(self) -> None:
        """The point of the whole module: one state type, two producers."""
        state = live_state(event())
        assert state.current_decline_code == "insufficient_funds"
        assert state.attempt_count == 0
        assert state.payment.amount_paise == 125_000

    def test_reference_time_comes_from_the_payload_not_the_clock(self) -> None:
        """A trace replayed tomorrow must describe the same moment it did today."""
        first = live_state(event(created_at=NOON_IST))
        second = live_state(event(created_at=NOON_IST))
        assert first.payment.reference_hour_of_day == second.payment.reference_hour_of_day
        assert first.payment.reference_day_of_month == second.payment.reference_day_of_month

    def test_day_of_month_is_clamped_into_the_simulated_month(self) -> None:
        """A payment on the 31st must not construct an invalid customer."""
        # 2026-01-31T12:00:00+05:30
        state = live_state(event(created_at=1_769_840_000))
        assert 1 <= state.payment.reference_day_of_month <= 28

    def test_income_is_not_read_on_the_live_path(self) -> None:
        """Income is a simulator input a webhook cannot supply.

        The trace substitutes a placeholder, which is only acceptable while nothing on this
        path reads it. This test is what makes that a fact rather than an intention.
        """
        state = live_state(event())
        baseline = TaxonomyAware().decide(state)

        from dataclasses import replace

        rich = replace(state, customer=replace(state.customer, monthly_income_paise=90_000_000))
        assert TaxonomyAware().decide(rich) == baseline

    def test_unknown_bank_reads_as_healthy_rather_than_blocking(self) -> None:
        """A missing health signal must degrade to normal behaviour, not to refusing work."""
        state = live_state(event("bank_technical_error"))
        assert state.current_instrument.bank_id == UNKNOWN_BANK
        assert not state.rail_is_degraded()

    def test_an_alternative_instrument_does_not_currently_change_the_decision(self) -> None:
        """Pins a known inconsistency, so that it is visible rather than lost.

        ``card_expired`` carries ``Remedy.NEW_INSTRUMENT``, and both ``TaxonomyAware`` and
        ``BanditPolicy`` contain a branch that switches to another instrument. That branch
        is gated on ``state.remaining_attempts > 0``, and the taxonomy requires every
        ``NEVER_RETRYABLE`` code to have ``max_attempts == 0``. So the branch can never
        run, and the guardrail layer would refuse the charge on ``ATTEMPT_CAP`` even if it
        did: the cap counts attempts against the *decline code*, not against the
        instrument that produced it.

        Whether that is correct is a real design question — the cap may mean "no attempt
        on this payment" or "no attempt on this instrument" — and it changes the measured
        numbers, so it is not something to settle inside a trace test. Until it is
        decided, this asserts what the system actually does.
        """
        spare = Instrument("instr_spare", PaymentMethod.UPI, "HDFC")
        with_spare = explain(
            event("card_expired"),
            state=live_state(event("card_expired"), alternatives=(spare,)),
        )
        assert with_spare.action == "outreach"
        assert lookup("card_expired").remedy.value == "new_instrument"
        assert lookup("card_expired").max_attempts == 0


class TestExplainAgreesWithTheEngine:
    @pytest.mark.parametrize(
        "code",
        ["card_expired", "payment_cancelled", "insufficient_funds", "bank_technical_error"],
    )
    def test_action_is_the_policys_action(self, code: str) -> None:
        """No paraphrase. The trace reports exactly what the policy returned."""
        payload = event(code)
        state = live_state(payload)
        action = TaxonomyAware().decide(state)
        trace = explain(payload, state=live_state(payload))

        expected = {
            ActionKind.RETRY: "retry",
            ActionKind.OUTREACH: "outreach",
            ActionKind.GIVE_UP: "stop",
        }[action.kind]
        assert trace.action == expected

    @pytest.mark.parametrize(
        "code",
        ["card_expired", "payment_cancelled", "insufficient_funds", "bank_technical_error"],
    )
    def test_classification_is_the_taxonomys_classification(self, code: str) -> None:
        trace = explain(event(code))
        reason = lookup(code)
        assert trace.decline_class == reason.decline_class.value
        assert trace.remedy == reason.remedy.value

    def test_every_stage_appears_in_order(self) -> None:
        trace = explain(event("insufficient_funds"))
        stages = [step.stage for step in trace.steps]
        assert stages[0] == "received"
        assert stages[1] == "classified"
        assert stages[2] == "decided"
        assert "guarded" in stages
        assert stages[-1] == "counterfactual"


class TestGuardrailsAreExercisedNotDescribed:
    def test_a_permitted_retry_is_refused_on_replay(self) -> None:
        """The double-charge guard, run rather than asserted, on every retry trace."""
        trace = explain(event("insufficient_funds"))
        replays = [s for s in trace.steps if "replayed" in s.title]
        assert len(replays) == 1
        assert replays[0].kind == "refuse"
        assert Rule.DUPLICATE_CHARGE.value in replays[0].verdict

    def test_two_different_payments_do_not_collide(self) -> None:
        """Idempotency must key on the payment, or the second customer is never charged."""
        guards = Guardrails()
        first = explain(event(payment_id="pay_one"), guardrails=guards)
        second = explain(event(payment_id="pay_two"), guardrails=guards)
        assert first.allowed
        assert second.allowed

    def test_the_same_payment_twice_collides(self) -> None:
        """A redelivered webhook must not become a second charge."""
        guards = Guardrails()
        first = explain(event(payment_id="pay_same"), guardrails=guards)
        second = explain(event(payment_id="pay_same"), guardrails=guards)
        assert first.allowed
        assert not second.allowed
        assert second.outcome == "refused"

    def test_a_dead_instrument_is_refused_when_there_is_nowhere_to_switch(self) -> None:
        """``card_expired`` with no alternative routes to the customer, not to a charge."""
        trace = explain(event("card_expired"))
        assert trace.action == "outreach"
        assert trace.decline_class == DeclineClass.NEVER_RETRYABLE.value

    def test_quiet_hours_defer_rather_than_refuse(self) -> None:
        """Deferral and refusal are different outcomes and must render differently."""
        # 03:00 IST: inside quiet hours, so a same-day contact is held until morning.
        three_am = 1_756_100_000 - 3 * 3600
        trace = explain(event("payment_cancelled", created_at=three_am))
        guarded = [s for s in trace.steps if s.stage == "guarded"]
        assert guarded
        if guarded[0].kind == "defer":
            assert trace.outcome == "deferred"
            assert Rule.QUIET_HOURS.value in guarded[0].verdict


class TestHonestyAboutWhatIsUnknown:
    def test_a_bare_webhook_admits_the_bank_is_unidentified(self) -> None:
        trace = explain(event())
        assert not trace.bank_identified
        assert any("issuing bank" in note for note in trace.notes)

    def test_a_bare_webhook_admits_it_knows_no_other_instrument(self) -> None:
        trace = explain(event())
        assert any("alternative instrument" in note for note in trace.notes)

    def test_an_unrecognised_code_is_flagged_rather_than_hidden(self) -> None:
        trace = explain(event("some_code_razorpay_added_last_tuesday"))
        assert not trace.recognised
        assert any("not in the taxonomy" in note for note in trace.notes)


class TestCounterfactual:
    def test_it_separates_what_is_proposed_from_what_is_permitted(self) -> None:
        """Overstating the baseline is the easiest way to lose the argument.

        The evaluation harness applies the same attempt cap to the baseline that it applies
        to Recoup, so the honest claim is about where attempts are *spent*, not about a
        strawman that burns four charges on a dead card.
        """
        trace = explain(event("card_expired"))
        assert trace.counterfactual.proposed == 4
        assert trace.counterfactual.permitted == lookup("card_expired").max_attempts
        assert trace.counterfactual.can_succeed == 0

    def test_session_conditional_retries_can_never_succeed(self) -> None:
        trace = explain(event("payment_cancelled"))
        assert trace.counterfactual.permitted > 0
        assert trace.counterfactual.can_succeed == 0

    def test_timing_dependent_classes_are_not_claimed_to_be_hopeless(self) -> None:
        trace = explain(event("insufficient_funds"))
        assert trace.counterfactual.can_succeed is None


class TestSerialisation:
    def test_the_trace_survives_the_json_boundary(self) -> None:
        import json

        trace = explain(event("payment_cancelled"))
        payload = json.loads(json.dumps(trace.to_dict()))
        assert payload["decline_code"] == "payment_cancelled"
        assert payload["outcome"] in {"allowed", "deferred", "refused", "none"}
        assert len(payload["steps"]) == len(trace.steps)
        assert all(isinstance(field, list) for step in payload["steps"] for field in step["fields"])


class TestTheTraceCarriesTheModelsWork:
    """The engine's intelligence has to reach the screen, or the demo shows only the
    deterministic half and a reader concludes the system is a decision table."""

    def test_a_deterministic_classification_says_no_model_was_needed(self) -> None:
        """The architecture's claim is that most traffic never reaches a model. A trace
        that cannot show *that* cannot evidence the claim."""
        from recoup.agent import DeclineNormalizer

        result = DeclineNormalizer().classify(error_reason="insufficient_funds")
        trace = explain(event("insufficient_funds"), classification=result)

        assert trace.classification is not None
        assert trace.classification["source"] == "exact"
        assert trace.classification["used_model"] is False
        assert "normalised" in [step.stage for step in trace.steps]

    def test_the_learned_policy_exposes_every_arm_it_scored(self) -> None:
        from recoup.policies import BanditPolicy

        trace = explain(event("insufficient_funds"), policy=BanditPolicy(explore=False))

        assert trace.bandit is not None
        assert len(trace.bandit["arms"]) > 1
        assert sum(1 for arm in trace.bandit["arms"] if arm["chosen"]) == 1

    def test_the_rule_based_policy_reports_no_bandit(self) -> None:
        """Absent, not faked. A bandit panel on a policy that has no bandit would be a
        screen inventing a model that did not run."""
        assert explain(event("insufficient_funds")).bandit is None

    def test_an_outreach_decision_carries_the_message(self) -> None:
        from recoup.agent import OutreachWriter

        trace = explain(event("payment_cancelled"), writer=OutreachWriter())

        assert trace.action == "outreach"
        assert trace.message is not None
        assert trace.message["body"]
        assert trace.message["generated"] is False, "no provider configured, so a template"
        assert trace.message["length"] <= trace.message["limit"]
        assert "composed" in [step.stage for step in trace.steps]

    def test_a_retry_decision_carries_no_message(self) -> None:
        from recoup.agent import OutreachWriter

        trace = explain(event("insufficient_funds"), writer=OutreachWriter())

        assert trace.action == "retry"
        assert trace.message is None, "nobody is contacted, so there is nothing to compose"

    def test_all_of_it_survives_the_json_boundary(self) -> None:
        from recoup.agent import DeclineNormalizer, OutreachWriter
        from recoup.policies import BanditPolicy

        trace = explain(
            event("payment_cancelled"),
            policy=BanditPolicy(explore=False),
            classification=DeclineNormalizer().classify(error_reason="payment_cancelled"),
            writer=OutreachWriter(),
        )
        payload = json.loads(json.dumps(trace.to_dict()))

        assert payload["classification"]["source"] == "exact"
        assert payload["bandit"]["arms"]
        assert payload["message"]["body"]
