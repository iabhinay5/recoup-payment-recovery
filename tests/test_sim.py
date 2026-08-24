"""Tests for the simulator.

Two kinds of test here. The mechanism tests check that balance, rail health and session
state behave the way the design says they do — these are what make the simulator's
behaviour explainable rather than emergent. The refusal tests check that the episode
runner enforces caps regardless of what a policy asks for, which is the runtime half of
the guarantee the taxonomy makes structurally.
"""

from __future__ import annotations

import numpy as np
import pytest

from recoup.sim import (
    Action,
    ContactChannel,
    Customer,
    FailedPayment,
    Instrument,
    Outage,
    OutcomeModel,
    RailHealth,
    SimParams,
    balance_fraction,
    generate_outages,
    generate_population,
    run_episode,
)
from recoup.sim.episode import MAX_CONSECUTIVE_REFUSALS, EpisodeState
from recoup.sim.outcomes import affordability_success_probability
from recoup.taxonomy import PaymentMethod


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- fixtures ------------------------------------------------------------------------


@pytest.fixture
def params() -> SimParams:
    return SimParams(n_customers=200, seed=7)


@pytest.fixture
def customer() -> Customer:
    return Customer(
        id="c1",
        instruments=(
            Instrument("i1", PaymentMethod.CARD, "hdfc"),
            Instrument("i2", PaymentMethod.UPI, "icici"),
        ),
        monthly_income_paise=5_000_000,
        salary_day_of_month=1,
    )


@pytest.fixture
def payment() -> FailedPayment:
    return FailedPayment(
        id="p1",
        customer_id="c1",
        instrument_id="i1",
        amount_paise=100_000,
        initial_decline_code="insufficient_funds",
        reference_day_of_month=15,
        reference_hour_of_day=10.0,
    )


# --- mechanisms ----------------------------------------------------------------------


class TestBalanceMechanism:
    """insufficient_funds recovers because balances recover, not because time passes."""

    def test_balance_is_full_on_salary_day(self) -> None:
        assert balance_fraction(1, salary_day=1, depletion_rate=0.9, floor=0.05) == 1.0

    def test_balance_depletes_through_the_cycle(self) -> None:
        early = balance_fraction(3, 1, 0.9, 0.05)
        late = balance_fraction(25, 1, 0.9, 0.05)
        assert early > late

    def test_balance_never_falls_below_the_floor(self) -> None:
        for day in range(1, 29):
            assert balance_fraction(day, 1, 0.9, 0.05) >= 0.05

    def test_cycle_wraps(self) -> None:
        """Day 1 with a salary on day 28 is one day after payday, not 27 days before."""
        just_after = balance_fraction(1, salary_day=28, depletion_rate=0.9, floor=0.05)
        long_after = balance_fraction(20, salary_day=28, depletion_rate=0.9, floor=0.05)
        assert just_after > long_after

    def test_affordability_saturates(self) -> None:
        assert affordability_success_probability(0, 1000) == 0.0
        assert affordability_success_probability(1000, 1000) == pytest.approx(0.5)
        assert affordability_success_probability(10_000_000, 1000) > 0.99

    def test_zero_amount_is_not_a_division_error(self) -> None:
        assert affordability_success_probability(500, 0) == 0.0


class TestRailHealth:
    def test_healthy_bank_reflects_its_decline_rate(self) -> None:
        rails = RailHealth({"hdfc": 0.004})
        assert rails.health_at("hdfc", 0.0) == pytest.approx(0.996)

    def test_outage_drives_health_down(self) -> None:
        rails = RailHealth({"hdfc": 0.004}, (Outage("hdfc", 10.0, 5.0),))
        assert rails.health_at("hdfc", 12.0) == 0.05
        assert rails.is_degraded("hdfc", 12.0)

    def test_health_recovers_after_the_outage(self) -> None:
        rails = RailHealth({"hdfc": 0.004}, (Outage("hdfc", 10.0, 5.0),))
        assert not rails.is_degraded("hdfc", 15.1)

    def test_unknown_bank_degrades_to_healthy_not_to_a_crash(self) -> None:
        """A missing health signal must not break the retry path."""
        assert RailHealth({}).health_at("who?", 0.0) == 1.0

    def test_outages_are_bank_specific(self) -> None:
        rails = RailHealth({"a": 0.0, "b": 0.0}, (Outage("a", 0.0, 10.0),))
        assert rails.is_degraded("a", 5.0)
        assert not rails.is_degraded("b", 5.0)

    def test_hours_until_healthy_is_none_when_healthy(self) -> None:
        assert RailHealth({"a": 0.0}).hours_until_healthy("a", 0.0) is None

    def test_generated_outages_stay_within_the_horizon(self) -> None:
        outages = generate_outages({"a": (0.5, 3.0), "b": (0.2, 5.0)}, 336.0, _rng())
        assert all(0.0 <= o.start_hours <= 336.0 for o in outages)
        assert all(o.duration_hours > 0 for o in outages)


class TestOutcomeMechanisms:
    def test_expired_card_never_succeeds(self, params: SimParams, payment: FailedPayment) -> None:
        expired = Instrument("i1", PaymentMethod.CARD, "hdfc", expires_at_hours=0.0)
        customer = Customer("c1", (expired,), 5_000_000, 1)
        model = OutcomeModel(params, RailHealth({"hdfc": 0.0}))

        for _ in range(50):
            attempt = model.resolve(
                payment, customer, expired, "insufficient_funds", 24.0, False, _rng()
            )
            assert not attempt.succeeded
            assert attempt.decline_code == "card_expired"

    def test_degraded_rail_fails_a_good_card(
        self, params: SimParams, payment: FailedPayment, customer: Customer
    ) -> None:
        rails = RailHealth({"hdfc": 0.0}, (Outage("hdfc", 0.0, 100.0),))
        model = OutcomeModel(params, rails)

        outcomes = [
            model.resolve(
                payment, customer, customer.instruments[0], "insufficient_funds", 10.0, False, rng
            )
            for rng in (_rng(s) for s in range(40))
        ]
        failures = [o for o in outcomes if not o.succeeded]
        assert len(failures) > 30, "a rail outage should dominate the outcome"
        assert any(o.decline_code == "bank_technical_error" for o in failures)

    def test_session_conditional_needs_a_session(
        self, params: SimParams, customer: Customer
    ) -> None:
        payment = FailedPayment("p", "c1", "i1", 1000, "payment_cancelled", 1)
        model = OutcomeModel(params, RailHealth({"hdfc": 0.0}))

        silent = [
            model.resolve(
                payment, customer, customer.instruments[0], "payment_cancelled", 5.0, False, _rng(s)
            )
            for s in range(30)
        ]
        assert not any(a.succeeded for a in silent), "a silent retry cannot resolve these"

        in_session = [
            model.resolve(
                payment, customer, customer.instruments[0], "payment_cancelled", 5.0, True, _rng(s)
            )
            for s in range(30)
        ]
        assert any(a.succeeded for a in in_session)

    def test_timing_matters_for_insufficient_funds(
        self, params: SimParams, customer: Customer
    ) -> None:
        """The whole premise of the project: when you retry changes the outcome."""
        rails = RailHealth({"hdfc": 0.0})
        model = OutcomeModel(params, rails)
        # Amount close to a full month of income, so balance position dominates.
        payment = FailedPayment("p", "c1", "i1", 4_000_000, "insufficient_funds", 20)

        def success_rate(elapsed: float) -> float:
            hits = sum(
                model.resolve(
                    payment,
                    customer,
                    customer.instruments[0],
                    "insufficient_funds",
                    elapsed,
                    False,
                    _rng(s),
                ).succeeded
                for s in range(200)
            )
            return hits / 200

        # Day 20 of the cycle, balance depleted, vs ~day 1 after a salary credit.
        depleted = success_rate(0.0)
        after_payday = success_rate(9 * 24.0)
        assert after_payday > depleted


# --- episode mechanics ---------------------------------------------------------------


class _ScriptedPolicy:
    """Replays a fixed list of actions, then gives up."""

    def __init__(self, actions: list[Action]) -> None:
        self._actions = list(actions)
        self.seen: list[EpisodeState] = []

    @property
    def name(self) -> str:
        return "scripted"

    def decide(self, state: EpisodeState) -> Action:
        self.seen.append(state)
        if self._actions:
            return self._actions.pop(0)
        return Action.give_up()


class TestActionValidation:
    def test_negative_delay_rejected(self) -> None:
        with pytest.raises(ValueError, match="time does not run backwards"):
            Action.retry(-1.0)

    def test_outreach_requires_a_channel(self) -> None:
        from recoup.sim.episode import ActionKind

        with pytest.raises(ValueError, match="must name a channel"):
            Action(ActionKind.OUTREACH, 1.0)


class TestEpisodeRunner:
    def _model(self, params: SimParams) -> tuple[OutcomeModel, RailHealth]:
        rails = RailHealth({"hdfc": 0.0, "icici": 0.0})
        return OutcomeModel(params, rails), rails

    def test_give_up_ends_the_episode(
        self, params: SimParams, payment: FailedPayment, customer: Customer
    ) -> None:
        model, rails = self._model(params)
        result = run_episode(payment, customer, _ScriptedPolicy([]), model, rails, params, _rng())
        assert not result.recovered
        assert result.attempt_count == 0

    def test_cap_is_enforced_against_a_greedy_policy(
        self, params: SimParams, customer: Customer
    ) -> None:
        """A policy that asks for more retries than the taxonomy permits is refused.

        card_expired permits zero attempts. The policy here asks for ten; none run.
        """
        payment = FailedPayment("p", "c1", "i1", 1000, "card_expired", 1)
        model, rails = self._model(params)
        policy = _ScriptedPolicy([Action.retry(1.0) for _ in range(10)])

        result = run_episode(payment, customer, policy, model, rails, params, _rng())

        assert result.attempt_count == 0, "no attempt may run against a zero-cap reason"
        assert not result.recovered
        # The episode ends once the same refusal repeats: an exhausted cap cannot be
        # un-exhausted without an attempt, and an attempt cannot run while it is exhausted.
        assert result.refused_actions == MAX_CONSECUTIVE_REFUSALS

    def test_horizon_stops_the_episode(
        self, params: SimParams, payment: FailedPayment, customer: Customer
    ) -> None:
        model, rails = self._model(params)
        beyond = params.horizon_days * 24.0 + 1.0
        policy = _ScriptedPolicy([Action.retry(beyond)])

        result = run_episode(payment, customer, policy, model, rails, params, _rng())
        assert result.attempt_count == 0

    def test_opted_out_customer_is_not_contacted(
        self, params: SimParams, payment: FailedPayment
    ) -> None:
        opted_out = Customer(
            "c1",
            (Instrument("i1", PaymentMethod.CARD, "hdfc"),),
            5_000_000,
            1,
            opted_out=True,
        )
        model, rails = self._model(params)
        policy = _ScriptedPolicy([Action.outreach(1.0, ContactChannel.SMS) for _ in range(5)])

        result = run_episode(payment, opted_out, policy, model, rails, params, _rng())

        assert result.contact_count == 0
        assert result.refused_actions == MAX_CONSECUTIVE_REFUSALS

    def test_successful_attempt_records_revenue_and_time(
        self, params: SimParams, customer: Customer
    ) -> None:
        # Tiny amount against a large income: affordability is overwhelming.
        payment = FailedPayment("p", "c1", "i1", 100, "insufficient_funds", 1)
        model, rails = self._model(params)
        policy = _ScriptedPolicy([Action.retry(6.0)])

        result = run_episode(payment, customer, policy, model, rails, params, _rng())

        assert result.recovered
        assert result.recovered_at_hours == pytest.approx(6.0)
        assert result.revenue_recovered_paise == 100
        assert result.wasted_attempts == 0

    def test_policy_only_sees_permitted_state(
        self, params: SimParams, payment: FailedPayment, customer: Customer
    ) -> None:
        """The policy must not be handed the outcome model or the outage schedule."""
        model, rails = self._model(params)
        policy = _ScriptedPolicy([Action.retry(1.0)])
        run_episode(payment, customer, policy, model, rails, params, _rng())

        state = policy.seen[0]
        assert not hasattr(state, "outcomes")
        assert hasattr(state, "rails"), "rail health is queryable, as a Downtime API would be"

    def test_repeated_refusal_ends_the_episode(
        self, params: SimParams, payment: FailedPayment, customer: Customer
    ) -> None:
        """A policy that keeps asking for a refused action does not spin.

        Before this bound existed, 35% of episodes burned the full action budget on
        refusals, which both wasted time and masked what the policy was really doing.
        """
        model, rails = self._model(params)
        opted_out = Customer(
            "c1", (Instrument("i1", PaymentMethod.CARD, "hdfc"),), 5_000_000, 1, opted_out=True
        )
        policy = _ScriptedPolicy([Action.outreach(0.0, ContactChannel.EMAIL)] * 50)

        result = run_episode(payment, opted_out, policy, model, rails, params, _rng())

        assert result.refused_actions == MAX_CONSECUTIVE_REFUSALS
        assert not result.hit_action_limit

    def test_a_policy_may_pivot_after_one_refusal(
        self, params: SimParams, customer: Customer
    ) -> None:
        """One refusal must not end the episode — switching tactics is legitimate."""
        payment = FailedPayment("p", "c1", "i1", 100, "card_expired", 1)
        model, rails = self._model(params)
        policy = _ScriptedPolicy([Action.retry(1.0), Action.outreach(1.0, ContactChannel.EMAIL)])

        result = run_episode(payment, customer, policy, model, rails, params, _rng())

        assert result.refused_actions == 1
        assert result.contact_count == 1, "the pivot to outreach must still be allowed"

    def test_action_limit_is_surfaced_not_swallowed(
        self, params: SimParams, payment: FailedPayment, customer: Customer
    ) -> None:
        """A policy that loops on *executing* actions still hits the safety bound."""
        from recoup.guardrails import Guardrails
        from recoup.sim.episode import MAX_ACTIONS_PER_EPISODE

        # No opt-out hazard and no contact rate limit, so zero-delay outreach is allowed
        # indefinitely and only the hard action bound can stop it.
        never_tires = params.with_overrides(opt_out_hazard_per_contact=0.0)
        model, rails = self._model(never_tires)
        policy = _ScriptedPolicy(
            [Action.outreach(0.0, ContactChannel.EMAIL)] * (MAX_ACTIONS_PER_EPISODE + 10)
        )

        result = run_episode(
            payment,
            customer,
            policy,
            model,
            rails,
            never_tires,
            _rng(),
            guardrails=Guardrails(max_contacts_per_window=10**6),
        )
        assert result.hit_action_limit


# --- population ----------------------------------------------------------------------


class TestPopulation:
    def test_one_payment_per_customer(self, params: SimParams) -> None:
        pop = generate_population(params)
        assert len(pop.payments) == len(pop.customers) == params.n_customers

    def test_reproducible_from_seed(self, params: SimParams) -> None:
        a = generate_population(params)
        b = generate_population(params)
        assert [p.amount_paise for p in a.payments] == [p.amount_paise for p in b.payments]
        assert [p.initial_decline_code for p in a.payments] == [
            p.initial_decline_code for p in b.payments
        ]

    def test_different_seeds_differ(self, params: SimParams) -> None:
        a = generate_population(params)
        b = generate_population(params.with_overrides(seed=params.seed + 1))
        assert [p.amount_paise for p in a.payments] != [p.amount_paise for p in b.payments]

    def test_expired_card_payments_hold_an_expired_card(self, params: SimParams) -> None:
        """State and decline reason must agree, or the episode is unrecoverable fiction."""
        pop = generate_population(params.with_overrides(n_customers=800))
        by_id = pop.customers_by_id

        checked = 0
        for pay in pop.payments:
            if pay.initial_decline_code != "card_expired":
                continue
            checked += 1
            instrument = next(
                i for i in by_id[pay.customer_id].instruments if i.id == pay.instrument_id
            )
            assert instrument.method is PaymentMethod.CARD
            assert instrument.is_expired_at(0.0)

        assert checked > 0, "sample should contain some expired-card failures"

    def test_amounts_are_positive(self, params: SimParams) -> None:
        assert all(p.amount_paise > 0 for p in generate_population(params).payments)

    def test_decline_mix_must_normalise(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SimParams(decline_mix={"insufficient_funds": 0.5, "card_expired": 0.2})
