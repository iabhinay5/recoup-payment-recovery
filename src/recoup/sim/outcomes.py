"""Whether a retry succeeds — modelled by mechanism rather than by fitted curve.

This is the most consequential design decision in the simulator, so it is worth stating
plainly.

The easy way to build this would be a curve: probability of recovery as a function of
hours since failure, fitted so the published benchmarks come out right. That approach has
a fatal flaw for our purposes. The thing we are trying to evaluate is *a policy that
chooses when to retry*, and a curve hands the answer to the policy directly — the optimum
is wherever the curve peaks, the policy finds it, and the result measures nothing except
our own curve-fitting.

So recovery is modelled through the mechanisms that actually drive it:

- ``insufficient_funds`` recovers because the customer's **balance** recovers, which is
  driven by a salary cycle. The policy has to infer that waiting for a payday helps.
- ``bank_technical_error`` recovers because the **rail** recovers, on an outage process
  the policy cannot see the future of.
- session-conditional declines recover only when the customer is **back in a session**,
  which only happens after successful outreach.
- never-retryable declines do not recover at all, whatever the timing.

The published Recurly benchmark then becomes a *check* rather than an input: if the
Day-1/3/5/7 schedule reproduces near 58% in this simulator, the mechanisms are calibrated
plausibly. If it does not, the calibration is wrong and gets fixed before any policy is
built on top. That gate is day 4 in docs/PLAN.md.
"""

from __future__ import annotations

import numpy as np

from recoup.sim.entities import Attempt, Customer, FailedPayment, Instrument
from recoup.sim.params import SimParams
from recoup.sim.rails import RailHealth
from recoup.taxonomy import DeclineClass, lookup

__all__ = ["OutcomeModel", "affordability_success_probability", "balance_fraction"]

# Probability that a hard decline resolves on its own. Low and constant: by definition
# these are refusals whose cause we cannot observe, so there is no mechanism to model.
HARD_DECLINE_SUCCESS_RATE = 0.06

# Probability that a customer in an active session completes payment, given the
# underlying instrument and rail are fine.
SESSION_COMPLETION_RATE = 0.82


def balance_fraction(
    day_of_month: int,
    salary_day: int,
    depletion_rate: float,
    floor: float,
) -> float:
    """Fraction of monthly income still available on a given day of the cycle.

    Income arrives on the salary day and is drawn down geometrically. This is what makes
    ``insufficient_funds`` a *timing* problem: the same customer with the same card has a
    very different chance of clearing a charge on day 2 of their cycle than on day 25.

    Returns a value in ``[floor, 1.0]``.
    """
    days_since_salary = (day_of_month - salary_day) % 28
    return floor + (1.0 - floor) * (depletion_rate**days_since_salary)


def affordability_success_probability(available_paise: float, amount_paise: int) -> float:
    """Probability a charge clears, given available balance and amount.

    A saturating function of the ratio: comfortably affordable charges nearly always
    clear, charges near the available balance are a coin flip, and charges well above it
    almost never clear. Smooth rather than a hard threshold, because a customer's true
    available balance is not observable and other debits land unpredictably.
    """
    if amount_paise <= 0:
        return 0.0
    ratio = max(0.0, available_paise / amount_paise)
    return ratio / (ratio + 1.0)


class OutcomeModel:
    """Resolves a retry attempt into a success or a specific decline reason."""

    def __init__(self, params: SimParams, rail_health: RailHealth) -> None:
        self._p = params
        self._rails = rail_health

    def resolve(
        self,
        payment: FailedPayment,
        customer: Customer,
        instrument: Instrument,
        current_decline_code: str,
        elapsed_hours: float,
        in_session: bool,
        rng: np.random.Generator,
    ) -> Attempt:
        """Resolve one charge attempt.

        ``current_decline_code`` is the reason the *most recent* attempt failed, which is
        not necessarily the reason the original payment failed. A card that was merely
        short of funds in week one can be expired by week two, and the model has to be
        able to produce that transition.
        """
        # 1. Instrument validity is checked first and independently of everything else.
        #    An expired card fails regardless of balance, rail health, or how well timed
        #    the retry was.
        if instrument.is_expired_at(elapsed_hours):
            return Attempt(elapsed_hours, instrument.id, False, "card_expired")

        # 2. Rail health is checked next, and it overrides the original decline reason.
        #    A perfectly good card fails when the issuing bank is down, which is exactly
        #    why retrying into a degraded rail wastes the attempt.
        health = self._rails.health_at(instrument.bank_id, elapsed_hours)
        if rng.random() > health:
            return Attempt(elapsed_hours, instrument.id, False, "bank_technical_error")

        # 3. Otherwise the original reason's own mechanism decides.
        reason = lookup(current_decline_code)
        probability = self._success_probability(
            reason.decline_class, payment, customer, elapsed_hours, in_session
        )

        if rng.random() < probability:
            return Attempt(elapsed_hours, instrument.id, True)
        return Attempt(elapsed_hours, instrument.id, False, current_decline_code)

    def _success_probability(
        self,
        decline_class: DeclineClass,
        payment: FailedPayment,
        customer: Customer,
        elapsed_hours: float,
        in_session: bool,
    ) -> float:
        if decline_class is DeclineClass.NEVER_RETRYABLE:
            # No mechanism exists. This is not a small probability, it is zero, and the
            # policy is rewarded for recognising that rather than for timing it well.
            return 0.0

        if decline_class is DeclineClass.RAIL_CONDITIONAL:
            # The rail check in resolve() has already passed, so the transient condition
            # that caused the original failure has cleared.
            return 0.95

        if decline_class is DeclineClass.TIME_CONDITIONAL:
            day = payment.day_of_month_at(elapsed_hours)
            fraction = balance_fraction(
                day,
                customer.salary_day_of_month,
                self._p.balance_depletion_rate,
                self._p.balance_floor_fraction,
            )
            available = fraction * customer.monthly_income_paise
            return affordability_success_probability(available, payment.amount_paise)

        if decline_class is DeclineClass.SESSION_CONDITIONAL:
            # A silent background retry cannot resolve these at all. Only a customer who
            # has come back through outreach can. Policies that treat these as ordinary
            # retryable failures burn their whole attempt budget for nothing.
            return SESSION_COMPLETION_RATE if in_session else 0.0

        return HARD_DECLINE_SUCCESS_RATE
