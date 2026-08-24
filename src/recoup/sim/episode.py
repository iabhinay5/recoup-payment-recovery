"""Episode mechanics: one failed payment, played forward under a policy.

An episode starts the moment a payment fails and ends when it is recovered, abandoned, or
runs past the horizon. The policy sees an ``EpisodeState`` and returns an ``Action``; the
simulator resolves the consequences and advances time.

The policy interface is deliberately narrow. It receives only what a production system
could actually observe at decision time — elapsed time, the attempt and contact history,
the current decline reason, and a rail-health query. It does not receive the outage
schedule, the customer's true balance, or the outcome model. A policy that could see those
would post results it could never reproduce against real traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from recoup.guardrails import Guardrails, idempotency_key
from recoup.sim.entities import (
    HOURS_PER_DAY,
    Attempt,
    Contact,
    ContactChannel,
    Customer,
    FailedPayment,
    Instrument,
)
from recoup.sim.outcomes import OutcomeModel
from recoup.sim.params import SimParams
from recoup.sim.rails import RailHealth
from recoup.taxonomy import lookup

__all__ = [
    "Action",
    "ActionKind",
    "EpisodeResult",
    "EpisodeState",
    "Policy",
    "run_episode",
]

MAX_ACTIONS_PER_EPISODE = 64
"""Safety bound on policy decisions per episode.

A policy that repeatedly requests a zero-delay action it is not permitted to take would
otherwise spin forever. Hitting this bound indicates a policy bug, so it is surfaced on
the result rather than silently swallowed.
"""

MAX_CONSECUTIVE_REFUSALS = 3
"""How many refusals in a row end the episode.

A refusal does not change the state that caused it — an exhausted attempt cap stays
exhausted until an attempt runs, and an attempt cannot run while the cap is exhausted. So
a policy that keeps asking gets the same answer forever.

The bound is not 1, because a policy is allowed to pivot: having a retry refused and then
trying outreach instead is reasonable behaviour, not a bug. Repeating the *same* refused
request is what this stops.
"""


class ActionKind(Enum):
    RETRY = "retry"
    OUTREACH = "outreach"
    GIVE_UP = "give_up"


@dataclass(frozen=True, slots=True)
class Action:
    """What the policy wants to do next.

    ``delay_hours`` is measured from *now*, not from the original failure. Policies think
    in terms of "wait six hours, then retry", and making that relative removes a whole
    class of off-by-one errors around what "now" means.
    """

    kind: ActionKind
    delay_hours: float = 0.0
    instrument_id: str | None = None
    channel: ContactChannel | None = None

    def __post_init__(self) -> None:
        if self.delay_hours < 0:
            raise ValueError("delay_hours cannot be negative; time does not run backwards")
        if self.kind is ActionKind.OUTREACH and self.channel is None:
            raise ValueError("an outreach action must name a channel")

    @classmethod
    def retry(cls, delay_hours: float, instrument_id: str | None = None) -> Action:
        return cls(ActionKind.RETRY, delay_hours, instrument_id=instrument_id)

    @classmethod
    def outreach(cls, delay_hours: float, channel: ContactChannel) -> Action:
        return cls(ActionKind.OUTREACH, delay_hours, channel=channel)

    @classmethod
    def give_up(cls) -> Action:
        return cls(ActionKind.GIVE_UP)


@dataclass(frozen=True, slots=True)
class EpisodeState:
    """Everything a policy is allowed to see at decision time."""

    payment: FailedPayment
    customer: Customer
    elapsed_hours: float
    attempts: tuple[Attempt, ...]
    contacts: tuple[Contact, ...]
    current_decline_code: str
    current_instrument_id: str
    in_session: bool
    opted_out: bool
    rails: RailHealth
    params: SimParams

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def contact_count(self) -> int:
        return len(self.contacts)

    @property
    def remaining_attempts(self) -> int:
        """Attempts still permitted for the current decline reason.

        Zero means no retry can be scheduled — the cap comes from the taxonomy and the
        runner enforces it regardless of what the policy asks for.
        """
        return max(0, lookup(self.current_decline_code).max_attempts - self.attempt_count)

    @property
    def hour_of_day(self) -> float:
        return self.payment.hour_of_day_at(self.elapsed_hours)

    @property
    def day_of_month(self) -> int:
        return self.payment.day_of_month_at(self.elapsed_hours)

    @property
    def current_instrument(self) -> Instrument:
        for instrument in self.customer.instruments:
            if instrument.id == self.current_instrument_id:
                return instrument
        return self.customer.primary_instrument

    def rail_is_degraded(self, instrument: Instrument | None = None) -> bool:
        """Whether the rail behind an instrument is currently degraded.

        This is the policy-visible form of ADR-006 and mirrors what a production Downtime
        API exposes: current state only.
        """
        target = instrument or self.current_instrument
        return self.rails.is_degraded(target.bank_id, self.elapsed_hours)

    def contacts_within(self, window_hours: float) -> int:
        """Contacts made in the preceding window, for fatigue-aware policies."""
        lower = self.elapsed_hours - window_hours
        return sum(1 for c in self.contacts if lower <= c.at_hours <= self.elapsed_hours)


class Policy(Protocol):
    """A recovery strategy.

    Implementations must be deterministic given the state, or must carry their own
    generator. The episode runner does not reseed between calls.

    A policy may optionally define ``observe(succeeded, opted_out)``. When present, the
    runner calls it after each action resolves, which is how a learning policy is credited
    for what it chose. This mirrors production: a real system learns from the webhook that
    reports whether a retry cleared, not from anything it knew at decision time.
    """

    @property
    def name(self) -> str: ...

    def decide(self, state: EpisodeState) -> Action: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """What happened over one recovery episode."""

    payment_id: str
    amount_paise: int
    recovered: bool
    recovered_at_hours: float | None
    attempts: tuple[Attempt, ...]
    contacts: tuple[Contact, ...]
    opted_out: bool
    refused_actions: int = 0
    hit_action_limit: bool = False

    @property
    def revenue_recovered_paise(self) -> int:
        return self.amount_paise if self.recovered else 0

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def contact_count(self) -> int:
        return len(self.contacts)

    @property
    def wasted_attempts(self) -> int:
        """Attempts that did not recover the payment.

        The metric that separates a good policy from a merely aggressive one. Two policies
        can recover the same revenue while one spends four times the attempts doing it,
        and attempts are not free — they consume issuer goodwill and rail capacity.
        """
        return sum(1 for a in self.attempts if not a.succeeded)


def _notify(policy: Policy, succeeded: bool, opted_out: bool) -> None:
    """Report an action's outcome back to a learning policy, if it wants one."""
    observe = getattr(policy, "observe", None)
    if observe is not None:
        observe(succeeded, opted_out)


def _resolve_instrument(customer: Customer, instrument_id: str | None) -> Instrument:
    if instrument_id is not None:
        for instrument in customer.instruments:
            if instrument.id == instrument_id:
                return instrument
    return customer.primary_instrument


def run_episode(
    payment: FailedPayment,
    customer: Customer,
    policy: Policy,
    outcomes: OutcomeModel,
    rails: RailHealth,
    params: SimParams,
    rng: np.random.Generator,
    guardrails: Guardrails | None = None,
) -> EpisodeResult:
    """Play one failed payment forward under ``policy``.

    Every action the policy proposes is checked by ``guardrails`` before it executes. A
    fresh set is created per episode when none is supplied; passing one in lets a caller
    inspect exactly which rules fired, which is what the audit trail is built from.
    """
    horizon_hours = params.horizon_days * HOURS_PER_DAY
    guards = guardrails if guardrails is not None else Guardrails()

    elapsed = 0.0
    attempts: list[Attempt] = []
    contacts: list[Contact] = []
    current_code = payment.initial_decline_code
    current_instrument_id = payment.instrument_id
    in_session = False
    opted_out = customer.opted_out
    refused = 0
    consecutive_refusals = 0

    for _ in range(MAX_ACTIONS_PER_EPISODE):
        state = EpisodeState(
            payment=payment,
            customer=customer,
            elapsed_hours=elapsed,
            attempts=tuple(attempts),
            contacts=tuple(contacts),
            current_decline_code=current_code,
            current_instrument_id=current_instrument_id,
            in_session=in_session,
            opted_out=opted_out,
            rails=rails,
            params=params,
        )

        action = policy.decide(state)

        if action.kind is ActionKind.GIVE_UP:
            break

        target = elapsed + action.delay_hours
        if target > horizon_hours:
            # The action would land past the point at which the payment is written off.
            break
        elapsed = target

        if action.kind is ActionKind.RETRY:
            instrument = _resolve_instrument(
                customer, action.instrument_id or current_instrument_id
            )
            key = idempotency_key(payment.id, len(attempts), instrument.id)

            verdict = guards.check_retry(
                key=key,
                attempts_made=len(attempts),
                decline_code=current_code,
                instrument_expired=instrument.is_expired_at(elapsed),
            )
            if not verdict.allowed:
                # The guardrail layer decides, not the policy. Asking again cannot change
                # the answer. See docs/DECISIONS.md ADR-007.
                refused += 1
                consecutive_refusals += 1
                if consecutive_refusals >= MAX_CONSECUTIVE_REFUSALS:
                    break
                continue

            current_instrument_id = instrument.id
            guards.record_charge(key)

            attempt = outcomes.resolve(
                payment=payment,
                customer=customer,
                instrument=instrument,
                current_decline_code=current_code,
                elapsed_hours=elapsed,
                in_session=in_session,
                rng=rng,
            )
            attempts.append(attempt)
            consecutive_refusals = 0
            _notify(policy, attempt.succeeded, opted_out)

            if attempt.succeeded:
                return EpisodeResult(
                    payment_id=payment.id,
                    amount_paise=payment.amount_paise,
                    recovered=True,
                    recovered_at_hours=elapsed,
                    attempts=tuple(attempts),
                    contacts=tuple(contacts),
                    opted_out=opted_out,
                    refused_actions=refused,
                )

            assert attempt.decline_code is not None
            current_code = attempt.decline_code
            # A session is consumed by the attempt it enabled, whether or not it worked.
            in_session = False

        elif action.kind is ActionKind.OUTREACH:
            window = params.horizon_days * HOURS_PER_DAY
            verdict = guards.check_outreach(
                hour_of_day=payment.hour_of_day_at(elapsed),
                opted_out=opted_out,
                contacts_in_window=sum(
                    1 for c in contacts if elapsed - guards.contact_window_hours <= c.at_hours
                ),
            )

            if verdict.deferred and verdict.defer_hours is not None:
                # Badly timed, not forbidden. Waiting until morning is the right answer;
                # dropping the message would lose a recovery to a clock.
                deferred_to = elapsed + verdict.defer_hours
                if deferred_to > window:
                    break
                elapsed = deferred_to
            elif not verdict.allowed:
                refused += 1
                consecutive_refusals += 1
                if consecutive_refusals >= MAX_CONSECUTIVE_REFUSALS:
                    break
                continue

            contacts.append(Contact(elapsed, action.channel or ContactChannel.EMAIL))
            consecutive_refusals = 0

            channel = action.channel or ContactChannel.EMAIL
            hazard = params.opt_out_hazard_per_contact * channel.intrusiveness
            if rng.random() < hazard:
                opted_out = True
                _notify(policy, False, True)
                continue

            responded = rng.random() < params.outreach_response_rate
            if responded:
                in_session = True
            _notify(policy, responded, opted_out)

    else:
        return EpisodeResult(
            payment_id=payment.id,
            amount_paise=payment.amount_paise,
            recovered=False,
            recovered_at_hours=None,
            attempts=tuple(attempts),
            contacts=tuple(contacts),
            opted_out=opted_out,
            refused_actions=refused,
            hit_action_limit=True,
        )

    return EpisodeResult(
        payment_id=payment.id,
        amount_paise=payment.amount_paise,
        recovered=False,
        recovered_at_hours=None,
        attempts=tuple(attempts),
        contacts=tuple(contacts),
        opted_out=opted_out,
        refused_actions=refused,
    )
