"""Baseline recovery policies.

These are what Recoup has to beat. They are implemented faithfully rather than as
strawmen — a comparison against a deliberately weakened baseline proves nothing, and the
first thing a payments engineer will check is whether the baseline is the one they
actually run.

``FixedSchedule`` with the Day-1/3/5/7 timing is the important one: Recurly published a
58% recovery rate for exactly that schedule across roughly 40 million subscription
transactions, which makes it both the baseline to beat and the calibration target for the
simulator (docs/DECISIONS.md ADR-008).
"""

from __future__ import annotations

from recoup.sim.entities import ContactChannel
from recoup.sim.episode import Action, EpisodeState

__all__ = [
    "AggressiveRetry",
    "ExponentialBackoff",
    "FixedSchedule",
    "NoRetry",
    "OutreachOnly",
]

HOURS_PER_DAY = 24.0

RECURLY_SCHEDULE_DAYS: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0)
"""Recurly's published retry schedule, reported to recover ~58% with no customer contact.

See docs/CALIBRATION.md section 3.
"""


class NoRetry:
    """Do nothing. The floor.

    Worth measuring rather than assuming zero: some payments in the simulator are already
    recoverable through a customer-initiated retry that the merchant never sees, and the
    honest uplift of any policy is measured against this, not against nothing.
    """

    @property
    def name(self) -> str:
        return "no_retry"

    def decide(self, state: EpisodeState) -> Action:
        return Action.give_up()


class FixedSchedule:
    """Retry at fixed offsets from the original failure, regardless of decline reason.

    This is the industry default and the direct target of the project's thesis. It treats
    an expired card, a bank outage and an insufficient balance as the same event, which
    means most of its attempts are spent where they cannot possibly work.
    """

    def __init__(self, days: tuple[float, ...] = RECURLY_SCHEDULE_DAYS, label: str = "") -> None:
        self._offsets = tuple(d * HOURS_PER_DAY for d in days)
        self._label = label or "_".join(str(int(d)) for d in days)

    @property
    def name(self) -> str:
        return f"fixed_{self._label}"

    def decide(self, state: EpisodeState) -> Action:
        n = state.attempt_count
        if n >= len(self._offsets):
            return Action.give_up()
        delay = self._offsets[n] - state.elapsed_hours
        return Action.retry(max(0.0, delay))


class ExponentialBackoff:
    """Retry with geometrically increasing delays.

    The reflex a backend engineer reaches for. It is a good pattern for transient
    infrastructure faults and a poor one for payments, because most payment failures are
    not transient — backing off politely from an expired card still never works.
    """

    def __init__(self, initial_hours: float = 2.0, factor: float = 3.0, max_attempts: int = 4):
        self._initial = initial_hours
        self._factor = factor
        self._max = max_attempts

    @property
    def name(self) -> str:
        return "exponential_backoff"

    def decide(self, state: EpisodeState) -> Action:
        n = state.attempt_count
        if n >= self._max:
            return Action.give_up()
        return Action.retry(self._initial * (self._factor**n))


class AggressiveRetry:
    """Retry as fast and as often as permitted.

    Included because it is what an unconstrained optimiser converges to when the reward
    function forgets that attempts have costs. Its results are the argument for why the
    reward includes a penalty at all — it should recover slightly more revenue than
    reasonable policies while burning far more attempts to do it.
    """

    def __init__(self, interval_hours: float = 4.0) -> None:
        self._interval = interval_hours

    @property
    def name(self) -> str:
        return "aggressive"

    def decide(self, state: EpisodeState) -> Action:
        if state.remaining_attempts <= 0:
            return Action.give_up()
        return Action.retry(self._interval)


class OutreachOnly:
    """Contact the customer, never retry silently.

    The dunning-email approach. Churnkey reports email and SMS alone recovering around
    42%, so this is the other published reference point — and it is the only baseline
    that can resolve session-conditional declines, which is where the fixed schedule
    wastes most of its attempts.
    """

    def __init__(
        self,
        offsets_days: tuple[float, ...] = (1.0, 3.0, 6.0),
        channel: ContactChannel = ContactChannel.EMAIL,
    ) -> None:
        self._offsets = tuple(d * HOURS_PER_DAY for d in offsets_days)
        self._channel = channel

    @property
    def name(self) -> str:
        return "outreach_only"

    def decide(self, state: EpisodeState) -> Action:
        # One retry immediately after the customer re-engages: the session is what makes
        # the retry viable, and it does not stay open.
        if state.in_session and state.remaining_attempts > 0:
            return Action.retry(0.0)

        n = state.contact_count
        if n >= len(self._offsets) or state.opted_out:
            return Action.give_up()

        delay = self._offsets[n] - state.elapsed_hours
        return Action.outreach(max(0.0, delay), self._channel)
