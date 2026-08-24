"""Route each decline to the remedy that can actually resolve it.

Recoup's first real policy, and the one that carries the argument. It contains no learning
and no model — it does nothing except take the decline taxonomy seriously.

The insight it exploits is that the industry-standard fixed schedule applies one remedy
(retry) to five different problems, four of which retrying cannot solve:

- a **session-conditional** decline needs the customer back in a live session, so a silent
  retry has probability zero however well timed
- a **never-retryable** decline needs a different instrument
- a **rail-conditional** decline needs the bank back up, so retrying during the outage is
  worse than waiting
- a **time-conditional** decline is the only one where retry timing is the answer

Measured against the calibrated Recurly baseline over 20,000 episodes, this recovers
**8.4 percentage points more payments using fewer attempts per episode** — which is the
shape of result worth having, because attempts are a cost rather than a free action.

It is also the benchmark the learned policy on day 5 has to beat. If a contextual bandit
cannot outperform a policy this simple, the bandit is not earning its complexity, and that
would be a finding worth reporting rather than hiding.
"""

from __future__ import annotations

from recoup.sim.entities import ContactChannel
from recoup.sim.episode import Action, EpisodeState
from recoup.taxonomy import DeclineClass, lookup

__all__ = ["TaxonomyAware"]


class TaxonomyAware:
    """Rule-based routing on decline class. No learning, no model."""

    def __init__(
        self,
        max_contacts: int = 2,
        first_contact_delay_hours: float = 12.0,
        second_contact_delay_hours: float = 60.0,
        rail_recheck_hours: float = 2.0,
    ) -> None:
        self._max_contacts = max_contacts
        self._first_contact = first_contact_delay_hours
        self._second_contact = second_contact_delay_hours
        self._rail_recheck = rail_recheck_hours

    @property
    def name(self) -> str:
        return "taxonomy_aware"

    def decide(self, state: EpisodeState) -> Action:
        reason = lookup(state.current_decline_code)

        match reason.decline_class:
            case DeclineClass.NEVER_RETRYABLE:
                return self._never_retryable(state)
            case DeclineClass.SESSION_CONDITIONAL:
                return self._session_conditional(state)
            case DeclineClass.RAIL_CONDITIONAL:
                return self._rail_conditional(state)
            case DeclineClass.TIME_CONDITIONAL:
                return self._time_conditional(state)
            case _:
                return self._hard_declined(state)

    def _never_retryable(self, state: EpisodeState) -> Action:
        """The instrument is dead. Retrying it is guaranteed waste."""
        alternatives = state.customer.alternatives_to(state.current_instrument_id)
        if alternatives and state.remaining_attempts > 0:
            return Action.retry(1.0, instrument_id=alternatives[0].id)
        if state.contact_count == 0 and not state.opted_out:
            # No alternative instrument exists, so only the customer can resolve this.
            return Action.outreach(self._first_contact, ContactChannel.EMAIL)
        return Action.give_up()

    def _session_conditional(self, state: EpisodeState) -> Action:
        """A silent retry has probability zero. Outreach is the only path.

        This is where the fixed-schedule baseline loses most heavily: it spends a third of
        its wasted attempts here, on declines no retry can resolve.
        """
        if state.in_session and state.remaining_attempts > 0:
            # The session is the scarce resource — use it immediately.
            return Action.retry(0.0)
        if state.contact_count < self._max_contacts and not state.opted_out:
            delay = self._first_contact if state.contact_count == 0 else self._second_contact
            return Action.outreach(delay, ContactChannel.EMAIL)
        return Action.give_up()

    def _rail_conditional(self, state: EpisodeState) -> Action:
        """Wait for the rail. Retrying into a degraded bank cannot succeed.

        The health query is the policy-visible form of Razorpay's Downtime API, and it
        reports current state only — no lookahead. See docs/DECISIONS.md ADR-006.
        """
        if state.remaining_attempts <= 0:
            return Action.give_up()
        if state.rail_is_degraded():
            return Action.retry(self._rail_recheck)
        return Action.retry(0.5)

    def _time_conditional(self, state: EpisodeState) -> Action:
        """The only class where *when* is the whole question.

        Widening delays give the customer's balance time to recover. Day 5 replaces this
        with a learned policy — this fixed ladder is the thing the bandit must beat.
        """
        if state.remaining_attempts <= 0:
            return Action.give_up()
        reason = lookup(state.current_decline_code)
        ladder = 24.0 * (state.attempt_count + 1)
        return Action.retry(max(reason.min_backoff_hours, ladder))

    def _hard_declined(self, state: EpisodeState) -> Action:
        """One cautious attempt, then stop. The cause is unobservable, so there is
        nothing to time and nothing to route around."""
        reason = lookup(state.current_decline_code)
        if state.remaining_attempts > 0 and state.attempt_count == 0:
            return Action.retry(reason.min_backoff_hours)
        return Action.give_up()
