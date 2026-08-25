"""Counterfactual variants of the routing policy, for attributing the uplift.

A single headline number says the system works. It does not say *which part* of it works,
and that is the question a panel asks. Each class here differs from ``TaxonomyAware`` by
exactly one branch, so the gap between it and the original is that branch's contribution
and nothing else.

Two of these remove a signal, which is an ablation in the usual sense:

- ``RailBlind`` stops consulting rail health, isolating what Razorpay's Downtime API is
  worth (ADR-006).
- ``SessionBlind`` treats session-conditional declines as ordinary timing problems,
  isolating the project's central claim — that a silent retry on a decline needing a live
  customer session has probability zero however well timed.

The third adds a rule rather than removing one, and is here because it is the same kind of
question:

- ``RemedyRouted`` routes hard-declined codes by the remedy the taxonomy already assigns
  them. ``card_declined`` and ``payment_failed`` carry ``Remedy.NEW_INSTRUMENT``, but
  ``_hard_declined`` never reads the remedy and retries the card that just failed.
  Razorpay's published resolution for ``card_declined`` is to "advise your customer to
  attempt the payment again using another card", which sides with the remedy. Whether
  following it costs or gains recovery is a measurement, not an opinion.
"""

from __future__ import annotations

from recoup.policies.taxonomy_aware import TaxonomyAware
from recoup.sim.entities import ContactChannel
from recoup.sim.episode import Action, EpisodeState
from recoup.taxonomy import Remedy, lookup

__all__ = ["RailBlind", "RemedyRouted", "SessionBlind"]


class RailBlind(TaxonomyAware):
    """Routes on decline class, but never asks whether the rail is degraded.

    Everything else is identical, so the difference measures one thing: what it is worth
    to know that the customer's bank is currently down. If that difference is small, the
    Downtime API integration is not carrying its weight and the honest thing is to say so.
    """

    @property
    def name(self) -> str:
        return "ablation_rail_blind"

    def _rail_conditional(self, state: EpisodeState) -> Action:
        if state.remaining_attempts <= 0:
            return Action.give_up()
        # The same prompt retry the original makes when the rail reads healthy — issued
        # unconditionally, because this policy cannot tell the difference.
        return Action.retry(0.5)


class SessionBlind(TaxonomyAware):
    """Treats a session-conditional decline as an ordinary timing problem.

    This is the fixed schedule's mistake, isolated. A ``payment_cancelled`` decline needs
    the customer back in a live session; retrying it silently cannot succeed at any delay.
    The gap between this and ``TaxonomyAware`` is the value of knowing that.
    """

    @property
    def name(self) -> str:
        return "ablation_session_blind"

    def _session_conditional(self, state: EpisodeState) -> Action:
        if state.remaining_attempts <= 0:
            return Action.give_up()
        if state.in_session:
            return Action.retry(0.0)
        ladder = 24.0 * (state.attempt_count + 1)
        return Action.retry(ladder)


class RemedyRouted(TaxonomyAware):
    """Routes hard-declined codes by their documented remedy rather than retrying once.

    ``_hard_declined`` in the original spends one attempt on the instrument that just
    failed, whatever the taxonomy says the remedy is. For the two codes carrying
    ``Remedy.NEW_INSTRUMENT`` that contradicts both the taxonomy and Razorpay's own
    guidance, which tells the merchant to ask the customer to use another card.

    Expected to *cost* recovery in the simulator, because a silent retry sometimes works
    and an outreach depends on the customer answering. That is the point of measuring it:
    if following the documented remedy is expensive, the size of the bill is what makes
    the decision defensible either way.
    """

    @property
    def name(self) -> str:
        return "ablation_remedy_routed"

    def _hard_declined(self, state: EpisodeState) -> Action:
        reason = lookup(state.current_decline_code)
        if reason.remedy is not Remedy.NEW_INSTRUMENT:
            return super()._hard_declined(state)
        if state.contact_count == 0 and not state.opted_out:
            return Action.outreach(self._first_contact, ContactChannel.EMAIL)
        return Action.give_up()
