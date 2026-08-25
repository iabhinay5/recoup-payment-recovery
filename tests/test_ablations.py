"""Tests for the counterfactual policy variants.

An ablation is only evidence if it differs from the original in the one way it claims to.
A variant that quietly changed two branches would attribute both to whichever it was named
after, and nothing in the measured numbers would show it. So each class here is pinned
twice: it must differ from ``TaxonomyAware`` on the branch it targets, and agree with it
everywhere else.
"""

from __future__ import annotations

import pytest

from recoup.policies import RailBlind, RemedyRouted, SessionBlind, TaxonomyAware
from recoup.sim.entities import Customer, FailedPayment, Instrument
from recoup.sim.episode import ActionKind, EpisodeState
from recoup.sim.params import SimParams
from recoup.sim.rails import Outage, RailHealth
from recoup.taxonomy import PaymentMethod

BANK = "hdfc"


def state(
    decline_code: str,
    *,
    degraded: bool = False,
    in_session: bool = False,
    attempts: tuple[object, ...] = (),
) -> EpisodeState:
    """A minimal episode state carrying one failed payment."""
    instrument = Instrument(id="instr_1", method=PaymentMethod.CARD, bank_id=BANK)
    customer = Customer(
        id="cust_1",
        instruments=(instrument,),
        monthly_income_paise=5_000_000,
        salary_day_of_month=1,
        opted_out=False,
    )
    payment = FailedPayment(
        id="pay_1",
        customer_id=customer.id,
        instrument_id=instrument.id,
        amount_paise=250_000,
        initial_decline_code=decline_code,
        reference_day_of_month=10,
        reference_hour_of_day=14.0,
    )
    rails = RailHealth({})
    if degraded:
        rails = rails.with_outage(Outage(BANK, -0.01, 6.0))

    return EpisodeState(
        payment=payment,
        customer=customer,
        elapsed_hours=0.0,
        attempts=attempts,  # type: ignore[arg-type]
        contacts=(),
        current_decline_code=decline_code,
        current_instrument_id=instrument.id,
        in_session=in_session,
        opted_out=False,
        rails=rails,
        params=SimParams(),
    )


class TestRailBlind:
    def test_it_retries_into_an_outage_the_original_waits_out(self) -> None:
        degraded = state("bank_technical_error", degraded=True)

        original = TaxonomyAware().decide(degraded)
        blind = RailBlind().decide(degraded)

        assert original.kind is ActionKind.RETRY
        assert blind.kind is ActionKind.RETRY
        assert blind.delay_hours < original.delay_hours, (
            "the point of the ablation is that it does not defer for the rail"
        )

    def test_it_matches_the_original_when_the_rail_is_healthy(self) -> None:
        healthy = state("bank_technical_error")

        original = TaxonomyAware().decide(healthy)
        blind = RailBlind().decide(healthy)

        assert (original.kind, original.delay_hours) == (blind.kind, blind.delay_hours)


class TestSessionBlind:
    def test_it_retries_a_decline_that_needs_the_customer(self) -> None:
        """The fixed schedule's mistake, isolated: a silent retry here cannot succeed."""
        idle = state("payment_cancelled")

        assert TaxonomyAware().decide(idle).kind is ActionKind.OUTREACH
        assert SessionBlind().decide(idle).kind is ActionKind.RETRY

    def test_it_matches_the_original_when_the_customer_is_present(self) -> None:
        live = state("payment_cancelled", in_session=True)

        original = TaxonomyAware().decide(live)
        blind = SessionBlind().decide(live)

        assert (original.kind, original.delay_hours) == (blind.kind, blind.delay_hours)


class TestRemedyRouted:
    def test_it_asks_the_customer_instead_of_retrying_a_dead_card(self) -> None:
        """`card_declined` carries Remedy.NEW_INSTRUMENT, and Razorpay's published
        resolution is to advise the customer to try another card. The original retries
        the card that just failed."""
        declined = state("card_declined")

        assert TaxonomyAware().decide(declined).kind is ActionKind.RETRY
        assert RemedyRouted().decide(declined).kind is ActionKind.OUTREACH

    def test_it_leaves_hard_declines_with_another_remedy_alone(self) -> None:
        """`vpa_resolution_failed` is hard-declined but carries Remedy.ESCALATE, so this
        variant must not touch it. Without this the ablation would be measuring two
        changes and attributing both to one."""
        other = state("vpa_resolution_failed")

        original = TaxonomyAware().decide(other)
        routed = RemedyRouted().decide(other)

        assert (original.kind, original.delay_hours) == (routed.kind, routed.delay_hours)


class TestEachVariantChangesOnlyItsOwnBranch:
    @pytest.mark.parametrize("variant", [RailBlind, RemedyRouted, SessionBlind])
    @pytest.mark.parametrize(
        "code", ["insufficient_funds", "transaction_limit_exceeded", "card_expired"]
    )
    def test_untargeted_classes_are_untouched(
        self, variant: type[TaxonomyAware], code: str
    ) -> None:
        subject = state(code)

        original = TaxonomyAware().decide(subject)
        ablated = variant().decide(subject)

        assert (original.kind, original.delay_hours) == (ablated.kind, ablated.delay_hours)

    @pytest.mark.parametrize("variant", [RailBlind, RemedyRouted, SessionBlind])
    def test_each_has_a_distinct_name(self, variant: type[TaxonomyAware]) -> None:
        """The results file is keyed by policy name; a collision would silently drop one."""
        assert variant().name != TaxonomyAware().name
        assert variant().name.startswith("ablation_")
