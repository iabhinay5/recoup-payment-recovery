"""One failed payment, explained: the chain from webhook to decision.

The dashboard needs to show *why* Recoup did what it did. The tempting way to build that
is to have the UI read a decline code and describe what the policy would probably do. That
produces a screen that is right until the policy changes, and then quietly lies.

So the trace is not a description of the decision. It **is** the decision: this module
constructs the same ``EpisodeState`` the simulator constructs, hands it to the same policy
object the evaluation harness measures, and puts the result through the same
``Guardrails`` instance that protects live traffic. Every field on a ``DecisionTrace`` is
something the engine returned. Nothing here re-derives a policy rule.

That has a second use beyond the dashboard. ``EpisodeState`` was written as the
simulator's interface to a policy, and if a real Razorpay webhook can construct one, then
the policy being evaluated offline is literally the policy that can run online — not a
reimplementation of it. This module is the evidence for that claim.

**What a webhook cannot tell us.** A ``payment.failed`` payload carries the payment, the
amount, the method and the error. It does not carry the customer's other instruments, the
issuing bank, or the customer's income. Those are joins a merchant integration performs
against its own records. Where they are unavailable the trace says so rather than
substituting a plausible value — a dashboard that invents a bank id is worse than one that
admits it does not know.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from recoup.agent.normalizer import Classification
from recoup.agent.outreach import MAX_CHARS, Language, OutreachWriter
from recoup.gateway.webhooks import FailedPaymentEvent
from recoup.guardrails import Guardrails, Verdict, idempotency_key
from recoup.policies.bandit import BanditPolicy
from recoup.policies.baselines import RECURLY_SCHEDULE_DAYS
from recoup.policies.taxonomy_aware import TaxonomyAware
from recoup.sim.entities import ContactChannel, Customer, FailedPayment, Instrument
from recoup.sim.episode import Action, ActionKind, EpisodeState, Policy
from recoup.sim.params import SimParams
from recoup.sim.rails import RailHealth
from recoup.taxonomy import DeclineClass, PaymentMethod, lookup

__all__ = [
    "UNKNOWN_BANK",
    "Counterfactual",
    "DecisionTrace",
    "TraceStep",
    "explain",
    "live_state",
]

IST = timezone(timedelta(hours=5, minutes=30))
"""Indian Standard Time.

Quiet hours are a fact about when a person is asleep, so they are evaluated in the
customer's local time, not the server's. Every merchant this is built for settles in INR.
"""

UNKNOWN_BANK = "__unidentified__"
"""Bank id used when the payload does not identify the issuer.

``RailHealth`` treats an unknown bank as healthy, which is the correct degradation: a
missing health signal must fall back to normal behaviour rather than to blocking every
retry. The trace surfaces the fallback so nobody reads a green rail as a measured one.
"""

UNKNOWN_INCOME_PAISE = 0
"""Placeholder for income, which a webhook never carries.

Read by the simulator's balance process and by nothing on this path. ``test_trace.py``
asserts the live decision is invariant to it, so this constant cannot silently start
mattering.
"""


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One stage of the chain, as it happened."""

    stage: str
    title: str
    verdict: str
    detail: str
    kind: str = "info"
    """``info``, ``pass``, ``refuse`` or ``defer`` — what the UI colours on."""

    fields: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "title": self.title,
            "verdict": self.verdict,
            "detail": self.detail,
            "kind": self.kind,
            "fields": [list(f) for f in self.fields],
        }


@dataclass(frozen=True, slots=True)
class Counterfactual:
    """What the industry-standard fixed schedule would have proposed for this payment.

    Stated as *proposals* and *permitted attempts* separately, because conflating them
    overstates the case. The Day-1/3/5/7 schedule proposes four retries for every decline
    reason alike; how many of those actually execute here depends on the attempt cap, and
    the evaluation harness applies that same cap to the baseline. Claiming the baseline
    burns four attempts on an expired card would be true of an unguarded production
    dunning system and false of the number this project reports.
    """

    schedule_days: tuple[float, ...]
    proposed: int
    permitted: int
    can_succeed: int | None
    """Permitted attempts that could succeed at all, or ``None`` when it depends on
    conditions at the time of the retry rather than on the decline class."""

    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_days": list(self.schedule_days),
            "proposed": self.proposed,
            "permitted": self.permitted,
            "can_succeed": self.can_succeed,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """A failed payment and everything Recoup concluded about it."""

    payment_id: str
    amount_paise: int
    method: str
    decline_code: str
    recognised: bool
    decline_class: str
    remedy: str
    policy_name: str
    action: str
    action_detail: str
    allowed: bool
    outcome: str
    """``allowed``, ``deferred``, ``refused``, or ``none`` when no action was proposed.

    Separate from ``allowed`` because a deferral is not a refusal. A message held until
    morning is still going to be sent; rendering it in the same red as an opt-out would
    teach a merchant to read a working system as a broken one.
    """

    steps: tuple[TraceStep, ...]
    counterfactual: Counterfactual
    received_at: float
    bank_identified: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    """Honest gaps: what the payload did not carry and what was assumed instead."""

    classification: dict[str, Any] | None = None
    """How the decline code was arrived at, when a normaliser was involved.

    Carries the source — exact lookup, formatting heuristic, model, or the conservative
    fallback — so a reader can see *whether the model was needed at all*. On most live
    traffic it is not, and that is the architecture's central claim rather than an
    incidental detail.
    """

    bandit: dict[str, Any] | None = None
    """Every arm the learned policy scored, not only the one it picked.

    Present when the policy is a ``BanditPolicy``. Showing the alternatives is what makes
    this a decision rather than an assertion: the argmax is checkable.
    """

    message: dict[str, Any] | None = None
    """The customer message, when the action was outreach.

    Generated by a language model where one is configured, and by a deterministic template
    otherwise — ``generated`` says which, because a demo that cannot tell you is not
    evidence of anything.
    """

    @property
    def amount_rupees(self) -> float:
        return self.amount_paise / 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "payment_id": self.payment_id,
            "amount_paise": self.amount_paise,
            "amount_rupees": self.amount_rupees,
            "method": self.method,
            "decline_code": self.decline_code,
            "recognised": self.recognised,
            "decline_class": self.decline_class,
            "remedy": self.remedy,
            "policy_name": self.policy_name,
            "action": self.action,
            "action_detail": self.action_detail,
            "allowed": self.allowed,
            "outcome": self.outcome,
            "steps": [s.to_dict() for s in self.steps],
            "counterfactual": self.counterfactual.to_dict(),
            "received_at": self.received_at,
            "bank_identified": self.bank_identified,
            "notes": list(self.notes),
            "classification": self.classification,
            "bandit": self.bandit,
            "message": self.message,
        }


def _reference_time(created_at: int) -> tuple[int, float]:
    """Day of month and hour of day of the failure, in IST.

    Derived from Razorpay's ``created_at`` rather than from the server clock, so a trace
    replayed tomorrow describes the same moment it did today. The day is clamped to 28
    because the simulator models a uniform 28-day month — see ``FailedPayment``.
    """
    moment = (
        datetime.now(tz=IST)
        if created_at <= 0
        else datetime.fromtimestamp(created_at, tz=UTC).astimezone(IST)
    )
    return min(moment.day, 28), moment.hour + moment.minute / 60.0


def live_state(
    event: FailedPaymentEvent,
    *,
    alternatives: tuple[Instrument, ...] = (),
    bank_id: str = UNKNOWN_BANK,
    rails: RailHealth | None = None,
    customer_id: str | None = None,
    opted_out: bool = False,
    params: SimParams | None = None,
) -> EpisodeState:
    """Build the policy's input from a real webhook.

    The point of this function is that its return type is the *simulator's* state object.
    A policy measured over 20,000 simulated episodes and a policy handed a live Razorpay
    failure are then provably the same object receiving the same type, rather than two
    implementations that are claimed to agree.

    Args:
        event: the verified ``payment.failed`` event.
        alternatives: other instruments the customer holds, from the merchant's vault.
            Empty means unknown, which is what a bare webhook gives you — and it changes
            the answer for a dead instrument, so it must not be guessed.
        bank_id: the issuer, if the integration can resolve it.
        rails: rail health source. An empty one is used when omitted, under which every
            bank reads healthy.
        customer_id: the merchant's customer id, if the order has been joined.
        opted_out: whether this customer has withdrawn contact consent.
        params: simulator parameters, used only for the horizon and contact windows.
    """
    method = event.payment_method or PaymentMethod.CARD
    instrument = Instrument(
        id=f"{event.payment_id}:instrument",
        method=method,
        bank_id=bank_id,
    )
    day, hour = _reference_time(event.created_at)

    customer = Customer(
        id=customer_id or f"unknown_customer_of_{event.payment_id}",
        instruments=(instrument, *alternatives),
        monthly_income_paise=UNKNOWN_INCOME_PAISE,
        salary_day_of_month=day,
        opted_out=opted_out,
    )
    payment = FailedPayment(
        id=event.payment_id,
        customer_id=customer.id,
        instrument_id=instrument.id,
        amount_paise=max(event.amount_paise, 1),
        initial_decline_code=event.decline_code,
        reference_day_of_month=day,
        reference_hour_of_day=hour,
    )

    return EpisodeState(
        payment=payment,
        customer=customer,
        elapsed_hours=0.0,
        attempts=(),
        contacts=(),
        current_decline_code=event.decline_code,
        current_instrument_id=instrument.id,
        in_session=False,
        opted_out=opted_out,
        rails=rails if rails is not None else RailHealth({}),
        params=params if params is not None else SimParams(),
    )


def _counterfactual(decline_code: str) -> Counterfactual:
    """What Day-1/3/5/7 would propose here, and how much of it could work."""
    reason = lookup(decline_code)
    proposed = len(RECURLY_SCHEDULE_DAYS)
    permitted = min(proposed, reason.max_attempts)

    match reason.decline_class:
        case DeclineClass.NEVER_RETRYABLE:
            return Counterfactual(
                RECURLY_SCHEDULE_DAYS,
                proposed,
                permitted,
                0,
                "Retrying this instrument cannot succeed on any schedule. Only a "
                "different instrument, or the customer, can resolve it.",
            )
        case DeclineClass.SESSION_CONDITIONAL:
            return Counterfactual(
                RECURLY_SCHEDULE_DAYS,
                proposed,
                permitted,
                0,
                "A silent retry has probability zero: this needs the customer back in a "
                "live session. Every permitted attempt is spent where it cannot work.",
            )
        case DeclineClass.RAIL_CONDITIONAL:
            return Counterfactual(
                RECURLY_SCHEDULE_DAYS,
                proposed,
                permitted,
                None,
                "Success depends on whether the rail has recovered by the scheduled hour, "
                "which a fixed schedule does not consult.",
            )
        case DeclineClass.TIME_CONDITIONAL:
            return Counterfactual(
                RECURLY_SCHEDULE_DAYS,
                proposed,
                permitted,
                None,
                "The only class where timing is the whole question, and the one a fixed "
                "schedule answers by calendar rather than by the customer's balance.",
            )
        case _:
            return Counterfactual(
                RECURLY_SCHEDULE_DAYS,
                proposed,
                permitted,
                None,
                "The cause is not observable from the decline code, so there is nothing "
                "to route around and nothing to time.",
            )


def _describe(action: Action, state: EpisodeState) -> tuple[str, str]:
    """Turn the policy's action into something a human reads."""
    match action.kind:
        case ActionKind.RETRY:
            switched = (
                action.instrument_id is not None
                and action.instrument_id != state.current_instrument_id
            )
            where = " on a different instrument" if switched else " on the same instrument"
            when = "immediately" if action.delay_hours == 0 else f"in {action.delay_hours:g}h"
            return "retry", f"retry {when}{where}"
        case ActionKind.OUTREACH:
            channel = action.channel.value if action.channel else "email"
            when = "immediately" if action.delay_hours == 0 else f"in {action.delay_hours:g}h"
            return "outreach", f"contact by {channel} {when}"
        case _:
            return "stop", "stop — no action can recover this payment"


def explain(
    event: FailedPaymentEvent,
    *,
    policy: Policy | None = None,
    guardrails: Guardrails | None = None,
    state: EpisodeState | None = None,
    received_at: float | None = None,
    classification: Classification | None = None,
    writer: OutreachWriter | None = None,
    language: Language = Language.ENGLISH,
) -> DecisionTrace:
    """Run one failed payment through the engine and record every stage.

    Nothing here decides anything itself. The taxonomy classifies, the policy chooses, the
    guardrails rule; this function only writes down what each of them returned, in order.

    Args:
        policy: the recovery policy. Defaults to ``TaxonomyAware``, whose structural
            routing is what ``BanditPolicy`` also applies — the learned policy differs
            only in the delay it picks inside the window this routing opens.
        guardrails: the guardrail instance to check against. Passing a shared one across
            calls is what makes a replayed charge collide with the original.
        state: a state built by ``live_state``. Build your own when the integration can
            supply the customer's other instruments or the issuing bank.
        received_at: unix timestamp of receipt, for ordering the feed.
        classification: how the decline code was resolved, when a normaliser produced it.
            Recorded so the trace can show whether a model was needed — usually it is not.
        writer: used to compose the customer message when the action is outreach. Falls
            back to a template when it has no provider, and says which it used.
        language: the language to write that message in.
    """
    resolved_policy: Policy = policy if policy is not None else TaxonomyAware()
    guards = guardrails if guardrails is not None else Guardrails()
    live = state if state is not None else live_state(event)
    reason = lookup(event.decline_code)

    notes: list[str] = []
    bank_identified = live.current_instrument.bank_id != UNKNOWN_BANK
    if not bank_identified:
        notes.append(
            "The payload does not identify the issuing bank, so rail health could not be "
            "consulted and the rail reads healthy by default."
        )
    if not live.customer.alternatives_to(live.current_instrument_id):
        notes.append(
            "No alternative instrument is known for this customer. A merchant integration "
            "would supply them from its vault, and for a dead instrument that changes the "
            "decision from outreach to an instrument switch."
        )
    if event.payment_method is None and event.method:
        notes.append(f"Method {event.method!r} is not one the taxonomy models by rail.")
    if not event.is_recognised:
        notes.append(
            "This decline code is not in the taxonomy. It is being handled under the "
            "conservative unknown default, and it should be classified before the next run."
        )

    steps: list[TraceStep] = [
        TraceStep(
            stage="received",
            title="What Razorpay reported",
            verdict=event.decline_code,
            detail=event.error_description or "no description supplied",
            kind="info",
            fields=(
                ("payment id", event.payment_id),
                ("amount", f"Rs {event.amount_rupees:,.2f}"),
                ("method", event.method or "-"),
                ("error_reason", event.error_reason),
                ("error_code", event.error_code or "-"),
                ("error_source", event.error_source or "-"),
                ("error_step", event.error_step or "-"),
            ),
        ),
        TraceStep(
            stage="classified",
            title="How the taxonomy classified it",
            verdict=reason.decline_class.value,
            detail=reason.description,
            kind="pass" if event.is_recognised else "defer",
            fields=(
                ("known code", "yes" if event.is_recognised else "NO — conservative default"),
                ("decline class", reason.decline_class.value),
                ("remedy", reason.remedy.value),
                ("attempts permitted", str(reason.max_attempts)),
                ("minimum backoff", f"{reason.min_backoff_hours:g}h"),
            ),
        ),
    ]

    classification_view: dict[str, Any] | None = None
    if classification is not None:
        classification_view = {
            "code": classification.code,
            "source": classification.source.value,
            "confidence": classification.confidence,
            "rationale": classification.rationale,
            "used_model": classification.used_model,
            "is_confident": classification.is_confident,
        }
        deterministic = not classification.used_model
        steps.insert(
            2,
            TraceStep(
                stage="normalised",
                title="How the code was resolved",
                verdict=classification.source.value.upper()
                + (" — no model call" if deterministic else " — language model"),
                detail=(
                    "Razorpay's error_reason is already the taxonomy's vocabulary, so the "
                    "common case is a dictionary lookup. The model is reached only when the "
                    "deterministic path cannot answer."
                    if deterministic
                    else "No deterministic match existed, so the free text was classified by "
                    "the model and the returned code validated against the taxonomy."
                ),
                kind="pass" if classification.is_confident else "defer",
                fields=(
                    ("source", classification.source.value),
                    ("confidence", f"{classification.confidence:.2f}"),
                    ("model used", "no" if deterministic else "yes"),
                    ("rationale", classification.rationale or "-"),
                ),
            ),
        )

    action = resolved_policy.decide(live)
    action_kind, action_detail = _describe(action, live)

    bandit_view: dict[str, Any] | None = None
    if isinstance(resolved_policy, BanditPolicy):
        bandit_view = resolved_policy.explain_dict(live)
    steps.append(
        TraceStep(
            stage="decided",
            title=f"What the policy chose — {resolved_policy.name}",
            verdict=action_detail,
            detail=(
                "This is the structural routing, taken from the decline class. The learned "
                "policy takes the same branch and differs only in the delay it picks."
            ),
            kind="pass",
            fields=(
                ("action", action_kind),
                ("delay", f"{action.delay_hours:g}h"),
                ("channel", action.channel.value if action.channel is not None else "-"),
            ),
        )
    )

    allowed = True
    outcome = "none"
    if action.kind is ActionKind.RETRY:
        target = live.current_instrument
        for instrument in live.customer.instruments:
            if instrument.id == action.instrument_id:
                target = instrument
        # A never-retryable decline is Razorpay telling us the instrument itself is
        # finished, so the dead-instrument rule is grounded in the payload rather than
        # assumed. It applies only to the instrument that actually failed.
        dead = (
            target.id == live.current_instrument_id
            and reason.decline_class is DeclineClass.NEVER_RETRYABLE
        )
        key = idempotency_key(event.payment_id, 0, target.id)
        verdict = guards.check_retry(
            key=key,
            attempts_made=0,
            decline_code=event.decline_code,
            instrument_expired=dead,
        )
        allowed = verdict.allowed
        outcome = _outcome(verdict)
        steps.append(_verdict_step("Guardrails — charge", verdict, key))

        if verdict.allowed:
            guards.record_charge(key)
            replay = guards.check_retry(
                key=key,
                attempts_made=0,
                decline_code=event.decline_code,
                instrument_expired=dead,
            )
            steps.append(
                _verdict_step(
                    "Guardrails — the same charge, replayed",
                    replay,
                    key,
                    detail_override=(
                        "The identical attempt is refused on the second submission. This "
                        "is the double-charge guard, exercised on every trace rather than "
                        "asserted in a slide."
                    ),
                )
            )
    elif action.kind is ActionKind.OUTREACH:
        lands_at = live.payment.hour_of_day_at(action.delay_hours)
        verdict = guards.check_outreach(
            hour_of_day=lands_at,
            opted_out=live.opted_out,
            contacts_in_window=live.contacts_within(guards.contact_window_hours),
        )
        allowed = verdict.allowed
        outcome = _outcome(verdict)
        steps.append(
            _verdict_step(
                "Guardrails — contact",
                verdict,
                None,
                extra=(("lands at", f"{lands_at:.1f}h local"),),
            )
        )
    else:
        steps.append(
            TraceStep(
                stage="guarded",
                title="Guardrails",
                verdict="nothing to check",
                detail="The policy proposed no action, so no rule has anything to rule on.",
                kind="info",
            )
        )

    message_view: dict[str, Any] | None = None
    if writer is not None and action.kind is ActionKind.OUTREACH:
        channel = action.channel if action.channel is not None else ContactChannel.EMAIL
        composed = writer.write(event.decline_code, event.amount_paise, channel, language)
        message_view = {
            "channel": composed.channel.value,
            "language": composed.language.name.title(),
            "subject": composed.subject,
            "body": composed.body,
            "generated": composed.generated,
            "length": composed.length,
            "limit": MAX_CHARS[composed.channel],
            "within_limit": composed.within_limit,
        }
        steps.append(
            TraceStep(
                stage="composed",
                title="The message the customer receives",
                verdict=("generated" if composed.generated else "template")
                + f" — {composed.length}/{MAX_CHARS[composed.channel]} chars",
                detail=composed.body,
                kind="pass",
                fields=(
                    ("channel", composed.channel.value),
                    ("language", composed.language.name.title()),
                    ("written by", "language model" if composed.generated else "template"),
                    ("subject", composed.subject or "-"),
                ),
            )
        )

    counterfactual = _counterfactual(event.decline_code)
    days = "/".join(f"{d:g}" for d in counterfactual.schedule_days)
    steps.append(
        TraceStep(
            stage="counterfactual",
            title="What a fixed Day-1/3/5/7 schedule would do",
            verdict=(
                f"{counterfactual.proposed} retries proposed on days {days}; "
                f"{counterfactual.permitted} permitted by the attempt cap"
            ),
            detail=counterfactual.note,
            kind="info",
            fields=(
                ("proposed retries", str(counterfactual.proposed)),
                ("permitted by cap", str(counterfactual.permitted)),
                (
                    "able to succeed",
                    "0" if counterfactual.can_succeed == 0 else "depends on conditions",
                ),
            ),
        )
    )

    return DecisionTrace(
        payment_id=event.payment_id,
        amount_paise=event.amount_paise,
        method=event.method,
        decline_code=event.decline_code,
        recognised=event.is_recognised,
        decline_class=reason.decline_class.value,
        remedy=reason.remedy.value,
        policy_name=resolved_policy.name,
        action=action_kind,
        action_detail=action_detail,
        allowed=allowed,
        outcome=outcome,
        steps=tuple(steps),
        counterfactual=counterfactual,
        received_at=received_at if received_at is not None else datetime.now(tz=UTC).timestamp(),
        bank_identified=bank_identified,
        notes=tuple(notes),
        classification=classification_view,
        bandit=bandit_view,
        message=message_view,
    )


def _outcome(verdict: Verdict) -> str:
    """Collapse a verdict into the word the UI badges."""
    if verdict.allowed:
        return "allowed"
    return "deferred" if verdict.deferred else "refused"


def _verdict_step(
    title: str,
    verdict: Verdict,
    key: str | None,
    *,
    detail_override: str | None = None,
    extra: tuple[tuple[str, str], ...] = (),
) -> TraceStep:
    """Render a guardrail outcome without restating the rule that produced it."""
    fields = list(extra)
    if key is not None:
        fields.append(("idempotency key", key[:16]))

    if verdict.allowed:
        return TraceStep(
            stage="guarded",
            title=title,
            verdict="ALLOWED",
            detail=detail_override or "Every rule cleared this action.",
            kind="pass",
            fields=tuple(fields),
        )

    assert verdict.refusal is not None
    if verdict.deferred and verdict.defer_hours is not None:
        return TraceStep(
            stage="guarded",
            title=title,
            verdict=f"DEFERRED {verdict.defer_hours:.1f}h — {verdict.refusal.rule.value}",
            detail=detail_override or verdict.refusal.detail,
            kind="defer",
            fields=tuple(fields),
        )

    return TraceStep(
        stage="guarded",
        title=title,
        verdict=f"REFUSED — {verdict.refusal.rule.value}",
        detail=detail_override or verdict.refusal.detail,
        kind="refuse",
        fields=tuple(fields),
    )
