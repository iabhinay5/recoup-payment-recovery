"""The layer every action passes through before it touches a customer.

A recovery policy proposes; this disposes. Nothing reaches a payment rail or an inbox
without clearing every rule here, and no policy — rule-based or learned — can reach past
it. That separation is the whole point: a guardrail a policy can talk its way around is
not a guardrail, it is a suggestion.

The worst thing this system could do is **charge someone twice**. Not "recover less
revenue" — take money that was never owed. That has to be impossible by construction
rather than unlikely by good behaviour, because the failure mode is not a bad number on a
chart, it is a real person who has to call their bank.

Every refusal is recorded with the rule that caused it, so a merchant can answer "why
didn't you retry my customer's payment?" with a reason rather than a shrug.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from recoup.taxonomy import lookup

HOURS_PER_DAY = 24.0
"""Defined here rather than imported from ``recoup.sim``.

The guardrail layer must not depend on the simulator: it protects real payment traffic,
and the simulator is one caller among several. Importing the other way round would also
make the cycle real — the episode runner needs this module.
"""

__all__ = [
    "Guardrails",
    "Refusal",
    "Rule",
    "Verdict",
    "idempotency_key",
]


class Rule(Enum):
    """Why an action was stopped."""

    DUPLICATE_CHARGE = "duplicate_charge"
    """An identical charge has already been executed. The double-charge guard."""

    ATTEMPT_CAP = "attempt_cap"
    """The decline reason's structural attempt limit is exhausted."""

    OPTED_OUT = "opted_out"
    """The customer asked not to be contacted."""

    QUIET_HOURS = "quiet_hours"
    """Outreach outside waking hours. Deferred rather than refused."""

    CONTACT_RATE = "contact_rate"
    """Too many contacts in the rolling window."""

    DEAD_INSTRUMENT = "dead_instrument"
    """The instrument cannot succeed, so charging it is pure waste."""


@dataclass(frozen=True, slots=True)
class Refusal:
    """A blocked action and the rule that blocked it."""

    rule: Rule
    detail: str

    def __str__(self) -> str:
        return f"{self.rule.value}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of a guardrail check.

    Three outcomes rather than two. Refusing and deferring are genuinely different: a
    customer who opted out must never be contacted, while a customer asleep at 3am must be
    contacted *later*. Collapsing them into one would either spam people at night or drop
    messages that were merely badly timed.
    """

    allowed: bool
    refusal: Refusal | None = None
    defer_hours: float | None = None

    @classmethod
    def allow(cls) -> Verdict:
        return cls(True)

    @classmethod
    def refuse(cls, rule: Rule, detail: str) -> Verdict:
        return cls(False, Refusal(rule, detail))

    @classmethod
    def defer(cls, rule: Rule, detail: str, hours: float) -> Verdict:
        return cls(False, Refusal(rule, detail), defer_hours=hours)

    @property
    def deferred(self) -> bool:
        return self.defer_hours is not None


def idempotency_key(payment_id: str, attempt_index: int, instrument_id: str) -> str:
    """Deterministic key identifying one logical charge attempt.

    Deterministic is the requirement, not uniqueness. The same logical attempt must
    produce the same key every time it is derived, so that a duplicate — a replayed
    webhook, a retried request after a timeout, a crash-and-restart that loses in-flight
    state — collides with the original instead of creating a second charge.

    A random key would defeat the entire mechanism while looking correct.
    """
    raw = f"{payment_id}:{attempt_index}:{instrument_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class Guardrails:
    """Enforces the rules. Holds the ledger of charges already executed.

    In production the ledger is a durable store shared across workers, because the case it
    exists for is precisely the one where a process died holding in-flight state. Here it
    is in-memory per episode, which models the same semantics.
    """

    quiet_start_hour: float = 22.0
    quiet_end_hour: float = 8.0
    max_contacts_per_window: int = 3
    contact_window_hours: float = 7 * HOURS_PER_DAY

    _charged: set[str] = field(default_factory=set, repr=False)
    refusals: list[Refusal] = field(default_factory=list, repr=False)

    # --- charges ---------------------------------------------------------------------

    def check_retry(
        self,
        key: str,
        attempts_made: int,
        decline_code: str,
        instrument_expired: bool,
    ) -> Verdict:
        """Decide whether a charge attempt may execute."""
        if key in self._charged:
            return self._record(
                Verdict.refuse(
                    Rule.DUPLICATE_CHARGE,
                    f"idempotency key {key[:12]} has already been charged",
                )
            )

        if instrument_expired:
            return self._record(
                Verdict.refuse(
                    Rule.DEAD_INSTRUMENT,
                    "the instrument cannot succeed; charging it can only waste an attempt",
                )
            )

        cap = lookup(decline_code).max_attempts
        if attempts_made >= cap:
            return self._record(
                Verdict.refuse(
                    Rule.ATTEMPT_CAP,
                    f"{decline_code} permits {cap} attempt(s); {attempts_made} already made",
                )
            )

        return Verdict.allow()

    def record_charge(self, key: str) -> None:
        """Mark a charge as executed.

        Called *after* the attempt is issued, so a replay of the same logical attempt
        collides on the next check. Recording before issuing would be safer against
        duplicates and worse against loss — this ordering matches how a payment processor
        reserves a key.
        """
        self._charged.add(key)

    def has_charged(self, key: str) -> bool:
        return key in self._charged

    @property
    def charge_count(self) -> int:
        return len(self._charged)

    # --- contact ---------------------------------------------------------------------

    def check_outreach(
        self,
        hour_of_day: float,
        opted_out: bool,
        contacts_in_window: int,
    ) -> Verdict:
        """Decide whether a customer may be contacted right now."""
        if opted_out:
            return self._record(
                Verdict.refuse(
                    Rule.OPTED_OUT,
                    "the customer asked not to be contacted",
                )
            )

        if contacts_in_window >= self.max_contacts_per_window:
            return self._record(
                Verdict.refuse(
                    Rule.CONTACT_RATE,
                    f"{contacts_in_window} contacts in the last "
                    f"{self.contact_window_hours / HOURS_PER_DAY:.0f} days",
                )
            )

        if self.in_quiet_hours(hour_of_day):
            wait = self.hours_until_wake(hour_of_day)
            return self._record(
                Verdict.defer(
                    Rule.QUIET_HOURS,
                    f"{hour_of_day:.1f}h is inside quiet hours; deferring {wait:.1f}h",
                    wait,
                )
            )

        return Verdict.allow()

    def in_quiet_hours(self, hour_of_day: float) -> bool:
        """Whether the given local hour falls inside the do-not-disturb window.

        The window wraps past midnight, which is the case a naive ``start <= h < end``
        comparison gets silently wrong — and getting it wrong means messaging people at
        three in the morning.
        """
        hour = hour_of_day % HOURS_PER_DAY
        if self.quiet_start_hour < self.quiet_end_hour:
            return self.quiet_start_hour <= hour < self.quiet_end_hour
        return hour >= self.quiet_start_hour or hour < self.quiet_end_hour

    def hours_until_wake(self, hour_of_day: float) -> float:
        """How long until outreach becomes permissible again."""
        hour = hour_of_day % HOURS_PER_DAY
        if not self.in_quiet_hours(hour):
            return 0.0
        delta = (self.quiet_end_hour - hour) % HOURS_PER_DAY
        return delta if delta > 0 else HOURS_PER_DAY

    # --- reporting -------------------------------------------------------------------

    def _record(self, verdict: Verdict) -> Verdict:
        if verdict.refusal is not None:
            self.refusals.append(verdict.refusal)
        return verdict

    def refusal_counts(self) -> dict[str, int]:
        """How often each rule fired. The merchant-facing audit summary."""
        counts: dict[str, int] = {}
        for refusal in self.refusals:
            counts[refusal.rule.value] = counts.get(refusal.rule.value, 0) + 1
        return counts
