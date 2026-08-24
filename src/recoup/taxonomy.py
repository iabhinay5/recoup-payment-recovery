"""Razorpay decline-reason taxonomy — the spine of every policy decision in Recoup.

Most failed-payment systems treat "payment failed" as one undifferentiated bucket and
apply a fixed retry schedule to all of it. That is wrong in both directions: an expired
card retried four times can never succeed, while a bank outage retried immediately wastes
an attempt on a rail that is still down.

This module encodes Razorpay's *published* decline codes and classifies each one by what
actually makes it recoverable. Every entry carries the documentation URL it came from —
see docs/CALIBRATION.md section 1 for why provenance is tracked this way.

The module has no third-party dependencies, deliberately. It is imported by the simulator,
the policy, the guardrail layer, and the LLM normalizer; none of them should have to pull
in numpy to ask whether a decline code is worth retrying.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CARD_ERRORS_SOURCE",
    "UNVERIFIED_REASONS",
    "UPI_ERRORS_SOURCE",
    "DeclineClass",
    "DeclineReason",
    "PaymentMethod",
    "Remedy",
    "all_reasons",
    "by_class",
    "is_retryable",
    "lookup",
    "max_attempts_for",
]

CARD_ERRORS_SOURCE = "https://razorpay.com/docs/errors/payments/cards/"
UPI_ERRORS_SOURCE = "https://razorpay.com/docs/errors/payments/upi/"


class PaymentMethod(Enum):
    """A payment rail.

    ``NETBANKING`` and ``WALLET`` are here because live sandbox traffic arrives on them,
    not because this module classifies rail-specific codes for them. Razorpay publishes
    separate error pages per method; only the card and UPI pages have been encoded so far,
    so a netbanking-specific reason would currently fall through to the conservative
    unknown default rather than being misclassified.
    """

    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class DeclineClass(Enum):
    """What determines whether a retry of this decline can ever succeed.

    This is the single most important classification in the system. The policy branches
    on it before considering anything else, because it bounds what any amount of clever
    timing can achieve.
    """

    NEVER_RETRYABLE = "never_retryable"
    """No retry can succeed. The instrument or the customer's account state must change
    first. Retrying burns attempts and customer goodwill for a guaranteed failure."""

    RAIL_CONDITIONAL = "rail_conditional"
    """Retryable only once the bank or gateway recovers. Retrying into a degraded rail
    cannot succeed and adds load to a system already failing. See DECISIONS.md ADR-006."""

    TIME_CONDITIONAL = "time_conditional"
    """Retryable, and the success probability depends strongly on *when*. This is the
    genuine sequential decision problem the bandit exists to solve."""

    SESSION_CONDITIONAL = "session_conditional"
    """Requires the customer back in a live payment session (fresh OTP, re-authorisation).
    A silent background retry cannot resolve it — this needs outreach, not a retry."""

    HARD_DECLINED = "hard_declined"
    """The issuer refused for reasons we cannot observe or influence. Low retry value;
    the productive move is a different instrument or human escalation."""


class Remedy(Enum):
    """The action that could actually resolve this decline."""

    RETRY_LATER = "retry_later"
    """Silent background retry at a better time. No customer contact required."""

    AWAIT_RAIL_RECOVERY = "await_rail_recovery"
    """Hold until the bank or gateway reports healthy, then retry."""

    CUSTOMER_SESSION = "customer_session"
    """Bring the customer back to complete a payment interactively."""

    CUSTOMER_ACTION = "customer_action"
    """The customer must act out-of-band: enable the card for online use, unblock it,
    or complete VPA registration. Nothing we do server-side changes the outcome."""

    NEW_INSTRUMENT = "new_instrument"
    """A different card or payment method is required."""

    ESCALATE = "escalate"
    """Route to human review. Used where retrying is both unlikely to help and
    potentially harmful."""


@dataclass(frozen=True, slots=True)
class DeclineReason:
    """One decline code and everything the policy needs to reason about it.

    ``max_attempts`` and ``min_backoff_hours`` are **structural caps, not a schedule**.
    The bandit chooses timing freely within them; it cannot exceed them. This is what
    makes the guardrail structural rather than advisory (DECISIONS.md ADR-007) — a policy
    literally cannot schedule a fifth attempt against a code capped at four, and cannot
    schedule any attempt against a code capped at zero.

    The cap values are engineering defaults, not calibrated measurements. They are
    deliberately conservative and are not swept during evaluation; the policy's job is to
    do well *within* them, not to discover them.
    """

    code: str

    methods: frozenset[PaymentMethod]
    """Which of Razorpay's error pages documented this code.

    **This is provenance, not an exhaustive claim about which rails can emit the code.**
    A live sandbox payment corrected that reading: ``payment_failed`` is documented on the
    card page and arrived on ``netbanking``. Generic codes cross rails freely, so nothing
    in the policy branches on this field — it exists so each entry can be traced back to
    the page it came from.
    """

    description: str
    decline_class: DeclineClass
    remedy: Remedy
    max_attempts: int
    min_backoff_hours: float
    source: str

    def __post_init__(self) -> None:
        if self.decline_class is DeclineClass.NEVER_RETRYABLE and self.max_attempts != 0:
            raise ValueError(
                f"{self.code}: NEVER_RETRYABLE codes must have max_attempts == 0, "
                f"got {self.max_attempts}. This invariant is what makes the "
                f"never-retry guarantee structural rather than advisory."
            )
        if self.max_attempts < 0:
            raise ValueError(f"{self.code}: max_attempts must be >= 0")
        if self.min_backoff_hours < 0:
            raise ValueError(f"{self.code}: min_backoff_hours must be >= 0")

    @property
    def is_retryable(self) -> bool:
        """Whether any automated retry of this decline could succeed."""
        return self.max_attempts > 0

    @property
    def needs_customer_contact(self) -> bool:
        """Whether resolving this requires reaching the customer rather than retrying."""
        return self.remedy in (Remedy.CUSTOMER_SESSION, Remedy.CUSTOMER_ACTION)


def _card(
    code: str,
    description: str,
    decline_class: DeclineClass,
    remedy: Remedy,
    max_attempts: int,
    min_backoff_hours: float,
) -> DeclineReason:
    return DeclineReason(
        code=code,
        methods=frozenset({PaymentMethod.CARD}),
        description=description,
        decline_class=decline_class,
        remedy=remedy,
        max_attempts=max_attempts,
        min_backoff_hours=min_backoff_hours,
        source=CARD_ERRORS_SOURCE,
    )


def _upi(
    code: str,
    description: str,
    decline_class: DeclineClass,
    remedy: Remedy,
    max_attempts: int,
    min_backoff_hours: float,
) -> DeclineReason:
    return DeclineReason(
        code=code,
        methods=frozenset({PaymentMethod.UPI}),
        description=description,
        decline_class=decline_class,
        remedy=remedy,
        max_attempts=max_attempts,
        min_backoff_hours=min_backoff_hours,
        source=UPI_ERRORS_SOURCE,
    )


def _shared(
    code: str,
    description: str,
    decline_class: DeclineClass,
    remedy: Remedy,
    max_attempts: int,
    min_backoff_hours: float,
) -> DeclineReason:
    """A code Razorpay documents on both the card and UPI error pages."""
    return DeclineReason(
        code=code,
        methods=frozenset({PaymentMethod.CARD, PaymentMethod.UPI}),
        description=description,
        decline_class=decline_class,
        remedy=remedy,
        max_attempts=max_attempts,
        min_backoff_hours=min_backoff_hours,
        source=f"{CARD_ERRORS_SOURCE} , {UPI_ERRORS_SOURCE}",
    )


_REASONS: tuple[DeclineReason, ...] = (
    # --- Never retryable -------------------------------------------------------------
    # The defining property of this group: max_attempts == 0. No timing strategy, however
    # clever, recovers these. Recognising them is where the cheapest wins in the whole
    # system live, because the industry default is to retry them on the same fixed
    # schedule as everything else.
    _card(
        "card_expired",
        "The card has expired. A retry cannot succeed until the customer supplies a new card.",
        DeclineClass.NEVER_RETRYABLE,
        Remedy.NEW_INSTRUMENT,
        0,
        0.0,
    ),
    _card(
        "debit_instrument_blocked",
        "The card was blocked by the customer or their bank.",
        DeclineClass.NEVER_RETRYABLE,
        Remedy.CUSTOMER_ACTION,
        0,
        0.0,
    ),
    _card(
        "debit_instrument_inactive",
        "The card is not activated for online transactions.",
        DeclineClass.NEVER_RETRYABLE,
        Remedy.CUSTOMER_ACTION,
        0,
        0.0,
    ),
    _card(
        "card_not_enrolled",
        "The card is not enrolled for online transactions. The customer must enable it "
        "via their banking app or portal.",
        DeclineClass.NEVER_RETRYABLE,
        Remedy.CUSTOMER_ACTION,
        0,
        0.0,
    ),
    _card(
        "card_disabled_for_online_payments",
        "The card is not enabled for online use. The customer must enable it via Card Control.",
        DeclineClass.NEVER_RETRYABLE,
        Remedy.CUSTOMER_ACTION,
        0,
        0.0,
    ),
    _upi(
        "invalid_vpa",
        "The customer is not a valid user on the UPI app. VPA registration with their "
        "bank must be completed first.",
        DeclineClass.NEVER_RETRYABLE,
        Remedy.CUSTOMER_ACTION,
        0,
        0.0,
    ),
    # --- Rail conditional ------------------------------------------------------------
    # Retryable, but only once the rail recovers. Razorpay's own documentation points at
    # the Downtime API for exactly this case. See DECISIONS.md ADR-006.
    _shared(
        "bank_technical_error",
        "The customer's bank (or the UPI provider) was experiencing downtime.",
        DeclineClass.RAIL_CONDITIONAL,
        Remedy.AWAIT_RAIL_RECOVERY,
        4,
        0.5,
    ),
    _shared(
        "gateway_technical_error",
        "A partner bank or gateway downtime caused the failure.",
        DeclineClass.RAIL_CONDITIONAL,
        Remedy.AWAIT_RAIL_RECOVERY,
        4,
        0.5,
    ),
    # --- Time conditional ------------------------------------------------------------
    # The real sequential decision problem. Success probability is a strong function of
    # when the retry lands, which is what the bandit is for.
    _shared(
        "insufficient_funds",
        "The customer's account did not have enough funds at the time of the attempt.",
        DeclineClass.TIME_CONDITIONAL,
        Remedy.RETRY_LATER,
        4,
        6.0,
    ),
    _card(
        "transaction_limit_exceeded",
        "The card's transaction limit was reached. Limits typically reset on a daily "
        "cycle, which makes this recoverable purely by waiting.",
        DeclineClass.TIME_CONDITIONAL,
        Remedy.RETRY_LATER,
        3,
        12.0,
    ),
    _upi(
        "payment_declined",
        "The funds could not be debited. Razorpay's documentation recommends retrying.",
        DeclineClass.TIME_CONDITIONAL,
        Remedy.RETRY_LATER,
        3,
        6.0,
    ),
    # --- Session conditional ---------------------------------------------------------
    # A silent retry cannot fix these: they need the customer back in a live session.
    # Treating them as retryable is a common and expensive mistake, because the retry
    # consumes budget while the only thing that could work is outreach.
    _shared(
        "payment_timed_out",
        "The customer exceeded the payment processing window.",
        DeclineClass.SESSION_CONDITIONAL,
        Remedy.CUSTOMER_SESSION,
        2,
        1.0,
    ),
    _shared(
        "payment_cancelled",
        "The customer cancelled the transaction or navigated away during processing.",
        DeclineClass.SESSION_CONDITIONAL,
        Remedy.CUSTOMER_SESSION,
        2,
        1.0,
    ),
    _card(
        "authentication_failed",
        "The customer entered an incorrect OTP, or closed the browser during authentication.",
        DeclineClass.SESSION_CONDITIONAL,
        Remedy.CUSTOMER_SESSION,
        2,
        1.0,
    ),
    _card(
        "incorrect_cvv",
        "An incorrect CVV was entered.",
        DeclineClass.SESSION_CONDITIONAL,
        Remedy.CUSTOMER_SESSION,
        2,
        1.0,
    ),
    _upi(
        "payment_collect_request_expired",
        "The customer did not act on the collect request within the time limit.",
        DeclineClass.SESSION_CONDITIONAL,
        Remedy.CUSTOMER_SESSION,
        2,
        1.0,
    ),
    # --- Hard declined ---------------------------------------------------------------
    # The issuer refused for reasons we cannot observe. Retry value is low, and in the
    # risk-flagged case repeated attempts are actively counterproductive.
    _card(
        "card_declined",
        "The bank declined the transaction without a more specific reason.",
        DeclineClass.HARD_DECLINED,
        Remedy.NEW_INSTRUMENT,
        1,
        24.0,
    ),
    _card(
        "payment_failed",
        "The bank declined the payment.",
        DeclineClass.HARD_DECLINED,
        Remedy.NEW_INSTRUMENT,
        1,
        24.0,
    ),
    _card(
        "payment_risk_check_failed",
        "The bank flagged the transaction as potentially fraudulent. Repeated attempts "
        "risk hardening the issuer's stance and are not retried automatically.",
        DeclineClass.HARD_DECLINED,
        Remedy.ESCALATE,
        0,
        0.0,
    ),
    _upi(
        "vpa_resolution_failed",
        "The transaction could not be processed using the customer's UPI ID. Razorpay's "
        "documentation directs this to a support ticket.",
        DeclineClass.HARD_DECLINED,
        Remedy.ESCALATE,
        1,
        24.0,
    ),
    _upi(
        "credit_failed",
        "The credit leg of the UPI transaction failed. Routed to reconciliation rather "
        "than retried, since the debit state is ambiguous and a blind retry risks a "
        "double charge.",
        DeclineClass.HARD_DECLINED,
        Remedy.ESCALATE,
        0,
        0.0,
    ),
)


# ``payment_risk_check_failed`` and ``credit_failed`` are HARD_DECLINED rather than
# NEVER_RETRYABLE but still carry max_attempts == 0. That is intentional: the class
# describes *why* recovery is hard, while the cap encodes a policy choice not to retry.
# The __post_init__ invariant only constrains NEVER_RETRYABLE in one direction, so a
# zero cap on another class is permitted.


UNVERIFIED_REASONS: tuple[str, ...] = ("Customer Bank Account Mismatch",)
"""Decline reasons known to exist in Razorpay's documentation whose exact machine-readable
code string has not yet been verified verbatim.

They are deliberately kept out of the registry rather than guessed at: an incorrect code
string in a lookup table is worse than an absent one, because it fails silently. Day 3 of
docs/PLAN.md resolves these against the live documentation.
"""


_BY_CODE: dict[str, DeclineReason] = {r.code: r for r in _REASONS}

UNKNOWN_REASON = DeclineReason(
    code="__unknown__",
    methods=frozenset({PaymentMethod.CARD, PaymentMethod.UPI}),
    description="Decline code not present in the taxonomy.",
    decline_class=DeclineClass.HARD_DECLINED,
    remedy=Remedy.ESCALATE,
    max_attempts=1,
    min_backoff_hours=24.0,
    source="n/a",
)
"""Conservative fallback for codes the taxonomy does not recognise.

Payment processors add and rename decline codes, so an unrecognised code is a routine
event rather than an exceptional one. The system must degrade safely rather than crash or
guess: one cautious attempt after a long backoff, then human escalation. Choosing the
permissive default here — treating unknown as freely retryable — would mean every new
code Razorpay ships silently becomes a retry storm.
"""


def lookup(code: str) -> DeclineReason:
    """Return the taxonomy entry for ``code``, or the conservative unknown fallback.

    Never raises. An unrecognised decline code is an expected operational condition,
    not a programming error.
    """
    return _BY_CODE.get(code, UNKNOWN_REASON)


def is_known(code: str) -> bool:
    """Whether ``code`` is present in the taxonomy."""
    return code in _BY_CODE


def is_retryable(code: str) -> bool:
    """Whether an automated retry of ``code`` could ever succeed."""
    return lookup(code).is_retryable


def max_attempts_for(code: str) -> int:
    """Structural cap on total automated attempts for ``code``."""
    return lookup(code).max_attempts


def all_reasons() -> tuple[DeclineReason, ...]:
    """Every decline reason in the taxonomy."""
    return _REASONS


def by_class(decline_class: DeclineClass) -> tuple[DeclineReason, ...]:
    """Every decline reason in the given class."""
    return tuple(r for r in _REASONS if r.decline_class is decline_class)


def by_method(method: PaymentMethod) -> tuple[DeclineReason, ...]:
    """Every decline reason documented for the given payment method."""
    return tuple(r for r in _REASONS if method in r.methods)
