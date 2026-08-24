"""The objects a recovery episode is made of.

Time is measured in **hours since the payment first failed**, not wall-clock dates. Every
policy decision is about elapsed time and time-of-day, never about a calendar date, and
carrying real datetimes through the simulator would add timezone handling and leap-year
edge cases in exchange for nothing. An absolute reference instant is kept on the payment
so that time-of-day and day-of-month effects (quiet hours, salary credits) still resolve
correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from recoup.taxonomy import PaymentMethod

__all__ = [
    "Attempt",
    "Bank",
    "Contact",
    "ContactChannel",
    "Customer",
    "FailedPayment",
    "Instrument",
]

HOURS_PER_DAY = 24.0


@dataclass(frozen=True, slots=True)
class Bank:
    """An issuing bank or UPI handle provider.

    ``base_technical_decline_rate`` is the bank's steady-state technical decline rate.
    NPCI publishes this per bank per month, and the dispersion across banks is the point:
    a policy that knows which rail it is retrying into can behave differently from one
    that does not. See docs/CALIBRATION.md section 2.
    """

    id: str
    name: str
    base_technical_decline_rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_technical_decline_rate <= 1.0:
            raise ValueError(f"{self.id}: technical decline rate must be a probability")


@dataclass(frozen=True, slots=True)
class Instrument:
    """A card or UPI handle a customer can pay with."""

    id: str
    method: PaymentMethod
    bank_id: str
    expires_at_hours: float | None = None
    """Hours from the episode reference instant at which this card expires.

    ``None`` means it does not expire within the horizon, which is always true for UPI.
    An instrument that expires mid-episode is the reason ``card_expired`` has to be
    handled as a state transition rather than a fixed property of the payment: a retry
    scheduled for day 5 can fail for a reason the original attempt could not have.
    """

    @property
    def can_expire(self) -> bool:
        return self.expires_at_hours is not None

    def is_expired_at(self, hours: float) -> bool:
        return self.expires_at_hours is not None and hours >= self.expires_at_hours


class ContactChannel(Enum):
    """How the customer is reached. Ordered loosely by intrusiveness."""

    EMAIL = "email"
    SMS = "sms"
    WHATSAPP = "whatsapp"

    @property
    def intrusiveness(self) -> float:
        """Relative contribution to contact fatigue.

        Email is cheap and easy to ignore; SMS and WhatsApp land on a lock screen and
        wear out their welcome faster. This is what stops the optimal policy from simply
        messaging everyone on every channel.
        """
        return {"email": 0.6, "sms": 1.0, "whatsapp": 1.2}[self.value]


@dataclass(frozen=True, slots=True)
class Contact:
    """One outreach event."""

    at_hours: float
    channel: ContactChannel


@dataclass(frozen=True, slots=True)
class Attempt:
    """One charge attempt and how it resolved."""

    at_hours: float
    instrument_id: str
    succeeded: bool
    decline_code: str | None = None
    """The decline reason, when the attempt failed. ``None`` on success.

    A retry can fail for a *different* reason than the original: a card that was merely
    short of funds in week one may be expired by week two. Policies that assume the
    decline reason is fixed for the life of an episode get this wrong.
    """

    def __post_init__(self) -> None:
        if self.succeeded and self.decline_code is not None:
            raise ValueError("a successful attempt cannot carry a decline code")
        if not self.succeeded and self.decline_code is None:
            raise ValueError("a failed attempt must carry a decline code")


@dataclass(frozen=True, slots=True)
class Customer:
    """A paying customer, with the state that determines whether recovery can succeed.

    The balance model is what makes ``insufficient_funds`` recoverable by *waiting* rather
    than by trying harder. Income arrives on a monthly cycle and is drawn down through the
    month, so a retry landing shortly after a salary credit meets a very different balance
    than one landing three weeks later.
    """

    id: str
    instruments: tuple[Instrument, ...]
    monthly_income_paise: int
    salary_day_of_month: int
    contacts: tuple[Contact, ...] = field(default_factory=tuple)
    opted_out: bool = False

    def __post_init__(self) -> None:
        if not self.instruments:
            raise ValueError(f"{self.id}: a customer needs at least one instrument")
        if not 1 <= self.salary_day_of_month <= 28:
            raise ValueError(f"{self.id}: salary_day_of_month must be in 1..28")

    @property
    def primary_instrument(self) -> Instrument:
        return self.instruments[0]

    def alternatives_to(self, instrument_id: str) -> tuple[Instrument, ...]:
        """Other instruments this customer could pay with.

        Instrument switching is the only remedy for a genuinely dead card, so a policy
        that cannot see alternatives cannot recover ``card_expired`` at all.
        """
        return tuple(i for i in self.instruments if i.id != instrument_id)

    def contacts_within(self, hours: float, window_hours: float) -> int:
        """How many times this customer was contacted in the preceding window."""
        return sum(1 for c in self.contacts if hours - window_hours <= c.at_hours <= hours)


@dataclass(frozen=True, slots=True)
class FailedPayment:
    """A payment that failed and is now a candidate for recovery.

    This is the unit of work: one of these arrives, and the policy decides what happens
    to it over the following days.
    """

    id: str
    customer_id: str
    instrument_id: str
    amount_paise: int
    initial_decline_code: str
    reference_day_of_month: int
    """Day of month on which the original failure occurred.

    Kept so that salary-cycle position can be computed from elapsed hours without
    carrying real dates through the simulation.
    """

    reference_hour_of_day: float = 12.0
    """Hour of day of the original failure, used for quiet-hours logic."""

    def __post_init__(self) -> None:
        if self.amount_paise <= 0:
            raise ValueError(f"{self.id}: amount must be positive")
        if not 1 <= self.reference_day_of_month <= 28:
            raise ValueError(f"{self.id}: reference_day_of_month must be in 1..28")

    def hour_of_day_at(self, elapsed_hours: float) -> float:
        """Local hour of day after ``elapsed_hours``, in [0, 24)."""
        return (self.reference_hour_of_day + elapsed_hours) % HOURS_PER_DAY

    def day_of_month_at(self, elapsed_hours: float) -> int:
        """Day of month after ``elapsed_hours``, wrapped to a 28-day cycle.

        A uniform 28-day month is a deliberate simplification: the effect being modelled
        is proximity to a salary credit, and real month lengths would add variation that
        obscures rather than sharpens that signal.
        """
        days_elapsed = int((self.reference_hour_of_day + elapsed_hours) // HOURS_PER_DAY)
        return ((self.reference_day_of_month - 1 + days_elapsed) % 28) + 1
