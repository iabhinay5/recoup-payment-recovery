"""Every simulator parameter, in one place, with its provenance.

The rule from docs/CALIBRATION.md is that nothing here is invented. Each field carries a
source tier in its comment:

    T1   verified directly against a primary source
    T2   secondary reporting of a primary study, to be upgraded where possible
    SWEPT no public source exists, so the parameter is varied across a plausible range
          and no headline claim is allowed to depend on any single value

``SWEPT`` is not an apology. It is the honest label for a quantity nobody publishes, and
sweeping it is what lets a result mean something anyway: if the ordering of policies holds
across the whole plausible region, that ordering is a finding rather than an artefact.
See docs/DECISIONS.md ADR-002.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = ["SimParams", "SweepRange"]


@dataclass(frozen=True, slots=True)
class SweepRange:
    """A parameter with no trustworthy point estimate.

    ``low`` and ``high`` bound what the value could plausibly be; ``default`` is used for
    single runs and is never treated as truth.
    """

    low: float
    high: float
    default: float
    note: str = ""

    def __post_init__(self) -> None:
        if not self.low <= self.default <= self.high:
            raise ValueError(
                f"default {self.default} is outside the sweep range [{self.low}, {self.high}]"
            )

    def steps(self, n: int = 5) -> tuple[float, ...]:
        """``n`` evenly spaced values across the range, inclusive of both endpoints."""
        if n < 2:
            return (self.default,)
        span = self.high - self.low
        return tuple(self.low + span * i / (n - 1) for i in range(n))


# Decline-reason mix, conditional on a payment having failed.
#
# T2, and SWEPT. Recurly's published mix is drawn from Western subscription businesses;
# the Indian rail mix genuinely differs, with UPI carrying a far larger share and
# different failure characteristics. This is the single most influential input to the
# simulator, so it is swept rather than trusted, and the discrepancy is disclosed in the
# README as a known limitation rather than smoothed over.
DEFAULT_DECLINE_MIX: dict[str, float] = {
    # Time-conditional. Recurly reports roughly half of failures are insufficient funds.
    "insufficient_funds": 0.42,
    "transaction_limit_exceeded": 0.04,
    "payment_declined": 0.06,
    # Rail-conditional. NPCI publishes technical decline rates; see rails.py.
    "bank_technical_error": 0.09,
    "gateway_technical_error": 0.03,
    # Session-conditional.
    "payment_timed_out": 0.07,
    "payment_cancelled": 0.06,
    "authentication_failed": 0.05,
    "incorrect_cvv": 0.02,
    "payment_collect_request_expired": 0.03,
    # Never retryable. Recurly puts card issues at 10-15% of failures.
    "card_expired": 0.05,
    "debit_instrument_blocked": 0.01,
    "debit_instrument_inactive": 0.01,
    "card_not_enrolled": 0.01,
    "card_disabled_for_online_payments": 0.01,
    "invalid_vpa": 0.01,
    # Hard declined.
    "card_declined": 0.02,
    "payment_failed": 0.01,
}


@dataclass(frozen=True, slots=True)
class SimParams:
    """Configuration for one simulation run.

    Immutable. ``with_overrides`` returns a modified copy, which keeps a sweep from
    accidentally mutating the baseline configuration it is varying against.
    """

    # --- Population ------------------------------------------------------------------
    n_customers: int = 10_000
    seed: int = 0

    # Transaction amounts, in rupees before conversion to paise. Log-normal is the
    # standard shape for payment amounts.
    #
    # T1 on the mean: NPCI's published monthly volume and value give an average UPI ticket
    # of Rs 1,296 over 16 months (Aug 2025 - Jul 2026), and amount_log_mean is set so the
    # distribution reproduces it. The previous invented value of 7.5 implied Rs 3,306 --
    # 2.6x too high, which inflated every revenue figure computed from it.
    #
    # SWEPT on the spread: NPCI publishes the mean but not the distribution, so sigma
    # remains an assumption.
    #
    # Stated limitation: this is the average across *all* UPI transactions, not across
    # *failed* ones. If failures skew large, this understates them. Anchoring to a
    # published mean is still better than an invented value off by 2.6x.
    amount_log_mean: float = 6.5620
    amount_log_sigma: float = 1.1

    # --- Decline mix -----------------------------------------------------------------
    decline_mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DECLINE_MIX))

    # --- Balance process (drives insufficient_funds recovery) ------------------------
    # SWEPT throughout. Nobody publishes the joint distribution of customer balance and
    # payment timing, and this is where the time-conditional recovery signal comes from.
    #
    # Salary credit timing is the one piece with a real-world anchor: Indian salaries
    # cluster at month end and the first working days of the month. That is why the
    # policy can learn to wait for a payday rather than retrying blindly.
    # The salary cycle is no longer an assumption. NPCI's daily statistics over 212 days
    # show average UPI ticket size peaking at index 121 on day 2 of the month and falling
    # to 86 by day 26 -- a +19.7% early-vs-late lift. The mechanism the time-conditional
    # recovery signal depends on is empirically present in real payment behaviour.
    salary_day_of_month: int = 1
    salary_day_jitter: int = 3

    # How far above the customer's available balance an insufficient-funds charge sat at
    # the moment it failed. CALIBRATED: the upper bound was fitted so that the published
    # Recurly Day-1/3/5/7 baseline reproduces at 58%. It is the one parameter in this file
    # tuned to hit a target rather than sampled from a source, and it is therefore the
    # first thing the sensitivity analysis varies.
    shortfall_low: float = 1.01
    shortfall_high: float = 1.5
    # Fraction of a customer's monthly income still available, by day of the salary
    # cycle. Balance falls through the month and is replenished at the credit.
    balance_floor_fraction: float = 0.05
    balance_depletion_rate: float = 0.9

    # --- Customer contact tolerance --------------------------------------------------
    # SWEPT. The cost side of the reward: contacting a customer too often produces
    # opt-outs and churn. Without this the optimal policy degenerates into contacting
    # everyone constantly, which is exactly the failure mode real dunning systems have.
    opt_out_hazard_per_contact: float = 0.04
    contact_fatigue_halflife_hours: float = 72.0
    # Probability a customer who receives outreach returns to complete a payment.
    outreach_response_rate: float = 0.28

    # --- Rail health (calibrated from NPCI data on day 3) ----------------------------
    # T1 source for the rates; SWEPT for outage duration, which NPCI does not publish.
    baseline_technical_decline_rate: float = 0.008
    outage_mean_duration_hours: float = 2.5
    outage_rate_per_bank_day: float = 0.05

    # --- Episode mechanics -----------------------------------------------------------
    horizon_days: int = 14
    """How long a recovery attempt stays open. Beyond this the payment is written off.

    Two weeks is a deliberate choice: Recurly's published schedule runs to day 7, so a
    14-day horizon gives any policy room to beat that baseline rather than being cut off
    at exactly the point the baseline stops.
    """

    def __post_init__(self) -> None:
        total = sum(self.decline_mix.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"decline_mix must sum to 1.0, got {total:.6f}. A mix that does not "
                f"normalise silently reweights every result computed from it."
            )
        if any(p < 0 for p in self.decline_mix.values()):
            raise ValueError("decline_mix probabilities must be non-negative")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if not 1 <= self.salary_day_of_month <= 28:
            raise ValueError("salary_day_of_month must be in 1..28 to exist in every month")

    def with_overrides(self, **kwargs: Any) -> SimParams:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)


# Sweep ranges for the parameters with no public source. Referenced by the sensitivity
# analysis on day 10; kept here so the honest-uncertainty surface lives next to the
# defaults it qualifies rather than in a separate file that can drift out of sync.
SWEPT_PARAMETERS: dict[str, SweepRange] = {
    "opt_out_hazard_per_contact": SweepRange(
        0.01,
        0.10,
        0.04,
        "How readily customers disengage after repeated contact. Drives the cost side "
        "of the reward, so the optimal contact frequency is highly sensitive to it.",
    ),
    "outreach_response_rate": SweepRange(
        0.15,
        0.45,
        0.28,
        "Fraction of contacted customers who return to complete payment. Bounds how "
        "much any outreach-based strategy can achieve.",
    ),
    "balance_depletion_rate": SweepRange(
        0.7,
        0.98,
        0.9,
        "How fast balances fall through the salary cycle. Determines how much signal "
        "there is in waiting for a payday.",
    ),
    "outage_mean_duration_hours": SweepRange(
        0.5,
        8.0,
        2.5,
        "NPCI publishes decline rates but not outage durations. Determines how long a "
        "rail-aware policy should defer.",
    ),
    "outage_rate_per_bank_day": SweepRange(
        0.01,
        0.15,
        0.05,
        "Frequency of bank outages. Together with duration this sets how much a "
        "rail-aware policy can win.",
    ),
    "shortfall_high": SweepRange(
        1.2,
        3.0,
        1.5,
        "How far a failed charge overshot the available balance. THE FITTED PARAMETER: "
        "its default was chosen so the published Recurly baseline reproduces at 58%. "
        "Because a headline number rests on it, it is swept the widest, and the "
        "calibration is re-checked at every point in the range.",
    ),
    "shortfall_low": SweepRange(
        1.001,
        1.1,
        1.01,
        "Lower bound of the same overshoot. Near-misses are common but unmeasured.",
    ),
    "amount_log_sigma": SweepRange(
        0.8,
        1.5,
        1.1,
        "Spread of payment amounts. NPCI publishes the mean ticket but not the shape, "
        "and the spread decides how much of the population sits near its balance limit.",
    ),
    "balance_floor_fraction": SweepRange(
        0.01,
        0.15,
        0.05,
        "Residual balance at the worst point of the salary cycle. Sets the floor on how "
        "badly a late-cycle retry can do.",
    ),
    "contact_fatigue_halflife_hours": SweepRange(
        24.0,
        168.0,
        72.0,
        "How quickly customers forget being contacted. Bounds how closely outreach can "
        "be spaced before it costs more than it recovers.",
    ),
    "salary_day_jitter": SweepRange(
        0.0,
        7.0,
        3.0,
        "Dispersion of salary credit dates across the population. At zero every customer "
        "is paid on the same day, which would make the cycle far easier to exploit than "
        "it is in reality.",
    ),
}
