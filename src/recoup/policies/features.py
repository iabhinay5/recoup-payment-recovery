"""Context features for the learned policy.

Every feature here must be observable by a production system at decision time. That
constraint is doing real work: it excludes the customer's true balance, their salary date,
and the outage schedule, all of which sit in the simulator and any of which would produce a
policy that scores well and cannot be deployed.

Day of month is included and is *not* private data — it is the calendar. The salary-cycle
effect it exposes is a population-level regularity confirmed in NPCI's published daily
statistics (+19.7% early-month ticket size), not a fact about any individual customer.
"""

from __future__ import annotations

import math

import numpy as np

from recoup.sim.episode import EpisodeState
from recoup.taxonomy import DeclineClass, lookup

__all__ = ["FEATURE_NAMES", "N_FEATURES", "extract_features"]

_CLASSES = (
    DeclineClass.NEVER_RETRYABLE,
    DeclineClass.RAIL_CONDITIONAL,
    DeclineClass.TIME_CONDITIONAL,
    DeclineClass.SESSION_CONDITIONAL,
    DeclineClass.HARD_DECLINED,
)

FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    *(f"class_{c.value}" for c in _CLASSES),
    "attempt_count",
    "contact_count",
    "elapsed_days",
    "days_to_month_start",
    "is_early_month",
    "hour_sin",
    "hour_cos",
    "rail_degraded",
    "in_session",
    "log_amount",
    "has_alternative_instrument",
    "remaining_attempts",
)

N_FEATURES = len(FEATURE_NAMES)

# Reference amount for scaling, in paise. Roughly the median simulated failed payment;
# only the scale matters, since the feature is a log ratio.
_REFERENCE_AMOUNT_PAISE = 350_000.0


def extract_features(state: EpisodeState) -> np.ndarray:
    """Build the context vector a bandit sees at one decision point."""
    reason = lookup(state.current_decline_code)
    day = state.day_of_month
    hour = state.hour_of_day

    # Distance to the next month start, which is where the salary credit lands. Wrapped,
    # so day 27 is three days away rather than twenty-four.
    days_to_start = min((day - 1) % 28, (1 - day) % 28)

    features = np.zeros(N_FEATURES, dtype=float)
    idx = 0

    features[idx] = 1.0  # bias
    idx += 1

    for decline_class in _CLASSES:
        features[idx] = 1.0 if reason.decline_class is decline_class else 0.0
        idx += 1

    features[idx] = min(state.attempt_count, 5) / 5.0
    idx += 1
    features[idx] = min(state.contact_count, 5) / 5.0
    idx += 1
    features[idx] = min(state.elapsed_hours / 24.0, 14.0) / 14.0
    idx += 1
    features[idx] = days_to_start / 14.0
    idx += 1
    features[idx] = 1.0 if day <= 7 else 0.0
    idx += 1

    # Hour of day as a circle, so 23:00 and 01:00 are near each other rather than at
    # opposite ends of a line. Quiet hours are a real constraint on outreach.
    features[idx] = math.sin(2 * math.pi * hour / 24.0)
    idx += 1
    features[idx] = math.cos(2 * math.pi * hour / 24.0)
    idx += 1

    features[idx] = 1.0 if state.rail_is_degraded() else 0.0
    idx += 1
    features[idx] = 1.0 if state.in_session else 0.0
    idx += 1
    features[idx] = math.log1p(state.payment.amount_paise / _REFERENCE_AMOUNT_PAISE)
    idx += 1
    features[idx] = 1.0 if state.customer.alternatives_to(state.current_instrument_id) else 0.0
    idx += 1
    features[idx] = min(state.remaining_attempts, 4) / 4.0

    return features
