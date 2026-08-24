"""Bank and rail health over time.

The single most wasteful thing a dunning system can do is retry into a bank that is
currently down. The attempt cannot succeed, it consumes a capped retry budget, and it adds
load to a rail that is already failing. Razorpay's own error documentation points
integrators at their Downtime API for exactly this reason — see docs/DECISIONS.md ADR-006.

Modelling that requires bank health to be *time-varying*, not a constant per bank. A
policy is only rewarded for consulting rail health if rail health actually changes.

Two components:

- a steady-state technical decline rate per bank (NPCI publishes this monthly, T1)
- an outage process that drives health to near zero for a period (SWEPT — NPCI publishes
  decline rates but not outage durations)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Outage", "RailHealth", "generate_outages"]


@dataclass(frozen=True, slots=True)
class Outage:
    """A period during which a bank is degraded."""

    bank_id: str
    start_hours: float
    duration_hours: float

    @property
    def end_hours(self) -> float:
        return self.start_hours + self.duration_hours

    def covers(self, hours: float) -> bool:
        return self.start_hours <= hours < self.end_hours


def generate_outages(
    banks: dict[str, tuple[float, float]],
    horizon_hours: float,
    rng: np.random.Generator,
) -> tuple[Outage, ...]:
    """Sample outages as a Poisson process per bank, with exponential durations.

    A Poisson arrival process is the standard model for independent failures over time,
    and it produces the property that matters here: outages are unpredictable in timing,
    so a policy cannot learn a schedule and must actually consult health at decision time.

    Rates and durations are per bank, taken from NPCI's published incident data. Sharing
    one rate across all banks would erase the 17x dispersion that makes knowing *which*
    bank you are retrying into worth anything.
    """
    outages: list[Outage] = []
    horizon_days = horizon_hours / 24.0

    for bank_id, (rate_per_day, mean_hours) in banks.items():
        n = int(rng.poisson(rate_per_day * horizon_days))
        for _ in range(n):
            start = float(rng.uniform(0.0, horizon_hours))
            duration = float(rng.exponential(mean_hours))
            outages.append(Outage(bank_id, start, duration))

    return tuple(sorted(outages, key=lambda o: (o.bank_id, o.start_hours)))


class RailHealth:
    """Queries the health of a bank at a point in simulated time.

    This is the interface the policy sees, and it deliberately mirrors what a real
    Downtime API offers: a health score now, not a schedule of past and future outages.
    A policy that could see the outage list would be able to plan around outages it has
    no way of knowing about in production, and any advantage it showed would be fiction.
    """

    def __init__(
        self,
        banks: dict[str, float],
        outages: tuple[Outage, ...] = (),
        degraded_health: float = 0.05,
    ) -> None:
        """
        Args:
            banks: bank id to steady-state technical decline rate.
            outages: sampled degraded periods.
            degraded_health: health during an outage. Not zero, because real outages are
                partial — some traffic still succeeds, which is precisely why they are
                hard to detect from a single failed transaction.
        """
        self._banks = banks
        self._degraded = degraded_health
        self._by_bank: dict[str, list[Outage]] = {}
        for outage in outages:
            self._by_bank.setdefault(outage.bank_id, []).append(outage)

    def health_at(self, bank_id: str, hours: float) -> float:
        """Probability that the rail itself will not fail an attempt right now.

        Returns a value in [0, 1]. An unknown bank is treated as healthy rather than
        raising: a missing health signal must degrade to normal behaviour, not to a
        crash in the retry path.
        """
        base = 1.0 - self._banks.get(bank_id, 0.0)
        for outage in self._by_bank.get(bank_id, ()):
            if outage.covers(hours):
                return self._degraded
        return base

    def is_degraded(self, bank_id: str, hours: float, threshold: float = 0.5) -> bool:
        """Whether the rail is currently degraded enough that retrying is wasteful."""
        return self.health_at(bank_id, hours) < threshold

    def hours_until_healthy(
        self, bank_id: str, hours: float, max_lookahead: float = 24.0
    ) -> float | None:
        """How long until the current outage ends, if one is active.

        Returns ``None`` when the rail is already healthy.

        A production Downtime API reports current state, not a recovery estimate, so a
        policy must not depend on this being exact. It is exposed so the *simulator* can
        model a realistic "check back later" pattern, and any policy using it is
        restricted to treating it as a hint. See the note in ``RailHealth``.
        """
        for outage in self._by_bank.get(bank_id, ()):
            if outage.covers(hours):
                remaining = outage.end_hours - hours
                return min(remaining, max_lookahead)
        return None

    def mean_health(self, bank_id: str, start: float, end: float, samples: int = 24) -> float:
        """Average health over a window. Used for reporting, not by policies."""
        if end <= start:
            return self.health_at(bank_id, start)
        step = (end - start) / samples
        return (
            math.fsum(self.health_at(bank_id, start + i * step) for i in range(samples)) / samples
        )

    def with_outage(self, outage: Outage) -> RailHealth:
        """Return a copy with one additional outage.

        Used to make an episode's rail state consistent with the failure that opened it:
        a payment that failed with ``bank_technical_error`` must have failed *during* an
        outage, otherwise the episode contradicts its own premise and a rail-aware policy
        is being asked to react to a problem that was never there.

        Episode-local rather than global because every episode measures time from its own
        failure, so one shared outage timeline cannot be simultaneously consistent with
        all of them.
        """
        merged = tuple(o for outages in self._by_bank.values() for o in outages) + (outage,)
        return RailHealth(dict(self._banks), merged, self._degraded)
