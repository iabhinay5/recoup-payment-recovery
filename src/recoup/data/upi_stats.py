"""NPCI UPI volume and value statistics, monthly and daily.

Source: https://npci.org.in/product/upi/product-statistics -> UPI and Daily Statistics tabs.
Raw files ship in ``data/npci/upi/`` and ``data/npci/upi-daily/``.

Two things come out of this that were previously invented:

**Average ticket size.** Monthly volume and value give the mean UPI transaction value
directly, which anchors the payment amount distribution.

**Empirical confirmation of the salary cycle.** This is the more important one. The balance
mechanism in ``sim/outcomes.py`` assumes Indian consumers have more spending capacity just
after a month-start salary credit and less by month end — and the entire time-conditional
recovery signal rests on that assumption. It was, until this data, pure invention.

Aggregating daily average ticket size by day of month across 212 days shows the pattern is
real: value peaks around day 2 and declines through the month. That does not make the
balance model correct in detail, but it moves the *existence* of the effect from assumed to
observed, which is the part that matters for whether waiting for a payday can help at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "DailyCycle",
    "MonthlyStats",
    "load_daily_cycle",
    "load_monthly",
]

DEFAULT_MONTHLY_DIR = Path("data/npci/upi")
DEFAULT_DAILY_DIR = Path("data/npci/upi-daily")

# NPCI reports volume in millions of transactions and value in crore rupees.
_MILLION = 1e6
_CRORE = 1e7


def _num(value: object) -> float:
    """Parse an NPCI figure. They are written with Indian digit grouping."""
    return float(str(value).replace(",", "").strip())


@dataclass(frozen=True, slots=True)
class MonthlyStats:
    """Aggregate UPI statistics for one month."""

    label: str
    banks_live: int
    volume_millions: float
    value_crore: float

    @property
    def average_ticket_rupees(self) -> float:
        return (self.value_crore * _CRORE) / (self.volume_millions * _MILLION)

    @property
    def average_ticket_paise(self) -> int:
        return int(round(self.average_ticket_rupees * 100))


def load_monthly(directory: Path | str = DEFAULT_MONTHLY_DIR) -> tuple[MonthlyStats, ...]:
    """Load monthly UPI volume/value files."""
    directory = Path(directory)
    seen: dict[str, MonthlyStats] = {}

    for path in sorted(directory.glob("*.xlsx")):
        frame = pd.read_excel(path, header=0)
        if frame.shape[1] < 4:
            continue
        frame.columns = ["month", "banks", "volume", "value"] + list(frame.columns[4:])
        for _, row in frame.dropna(subset=["month"]).iterrows():
            label = str(row["month"]).strip()
            try:
                stats = MonthlyStats(
                    label=label,
                    banks_live=int(_num(row["banks"])),
                    volume_millions=_num(row["volume"]),
                    value_crore=_num(row["value"]),
                )
            except (TypeError, ValueError):
                continue
            # The 2025-26 and 2026-27 workbooks overlap on shared months.
            seen.setdefault(label, stats)

    return tuple(seen.values())


@dataclass(frozen=True, slots=True)
class DailyCycle:
    """Day-of-month pattern in UPI transaction value, aggregated across months."""

    ticket_index_by_day: dict[int, float]
    """Average ticket size by day of month, indexed to 100 = period mean."""

    volume_index_by_day: dict[int, float]
    n_days: int
    n_months: int

    def _mean_index(self, days: range) -> float:
        values = [self.ticket_index_by_day[d] for d in days if d in self.ticket_index_by_day]
        return sum(values) / len(values) if values else 0.0

    @property
    def early_month_index(self) -> float:
        """Mean ticket index over days 1-7, the post-salary window."""
        return self._mean_index(range(1, 8))

    @property
    def late_month_index(self) -> float:
        """Mean ticket index over days 21-28."""
        return self._mean_index(range(21, 29))

    @property
    def salary_cycle_lift(self) -> float:
        """Fractional lift in average ticket, early month vs late month.

        The empirical size of the salary effect. Positive means consumers transact for
        larger amounts just after month start, which is what makes waiting for a payday a
        real strategy rather than a guess.
        """
        late = self.late_month_index
        return (self.early_month_index / late - 1.0) if late else 0.0

    def implied_depletion_rate(self) -> float:
        """Daily decay rate implied by the observed early-to-late decline.

        **Read this as a lower bound on the balance effect, not a measurement of it.**
        Average ticket size measures what people *spend*, and spending is stickier than
        available balance: people defer purchases rather than scaling every transaction
        down in proportion to their bank balance. The true balance decay is steeper than
        this figure, which is why ``balance_depletion_rate`` is swept upward from here
        rather than pinned to it.
        """
        early_day, late_day = 4.0, 24.5  # midpoints of the two windows
        ratio = self.late_month_index / self.early_month_index
        return float(ratio ** (1.0 / (late_day - early_day)))

    def summary(self) -> str:
        return "\n".join(
            (
                f"UPI daily cycle ({self.n_days} days, {self.n_months} months)",
                f"  days 1-7 ticket index      {self.early_month_index:.1f}",
                f"  days 21-28 ticket index    {self.late_month_index:.1f}",
                f"  salary-cycle lift          {self.salary_cycle_lift:+.1%}",
                f"  implied daily decay        {self.implied_depletion_rate():.4f} "
                f"(lower bound; see docstring)",
            )
        )


_DAY_RE = re.compile(r"(\d{1,2}),")


def load_daily_cycle(directory: Path | str = DEFAULT_DAILY_DIR) -> DailyCycle:
    """Aggregate daily UPI statistics into a day-of-month index."""
    directory = Path(directory)
    by_day: dict[int, list[tuple[float, float]]] = {}
    n_days = 0
    files = sorted(directory.glob("*.xlsx"))

    for path in files:
        frame = pd.read_excel(path, header=0)
        if frame.shape[1] < 3:
            continue
        frame.columns = ["day", "volume", "value"] + list(frame.columns[3:])
        for _, row in frame.dropna(subset=["day"]).iterrows():
            match = _DAY_RE.search(str(row["day"]))
            if not match:
                continue
            try:
                volume, value = _num(row["volume"]), _num(row["value"])
            except (TypeError, ValueError):
                continue
            if volume <= 0:
                continue
            ticket = (value * _CRORE) / (volume * _MILLION)
            by_day.setdefault(int(match.group(1)), []).append((volume, ticket))
            n_days += 1

    if not by_day:
        raise ValueError(f"no daily statistics found in {directory}")

    mean_ticket_by_day = {d: sum(t for _, t in rows) / len(rows) for d, rows in by_day.items()}
    mean_volume_by_day = {d: sum(v for v, _ in rows) / len(rows) for d, rows in by_day.items()}

    overall_ticket = sum(mean_ticket_by_day.values()) / len(mean_ticket_by_day)
    overall_volume = sum(mean_volume_by_day.values()) / len(mean_volume_by_day)

    return DailyCycle(
        ticket_index_by_day={d: v / overall_ticket * 100 for d, v in mean_ticket_by_day.items()},
        volume_index_by_day={d: v / overall_volume * 100 for d, v in mean_volume_by_day.items()},
        n_days=n_days,
        n_months=len(files),
    )
