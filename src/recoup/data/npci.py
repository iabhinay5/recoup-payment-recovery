"""Load NPCI's published UPI downtime data and derive simulator parameters from it.

Source: https://npci.org.in/product/upi/product-statistics -> Downtime/Incidents tab.
Downloaded per month; the raw files ship in ``data/npci/downtime/`` so the calibration is
reproducible by anyone who clones the repository.

Each file lists, for one month, every member bank that experienced at least one reportable
incident, with an incident count and total downtime. Banks with no incidents are **absent
rather than zero**, which is the single most important thing to get right when reading it:
treating the file as the population would compute an outage rate over only the banks that
failed, and overstate it by roughly two orders of magnitude.

This replaces two parameters that were previously invented. See docs/CALIBRATION.md
section 2.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recoup.sim.entities import Bank

__all__ = [
    "banks_from_downtime",
    "DowntimeCalibration",
    "normalise_bank",
    "DowntimeRecord",
    "calibrate_from_downtime",
    "load_downtime",
]

DEFAULT_DOWNTIME_DIR = Path("data/npci/downtime")

BANKS_LIVE_ON_UPI = 741
"""Member banks live on UPI, from the UPI tab of the same NPCI page (July 2026).

The denominator for the outage rate, and the number that decides whether the rate comes
out at 0.001 or 0.08 per bank-day. It drifted from about 720 to 741 over the loaded
period; the variation is far smaller than the uncertainty it feeds into.

**Stated limitation.** This counts every member bank equally, including several hundred
small cooperative banks carrying negligible volume. A volume-weighted denominator would
give a higher effective rate among the banks that actually process payments. The raw ratio
is therefore a lower bound on the outage rate a real merchant would experience.
"""

_MONTH_RE = re.compile(r"-([A-Z][a-z]+)-(\d{4})-", re.ASCII)
_MONTHS = {name: i for i, name in enumerate(calendar.month_name) if name}


def normalise_bank(name: str) -> str:
    """Canonical form of a bank name.

    NPCI's files are typed by hand and the same institution appears with different
    capitalisation across months — "Punjab and Sind Bank" and "Punjab And Sind Bank" are
    one bank, and counting them separately splits its incident history in half and halves
    its apparent failure rate. Suffix variants are folded for the same reason.
    """
    cleaned = " ".join(str(name).split()).strip().rstrip(".")
    for suffix in (" Limited", " Ltd", " Ltd."):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.title()


@dataclass(frozen=True, slots=True)
class DowntimeRecord:
    """One bank's reportable downtime in one month."""

    year: int
    month: int
    bank: str
    incidents: int
    hours: float

    @property
    def hours_per_incident(self) -> float:
        return self.hours / self.incidents if self.incidents else 0.0

    @property
    def days_in_month(self) -> int:
        return calendar.monthrange(self.year, self.month)[1]


def _parse_hours(value: object) -> float:
    """NPCI writes downtime as ``HH:MM``; pandas may hand it back as a time or a string."""
    if isinstance(value, str) and ":" in value:
        parts = value.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0
    if hasattr(value, "hour"):
        return float(value.hour) + float(value.minute) / 60.0  # type: ignore[attr-defined]
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _month_from_filename(path: Path) -> tuple[int, int]:
    match = _MONTH_RE.search(path.name)
    if not match or match.group(1) not in _MONTHS:
        raise ValueError(f"cannot read a month from {path.name!r}")
    return int(match.group(2)), _MONTHS[match.group(1)]


def load_downtime(directory: Path | str = DEFAULT_DOWNTIME_DIR) -> tuple[DowntimeRecord, ...]:
    """Load every monthly downtime spreadsheet in ``directory``."""
    directory = Path(directory)
    records: list[DowntimeRecord] = []

    for path in sorted(directory.glob("*.xlsx")):
        year, month = _month_from_filename(path)
        frame = pd.read_excel(path, header=0)
        frame.columns = [str(c).strip().lower() for c in frame.columns]

        bank_col = next((c for c in frame.columns if "bank" in c), None)
        inc_col = next((c for c in frame.columns if "incident" in c), None)
        down_col = next((c for c in frame.columns if "downtime" in c), None)
        if not (bank_col and inc_col and down_col):
            raise ValueError(f"{path.name}: unexpected columns {list(frame.columns)}")

        for _, row in frame.dropna(subset=[bank_col]).iterrows():
            try:
                incidents = int(row[inc_col])
            except (TypeError, ValueError):
                continue
            if incidents <= 0:
                continue
            records.append(
                DowntimeRecord(
                    year=year,
                    month=month,
                    bank=normalise_bank(str(row[bank_col])),
                    incidents=incidents,
                    hours=_parse_hours(row[down_col]),
                )
            )

    return tuple(records)


@dataclass(frozen=True, slots=True)
class DowntimeCalibration:
    """Simulator parameters derived from NPCI's published incident data."""

    mean_hours_per_incident: float
    median_hours_per_incident: float
    outage_rate_per_bank_day: float
    bank_hours_per_incident: dict[str, float]
    n_incidents: int
    n_months: int
    n_distinct_banks: int
    total_hours: float

    @property
    def relative_standard_error(self) -> float:
        """Approximate relative standard error of the mean duration.

        Exponential-ish durations give ``1/sqrt(n)``. Reported alongside the estimate so
        the precision is visible rather than implied.
        """
        return self.n_incidents**-0.5 if self.n_incidents else float("inf")

    def summary(self) -> str:
        return "\n".join(
            (
                f"NPCI downtime calibration ({self.n_months} months, {self.n_incidents} incidents)",
                f"  mean hours per incident      {self.mean_hours_per_incident:.2f} "
                f"(+/- {self.relative_standard_error:.1%})",
                f"  median hours per incident    {self.median_hours_per_incident:.2f}",
                f"  outage rate per bank-day     {self.outage_rate_per_bank_day:.5f}",
                f"  distinct banks affected      {self.n_distinct_banks}",
                f"  total downtime hours         {self.total_hours:.1f}",
            )
        )


def calibrate_from_downtime(
    records: tuple[DowntimeRecord, ...],
    banks_live: int = BANKS_LIVE_ON_UPI,
) -> DowntimeCalibration:
    """Derive outage duration and rate from loaded records."""
    if not records:
        raise ValueError("no downtime records; check the data directory")

    months = {(r.year, r.month) for r in records}
    total_incidents = sum(r.incidents for r in records)
    total_hours = sum(r.hours for r in records)

    # Weighted by incident count: a bank with four incidents contributes four
    # observations, not one. Averaging the per-bank averages would silently
    # over-weight banks that failed once.
    mean_hours = total_hours / total_incidents

    per_incident = sorted(r.hours_per_incident for r in records for _ in range(r.incidents))
    mid = len(per_incident) // 2
    median_hours = (
        per_incident[mid]
        if len(per_incident) % 2
        else (per_incident[mid - 1] + per_incident[mid]) / 2
    )

    # Bank-days observed across the whole period, over ALL live banks — not only the
    # banks that appear in the files. See the note on BANKS_LIVE_ON_UPI.
    bank_days = sum(
        calendar.monthrange(year, month)[1] * banks_live for year, month in sorted(months)
    )

    by_bank: dict[str, list[tuple[int, float]]] = {}
    for record in records:
        by_bank.setdefault(record.bank, []).append((record.incidents, record.hours))
    bank_hours = {
        bank: sum(h for _, h in rows) / sum(i for i, _ in rows)
        for bank, rows in by_bank.items()
        if sum(i for i, _ in rows) > 0
    }

    return DowntimeCalibration(
        mean_hours_per_incident=mean_hours,
        median_hours_per_incident=median_hours,
        outage_rate_per_bank_day=total_incidents / bank_days,
        bank_hours_per_incident=bank_hours,
        n_incidents=total_incidents,
        n_months=len(months),
        n_distinct_banks=len(by_bank),
        total_hours=total_hours,
    )


def banks_from_downtime(
    records: tuple[DowntimeRecord, ...],
    top_n: int = 10,
    base_technical_decline_rate: float = 0.008,
) -> tuple[Bank, ...]:
    """Build simulator banks from NPCI's incident data, each with its own outage profile.

    Selects the ``top_n`` banks by incident count. Those are India's large public-sector
    issuers — State Bank of India, Punjab National Bank, Central Bank of India, Bank of
    India — which carry both the most incidents and the most volume, so they are the banks
    a real merchant's customers actually hold accounts with.

    **This is the volume-weighting fix**, and it matters more than it looks. Pooled across
    all 741 member banks the outage rate is 0.00121 per bank-day, but that denominator is
    dominated by several hundred small cooperative banks processing almost nothing. State
    Bank of India's own measured rate is roughly eighty times higher. A merchant does not
    experience the average bank; it experiences the banks its customers use, and modelling
    the diluted rate would understate rail risk by two orders of magnitude.

    ``base_technical_decline_rate`` stays uniform at NPCI's system-wide figure. Reportable
    outages and steady-state technical declines are different phenomena, and NPCI publishes
    per-bank data only for the former.
    """
    from collections import defaultdict

    months = {(r.year, r.month) for r in records}
    observed_days = sum(calendar.monthrange(y, m)[1] for y, m in months)

    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for record in records:
        totals[record.bank][0] += record.incidents
        totals[record.bank][1] += record.hours

    ranked = sorted(totals.items(), key=lambda kv: -kv[1][0])[:top_n]

    return tuple(
        Bank(
            id=re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"),
            name=name,
            base_technical_decline_rate=base_technical_decline_rate,
            outage_rate_per_day=incidents / observed_days,
            mean_outage_hours=hours / incidents,
        )
        for name, (incidents, hours) in ranked
    )
