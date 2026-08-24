"""Tests for the NPCI UPI volume/value loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from recoup.data.upi_stats import MonthlyStats, load_daily_cycle, load_monthly

MONTHLY_DIR = Path("data/npci/upi")
DAILY_DIR = Path("data/npci/upi-daily")

has_monthly = pytest.mark.skipif(not list(MONTHLY_DIR.glob("*.xlsx")), reason="no monthly files")
has_daily = pytest.mark.skipif(not list(DAILY_DIR.glob("*.xlsx")), reason="no daily files")


class TestTicketArithmetic:
    def test_unit_conversion(self) -> None:
        """Volume is in millions of transactions, value in crore rupees."""
        stats = MonthlyStats("Test", 700, volume_millions=100.0, value_crore=10_000.0)
        # 10,000 crore = 1e11 rupees over 1e8 transactions = Rs 1,000 each.
        assert stats.average_ticket_rupees == pytest.approx(1000.0)
        assert stats.average_ticket_paise == 100_000


@has_monthly
class TestMonthly:
    def test_ticket_matches_published_magnitude(self) -> None:
        """UPI is a low-value, high-volume rail: the average ticket is Rs 1,000-1,500."""
        tickets = [s.average_ticket_rupees for s in load_monthly(MONTHLY_DIR)]
        assert tickets
        mean = sum(tickets) / len(tickets)
        assert 1000 < mean < 1600, f"average ticket Rs {mean:.0f} is outside the plausible range"

    def test_overlapping_workbooks_are_deduplicated(self) -> None:
        """The 2025-26 and 2026-27 files share months; counting one twice would skew
        any average taken across them."""
        months = [s.label for s in load_monthly(MONTHLY_DIR)]
        assert len(months) == len(set(months))


@has_daily
class TestDailyCycle:
    def test_the_salary_cycle_is_present_in_real_data(self) -> None:
        """The mechanism the time-conditional recovery signal rests on.

        If this ever fails, the balance model has no empirical support and the
        time-conditional half of the thesis should be withdrawn rather than defended.
        """
        cycle = load_daily_cycle(DAILY_DIR)
        assert cycle.early_month_index > cycle.late_month_index
        assert cycle.salary_cycle_lift > 0.05, "expected a clear early-month lift"

    def test_index_is_normalised_around_100(self) -> None:
        cycle = load_daily_cycle(DAILY_DIR)
        values = list(cycle.ticket_index_by_day.values())
        assert 95 < sum(values) / len(values) < 105

    def test_implied_decay_is_a_lower_bound_not_a_measurement(self) -> None:
        """Spending is stickier than balance, so the observed decay understates the real
        one. The simulator's default must therefore be steeper, not equal."""
        from recoup.sim.params import SimParams

        implied = load_daily_cycle(DAILY_DIR).implied_depletion_rate()
        assert 0.9 < implied < 1.0
        assert SimParams().balance_depletion_rate < implied
