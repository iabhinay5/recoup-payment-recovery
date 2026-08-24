"""Tests for the NPCI downtime loader.

The tests that matter here are about *reading the data correctly*, not about parsing
spreadsheets. Two mistakes in this file would each silently move a headline number by
orders of magnitude, and neither would raise: counting a bank twice because its name was
typed differently, and dividing by the wrong denominator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recoup.data.npci import (
    DowntimeRecord,
    banks_from_downtime,
    calibrate_from_downtime,
    load_downtime,
    normalise_bank,
)

DATA_DIR = Path("data/npci/downtime")
has_data = pytest.mark.skipif(
    not DATA_DIR.exists() or not list(DATA_DIR.glob("*.xlsx")),
    reason="NPCI spreadsheets not present in this checkout",
)


class TestBankNameNormalisation:
    """NPCI's files are typed by hand and the same bank appears spelled several ways."""

    def test_case_variants_collapse(self) -> None:
        assert normalise_bank("Punjab and Sind Bank") == normalise_bank("Punjab And Sind Bank")

    def test_company_suffixes_collapse(self) -> None:
        assert normalise_bank("Airtel Payments Bank Ltd") == normalise_bank(
            "Airtel Payments Bank Limited"
        )

    def test_whitespace_collapses(self) -> None:
        assert normalise_bank("  State   Bank  of India ") == normalise_bank("State Bank of India")

    def test_distinct_banks_stay_distinct(self) -> None:
        """Over-merging would be as wrong as under-merging."""
        assert normalise_bank("Bank of India") != normalise_bank("Bank of Baroda")
        assert normalise_bank("Indian Bank") != normalise_bank("Indian Overseas Bank")


class TestCalibrationArithmetic:
    """Constructed records, so the expected answer is known exactly."""

    def _records(self) -> tuple[DowntimeRecord, ...]:
        # January 2026 has 31 days.
        return (
            DowntimeRecord(2026, 1, "Bank A", incidents=2, hours=4.0),
            DowntimeRecord(2026, 1, "Bank B", incidents=3, hours=3.0),
        )

    def test_mean_is_weighted_by_incident_count(self) -> None:
        """5 incidents, 7 hours. Averaging the two per-bank means would give 1.5, which
        would over-weight the bank that failed less often."""
        cal = calibrate_from_downtime(self._records(), banks_live=100)
        assert cal.mean_hours_per_incident == pytest.approx(7.0 / 5.0)

    def test_rate_uses_all_live_banks_not_just_affected_ones(self) -> None:
        """The denominator that matters.

        Banks with no incidents are absent from the file rather than present with zero.
        Dividing by the two banks that appear instead of the full population overstates
        the outage rate by the ratio between them — here, fiftyfold.
        """
        cal = calibrate_from_downtime(self._records(), banks_live=100)
        assert cal.outage_rate_per_bank_day == pytest.approx(5 / (31 * 100))

        wrong = calibrate_from_downtime(self._records(), banks_live=2)
        assert wrong.outage_rate_per_bank_day == pytest.approx(5 / (31 * 2))
        assert wrong.outage_rate_per_bank_day / cal.outage_rate_per_bank_day == pytest.approx(50)

    def test_standard_error_shrinks_with_sample_size(self) -> None:
        small = calibrate_from_downtime((DowntimeRecord(2026, 1, "A", 4, 8.0),), banks_live=10)
        large = calibrate_from_downtime((DowntimeRecord(2026, 1, "A", 400, 800.0),), banks_live=10)
        assert large.relative_standard_error < small.relative_standard_error

    def test_empty_records_raise(self) -> None:
        with pytest.raises(ValueError, match="no downtime records"):
            calibrate_from_downtime(())


@has_data
class TestRealData:
    def test_loads_every_month(self) -> None:
        records = load_downtime(DATA_DIR)
        months = {(r.year, r.month) for r in records}
        assert len(months) == len(list(DATA_DIR.glob("*.xlsx")))

    def test_every_record_is_plausible(self) -> None:
        for record in load_downtime(DATA_DIR):
            assert record.incidents > 0
            assert record.hours > 0
            assert 1 <= record.month <= 12
            # A single reportable incident lasting over a week would indicate a parsing
            # error rather than an outage.
            assert record.hours_per_incident < 168

    def test_banks_carry_distinct_outage_profiles(self) -> None:
        """If every bank behaved identically, rail-awareness would be worthless."""
        banks = banks_from_downtime(load_downtime(DATA_DIR))
        assert len(banks) >= 5
        durations = {b.mean_outage_hours for b in banks}
        assert len(durations) == len(banks), "banks should differ in outage duration"
        assert max(durations) / min(durations) > 2.0, "expected real dispersion across banks"

    def test_selected_banks_are_higher_risk_than_the_pooled_average(self) -> None:
        """The volume-weighting correction.

        The pooled rate is diluted by hundreds of tiny cooperative banks. The banks a
        merchant's customers actually use fail far more often than that average, and
        modelling the diluted figure would understate rail risk badly.
        """
        records = load_downtime(DATA_DIR)
        pooled = calibrate_from_downtime(records).outage_rate_per_bank_day
        banks = banks_from_downtime(records)
        assert all(b.outage_rate_per_day > pooled * 10 for b in banks)


class TestParameterAudit:
    """The audit is a guarantee, not a report. It must fail when it should."""

    def test_audit_passes_on_the_current_parameter_set(self) -> None:
        from scripts.audit_parameters import main

        assert main() == 0

    def test_every_invented_parameter_is_swept(self) -> None:
        """The property the audit exists to enforce.

        An invented parameter left out of the sweep can silently carry a headline result,
        which is exactly the failure mode simulation-based work is accused of.
        """
        from scripts.audit_parameters import LEDGER, NOT_EMPIRICAL

        from recoup.sim.params import SWEPT_PARAMETERS, SimParams

        for name in SimParams.__dataclass_fields__:
            prov = LEDGER.get(name)
            assert prov is not None, f"{name} has no provenance entry"
            if prov.tier == "INVENTED" and name not in NOT_EMPIRICAL:
                assert name in SWEPT_PARAMETERS, f"{name} is invented but never swept"

    def test_the_fitted_parameter_is_swept_widest(self) -> None:
        """shortfall_high was tuned to hit 58%, so it carries the most risk."""
        from recoup.sim.params import SWEPT_PARAMETERS

        fitted = SWEPT_PARAMETERS["shortfall_high"]
        assert fitted.high / fitted.low >= 2.0, "the fitted parameter needs a wide sweep"
        assert "FITTED" in fitted.note
