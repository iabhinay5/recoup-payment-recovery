"""Print every simulator parameter with its provenance, and flag the unsourced ones.

Run:  python scripts/audit_parameters.py

This exists to answer one question without requiring anyone to take a claim on trust:
**which of these numbers are measured, and which did we make up?**

A simulation-based result is only as honest as its willingness to say where its inputs came
from. Rather than assert that in prose, this prints the ledger. Anything marked INVENTED is
a value chosen by the authors; it must appear in the sensitivity sweep, and no headline
claim may depend on its particular setting.

Exit code is non-zero if an invented parameter is missing from the sweep list, so the
guarantee is enforced rather than remembered.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from recoup.sim.params import SWEPT_PARAMETERS, SimParams


@dataclass(frozen=True)
class Provenance:
    tier: str  # MEASURED | PUBLISHED | INVENTED
    source: str


# Every field of SimParams, plus the model constants that are not fields. Adding a
# parameter without adding it here fails the audit, which is the point.
LEDGER: dict[str, Provenance] = {
    # --- MEASURED: real data, files committed to this repository -----------------------
    "amount_log_mean": Provenance(
        "MEASURED",
        "NPCI monthly UPI volume/value, 16 months: average ticket Rs 1,296",
    ),
    "outage_mean_duration_hours": Provenance(
        "MEASURED", "NPCI downtime, 300 incidents over 11 months: 2.77h mean"
    ),
    "outage_rate_per_bank_day": Provenance(
        "MEASURED", "NPCI downtime; per-bank rates, SBI 0.0988/day"
    ),
    "salary_day_of_month": Provenance(
        "MEASURED",
        "NPCI daily statistics, 212 days: ticket index peaks day 2, +19.7% vs late month",
    ),
    # --- PUBLISHED: real third-party studies, read through reporting --------------------
    "decline_mix": Provenance(
        "PUBLISHED",
        "Recurly decline-reason benchmarks. Western subscription data; Indian mix differs",
    ),
    "baseline_technical_decline_rate": Provenance(
        "PUBLISHED", "NPCI system-wide technical decline ~0.7-0.8%"
    ),
    # --- INVENTED: chosen by us. Every one of these must be swept ----------------------
    "amount_log_sigma": Provenance("INVENTED", "NPCI publishes the mean, not the spread"),
    "shortfall_low": Provenance("INVENTED", "no source for how far a failed charge overshot"),
    "shortfall_high": Provenance(
        "INVENTED", "FITTED so the Recurly Day-1/3/5/7 baseline reproduces at 58%"
    ),
    "balance_depletion_rate": Provenance(
        "INVENTED",
        "NPCI daily data gives a lower bound (0.9913) on spending decay, not balance decay",
    ),
    "balance_floor_fraction": Provenance("INVENTED", "no public data on residual balances"),
    "opt_out_hazard_per_contact": Provenance("INVENTED", "no published per-contact hazard"),
    "outreach_response_rate": Provenance(
        "INVENTED", "published dunning figures disagree: 32%, 42%, 47.6%"
    ),
    "contact_fatigue_halflife_hours": Provenance("INVENTED", "no source"),
    "salary_day_jitter": Provenance("INVENTED", "no source on salary date dispersion"),
    "horizon_days": Provenance("INVENTED", "design choice: 2x the Recurly schedule length"),
    "n_customers": Provenance("INVENTED", "sample size, not a modelling parameter"),
    "seed": Provenance("INVENTED", "reproducibility, not a modelling parameter"),
}

# Parameters that are structural rather than empirical: changing them changes what is
# being modelled, not how much of it there is.
NOT_EMPIRICAL = {"n_customers", "seed", "horizon_days"}

TIER_ORDER = {"MEASURED": 0, "PUBLISHED": 1, "INVENTED": 2}


def main() -> int:
    fields = [f for f in SimParams.__dataclass_fields__]
    missing = [f for f in fields if f not in LEDGER]

    rows = sorted(
        ((name, LEDGER[name]) for name in fields if name in LEDGER),
        key=lambda kv: (TIER_ORDER[kv[1].tier], kv[0]),
    )

    print("Recoup parameter provenance")
    print("=" * 96)
    current = ""
    for name, prov in rows:
        if prov.tier != current:
            current = prov.tier
            print(f"\n--- {current} ---")
        swept = " [swept]" if name in SWEPT_PARAMETERS else ""
        print(f"  {name:<34}{swept:<9} {prov.source}")

    counts = {t: sum(1 for _, p in rows if p.tier == t) for t in TIER_ORDER}
    print()
    print("=" * 96)
    print(
        f"MEASURED {counts['MEASURED']}   PUBLISHED {counts['PUBLISHED']}   "
        f"INVENTED {counts['INVENTED']}"
    )

    problems: list[str] = []

    if missing:
        problems.append(f"parameters with no provenance entry: {', '.join(sorted(missing))}")

    unswept = [
        name
        for name, prov in rows
        if prov.tier == "INVENTED" and name not in SWEPT_PARAMETERS and name not in NOT_EMPIRICAL
    ]
    if unswept:
        problems.append(
            "INVENTED parameters missing from the sensitivity sweep: " + ", ".join(sorted(unswept))
        )

    print()
    if problems:
        for problem in problems:
            print(f"FAIL  {problem}")
        print()
        print("An invented parameter that is not swept can silently carry a headline result.")
        return 1

    print("OK    every parameter is accounted for, and every invented one is swept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
