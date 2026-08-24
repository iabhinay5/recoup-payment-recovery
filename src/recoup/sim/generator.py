"""Population generation: banks, customers, instruments, and failed payments.

Everything is derived from a single seeded generator, so a run is reproducible from its
seed alone. That matters more than it might seem: the evaluation compares policies on the
*same* population, and a difference that came from resampling customers rather than from
the policy would be indistinguishable from a real effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recoup.sim.entities import HOURS_PER_DAY, Bank, Customer, FailedPayment, Instrument
from recoup.sim.params import SimParams
from recoup.taxonomy import PaymentMethod

__all__ = ["Population", "generate_population"]

# Placeholder issuers with plausible technical-decline dispersion. Day 3 of docs/PLAN.md
# replaces these with real NPCI bank-wise monthly figures, which is what upgrades this
# input from an assumption to a measurement (docs/CALIBRATION.md section 2).
_PLACEHOLDER_BANKS: tuple[tuple[str, str, float], ...] = (
    ("hdfc", "HDFC Bank", 0.004),
    ("icici", "ICICI Bank", 0.006),
    ("sbi", "State Bank of India", 0.012),
    ("axis", "Axis Bank", 0.007),
    ("kotak", "Kotak Mahindra Bank", 0.005),
    ("pnb", "Punjab National Bank", 0.015),
)

UPI_SHARE = 0.62
"""Share of instruments that are UPI rather than card.

UPI dominates Indian retail payments, and the mix matters because the two rails fail
differently — a card can expire, a VPA cannot. Swept as part of the decline-mix
sensitivity on day 10.
"""


@dataclass(frozen=True, slots=True)
class Population:
    """A generated population and the failed payments to be recovered from it."""

    banks: tuple[Bank, ...]
    customers: tuple[Customer, ...]
    payments: tuple[FailedPayment, ...]

    def __post_init__(self) -> None:
        if len(self.payments) != len(self.customers):
            raise ValueError("expected exactly one failed payment per customer")

    @property
    def customers_by_id(self) -> dict[str, Customer]:
        return {c.id: c for c in self.customers}

    @property
    def bank_decline_rates(self) -> dict[str, float]:
        """Bank id to steady-state technical decline rate, for ``RailHealth``."""
        return {b.id: b.base_technical_decline_rate for b in self.banks}


def _sample_decline_codes(params: SimParams, n: int, rng: np.random.Generator) -> list[str]:
    codes = list(params.decline_mix.keys())
    weights = np.array([params.decline_mix[c] for c in codes], dtype=float)
    weights /= weights.sum()
    return [codes[i] for i in rng.choice(len(codes), size=n, p=weights)]


def _make_instruments(
    customer_index: int,
    n_instruments: int,
    bank_ids: list[str],
    horizon_hours: float,
    force_expired: bool,
    rng: np.random.Generator,
) -> tuple[Instrument, ...]:
    instruments: list[Instrument] = []
    for j in range(n_instruments):
        is_upi = rng.random() < UPI_SHARE
        method = PaymentMethod.UPI if is_upi else PaymentMethod.CARD

        expires_at: float | None = None
        if method is PaymentMethod.CARD:
            if force_expired and j == 0:
                # The payment's decline reason and the instrument state have to agree.
                # An episode that reports card_expired while holding a valid card would
                # let a policy "recover" something that is not recoverable.
                expires_at = 0.0
            elif rng.random() < 0.04:
                # A small share of cards expire partway through the horizon, which is
                # what produces mid-episode reason transitions.
                expires_at = float(rng.uniform(0.0, horizon_hours))

        instruments.append(
            Instrument(
                id=f"inst_{customer_index}_{j}",
                method=method,
                bank_id=str(rng.choice(bank_ids)),
                expires_at_hours=expires_at,
            )
        )
    return tuple(instruments)


def generate_population(params: SimParams) -> Population:
    """Generate banks, customers, instruments, and one failed payment per customer."""
    rng = np.random.default_rng(params.seed)
    horizon_hours = params.horizon_days * HOURS_PER_DAY

    banks = tuple(
        Bank(id=bid, name=name, base_technical_decline_rate=rate)
        for bid, name, rate in _PLACEHOLDER_BANKS
    )
    bank_ids = [b.id for b in banks]

    decline_codes = _sample_decline_codes(params, params.n_customers, rng)

    customers: list[Customer] = []
    payments: list[FailedPayment] = []

    for i in range(params.n_customers):
        code = decline_codes[i]

        # Most customers hold one instrument; a minority hold two or three. Instrument
        # switching is the only remedy for a dead card, so this distribution bounds how
        # much of the never-retryable population is recoverable at all.
        n_instruments = int(rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10]))

        instruments = _make_instruments(
            customer_index=i,
            n_instruments=n_instruments,
            bank_ids=bank_ids,
            horizon_hours=horizon_hours,
            force_expired=(code == "card_expired"),
            rng=rng,
        )

        # A card-only decline reason requires a card to have been used.
        if code in ("card_expired", "incorrect_cvv") and instruments[0].method is PaymentMethod.UPI:
            instruments = (
                Instrument(
                    id=instruments[0].id,
                    method=PaymentMethod.CARD,
                    bank_id=instruments[0].bank_id,
                    expires_at_hours=0.0 if code == "card_expired" else None,
                ),
                *instruments[1:],
            )

        monthly_income = int(rng.lognormal(mean=11.6, sigma=0.7) * 100)
        salary_day = int(
            np.clip(
                params.salary_day_of_month + rng.integers(0, params.salary_day_jitter + 1),
                1,
                28,
            )
        )

        customer = Customer(
            id=f"cust_{i}",
            instruments=instruments,
            monthly_income_paise=monthly_income,
            salary_day_of_month=salary_day,
        )
        customers.append(customer)

        amount = int(rng.lognormal(params.amount_log_mean, params.amount_log_sigma) * 100)
        payments.append(
            FailedPayment(
                id=f"pay_{i}",
                customer_id=customer.id,
                instrument_id=instruments[0].id,
                amount_paise=max(100, amount),
                initial_decline_code=code,
                reference_day_of_month=int(rng.integers(1, 29)),
                reference_hour_of_day=float(rng.uniform(0.0, 24.0)),
            )
        )

    return Population(banks=banks, customers=tuple(customers), payments=tuple(payments))
