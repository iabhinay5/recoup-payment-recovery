# Recoup — Calibration & Data Provenance

Every parameter in the simulator traces to a source in this file. Nothing is invented.
Where a value is uncertain, the uncertainty is recorded as a **sweep range**, and results
are reported across that range rather than at a point estimate (see DECISIONS.md ADR-002).

**Source tiers.** T1 = primary source, verified directly. T2 = secondary reporting of a
primary study; to be upgraded to T1 during the build by locating the original.
Any parameter still at T2 at submission is disclosed as such in the README.

---

## 1. Decline taxonomy — T1

Source: Razorpay's published error documentation.
- Cards: https://razorpay.com/docs/errors/payments/cards/
- UPI: https://razorpay.com/docs/errors/payments/upi/
- Index: https://razorpay.com/docs/errors/payments/list/

Sixteen card decline codes, verified directly from the cards page:

`payment_timed_out` · `gateway_technical_error` · `payment_cancelled` · `card_declined` ·
`insufficient_funds` · `card_not_enrolled` · `bank_technical_error` ·
`card_disabled_for_online_payments` · `authentication_failed` · `payment_risk_check_failed` ·
`payment_failed` · `incorrect_cvv` · `debit_instrument_inactive` ·
`debit_instrument_blocked` · `card_expired` · `transaction_limit_exceeded`

**Retryability classification** — derived from the documented semantics, not assumed:

| Class | Codes | Policy implication |
|---|---|---|
| **Never retryable** | `card_expired`, `debit_instrument_blocked`, `card_not_enrolled`, `card_disabled_for_online_payments`, `debit_instrument_inactive` | Retry cannot succeed. Requires a new instrument or customer action. Retrying is pure waste. |
| **Rail-conditional** | `bank_technical_error`, `gateway_technical_error` | Retryable *only once the rail recovers*. Drives ADR-006. |
| **Time-conditional** | `insufficient_funds`, `transaction_limit_exceeded` | Retryable; success probability depends strongly on *when*. The core bandit problem. |
| **Session-conditional** | `payment_timed_out`, `payment_cancelled`, `authentication_failed`, `incorrect_cvv` | Requires a fresh customer session — outreach, not a silent retry. |
| **Hard-declined** | `payment_risk_check_failed`, `card_declined`, `payment_failed` | Low retry value; escalate or switch instrument. |

This table is the spine of the entire policy. It is derived entirely from T1 documentation.

## 2. Bank / rail health — T1 source, sweep on detail

Source: NPCI UPI ecosystem statistics — https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics

NPCI publishes **bank-wise technical decline (TD%) and business decline (BD%) monthly**.

| Parameter | Value | Tier | Note |
|---|---|---|---|
| System-wide TD, current | ~0.7–0.8% | T2 | Down from 8–10% in 2016 |
| NPCI target, TD | < 1% | T2 | Circular OC-149, June 2022 — verify primary |
| NPCI target, BD | < 5% | T2 | Same circular |
| Per-bank TD dispersion | **sweep** | — | Bank-wise variance is the point; download actual monthly data during build |
| Outage duration distribution | **sweep** | — | Not published; swept across plausible range |

**Action during build:** download real bank-wise monthly TD/BD data and drive the
simulator's bank-health process from actual observed values rather than a fitted
distribution. This upgrades the most important calibration input to T1.

## 3. Recovery baselines — T2, upgrade to T1

Source: Recurly's published analysis of ~40 million subscription transactions.
- https://recurly.com/blog/failed-payment-recovery-data-based-strategy/
- https://recurly.com/research/subscription-benchmarks-top-payment-decline-reasons/

| Parameter | Value | Tier | Use |
|---|---|---|---|
| Day-1/3/5/7 schedule recovery, no customer contact | **58%** | T2 | **The baseline to beat.** Primary benchmark. |
| First retry at 24h vs 2h | **+6.5%** | T2 | Validates that timing alone carries real signal |
| Industry median recovery | ~47.6% | T2 | Context |
| Top-performer layered programs | 70–85% | T2 | Upper reference |
| Dunning email/SMS alone | ~42% | T2 | Churnkey, State of Retention 2025 — outreach arm |

## 4. Decline reason mix — T2

| Class | Share | Tier |
|---|---|---|
| Insufficient funds (soft) | ~50% | T2 |
| Risk-management flags | 25–33% | T2 |
| Card issues (expiry, loss, theft) | 10–15% | T2 |

**Swept**, since this mix is the single most influential simulator input and is reported
for Western subscription businesses rather than Indian payment rails. Indian mix will
differ — UPI is a far larger share and has different failure characteristics. Disclosed
in the README as a known limitation.

## 5. Live sandbox — T1

Razorpay test mode, verified from documentation:
- `failure@razorpay` as UPI ID → deterministic decline
- `success@razorpay` → success
- Test cards with a mock bank page exposing explicit Success / Failure paths
- OTP of 4–10 digits succeeds; under 4 digits fails authentication
- Real webhooks fire in test mode

Docs: https://razorpay.com/docs/payments/payments/test-upi-details/ ·
https://razorpay.com/docs/payments/payments/test-card-details/

## 6. Parameters with no public source

Declared openly. Each is **swept**, never point-estimated, and no headline claim
depends on any single setting:

- P(recovery | decline_reason, attempt_n, elapsed_time) — the response surface itself
- Salary-credit timing effects on `insufficient_funds` recovery
- Customer annoyance / opt-out hazard as a function of contact frequency
- Instrument-switch success rates

**These are the honest weak point of the project, and the README says so.**
The defense is not that they are right — it is that the policy ordering is stable
across the entire plausible region, and that two of the core mechanisms
(never retrying `card_expired`, deferring during a rail outage) are provably
non-lossy without reference to any of them.
