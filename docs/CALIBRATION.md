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

**RESOLVED — now T1.** 11 monthly files (Sep 2025 – Jul 2026) from the Downtime/Incidents
tab ship in `data/npci/downtime/`, so this calibration is reproducible from a clone.
Loader: `recoup.data.npci`.

**300 incidents, 37 distinct banks, 831.3 downtime hours.**

| Parameter | Was (invented) | Now (measured) | Tier |
|---|---|---|---|
| Mean hours per incident | 2.50 | **2.77** (±5.8%) | T1 |
| Median hours per incident | — | **1.67** | T1 |
| Outage rate, pooled over all 741 banks | 0.05 | **0.00121** /bank/day | T1 |
| Per-bank outage rate and duration | one shared value | **per bank, measured** | T1 |
| System-wide TD | ~0.7–0.8% | unchanged | T2 |

**The 41× correction, and why the naive reading is wrong.** Pooled across all 741 member
banks the rate is 0.00121/bank/day — 41× below the invented figure. But that denominator is
dominated by several hundred small cooperative banks processing almost nothing. State Bank
of India's own measured rate is **0.0988/day, roughly 82× the pooled average**. A merchant
does not meet the average bank; it meets the banks its customers hold accounts with. The
simulator therefore uses the ten most incident-prone banks with their individually measured
profiles, which are also India's largest issuers by volume.

**Dispersion is the finding.** Mean incident duration ranges from 0.87h (Indian Bank) to
5.31h (Uttar Pradesh Gramin Bank) among the selected banks, and up to 14.20h across all 37.
That ~17× spread is the entire reason knowing *which* rail you are retrying into is worth
anything — if every bank failed identically, ADR-006 would be pointless.

**Two reading errors this data invites**, both silent and both worth stating:
1. Banks with no incidents are **absent, not zero**. Computing the rate over only the banks
   that appear overstates it by roughly two orders of magnitude.
2. Bank names are typed by hand and vary in case and suffix across months. "Punjab and Sind
   Bank" and "Punjab And Sind Bank" are one bank; not merging them splits a bank's history
   and halves its apparent failure rate. Normalisation collapsed 43 apparent banks to 37.

**Uptime tab — checked, not usable.** The Uptime files report NPCI's *own* central
infrastructure uptime (100% in the months sampled), not per-bank uptime. Useful as a
negative result: UPI failures originate at the bank edge, not in NPCI's core, which is why
the model puts all rail variance on the issuer.

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
