# Recoup — Execution Plan

**24 Aug → submit 4 Sep 2026.** Deadline 5 Sep; 4 Sep is the target so one buffer day exists.
User capacity: ~6h/day. Implementation, testing, and documentation drafting are done by Claude.

---

## The one risk this plan is built around

All code being written by an assistant creates a specific failure mode: **a candidate who
cannot defend their own system.** The submission is not the repo — it is the panel
interview that follows it. A perfect repo the candidate cannot explain fails outright.

So the daily loop is deliberately not "review the diff":

> **Build (Claude) → Read (user) → Explain-back (user, in writing) → Correct (both)**

The explain-back is written into `docs/panel-prep/dayN.md` each day, in the user's own
words, without looking at the code. Anything that cannot be explained without looking is
flagged and reworked until it can be. This is not overhead — **it is the deliverable.**
Roughly 2 of the 6 daily hours belong to it.

---

## Checkpoints

Each checkpoint is a state where the project is demonstrable even if everything after it fails.

| # | By end of | Demonstrable state |
|---|---|---|
| **A** | Day 4 (28 Aug) | Calibrated simulator + baselines reproduced. "Here is the 58% benchmark, reproduced." |
| **B** | Day 6 (30 Aug) | Bandit policy + guardrails. **"Here is measured uplift over that baseline."** |
| **C** | Day 9 (2 Sep) | End-to-end: real Razorpay webhook → agent → policy → guardrails → outreach → dashboard. |
| **D** | Day 11 (4 Sep) | Submitted: repo, README, architecture doc, results, 5-min video. |

If we slip, we cut from SCOPE.md's cut-lines — never from a checkpoint.

---

## Day 0 — 24 Aug — Planning and accounts *(today)*

- **Claude:** SCOPE.md, DECISIONS.md, CALIBRATION.md, PLAN.md. Repo skeleton. No implementation code.
- **User:** Create Razorpay **test-mode** account (lead time — do first). Create Groq API key.
  Claim GitHub Student Pack. Create the public GitHub repo. Read all four planning docs and
  challenge anything that looks wrong.
- **Ship:** Planning committed. Repo exists.

## Day 1 — 25 Aug — Taxonomy and foundations

- **Claude:** Scrape and encode the Razorpay decline taxonomy into a typed module with the
  retryability classification from CALIBRATION.md section 1. Project skeleton, config,
  provider-abstracted LLM interface (ADR-004), test harness, CI.
- **User:** Read the taxonomy end to end. **This is the spine of the project** — explain-back
  must cover why each code falls into its retryability class.
- **Ship:** `recoup.taxonomy` — the decision spine, fully tested.

## Day 2 — 26 Aug — Simulator core

- **Claude:** Transaction generator, decline-reason sampling, customer/merchant/instrument model,
  episode mechanics (attempt → outcome → state transition).
- **User:** Download **real NPCI bank-wise TD/BD monthly data** — this upgrades our most important
  calibration input to T1. Explain-back on episode mechanics.
- **Ship:** Simulator produces episodes.

## Day 3 — 27 Aug — Calibration and bank health

- **Claude:** Wire NPCI data into a time-varying bank-health process. Calibrate decline mix and
  recovery surface to CALIBRATION.md. Implement every sweep range.
- **User:** Verify each cited number against its source; upgrade T2 to T1 where possible.
  Anything that cannot be sourced gets moved to the swept list.
- **Ship:** Calibrated simulator, every parameter traceable.

## Day 4 — 28 Aug — Eval harness and baselines · **CHECKPOINT A**

- **Claude:** Evaluation harness (deterministic, no LLM — ADR-005). Baselines: no-retry,
  fixed-interval, exponential backoff, **Recurly Day-1/3/5/7**. Metrics: recovery rate, revenue
  recovered, attempts wasted, customer contacts, time-to-recovery, opt-out rate.
- **User:** Confirm the Day-1/3/5/7 baseline reproduces near the published 58%. If it does not,
  **the calibration is wrong and we fix it before building any policy on top.**
- **Ship:** Baselines reproduced and measured. Everything after this is compared against them.

## Day 5 — 29 Aug — Decision core

- **Claude:** Recovery-probability model (LightGBM, CPU). Contextual bandit for retry timing —
  LinUCB and Thompson Sampling, both evaluated. Reward shaping including annoyance cost.
- **User:** Deep session on the bandit. **The highest-probability panel question is "why a bandit
  and not an LLM"** (ADR-003) and "why this reward function". Both must be fluent.
- **Ship:** A policy that beats baseline in the harness.

## Day 6 — 30 Aug — Guardrails · **CHECKPOINT B**

- **Claude:** Structural guardrail layer (ADR-007): idempotency keys, retry caps, quiet hours,
  opt-out, bank-down deferral. Adversarial tests that *attempt* double-charge and are rejected.
- **User:** Explain-back on why each guardrail is structural rather than advisory.
- **Ship:** **Measured uplift over the published baseline, with harm made impossible by construction.**

## Day 7 — 31 Aug — Razorpay sandbox integration

- **Claude:** Test-mode client, webhook receiver, `failure@razorpay` flows, real payload
  normalization into the internal event model, retry execution against the sandbox.
- **User:** Run a real failed payment end-to-end yourself. Confirm the webhook lands.
- **Ship:** Real Razorpay events flowing through the system.

## Day 8 — 1 Sep — LLM layer

- **Claude:** Decline normalizer (cached, Groq). Agentic orchestrator with a tight tool surface:
  `get_bank_health`, `get_customer_history`, `get_policy_recommendation`, `schedule_retry`,
  `send_outreach`, `escalate_to_human`. Outreach copy generation. Decision narration for audit.
- **User:** Explain-back on **where the LLM is and is not**, and why the tool surface is small.
- **Ship:** AI-native layer, provider-abstracted, cache-backed.

## Day 9 — 2 Sep — Dashboard · **CHECKPOINT C**

- **Claude:** Demo UI — live event feed, per-transaction decision trace with reasoning, guardrail
  rejections shown explicitly, eval results and sensitivity plots.
- **User:** Rehearse the demo path start to finish. Identify what is confusing on screen.
- **Ship:** End-to-end demonstrable system.

## Day 10 — 3 Sep — Results

- **Claude:** Full eval runs. Sensitivity sweeps across every uncertain parameter. Ablations
  (taxonomy-aware vs not; bank-health-aware vs not; bandit vs fixed). Charts. RESULTS.md.
- **User:** Interrogate the results. **Find the weakest claim and either strengthen it or
  soften the wording.** Overclaiming is the fastest way to fail a panel.
- **Ship:** Defensible, robustness-checked numbers.

## Day 11 — 4 Sep — Documentation, video, submit · **CHECKPOINT D**

- **Claude:** README (problem, architecture diagram, setup, results, **stated limitations**),
  ARCHITECTURE.md, final pass on all docs.
- **User:** Record the 5-minute video. Structure: problem → solution → architecture → live demo.
  Submit.
- **Ship:** Submitted, with a day to spare.

---

## Video plan (drafted day 9, recorded day 11)

| Time | Content |
|---|---|
| 0:00–0:45 | The problem — decline reasons are not interchangeable; `card_expired` retried 4x is pure waste |
| 0:45–1:30 | Architecture — where the LLM is, and **why it is not in the decision path** |
| 1:30–3:00 | Live demo — real sandbox failure → webhook → agent trace → policy decision → outreach |
| 3:00–3:30 | Guardrail demo — attempt a double-charge, watch it be rejected |
| 3:30–4:30 | Results — uplift over the Recurly baseline, plus the sensitivity sweep |
| 4:30–5:00 | Limitations, stated plainly, and what production data would settle |

Ending on limitations rather than a victory lap is deliberate. It is the "understands
failure modes" signal, delivered as the last thing they hear.

---

## Guardrails on how Claude works

Carried from the user's stated working style:

1. **No implementation before a decision is written down.** Anything non-obvious becomes an ADR first.
2. **No unverified claims.** Nothing is reported as working without command output showing it.
   No invented benchmark numbers, no invented API shapes, no invented sources.
3. **Cut-lines are honoured.** Falling behind means cutting from SCOPE.md's list, in order —
   never quietly dropping a core component or a checkpoint.
4. **Every uncertain parameter is swept, never point-estimated.**
5. **Scope creep is refused out loud.** A good idea outside SCOPE.md gets logged, not built.
