# Recoup — Scope

**Project:** Recoup — agentic failed-payment recovery for Indian payment rails
**Track:** AI Revenue Recovery (Razorpay AI Buildathon 2026)
**Window:** 24 Aug 2026 → submit 4 Sep 2026 (deadline 5 Sep, one buffer day)

---

## The problem, stated precisely

A payment fails. Something must decide: retry or not, *when*, on *which* instrument,
and whether to contact the customer — before the money is lost or the customer churns.

Most systems treat "failed payment" as one undifferentiated bucket and apply a fixed
retry schedule. That is wrong in both directions:

- `card_expired` is retried repeatedly — it can **never** succeed. Pure waste.
- `bank_technical_error` is retried immediately into a bank that is **still down**.
  Wastes the attempt and adds load to a struggling rail.
- `insufficient_funds` is retried at a random hour instead of near a salary credit.

Recoup makes that decision per-transaction, grounded in the actual decline reason,
live bank health, and a learned timing policy — with hard guardrails that make
customer-harming actions structurally impossible.

## Success criteria

This project succeeds if, at the panel, the following are all true:

1. A real Razorpay sandbox payment failure flows end-to-end through the system, live.
2. Measured uplift over a published industry baseline, with sensitivity analysis
   showing the result is not an artifact of one parameter setting.
3. Every guardrail (idempotency, retry caps, bank-down deferral, opt-out) is
   demonstrable — including a live attempt to violate one, and its rejection.
4. Every architectural decision in DECISIONS.md can be defended out loud.

Note that "impressive demo" is not on that list. Defensibility is the product.

---

## In scope

| # | Component | Why it must exist |
|---|---|---|
| 1 | Decline taxonomy grounded in Razorpay's published error codes | Speaks the panel's own vocabulary; drives all policy branching |
| 2 | Transaction simulator calibrated to published NPCI + Recurly statistics | The evaluation environment; every parameter is cited, not invented |
| 3 | Baseline policies incl. the Recurly Day-1/3/5/7 schedule | You cannot claim uplift without a real number to beat |
| 4 | Recovery-probability model (LightGBM, CPU) | Turns decline context into P(success) |
| 5 | Retry-timing policy — contextual bandit | The core decision. Numeric, defensible, fast to evaluate |
| 6 | Guardrail engine | Idempotency, retry caps, quiet hours, opt-out, bank-down deferral |
| 7 | Razorpay sandbox integration — real webhooks | Proves this is a system, not a notebook |
| 8 | LLM layer — normalization, orchestration, outreach copy | The AI-native surface, provider-abstracted |
| 9 | Evaluation harness — sensitivity sweeps + ablations | The single biggest differentiator vs. the applicant pool |
| 10 | Demo dashboard | Carries the 5-minute video |
| 11 | README + ARCHITECTURE.md + CALIBRATION.md | Judged artifacts in their own right |

## Explicitly OUT of scope

Stated here so they are never silently attempted, and so the README can say them plainly.

- **Not a production payment system.** No live mode, no real money, ever.
- **No PCI-scope data.** No card numbers touch this system at any point.
- **No claim of absolute recovery percentages.** The simulator supports claims about
  the *ordering* and *robustness* of policies, not real-world magnitudes. See DECISIONS.md ADR-002.
- **Not multi-tenant.** Single-merchant model throughout.
- **No model training beyond CPU-minutes.** No GPU, no fine-tuning, no pretraining.
- **Not a full merchant dashboard.** The UI exists to demonstrate the engine, nothing more.
- **No mobile app, no auth system, no billing.**

## Cut-lines

If we fall behind, we cut from this list **in this order**. Core is never cut.

1. GraphRAG layer over the merchant/customer/bank/instrument graph *(stretch from day one)*
2. Multilingual outreach generation
3. Docker Compose packaging
4. Dashboard richness — falls back to Streamlit if the richer UI is at risk
5. Live Razorpay Downtime API — falls back to simulated bank health

**Never cut:** taxonomy, simulator, baselines, bandit policy, guardrails, eval harness,
sandbox integration, README/architecture docs, video.

## Risks

| Risk | Mitigation |
|---|---|
| Simulator reads as hand-waving | Every parameter cited in CALIBRATION.md; sensitivity sweeps prove robustness; limitations stated aloud |
| Razorpay sandbox activation delayed | Account created day 1; simulator path is fully independent of it |
| Scope creep into a "cool agent demo" | This document; cut-lines; ADR-003 keeps the LLM out of the decision core |
| Groq rate limits during eval runs | Eval harness never calls an LLM live (ADR-005) |
| Falling behind and panicking into a code dump | Daily shippable checkpoints in PLAN.md — always something demoable |
