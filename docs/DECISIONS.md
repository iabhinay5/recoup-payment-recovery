# Recoup — Architecture Decision Record

Every entry is a decision the panel may challenge. Each states the decision, the
reasoning, the alternative rejected, and the honest cost. If a decision cannot be
defended out loud, it does not belong in the system.

---

## ADR-001 — Track: AI Revenue Recovery

**Decision.** Build in the AI Revenue Recovery track.

**Why.** Razorpay's revenue is a percentage of *successful* transactions, so failed-payment
recovery sits directly on their P&L. The track is also the least likely to be crowded:
it requires payments-domain knowledge that most applicants will not invest in, whereas
fraud detection and shopping agents are well-trodden student territory.

**Rejected.** *AI Risk Manager* — high applicant density, and tends to collapse into a
classifier with an LLM bolted on. *Agentic Commerce* — highest density, hardest to
demonstrate a hard metric. *Finance Controller* — good and under-picked, but the honest
solution is mostly deterministic matching, which demos poorly on video.

**Cost.** Requires learning payments domain vocabulary before building. Accepted.

---

## ADR-002 — Simulation-based offline policy evaluation

**Decision.** Evaluate policies in a transaction simulator calibrated to published
statistics, not on a proprietary transaction log.

**Why.** No public dataset pairs decline reasons with retry outcomes, and no such dataset
can exist outside a payment processor. This is the standard situation for evaluating a
decision policy without production access, and it has a standard answer: offline policy
evaluation. Rigor comes from three commitments, all enforced in CALIBRATION.md:

1. Every simulator parameter traces to a cited public source — never invented.
2. Results are reported across **sensitivity sweeps**, not at a single parameterization.
3. Limitations are stated in the README before anyone has to ask.

**Rejected.** Fabricating a dataset and presenting it as real — dishonest and instantly
detectable. Using an unrelated Kaggle fraud dataset — measures the wrong thing.

**Cost — stated plainly.** The simulator cannot establish absolute recovery percentages.
It supports claims about the *ordering* of policies and the *robustness* of that ordering.
The uplift magnitude requires production data; the direction does not.

**Panel answer.** "I don't know the absolute numbers — a simulator can't give me those.
What I can show is the ordering of policies, stable across every plausible parameterization
I swept. And the mechanism is independently verifiable without any simulator at all: not
retrying `card_expired` cannot lose money, and deferring a retry during a bank outage
cannot lose money."

---

## ADR-003 — The LLM does not choose the retry time

**Decision.** Retry timing, channel, and instrument selection are decided by a contextual
bandit over numeric features. The LLM is structurally excluded from this path.

**Why.** Three independent reasons:

1. **Defensibility.** If an LLM outputs "retry in 27 hours", there is no way to justify 27.
   A bandit exposes its posterior and its regret curve.
2. **Evaluability.** The harness runs 10^4–10^5 episodes. LLM-in-the-loop makes that slow,
   costly, and non-deterministic — the eval harness would not exist.
3. **Correctness.** This is numeric optimization against a well-defined reward. It is not
   a language task, and an LLM is the wrong instrument for it.

**Where the LLM does belong:** normalizing heterogeneous decline signals into the canonical
taxonomy; agentic orchestration (deciding *which tools to call*); generating customer
outreach copy; and narrating decisions in natural language for the audit trail.

**Rejected.** An LLM agent that reasons its way to a retry schedule. This is the default
failure mode of this track and it is what most submissions will do.

**Cost.** Less superficially impressive than "the agent decided everything." Accepted
deliberately — knowing where *not* to use an LLM is the point.

---

## ADR-004 — Provider-abstracted LLM layer; Groq primary, Ollama documented

**Decision.** All model calls go through one internal interface. Groq's free tier is the
development default; a local Ollama path is implemented and documented.

**Why.** Transaction data is among the most sensitive data a merchant holds. A recovery
system that requires shipping decline data to a third-party API to make a retry decision
imposes a data-residency cost that a payment processor may not accept. Abstracting the
provider means the identical system runs fully on-premise with a config change.

Vendor-decoupling is good engineering independent of that argument. Zero cost is a
convenient consequence, not the reason.

**Rejected.** Hard-coding a single vendor SDK throughout the codebase.

**Cost.** A thin indirection layer, and we cannot use provider-specific features. Both cheap.

---

## ADR-005 — The evaluation harness never makes a live LLM call

**Decision.** LLM outputs are cached (or mocked) before any evaluation run. The harness
executes pure numeric code.

**Why.** Evaluations must be **deterministic and reproducible** — a result that changes
between runs because of sampling noise in an unrelated component is not a result. This is
a correctness requirement first; that it also makes eval runs free and instant is secondary.

This is tractable because the decline taxonomy is **finite** (~16 card codes plus the UPI
set). Normalization is a cache hit in the overwhelming majority of cases.

**Rejected.** Calling the LLM inside the episode loop. Would be slow, non-reproducible,
and would hit Groq's 6,000 TPM ceiling immediately.

---

## ADR-006 — Bank-health-aware retry deferral

**Decision.** The policy consults bank/rail health before scheduling a retry and defers
attempts targeted at a rail that is currently degraded.

**Why.** Razorpay's own error documentation, under `bank_technical_error`, directs
integrators to their Downtime API. Retrying into a bank that is currently down is the
single most wasteful action a dunning system can take: the attempt cannot succeed, it
consumes a limited retry budget, and it adds load to an already-failing rail.

NPCI publishes bank-wise technical decline rates monthly, so bank health is both real
and time-varying — it is a legitimate feature, not a synthetic one.

**Cost.** Adds a dependency on a health signal that may be stale or unavailable. Handled:
the policy degrades to a conservative default when health data is missing.

---

## ADR-007 — Guardrails are structural, not advisory

**Decision.** Idempotency, retry caps, quiet hours, and opt-out are enforced in a layer the
policy cannot bypass — not as instructions in a prompt or checks the policy is asked to respect.

**Why.** The catastrophic failure of a payment retry system is **double-charging a customer**.
That must be impossible by construction, not unlikely by persuasion. Any guardrail
implemented as a prompt instruction is a guardrail that fails under distribution shift.

**Demonstrated, not asserted.** The demo includes a deliberate attempt to trigger a
double-charge, and shows the idempotency layer rejecting it.
