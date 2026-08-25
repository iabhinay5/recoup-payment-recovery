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

---

## ADR-008 — Recovery is modelled by mechanism, not by fitted curve

**Decision.** The simulator does not contain a "probability of recovery vs. hours since
failure" curve. Recovery emerges from the processes that actually drive it: a customer
balance process on a salary cycle, a rail outage process, and a session state that only
outreach can open.

**Why.** The easy implementation is a curve fitted so the published benchmarks come out
right. It is also self-defeating. The thing under evaluation is *a policy that chooses
when to retry* — hand it a curve and the optimum is wherever the curve peaks, the policy
finds it immediately, and the experiment measures nothing but our own curve-fitting.

Modelling mechanisms means the policy has to infer something real: that balances recover
on paydays, that outages end, that a cancelled payment needs a human back in a session.
None of those are visible to it directly.

It also changes what the Recurly benchmark is *for*. Under a fitted curve, 58% would be an
input. Under this design it is an independent **check**: if a Day-1/3/5/7 schedule
reproduces near 58% in a simulator that was never told about that number, the mechanisms
are plausibly calibrated. That check is the day 4 gate in docs/PLAN.md.

**Rejected.** Fitting a hazard curve per decline reason. Faster to build, and it would
have produced better-looking results sooner — which is precisely the problem.

**Cost.** More parameters to justify, and calibration is genuinely harder: the mechanisms
have to be tuned jointly until an *emergent* quantity matches a published one. That work
is real, and it is the work that makes the eventual number mean anything.

**First observation, before calibration.** A naive fixed Day-1/3/5/7 schedule over 5,000
simulated episodes spends **78.8% of its wasted attempts on session-conditional declines**
— failures that a silent retry can never resolve, no matter how well timed. That is the
project's thesis appearing unprompted in the first end-to-end run, and it is the number
the pitch should lead with.

---

## ADR-009 — The dashboard renders traces produced by the engine, never its own account of them

**Decision.** The demo UI holds no decision logic. `recoup.trace.explain` runs the real
policy against the real guardrails and returns a `DecisionTrace`; the dashboard, and the
sandbox CLI demo, only lay that object out on a screen.

**Why.** The natural way to build a demo screen is to read a decline code and describe what
the policy would probably do with it. That produces a UI that is correct on the day it is
written and silently wrong afterwards — the policy changes, no test fails, and the first
person to notice the disagreement is a panellist reading the code next to the screen. A
dashboard that *is* the decision cannot drift from it.

It also settles a question the panel is entitled to ask: is the policy you evaluated the
policy you could deploy? `EpisodeState` was written as the simulator's interface to a
policy. `recoup.trace.live_state` constructs the same type from a real Razorpay webhook, so
the object measured over 20,000 simulated episodes is literally the object handed a live
failure — not a reimplementation claimed to agree with it.

**Honest about what a webhook cannot say.** A `payment.failed` payload does not carry the
issuing bank, the customer's other instruments, or their income. Those are joins a merchant
performs against its own records. Where they are missing the trace records the gap and the
screen prints it, because a dashboard that quietly substitutes a plausible bank id is worse
than one that admits it does not know: the first teaches you to trust a number that is not
there.

**Rejected.** Streamlit, which is the cut-line fallback in SCOPE.md. Not needed — FastAPI
was already a dependency for the webhook receiver, so the dashboard and the endpoint being
demonstrated run in one process on one port, and the endpoint on screen is the endpoint
under test. Also rejected: a JS framework, which would add a build step to a page that
polls one JSON endpoint.

**Cost.** `recoup.trace` depends on the gateway, the policies, the guardrails and the
simulator's entity types at once. That is a wide dependency for one module. It is the right
place for it — the module sits above all four and none of them import it — but it does mean
a change to `EpisodeState` now breaks the live path as well as the simulator, which is the
coupling the decision is deliberately buying.

---

## ADR-010 — Measured results are a file, not a scrollback

**Decision.** `scripts/run_eval.py` writes `data/results/eval.json`, recording the metrics
alongside the parameters, the seed and the git commit that produced them. Every surface
that displays a result reads that file. Nothing transcribes a number by hand.

**Why.** This was found rather than foreseen. The headline figures in STATUS.md had been
copied out of a terminal, and when the comparison was made reproducible it did not
reproduce them — the recorded uplift was +9.8pp against a 58.2% baseline, and a run with
every policy measured on the same held-out half gives +10.8pp against 58.4%. The direction
and the argument are unchanged, but the earlier number came from a configuration nobody
wrote down, which makes it indefensible whether or not it was right.

A results file with its own provenance turns "here is our number" into "here is our number,
here is the seed and the commit that produced it, run it yourself". That is the difference
between a claim and a result.

**Rejected.** Committing plots or a static HTML table as the source of truth. Both are
renderings; both go stale silently the moment the harness changes, and neither can be
diffed usefully.

**Cost.** The full run takes minutes, so the file is regenerated deliberately rather than
on every change — which means it can lag the code. The recorded commit is what makes that
lag visible instead of invisible.

**Postscript — writing the file was not sufficient.** The first results file this ADR
produced did not reproduce. `_episode_rails` seeded its per-episode generator from
`hash(payment.id)`, and Python salts `hash()` on a str per process, so the outage drawn for
an episode differed between runs. The failure was well hidden: every policy within one
process met the same outage, so the comparison stayed internally valid and only the
published figure moved — between +10.7pp and +10.8pp. A results file makes a number
*checkable*; it does not make it *stable*, and the two were conflated here.

`stable_seed()` uses a blake2b digest, and `tests/test_harness.py` re-runs the evaluation
under three `PYTHONHASHSEED` values and fails if they disagree. That guards the property
rather than the instance, so the next unstable hash is caught by the same test.

The file also records `git_dirty` now. The commit alone overstated what it promised: this
ADR offers "here is the commit, run it yourself", and from a tree with uncommitted changes
a reader cannot.
