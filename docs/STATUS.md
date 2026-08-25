# Status — end of 25 Aug 2026

**Day 9 of 11 complete. Checkpoint C reached. Still one day ahead of the plan.**
Deadline **5 Sep**; target submission **4 Sep**, leaving one buffer day.

---

## The headline number

Measured on 20,000 held-out episodes, exploration off, common random numbers across
policies. **Regenerate with `python scripts/run_eval.py`**; the numbers below are read
from `data/results/eval.json`, which records its own seed, parameters and commit.

| policy | recovery | revenue | attempts/ep | wasted |
|---|---|---|---|---|
| fixed 1/3/5/7 *(baseline)* | 58.4% | 82.2% | 1.63 | 64.2% |
| exponential backoff | 59.6% | 86.8% | 1.68 | 64.5% |
| aggressive retry | 59.8% | 87.6% | 1.65 | 63.8% |
| outreach only | 36.9% | 28.3% | 0.60 | 38.5% |
| taxonomy routing | 67.1% | 81.9% | 1.36 | 50.6% |
| **routing + bandit** | **69.2%** | **85.7%** | **1.26** | **45.0%** |

**+10.8 points of recovery on fewer attempts.** Calibration gate passes: the simulator
reproduces Recurly's published 58.0% at 58.4%, within the ±3pp tolerance.

### This is not the number that was here yesterday, and that matters

Yesterday's file recorded **+9.8pp against a 58.2% baseline**. That figure was copied out
of a terminal from a run whose configuration was never written down, and it does not
reproduce. Making the comparison reproducible — every policy scored on the same held-out
half, parameters and seed recorded — gives **+10.8pp against 58.4%**.

The uplift got *larger*, so nothing about the argument weakens. But the old number was
indefensible regardless of whether it happened to be right, which is the whole reason
ADR-010 now exists. **Do not quote a figure that is not in `data/results/eval.json`.**

---

## What exists

| | Component | State |
|---|---|---|
| 1 | Decline taxonomy — 21 codes, 5 classes, structural caps | done |
| 2 | Simulator — mechanism-based, NPCI-calibrated | done |
| 3 | Evaluation harness — baselines, common random numbers | done |
| 4 | Contextual bandit (LinUCB) + train/test split | done |
| 5 | Guardrail layer — idempotency, caps, opt-out, quiet hours | done |
| 6 | Razorpay integration — test-mode client, signed webhooks | done |
| 7 | LLM layer — normaliser, outreach writer, agent | done |
| 8 | Dashboard — live feed, decision trace, results, guardrail ledger | **done** |
| 9 | **Sensitivity sweep + ablations** | **day 10 — not started** |
| 10 | **README, architecture doc, video, submission** | **day 11 — not started** |

**241 tests passing.** ruff, ruff format, mypy strict all clean.

*(The taxonomy has 21 codes, not the 26 this file claimed yesterday. Counted, not
remembered: `all_reasons()` returns 21 across 5 classes.)*

---

## Runnable demos

```
.venv\Scripts\python.exe scripts\dashboard.py          # the demo screen, http://127.0.0.1:8000
.venv\Scripts\python.exe scripts\run_eval.py           # regenerate data/results/eval.json (~3.5 min)
.venv\Scripts\python.exe scripts\demo_guardrails.py    # five attacks, all refused
.venv\Scripts\python.exe scripts\sandbox_demo.py       # real Razorpay payment -> decision chain
.venv\Scripts\python.exe scripts\audit_parameters.py   # provenance ledger, exits 1 if unsourced
.venv\Scripts\python.exe scripts\verify_razorpay.py    # credential check, redacted output
```

### The dashboard, in one paragraph

`scripts/dashboard.py` serves the demo screen and the webhook receiver on one port. Three
streams feed the event list and each is badged differently: **webhook** (a signed
`payment.failed` delivery), **sandbox** (a real failed payment pulled back from the
Razorpay API), and **simulated** (an injected decline, for showing classes the sandbox
will not produce on demand). Selecting any event shows the full decision chain, a
*What this payload could not tell us* panel, and the measured results table underneath.
Every verdict on screen is returned by the engine — see ADR-009.

**Verified end to end on 25 Aug:** the real sandbox payment `pay_TTh0P3vWkSpZtu`
(`netbanking`, `payment_failed`) was pulled from the Razorpay API into the dashboard and
classified `hard_declined` / 1 permitted attempt, its retry allowed and its replay refused
on `duplicate_charge`.

---

## Things tomorrow's session must know

**`RAZORPAY_WEBHOOK_SECRET` is not set in `.env`.** The receiver therefore refuses every
delivery, which is correct behaviour and looks exactly like a broken demo. The dashboard
header shows `webhook secret: NOT SET` when this is the case. Set it before recording if
the video shows the webhook path; the sandbox-pull button does not need it.

**The Groq API key expires around 31 Aug** — created as a 7-day key on 24 Aug, which is
before the 5 Sep deadline. Regenerate at console.groq.com and update `.env` before the
video is recorded, or the live agent demo will fail on camera.

**Groq model is `openai/gpt-oss-120b`.** It is a *reasoning* model: it spends most of its
token budget thinking before answering. `REASONING_BUDGET = 1600` exists for that reason —
lowering it produces empty responses with `finish_reason: length`, not errors.

**Razorpay test account works but is activation-pending**, so UPI does not appear at
checkout. **Use Netbanking** — pick any bank, then click *Failure* on the mock page. No
credentials needed.

**`docs/benchmark.html` is now stale** — it carries the retired +9.8pp figures. Regenerate
it from `data/results/eval.json` after the day 10 sweep and redeploy to the same artifact
URL (https://claude.ai/code/artifact/b7458ae4-ba93-4758-90f8-f3c980eb9320).

---

## Open risks

**A documented remedy has no reachable code path.** `card_expired` carries
`Remedy.NEW_INSTRUMENT`, and both `TaxonomyAware` and `BanditPolicy` contain a branch that
switches to another instrument. That branch is gated on `remaining_attempts > 0`, and the
taxonomy *requires* every `NEVER_RETRYABLE` code to have `max_attempts == 0`. So it can
never run, and the guardrail layer would refuse the charge on `ATTEMPT_CAP` anyway: the cap
counts attempts against the **decline code**, not against the instrument that produced it.

The question is whether the cap means *"no attempt on this payment"* or *"no attempt on
this instrument"*. The second reading is what `Remedy.NEW_INSTRUMENT` and both policies
plainly intended. Resolving it would give never-retryable declines a recovery path and
would probably *increase* the measured uplift — so it is a day-10 decision, not a quiet
fix. Pinned by `test_an_alternative_instrument_does_not_currently_change_the_decision` so
it cannot be lost. **This is a likely panel question: "show me the new-instrument path."**

**The +10.8pp magnitude is still not sensitivity-tested.** It rests on three invented
parameters — the session-conditional share of the decline mix, `outreach_response_rate`,
and `SESSION_COMPLETION_RATE`. Day 10's sweep is what turns it from provisional into
defensible, and it may move the number. The *direction* is safe for a separate reason:
retrying a `payment_cancelled` decline has probability exactly zero by Razorpay's own
documentation, so any non-zero outreach response rate makes outreach the better action.

**Revenue figures carry a heavy tail** — simulated failed-payment amounts have a mean far
above their median. Recovery rate by count is robust; revenue-weighted figures are not.
Note that `taxonomy_aware` recovers *less* revenue than the baseline while recovering more
payments, which is this tail, not a defect.

**The salary-cycle story is not the win.** The bandit converged on retrying at the earliest
permitted delay, and a payday-targeting policy tested *worse* than baseline. The win is
routing, not timing. Do not put the salary-cycle narrative in the video; it is the more
appealing story and the wrong one.

---

## Deferred by the user

The daily explain-backs in `docs/panel-prep/` have not been written. The user chose to
prioritise building and read the code later. That is their call, and it is recorded here
because the panel interview is what the submission is actually judged on, and nobody else
can do that part.

---

## Day 10 plan

Sensitivity sweeps across every uncertain parameter, and ablations: taxonomy-aware vs not,
bank-health-aware vs not, bandit vs fixed. Decide the never-retryable cap question above
before the sweep, because it changes what is being swept. Regenerate
`data/results/eval.json` and `docs/benchmark.html` from the result, and write RESULTS.md.
