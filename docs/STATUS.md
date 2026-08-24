# Status — end of 24 Aug 2026

**Day 8 of 11 complete. One day ahead of the plan.**
Deadline **5 Sep**; target submission **4 Sep**, leaving one buffer day.

---

## The headline number

Measured on 20,000 held-out episodes, exploration off, common random numbers across policies:

| policy | recovery | revenue | attempts/ep | wasted |
|---|---|---|---|---|
| fixed 1/3/5/7 *(baseline)* | 58.2% | 81.7% | 1.64 | 64.4% |
| aggressive retry | 59.3% | 87.0% | 1.65 | 64.0% |
| taxonomy routing | 66.4% | 81.1% | 1.36 | 51.3% |
| **routing + bandit** | **68.0%** | **86.6%** | **1.26** | **46.1%** |

**+9.8 points of recovery on fewer attempts.** Calibration gate passes: the simulator
reproduces Recurly's published 58.0% at 58.2%, within a ±3pp tolerance.

---

## What exists

| | Component | State |
|---|---|---|
| 1 | Decline taxonomy — 26 codes, 5 classes, structural caps | done |
| 2 | Simulator — mechanism-based, NPCI-calibrated | done |
| 3 | Evaluation harness — baselines, common random numbers | done |
| 4 | Contextual bandit (LinUCB) + train/test split | done |
| 5 | Guardrail layer — idempotency, caps, opt-out, quiet hours | done |
| 6 | Razorpay integration — test-mode client, signed webhooks | done |
| 7 | LLM layer — normaliser, outreach writer, agent | done |
| 8 | **Dashboard** | **day 9 — not started** |
| 9 | **Sensitivity sweep + ablations** | **day 10 — not started** |
| 10 | **README, architecture doc, video, submission** | **day 11 — not started** |

**200 tests passing.** ruff, ruff format, mypy strict all clean. CI green.

---

## Runnable demos

```
.venv\Scripts\python.exe scripts\demo_guardrails.py     # five attacks, all refused
.venv\Scripts\python.exe scripts\sandbox_demo.py        # real Razorpay payment -> decision chain
.venv\Scripts\python.exe scripts\audit_parameters.py    # provenance ledger, exits 1 if unsourced
.venv\Scripts\python.exe scripts\verify_razorpay.py     # credential check, redacted output
```

`docs/benchmark.html` — results page, open in a browser. Also published as an artifact at
https://claude.ai/code/artifact/b7458ae4-ba93-4758-90f8-f3c980eb9320 (needs claude.ai login).

---

## Things tomorrow's session must know

**The Groq API key expires around 31 Aug** — it was created as a 7-day key on 24 Aug.
That is **before the 5 Sep deadline**. Regenerate at console.groq.com and update `.env`
before the video is recorded, or the live agent demo will fail on camera.

**Groq model is `openai/gpt-oss-120b`.** It is a *reasoning* model: it spends most of its
token budget thinking before answering. `REASONING_BUDGET = 1600` exists for that reason —
lowering it produces empty responses with `finish_reason: length`, not errors.

**Razorpay test account works but is activation-pending**, so UPI does not appear at
checkout. **Use Netbanking** — pick any bank, then click *Failure* on the mock page. No
credentials needed. Verified working: `pay_TTh0P3vWkSpZtu`, `error_reason: payment_failed`.

**The benchmark artifact is stale the moment day 10 runs.** Regenerate `docs/benchmark.html`
after the sensitivity sweep, and redeploy to the same artifact URL.

---

## Open risks

**The +9.8pp magnitude is not yet trustworthy.** It rests on three invented parameters —
the session-conditional share of the decline mix, `outreach_response_rate`, and
`SESSION_COMPLETION_RATE`. Day 10's sweep is what turns it from provisional into
defensible, and it may move the number. The *direction* is safe for a separate reason:
retrying a `payment_cancelled` decline has probability exactly zero by Razorpay's own
documentation, so any non-zero outreach response rate makes outreach the better action.

**Revenue figures carry a heavy tail** — simulated failed-payment amounts have a mean of
Rs 31,009 against a median of Rs 3,479. Recovery rate by count is robust; revenue-weighted
figures are not.

**The salary-cycle story is not the win.** The bandit converged on retrying at +6h — the
earliest permitted — and a payday-targeting policy tested *worse* than baseline. The win is
routing, not timing. Do not put the salary-cycle narrative in the video; it is the more
appealing story and the wrong one.

**`docs/benchmark.html` is untracked** — decide whether to commit it.

---

## Deferred by the user

The daily explain-backs in `docs/panel-prep/` have not been written. The user chose to
prioritise building and read the code later. That is their call, and it is recorded here
because the panel interview is what the submission is actually judged on, and nobody else
can do that part.

---

## Day 9 plan

Dashboard. A single screen showing: live event feed from the webhook receiver, the decision
trace per payment (classification, policy choice, guardrail verdicts), and the evaluation
results with the waste breakdown. Streamlit is the fallback if a richer UI is at risk —
see the cut-lines in `SCOPE.md`.
