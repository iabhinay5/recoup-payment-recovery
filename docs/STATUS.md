# Status — end of 25 Aug 2026

**Day 10 of 11 complete. Still one day ahead of the plan.**
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
| aggressive retry | 59.7% | 87.6% | 1.65 | 63.8% |
| outreach only | 36.9% | 28.3% | 0.60 | 38.5% |
| taxonomy routing | 67.1% | 81.9% | 1.36 | 50.7% |
| **routing + bandit** | **69.1%** | **85.6%** | **1.26** | **45.2%** |

**+10.7 points of recovery on fewer attempts.** Calibration gate passes: the simulator
reproduces Recurly's published 58.0% at 58.4%, within the ±3pp tolerance.

### The number moved twice, and the second time is the one that mattered

It recorded **+9.8pp against 58.2%** on day 8, copied out of a terminal from a run whose
configuration was never written down. Day 9 made the comparison reproducible and got
**+10.8pp against 58.4%**.

That second figure did not reproduce either. Re-running the identical command gave a
different answer, because `_episode_rails` seeded its generator from `hash(payment.id)` and
Python salts `hash()` on a str per process. Every policy inside one run met the same
outage, so each comparison was internally valid and only the published figure drifted —
which is why nothing caught it. The uplift moved between +10.7pp and +10.8pp depending on
the process.

`stable_seed()` replaces it with a blake2b digest. **Two independent full runs now agree on
every metric, to the last integer.** The honest number is **+10.7pp**, and it is now a
number that can be checked rather than believed.

`tests/test_harness.py` re-runs the evaluation under three `PYTHONHASHSEED` values and
fails if they disagree, which guards any future unstable hash rather than this one. The
harness had no test file at all before this.

**Do not quote a figure that is not in `data/results/eval.json`.**

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
| 9 | Sensitivity sweep + ablations — 45 configurations, 3 ablations | done |
| 10 | **README, architecture doc, video, submission** | **day 11 — not started** |

**274 tests passing.** ruff, ruff format, mypy strict all clean.

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

**A documented remedy has no reachable code path — day 9 was right, for a narrower
reason.** Measured, not read: over 8,000 simulated customers both `TaxonomyAware` and
`BanditPolicy` return **zero** instrument switches. But the branch is not reachable-and-
refused; it is never entered. It sits inside the `NEVER_RETRYABLE` case in both policies
(`taxonomy_aware.py:70`, `bandit.py:186`), gated on `remaining_attempts > 0`, and every
`NEVER_RETRYABLE` code has `max_attempts == 0` by construction.

Underneath it is a real inconsistency. Three codes carry `Remedy.NEW_INSTRUMENT`:

| code | class | max_attempts | what the policy actually does |
|---|---|---|---|
| `card_expired` | never_retryable | 0 | outreach carrying a payment link |
| `card_declined` | hard_declined | 1 | **one silent retry on the same card** |
| `payment_failed` | hard_declined | 1 | **one silent retry on the same card** |

The two hard-declined codes take `_hard_declined`, which never consults the remedy: the
taxonomy says *new instrument* and the policy retries the one that just failed. Razorpay's
published resolution for `card_declined` is "advise your customer to attempt the payment
again using another card" — which sides with the remedy, not with the retry.

Measured cost: **245 such retries per 8,000 customers**, so this is a small effect and not a
headline one. Ablate it on day 10; routing those two codes to outreach is what the taxonomy
already promises and what Razorpay documents.

The `card_expired` path, by contrast, is now *validated* rather than merely untested. It
sends outreach carrying a payment link, and Razorpay's own failed-subscription flow sends
"a link that the customer can use to change the card details" — same remedy, same
mechanism. **Panel answer: we do not silently move a charge to another card, because
Razorpay doesn't either.** Pinned by
`test_an_alternative_instrument_does_not_currently_change_the_decision`.

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

## What day 10 established

**The uplift survives every parameter nobody could source.** 45 configurations, nine
uncertain parameters walked across their full plausible ranges, the bandit retrained at
every point. The uplift never goes negative: **+6.19pp at worst, +15.18pp at best**. The
worst case is `outreach_response_rate` at 0.15, the pessimistic end of the parameter that
most directly drives the mechanism. That turns the claim from "+10.7pp under our
assumptions" into "at least +6.2pp under any plausible assumption".

Two of 45 configurations fall outside the calibration gate, both at `shortfall_high` >= 2.55
— the fitted parameter, swept widest on purpose. They are reported in `docs/RESULTS.md`
rather than dropped, and no claim rests on them.

**The ablations say which part earns it.** Each variant differs from the routing policy by
exactly one branch:

| removed | recovery | worth |
|---|---|---|
| session-awareness | 57.8% — *below the 58.1% baseline* | **+9.46pp** |
| rail-awareness | 63.4% | +3.87pp |
| following Razorpay's guidance | 67.1% | costs only 0.22pp |

Strip out session-awareness and the policy is worse than the fixed schedule it is supposed
to beat. **That is the pitch**: nearly the whole advantage comes from recognising that some
declines need the customer back rather than another retry — not from clever timing.

**Two swept parameters were measuring nothing.** `contact_fatigue_halflife_hours` was read
by no code at all, and `outage_rate_per_bank_day` had been superseded by NPCI's per-bank
rates on `Bank`. Ten of the original 55 configurations were inert, and
`audit_parameters.py` reported OK throughout, because it checks that invented parameters
are *listed* in the sweep, not that the simulator *reads* them. Both are removed;
`TestEverySweptParameterIsLive` now fails if a swept parameter does not move the outcome.
Removing them changed no measured number — verified by re-running the evaluation and
diffing against the committed file: zero mismatches.

**`docs/RESULTS.md` is generated, not written.** `scripts/render_results.py` renders it
from `eval.json` and `sweep.json`; `--check` fails if they have drifted. That check and
`audit_parameters.py` now both run in CI, where neither ran before.

---

## Day 11 plan

README and ARCHITECTURE.md, then record the video and submit. The numbers all come from
`docs/RESULTS.md` — do not retype any of them. Regenerate `docs/benchmark.html` from
`data/results/eval.json` and redeploy to the same artifact URL
(https://claude.ai/code/artifact/b7458ae4-ba93-4758-90f8-f3c980eb9320).
