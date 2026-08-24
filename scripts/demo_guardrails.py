"""Try to make Recoup harm a customer, and watch it refuse.

Run:  python scripts/demo_guardrails.py

Five deliberate attacks against the guardrail layer, narrated. This is the demonstration
worth putting on screen, because the interesting claim about a payment system is not how
much it recovers — it is what it refuses to do when something upstream goes wrong.

Nothing here is mocked. Every refusal comes from the same code path that runs in the
evaluation harness.
"""

from __future__ import annotations

import sys

from recoup.guardrails import Guardrails, Rule, idempotency_key

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"


def header(n: int, title: str, attack: str) -> None:
    print()
    print(f"{BOLD}[{n}] {title}{OFF}")
    print(f"    {DIM}attack: {attack}{OFF}")


def refused(verdict, expected: Rule) -> bool:
    """Print the verdict and confirm it matches the rule we expected to fire."""
    if verdict.allowed:
        print(f"    {RED}ALLOWED{OFF}  <- the guardrail failed to stop this")
        return False
    assert verdict.refusal is not None
    ok = verdict.refusal.rule is expected
    label = "DEFERRED" if verdict.deferred else "REFUSED"
    colour = GREEN if ok else RED
    print(f"    {colour}{label}{OFF}  {verdict.refusal.rule.value}")
    print(f"    {DIM}{verdict.refusal.detail}{OFF}")
    return ok


def allowed(verdict) -> bool:
    if not verdict.allowed:
        print(f"    {RED}REFUSED{OFF}  <- a legitimate action was blocked")
        return False
    print(f"    {GREEN}ALLOWED{OFF}  {DIM}legitimate action, permitted{OFF}")
    return True


def main() -> int:
    print(f"{BOLD}Recoup - guardrail demonstration{OFF}")
    print(f"{DIM}Every refusal below comes from the live code path, not a mock.{OFF}")

    results: list[bool] = []
    guards = Guardrails()

    # -- 1 ---------------------------------------------------------------------------
    header(
        1,
        "The double charge",
        "a webhook is delivered twice, so the same charge is issued twice",
    )
    key = idempotency_key("pay_88213", 0, "card_9f21")
    print(f"    {DIM}idempotency key: {key[:24]}...{OFF}")
    results.append(allowed(guards.check_retry(key, 0, "insufficient_funds", False)))
    guards.record_charge(key)
    print(f"    {DIM}charge executed. now the duplicate arrives:{OFF}")
    results.append(
        refused(guards.check_retry(key, 0, "insufficient_funds", False), Rule.DUPLICATE_CHARGE)
    )

    # -- 2 ---------------------------------------------------------------------------
    header(
        2,
        "The dead card",
        "a policy asks to retry a card that has already expired",
    )
    results.append(
        refused(
            guards.check_retry(
                idempotency_key("pay_88214", 0, "card_expired_x"),
                0,
                "insufficient_funds",
                instrument_expired=True,
            ),
            Rule.DEAD_INSTRUMENT,
        )
    )

    # -- 3 ---------------------------------------------------------------------------
    header(
        3,
        "The runaway policy",
        "a buggy or greedy policy demands a 5th retry where 4 are permitted",
    )
    fresh = Guardrails()
    for n in range(4):
        k = idempotency_key("pay_88215", n, "card_1a03")
        fresh.check_retry(k, n, "insufficient_funds", False)
        fresh.record_charge(k)
    print(f"    {DIM}4 attempts made, all permitted. the 5th:{OFF}")
    results.append(
        refused(
            fresh.check_retry(
                idempotency_key("pay_88215", 4, "card_1a03"), 4, "insufficient_funds", False
            ),
            Rule.ATTEMPT_CAP,
        )
    )

    # -- 4 ---------------------------------------------------------------------------
    header(
        4,
        "The 3am message",
        "outreach is scheduled while the customer is asleep",
    )
    results.append(
        refused(
            guards.check_outreach(hour_of_day=3.0, opted_out=False, contacts_in_window=0),
            Rule.QUIET_HOURS,
        )
    )
    print(f"    {DIM}note: deferred, not discarded - it sends at 08:00{OFF}")

    # -- 5 ---------------------------------------------------------------------------
    header(
        5,
        "The customer who said stop",
        "outreach is attempted after the customer opted out",
    )
    results.append(
        refused(
            guards.check_outreach(hour_of_day=12.0, opted_out=True, contacts_in_window=0),
            Rule.OPTED_OUT,
        )
    )

    # -- audit -----------------------------------------------------------------------
    print()
    print(f"{BOLD}Audit trail{OFF}")
    print(f"    {DIM}every refusal is recorded with the rule that caused it, so a merchant")
    print(f'    asking "why wasn\'t my customer retried?" gets an answer.{OFF}')
    for rule, count in sorted(guards.refusal_counts().items()):
        print(f"      {rule:<20} {count}")

    print()
    passed = all(results)
    if passed:
        print(f"{GREEN}{BOLD}All {len(results)} checks behaved correctly.{OFF}")
        print(f"{DIM}Harm is structurally unavailable, not merely unlikely.{OFF}")
    else:
        print(f"{RED}{BOLD}A guardrail did not hold. This is a release blocker.{OFF}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
