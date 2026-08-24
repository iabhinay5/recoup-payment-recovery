"""Terminal rendering of evaluation results.

Kept separate from the harness so that measurement and presentation cannot drift into
each other. The harness produces numbers; nothing here computes one.
"""

from __future__ import annotations

from recoup.eval.harness import EvalResult

__all__ = ["format_comparison", "format_waste_breakdown"]

RECURLY_PUBLISHED_RECOVERY = 0.58
"""Recurly's published recovery rate for a Day-1/3/5/7 schedule with no customer contact.

The calibration target, not an input to the simulator. See docs/DECISIONS.md ADR-008.
"""


def _rupees(paise: int) -> str:
    return f"Rs {paise / 100:,.0f}"


def format_comparison(results: list[EvalResult], baseline: str | None = None) -> str:
    """Render a comparison table, optionally with uplift against a named baseline."""
    if not results:
        return "(no results)"

    base = next((r for r in results if r.policy_name == baseline), None)

    header = (
        f"{'policy':<22} {'recovery':>9} {'revenue':>9} {'att/ep':>7} "
        f"{'wasted':>8} {'contacts':>9} {'opt-out':>8}"
    )
    lines = [header, "-" * len(header)]

    for result in results:
        lines.append(
            f"{result.policy_name:<22} "
            f"{result.recovery_rate:>8.1%} "
            f"{result.revenue_recovery_rate:>8.1%} "
            f"{result.attempts_per_episode:>7.2f} "
            f"{result.wasted_attempt_rate:>7.1%} "
            f"{result.contacts_per_episode:>9.2f} "
            f"{result.opt_out_rate:>7.1%}"
        )

    if base is not None:
        lines.append("")
        lines.append(f"uplift vs {base.policy_name}:")
        for result in results:
            if result.policy_name == base.policy_name:
                continue
            delta_recovery = result.recovery_rate - base.recovery_rate
            delta_revenue = result.revenue_recovered_paise - base.revenue_recovered_paise
            delta_attempts = result.attempts_per_episode - base.attempts_per_episode
            lines.append(
                f"  {result.policy_name:<22} "
                f"{delta_recovery:+.1%} recovery   "
                f"{_rupees(delta_revenue):>14} revenue   "
                f"{delta_attempts:+.2f} attempts/episode"
            )

    return "\n".join(lines)


def format_waste_breakdown(result: EvalResult) -> str:
    """Show where a policy's wasted attempts went, by decline class.

    This is the diagnostic the whole project turns on. A policy that spends most of its
    attempts on session-conditional or never-retryable declines is not unlucky — it is
    ignoring information it already has.
    """
    total = result.wasted_attempts
    if not total:
        return f"{result.policy_name}: no wasted attempts"

    lines = [f"{result.policy_name}: {total:,} wasted attempts"]
    for name, count in sorted(result.wasted_by_class.items(), key=lambda kv: -kv[1]):
        bar = "#" * round(40 * count / total)
        lines.append(f"  {name:<22} {count:>6,}  {count / total:>6.1%}  {bar}")
    return "\n".join(lines)


def format_calibration_check(result: EvalResult) -> str:
    """Compare a fixed-schedule baseline against Recurly's published figure.

    This is the day 4 gate in docs/PLAN.md: if the simulator does not reproduce the
    published benchmark, its calibration is wrong and no policy built on top of it means
    anything.
    """
    observed = result.recovery_rate
    delta = observed - RECURLY_PUBLISHED_RECOVERY
    verdict = "PASS" if abs(delta) <= 0.03 else "FAIL"

    return "\n".join(
        (
            "calibration check (day 4 gate)",
            f"  published (Recurly, ~40M transactions)  {RECURLY_PUBLISHED_RECOVERY:.1%}",
            f"  simulated ({result.policy_name})        {observed:.1%}",
            f"  difference                              {delta:+.1%}",
            f"  verdict                                 {verdict}  (tolerance +/-3.0pp)",
        )
    )
