"""Run the full policy comparison and write the result to disk.

Run:  python scripts/run_eval.py            (40,000 customers, 20,000 held out)
      python scripts/run_eval.py --quick    (4,000 customers, for a smoke check)

Until now the headline numbers lived in a terminal scrollback and were copied by hand into
documents. That is the shape of mistake that ends a panel: a figure nobody can regenerate,
differing by a point from the one in the README, with no way to tell which is current.

So the comparison writes a JSON file, and everything that displays a number reads it. The
file records the parameters and the git commit it was produced under, because a result
without the configuration that produced it is an anecdote.

**Every policy is measured on the same held-out half.** The bandit learns on the training
half; the baselines have nothing to learn but are still evaluated on the test half only,
so the comparison is like for like rather than one policy being scored on a different
population from the others.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recoup.eval.harness import EvalResult, build_world, evaluate
from recoup.eval.report import (
    RECURLY_PUBLISHED_RECOVERY,
    format_calibration_check,
    format_comparison,
    format_waste_breakdown,
)
from recoup.policies import (
    OUTREACH_ARMS,
    RETRY_DELAYS_HOURS,
    AggressiveRetry,
    BanditPolicy,
    ExponentialBackoff,
    FixedSchedule,
    NoRetry,
    OutreachOnly,
    TaxonomyAware,
    split_world,
    train_bandit,
)
from recoup.sim.episode import Policy
from recoup.sim.params import SimParams

DEFAULT_CUSTOMERS = 40_000
QUICK_CUSTOMERS = 4_000
CALIBRATION_TOLERANCE = 0.03
OUTPUT = Path("data/results/eval.json")

BASELINE = "fixed_1_3_5_7"
"""The policy every uplift is quoted against. Recurly's published schedule."""


def git_commit() -> str | None:
    """The commit these numbers were produced under, if this is a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def as_dict(result: EvalResult) -> dict[str, Any]:
    """One policy's measured outcome, flattened for the results file."""
    return {
        "policy": result.policy_name,
        "n_episodes": result.n_episodes,
        "recovery_rate": result.recovery_rate,
        "revenue_recovery_rate": result.revenue_recovery_rate,
        "revenue_recovered_paise": result.revenue_recovered_paise,
        "revenue_total_paise": result.revenue_total_paise,
        "attempts_per_episode": result.attempts_per_episode,
        "wasted_attempt_rate": result.wasted_attempt_rate,
        "wasted_attempts": result.wasted_attempts,
        "total_attempts": result.total_attempts,
        "contacts_per_episode": result.contacts_per_episode,
        "opt_out_rate": result.opt_out_rate,
        "refused_actions": result.refused_actions,
        "median_hours_to_recovery": result.median_hours_to_recovery,
        "wasted_by_class": result.wasted_by_class,
    }


def run(customers: int, seed: int, epochs: int) -> dict[str, Any]:
    """Train, evaluate, and assemble the results document."""
    params = SimParams(n_customers=customers, seed=seed)

    print(f"building world: {customers:,} customers, seed {seed}")
    started = time.time()
    world = build_world(params)
    train, test = split_world(world, train_fraction=0.5)
    print(f"  train {len(train.population.payments):,}  test {len(test.population.payments):,}")

    print(f"training bandit: {epochs} epoch(s)")
    bandit = BanditPolicy()
    report = train_bandit(bandit, train, epochs=epochs)
    print(
        report.summary(
            retry_labels=tuple(f"+{h:g}h" for h in RETRY_DELAYS_HOURS),
            outreach_labels=tuple(f"+{h:g}h {c.value}" for h, c in OUTREACH_ARMS),
        )
    )
    # Exploration off before measurement: the reported figure describes the policy that
    # would be deployed, not one still paying for information.
    bandit.explore = False

    policies: list[Policy] = [
        NoRetry(),
        FixedSchedule(),
        ExponentialBackoff(),
        AggressiveRetry(),
        OutreachOnly(),
        TaxonomyAware(),
        bandit,
    ]

    results: list[EvalResult] = []
    for policy in policies:
        print(f"evaluating {policy.name}")
        results.append(evaluate(policy, test))

    baseline = next(r for r in results if r.policy_name == BASELINE)
    observed = baseline.recovery_rate
    delta = observed - RECURLY_PUBLISHED_RECOVERY

    document: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "elapsed_seconds": round(time.time() - started, 1),
        "params": {
            "n_customers": customers,
            "seed": seed,
            "horizon_days": params.horizon_days,
            "train_fraction": 0.5,
            "bandit_epochs": epochs,
        },
        "n_episodes": baseline.n_episodes,
        "baseline": BASELINE,
        "calibration": {
            "published_recovery": RECURLY_PUBLISHED_RECOVERY,
            "observed_recovery": observed,
            "delta": delta,
            "tolerance": CALIBRATION_TOLERANCE,
            "verdict": "PASS" if abs(delta) <= CALIBRATION_TOLERANCE else "FAIL",
            "source": "Recurly, ~40M subscription transactions, Day-1/3/5/7, no contact",
        },
        "training": {
            "epochs": report.epochs,
            "train_episodes": report.train_episodes,
            "decisions": report.total_decisions,
            "retry_pulls": list(report.retry_pulls),
            "outreach_pulls": list(report.outreach_pulls),
        },
        "policies": [as_dict(r) for r in results],
    }
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small run, for a smoke check")
    parser.add_argument("--customers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    customers = args.customers or (QUICK_CUSTOMERS if args.quick else DEFAULT_CUSTOMERS)
    document = run(customers, args.seed, args.epochs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")

    results = [
        EvalResult(
            policy_name=p["policy"],
            n_episodes=p["n_episodes"],
            recovered=round(p["recovery_rate"] * p["n_episodes"]),
            revenue_recovered_paise=p["revenue_recovered_paise"],
            revenue_total_paise=p["revenue_total_paise"],
            total_attempts=p["total_attempts"],
            wasted_attempts=p["wasted_attempts"],
            total_contacts=round(p["contacts_per_episode"] * p["n_episodes"]),
            opt_outs=round(p["opt_out_rate"] * p["n_episodes"]),
            refused_actions=p["refused_actions"],
            wasted_by_class=p["wasted_by_class"],
        )
        for p in document["policies"]
    ]

    print()
    print(format_comparison(results, baseline=BASELINE))
    print()
    print(format_calibration_check(next(r for r in results if r.policy_name == BASELINE)))
    print()
    for name in (BASELINE, "bandit_greedy"):
        result = next((r for r in results if r.policy_name == name), None)
        if result is not None:
            print(format_waste_breakdown(result))
            print()

    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
