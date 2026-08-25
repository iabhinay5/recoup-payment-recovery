"""Sensitivity sweeps and ablations: what the headline number rests on.

Run:  python scripts/run_sweep.py            (~50 min)
      python scripts/run_sweep.py --smoke    (tiny, for checking the plumbing)

A single measured uplift says the system works. It does not say whether it would still
work if the parameters nobody could source had been chosen differently, and it does not
say which part of the system earned it. Those are the two questions this answers, and both
are questions a panel asks.

**Sensitivity.** Each parameter in ``SWEPT_PARAMETERS`` is walked across its full plausible
range with everything else held at default, and the whole comparison is re-run at every
point — including retraining the bandit, because a policy tuned under one set of dynamics
is not evidence about another. What matters is not the uplift at the default point but the
*worst* uplift anywhere in the ranges. The calibration gate is re-checked at every point
too: a parameter setting that makes the simulator stop reproducing the published Recurly
baseline is not a setting under which our own number means anything.

**Ablations.** Each variant in ``recoup.policies.ablations`` differs from ``TaxonomyAware``
by exactly one branch, so the gap between them attributes the uplift to that branch rather
than to the policy as a whole.

Runs smaller than ``scripts/run_eval.py``: 16,000 customers rather than 40,000, and two
bandit epochs rather than three. At that size the uplift lands within about a tenth of a
point of the full run, which is far finer than the effects being looked for here, and it
turns a two-day sweep into an hour.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recoup.eval.harness import EvalResult, build_world, evaluate
from recoup.eval.provenance import git_commit, git_dirty
from recoup.eval.report import RECURLY_PUBLISHED_RECOVERY
from recoup.policies import (
    BanditPolicy,
    FixedSchedule,
    RailBlind,
    RemedyRouted,
    SessionBlind,
    TaxonomyAware,
    split_world,
    train_bandit,
)
from recoup.sim.episode import Policy
from recoup.sim.params import SWEPT_PARAMETERS, SimParams

DEFAULT_CUSTOMERS = 16_000
DEFAULT_EPOCHS = 2
DEFAULT_POINTS = 5
CALIBRATION_TOLERANCE = 0.03
OUTPUT = Path("data/results/sweep.json")

BASELINE = "fixed_1_3_5_7"
BANDIT = "bandit_greedy"


def _metrics(result: EvalResult) -> dict[str, Any]:
    return {
        "policy": result.policy_name,
        "recovery_rate": result.recovery_rate,
        "revenue_recovery_rate": result.revenue_recovery_rate,
        "attempts_per_episode": result.attempts_per_episode,
        "wasted_attempt_rate": result.wasted_attempt_rate,
        "contacts_per_episode": result.contacts_per_episode,
        "opt_out_rate": result.opt_out_rate,
        "wasted_by_class": result.wasted_by_class,
    }


def measure(
    params: SimParams,
    epochs: int,
    extra: tuple[Policy, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Train and evaluate one configuration. Returns metrics keyed by policy name.

    The bandit is retrained inside every call. Reusing one trained under the default
    parameters would measure how well a policy fitted to one world transfers to another,
    which is a different and much weaker claim than the one being made.
    """
    world = build_world(params)
    train, test = split_world(world, train_fraction=0.5)

    bandit = BanditPolicy()
    train_bandit(bandit, train, epochs=epochs)
    bandit.explore = False

    policies: list[Policy] = [FixedSchedule(), TaxonomyAware(), *extra, bandit]
    return {p.name: _metrics(evaluate(p, test)) for p in policies}


def sensitivity(customers: int, seed: int, epochs: int, points: int) -> list[dict[str, Any]]:
    """Walk every uncertain parameter across its range, re-running the whole comparison."""
    out: list[dict[str, Any]] = []
    total = len(SWEPT_PARAMETERS)

    for index, (name, span) in enumerate(SWEPT_PARAMETERS.items(), start=1):
        entry: dict[str, Any] = {
            "parameter": name,
            "note": span.note,
            "low": span.low,
            "high": span.high,
            "default": span.default,
            "points": [],
        }
        for value in span.steps(points):
            params = replace(SimParams(n_customers=customers, seed=seed), **{name: float(value)})
            started = time.time()
            measured = measure(params, epochs)

            base = measured[BASELINE]["recovery_rate"]
            bandit = measured[BANDIT]["recovery_rate"]
            delta = base - RECURLY_PUBLISHED_RECOVERY
            calibrated = abs(delta) <= CALIBRATION_TOLERANCE
            entry["points"].append(
                {
                    "value": float(value),
                    "baseline_recovery": base,
                    "taxonomy_recovery": measured["taxonomy_aware"]["recovery_rate"],
                    "bandit_recovery": bandit,
                    "uplift_pp": (bandit - base) * 100.0,
                    "baseline_attempts": measured[BASELINE]["attempts_per_episode"],
                    "bandit_attempts": measured[BANDIT]["attempts_per_episode"],
                    "calibration_delta": delta,
                    "calibration_ok": calibrated,
                }
            )
            gate = "ok" if calibrated else "OUT OF CALIBRATION"
            elapsed = time.time() - started
            print(
                f"  [{index:2d}/{total}] {name}={value:<8.4g} "
                f"base={base * 100:5.2f}% bandit={bandit * 100:5.2f}% "
                f"uplift={(bandit - base) * 100:+5.2f}pp  cal={gate}  ({elapsed:.0f}s)",
                flush=True,
            )
        out.append(entry)

    return out


def run(customers: int, seed: int, epochs: int, points: int) -> dict[str, Any]:
    started = time.time()
    params = SimParams(n_customers=customers, seed=seed)

    print(f"ablations at default parameters ({customers:,} customers, {epochs} epochs)")
    ablations = measure(params, epochs, extra=(RailBlind(), SessionBlind(), RemedyRouted()))
    reference = ablations[BASELINE]["recovery_rate"]
    for name, metrics in ablations.items():
        gap = (metrics["recovery_rate"] - reference) * 100.0
        print(
            f"  {name:<26} {metrics['recovery_rate'] * 100:5.2f}%  "
            f"{metrics['attempts_per_episode']:.2f} att/ep  ({gap:+.2f}pp vs baseline)",
            flush=True,
        )

    print(f"\nsensitivity: {len(SWEPT_PARAMETERS)} parameters x {points} points")
    curves = sensitivity(customers, seed, epochs, points)

    every = [(curve["parameter"], point) for curve in curves for point in curve["points"]]
    worst_name, worst = min(every, key=lambda pair: pair[1]["uplift_pp"])
    best_name, best = max(every, key=lambda pair: pair[1]["uplift_pp"])
    uncalibrated = [
        {
            "parameter": name,
            "value": point["value"],
            "baseline_recovery": point["baseline_recovery"],
        }
        for name, point in every
        if not point["calibration_ok"]
    ]

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "elapsed_seconds": round(time.time() - started, 1),
        "config": {
            "n_customers": customers,
            "seed": seed,
            "bandit_epochs": epochs,
            "points_per_parameter": points,
            "train_fraction": 0.5,
        },
        "baseline": BASELINE,
        "calibration_tolerance": CALIBRATION_TOLERANCE,
        "ablations": ablations,
        "sensitivity": curves,
        "summary": {
            "n_configurations": len(every),
            "uplift_min_pp": worst["uplift_pp"],
            "uplift_max_pp": best["uplift_pp"],
            "worst_parameter": worst_name,
            "worst_value": worst["value"],
            "best_parameter": best_name,
            "uncalibrated_points": uncalibrated,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny run, checks the plumbing")
    parser.add_argument("--customers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    # Same guard as run_eval.py: a smoke run must not overwrite published results.
    if args.smoke and args.out == OUTPUT:
        args.out = OUTPUT.with_name("sweep_smoke.json")
        print(f"--smoke: writing to {args.out} so {OUTPUT} is left alone")

    customers = args.customers or (2_000 if args.smoke else DEFAULT_CUSTOMERS)
    points = 2 if args.smoke else args.points
    epochs = 1 if args.smoke else args.epochs

    document = run(customers, args.seed, epochs, points)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")

    summary = document["summary"]
    print(
        f"\nuplift across {summary['n_configurations']} configurations: "
        f"{summary['uplift_min_pp']:+.2f}pp to {summary['uplift_max_pp']:+.2f}pp "
        f"(worst at {summary['worst_parameter']}={summary['worst_value']:.4g})"
    )
    if summary["uncalibrated_points"]:
        print(f"{len(summary['uncalibrated_points'])} point(s) outside the calibration gate")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
