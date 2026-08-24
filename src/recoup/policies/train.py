"""Training the learned policy, with the split that keeps its result honest.

A bandit trained and evaluated on the same episodes reports how well it memorised them,
not how well it works. The population is therefore partitioned: the policy learns on one
half and is measured on the other, with exploration switched off at evaluation time so the
reported number reflects the policy someone would actually deploy rather than one still
paying for information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recoup.eval.harness import EvalResult, World, evaluate
from recoup.policies.bandit import BanditPolicy
from recoup.sim.episode import run_episode
from recoup.sim.generator import Population
from recoup.sim.outcomes import OutcomeModel

__all__ = ["TrainingReport", "split_world", "train_bandit"]


def split_world(world: World, train_fraction: float = 0.5) -> tuple[World, World]:
    """Partition a world into disjoint training and evaluation halves.

    Split by customer, not by episode, so that no customer appears on both sides. With one
    payment per customer the distinction is currently cosmetic, but it stops being cosmetic
    the moment a customer has more than one failed payment, and getting it wrong then would
    be a subtle leak rather than an obvious one.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    payments = world.population.payments
    cut = int(len(payments) * train_fraction)

    customers = world.population.customers_by_id
    train_payments = payments[:cut]
    test_payments = payments[cut:]

    def _sub(subset: tuple) -> World:  # type: ignore[type-arg]
        return World(
            population=Population(
                banks=world.population.banks,
                customers=tuple(customers[p.customer_id] for p in subset),
                payments=subset,
            ),
            rails=world.rails,
            outcomes=world.outcomes,
            params=world.params,
        )

    return _sub(train_payments), _sub(test_payments)


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """What happened during training. Reported so the run is inspectable, not asserted."""

    epochs: int
    train_episodes: int
    retry_pulls: tuple[int, ...]
    outreach_pulls: tuple[int, ...]

    @property
    def total_decisions(self) -> int:
        return sum(self.retry_pulls) + sum(self.outreach_pulls)

    def summary(self, retry_labels: tuple[str, ...], outreach_labels: tuple[str, ...]) -> str:
        lines = [
            f"trained {self.epochs} epoch(s) over {self.train_episodes:,} episodes "
            f"({self.total_decisions:,} decisions)",
            "  retry arm pulls:",
        ]
        total = max(sum(self.retry_pulls), 1)
        for label, pulls in zip(retry_labels, self.retry_pulls, strict=True):
            bar = "#" * round(30 * pulls / total)
            lines.append(f"    {label:<14} {pulls:>7,}  {pulls / total:>5.1%}  {bar}")
        lines.append("  outreach arm pulls:")
        total = max(sum(self.outreach_pulls), 1)
        for label, pulls in zip(outreach_labels, self.outreach_pulls, strict=True):
            bar = "#" * round(30 * pulls / total)
            lines.append(f"    {label:<14} {pulls:>7,}  {pulls / total:>5.1%}  {bar}")
        return "\n".join(lines)


def train_bandit(
    policy: BanditPolicy,
    world: World,
    epochs: int = 3,
    seed_offset: int = 10_000,
) -> TrainingReport:
    """Run training episodes, letting the policy learn from each outcome.

    Multiple epochs replay the same population with different random draws. That is not
    the same as more data — it is the same customers meeting different luck — so it
    improves the estimate of each arm's value without inventing new situations.

    Episodes use generators offset from the evaluation seeds, so training never consumes
    the exact random draws the policy is later measured against.
    """
    customers = world.population.customers_by_id
    policy.explore = True

    for epoch in range(epochs):
        for index, payment in enumerate(world.population.payments):
            rng = np.random.default_rng([world.params.seed, seed_offset + epoch, index])
            run_episode(
                payment=payment,
                customer=customers[payment.customer_id],
                policy=policy,
                outcomes=OutcomeModel(world.params, world.rails),
                rails=world.rails,
                params=world.params,
                rng=rng,
            )

    return TrainingReport(
        epochs=epochs,
        train_episodes=len(world.population.payments),
        retry_pulls=tuple(int(n) for n in policy.retry_bandit.pulls),
        outreach_pulls=tuple(int(n) for n in policy.outreach_bandit.pulls),
    )


def train_and_evaluate(
    world: World,
    epochs: int = 3,
    train_fraction: float = 0.5,
) -> tuple[BanditPolicy, TrainingReport, EvalResult]:
    """Train on one half of ``world`` and measure on the other.

    Exploration is disabled before evaluation: the reported figure describes the policy
    that would be deployed, not one still spending attempts to learn.
    """
    train, test = split_world(world, train_fraction)
    policy = BanditPolicy()
    report = train_bandit(policy, train, epochs=epochs)

    policy.explore = False
    return policy, report, evaluate(policy, test)
