"""Evaluation harness: run policies over a population and measure what happened.

Two methodological commitments are worth calling out, because both are the difference
between a number and a result.

**Common random numbers.** Every policy faces an identically seeded world — the same
customers, the same outages, and the same per-episode random draws. Each episode derives
its generator from ``(seed, episode_index)`` rather than consuming a shared stream,
because policies make different numbers of random calls and a shared stream would
desynchronise them: policy B would face different luck than policy A purely because policy
A retried once more. Pairing the draws removes that noise, so a measured difference is
attributable to the policy rather than to sampling.

**No live model calls.** The harness never touches an LLM. Anything the model contributes
is resolved and cached beforehand (docs/DECISIONS.md ADR-005), which is what makes a run
reproducible rather than merely repeatable.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from recoup.sim.entities import HOURS_PER_DAY, Customer, FailedPayment
from recoup.sim.episode import EpisodeResult, Policy, run_episode
from recoup.sim.generator import Population, generate_population
from recoup.sim.outcomes import OutcomeModel
from recoup.sim.params import SimParams
from recoup.sim.rails import Outage, RailHealth, generate_outages
from recoup.taxonomy import DeclineClass, lookup

__all__ = ["EvalResult", "World", "build_world", "compare", "evaluate", "stable_seed"]


def stable_seed(text: str) -> int:
    """A seed derived from ``text`` that is identical in every process.

    ``hash()`` on a ``str`` is salted per process by ``PYTHONHASHSEED``. An RNG seeded
    with it is therefore stable *within* one run and different *between* runs, which is
    close to invisible: every policy in a single process meets the same outage, so the
    comparison stays internally valid and only the published number moves. It was moving
    the headline uplift between +10.7pp and +10.8pp before this function existed.

    A digest carries no per-process salt, so a results file can be regenerated and
    compared against its predecessor — which is the whole promise of ADR-010.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


@dataclass(frozen=True, slots=True)
class World:
    """A fixed simulated world that every policy is evaluated against.

    Built once and shared, so no policy gets an easier population than another.
    """

    population: Population
    rails: RailHealth
    outcomes: OutcomeModel
    params: SimParams


def build_world(params: SimParams) -> World:
    """Construct the shared evaluation world from ``params``."""
    population = generate_population(params)
    # A fixed offset from the run seed, so the outage schedule is reproducible and
    # distinct from the per-episode streams below.
    rng = np.random.default_rng([params.seed, 991])
    outages = generate_outages(
        population.bank_outage_profiles,
        params.horizon_days * HOURS_PER_DAY,
        rng,
    )
    rails = RailHealth(population.bank_decline_rates, outages)
    return World(population, rails, OutcomeModel(params, rails), params)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Aggregate outcome of running one policy over one world."""

    policy_name: str
    n_episodes: int
    recovered: int
    revenue_recovered_paise: int
    revenue_total_paise: int
    total_attempts: int
    wasted_attempts: int
    total_contacts: int
    opt_outs: int
    refused_actions: int
    hours_to_recovery: tuple[float, ...] = field(default_factory=tuple)
    wasted_by_class: dict[str, int] = field(default_factory=dict)

    @property
    def recovery_rate(self) -> float:
        return self.recovered / self.n_episodes if self.n_episodes else 0.0

    @property
    def revenue_recovery_rate(self) -> float:
        return (
            self.revenue_recovered_paise / self.revenue_total_paise
            if self.revenue_total_paise
            else 0.0
        )

    @property
    def attempts_per_episode(self) -> float:
        return self.total_attempts / self.n_episodes if self.n_episodes else 0.0

    @property
    def wasted_attempt_rate(self) -> float:
        """Share of all attempts that did not recover anything.

        The efficiency number. Two policies can recover identical revenue while one spends
        four times the attempts, and attempts are not free — they consume issuer goodwill,
        rail capacity, and the customer's patience.
        """
        return self.wasted_attempts / self.total_attempts if self.total_attempts else 0.0

    @property
    def contacts_per_episode(self) -> float:
        return self.total_contacts / self.n_episodes if self.n_episodes else 0.0

    @property
    def opt_out_rate(self) -> float:
        return self.opt_outs / self.n_episodes if self.n_episodes else 0.0

    @property
    def median_hours_to_recovery(self) -> float | None:
        return statistics.median(self.hours_to_recovery) if self.hours_to_recovery else None


def _summarise(name: str, results: list[EpisodeResult]) -> EvalResult:
    wasted_by_class: Counter[str] = Counter()
    for result in results:
        for attempt in result.attempts:
            if attempt.succeeded or attempt.decline_code is None:
                continue
            wasted_by_class[lookup(attempt.decline_code).decline_class.value] += 1

    return EvalResult(
        policy_name=name,
        n_episodes=len(results),
        recovered=sum(r.recovered for r in results),
        revenue_recovered_paise=sum(r.revenue_recovered_paise for r in results),
        revenue_total_paise=sum(r.amount_paise for r in results),
        total_attempts=sum(r.attempt_count for r in results),
        wasted_attempts=sum(r.wasted_attempts for r in results),
        total_contacts=sum(r.contact_count for r in results),
        opt_outs=sum(r.opted_out for r in results),
        refused_actions=sum(r.refused_actions for r in results),
        hours_to_recovery=tuple(
            r.recovered_at_hours for r in results if r.recovered_at_hours is not None
        ),
        wasted_by_class=dict(wasted_by_class),
    )


def _episode_rails(world: World, payment: FailedPayment, customer: Customer) -> RailHealth:
    """Rail health for one episode, made consistent with the failure that opened it.

    A payment that failed with ``bank_technical_error`` failed *during an outage*. If the
    episode starts with a healthy rail, it contradicts its own premise: a rail-aware policy
    is asked to react to a problem that was never there, and the value of consulting rail
    health is measured as zero for the wrong reason.

    The outage is episode-local because every episode measures time from its own failure,
    so a single shared timeline cannot be consistent with all of them at once.
    """
    if lookup(payment.initial_decline_code).decline_class is not DeclineClass.RAIL_CONDITIONAL:
        return world.rails

    instrument = next(
        (i for i in customer.instruments if i.id == payment.instrument_id),
        customer.primary_instrument,
    )
    # Seeded from the payment id so the outage is stable across policies — otherwise a
    # policy would be rewarded or punished by a different outage than its rivals faced.
    # stable_seed rather than hash() so it is also stable across *runs*; see its docstring.
    rng = np.random.default_rng([world.params.seed, stable_seed(payment.id)])
    remaining = float(rng.exponential(world.params.outage_mean_duration_hours))
    return world.rails.with_outage(Outage(instrument.bank_id, -0.01, remaining + 0.01))


def evaluate(policy: Policy, world: World) -> EvalResult:
    """Run ``policy`` over every payment in ``world`` and aggregate the outcome."""
    customers = world.population.customers_by_id
    results: list[EpisodeResult] = []

    for index, payment in enumerate(world.population.payments):
        customer = customers[payment.customer_id]
        # Derived per episode rather than drawn from a shared stream, so every policy
        # meets identical luck on identical episodes. See the module docstring.
        rng = np.random.default_rng([world.params.seed, index])
        rails = _episode_rails(world, payment, customer)
        results.append(
            run_episode(
                payment=payment,
                customer=customer,
                policy=policy,
                outcomes=OutcomeModel(world.params, rails),
                rails=rails,
                params=world.params,
                rng=rng,
            )
        )

    return _summarise(policy.name, results)


def compare(policies: list[Policy], params: SimParams) -> list[EvalResult]:
    """Evaluate several policies against one shared world."""
    world = build_world(params)
    return [evaluate(policy, world) for policy in policies]
