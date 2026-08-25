"""Tests for the evaluation harness — specifically, that its numbers reproduce.

A result that cannot be regenerated is an anecdote (docs/DECISIONS.md ADR-010), and until
these existed nothing checked the claim. The harness *was* reproducible within a single
process and was not reproducible between processes, which is the hard version of this bug
to notice: every policy in one run meets the same luck, so the comparison stays internally
valid and only the published figure drifts. It drifted the headline uplift between
+10.7pp and +10.8pp.
"""

from __future__ import annotations

import os
import subprocess
import sys

from recoup.eval.harness import build_world, evaluate, stable_seed
from recoup.policies import TaxonomyAware
from recoup.sim.params import SimParams

SMALL = SimParams(n_customers=400, seed=0)


class TestStableSeed:
    def test_it_is_pinned_to_a_known_value(self) -> None:
        """A literal, because the property under test is that this never changes.

        ``hash()`` would satisfy any test that only compared two calls in one process.
        Only a constant carried across processes can catch a per-process salt.
        """
        assert stable_seed("pay_TTh0P3vWkSpZtu") == 1948882479

    def test_different_payments_get_different_streams(self) -> None:
        seeds = {stable_seed(f"pay_{i}") for i in range(500)}
        assert len(seeds) == 500, "collisions would make two episodes share an outage"

    def test_it_stays_inside_the_seed_range(self) -> None:
        assert all(0 <= stable_seed(f"pay_{i}") < 2**31 for i in range(200))


class TestTheHarnessReproduces:
    def test_the_same_world_measures_the_same_twice(self) -> None:
        world = build_world(SMALL)
        first = evaluate(TaxonomyAware(), world)
        second = evaluate(TaxonomyAware(), world)

        assert first.recovered == second.recovered
        assert first.total_attempts == second.total_attempts
        assert first.wasted_by_class == second.wasted_by_class

    def test_a_rebuilt_world_measures_the_same(self) -> None:
        first = evaluate(TaxonomyAware(), build_world(SMALL))
        second = evaluate(TaxonomyAware(), build_world(SMALL))

        assert first.recovered == second.recovered
        assert first.wasted_by_class == second.wasted_by_class

    def test_it_reproduces_under_a_different_hash_seed(self) -> None:
        """The one that matters, and the one an in-process test cannot express.

        Python salts ``hash()`` on str per process unless PYTHONHASHSEED is pinned, so a
        harness seeded from ``hash(payment.id)`` returns different numbers on every run
        while looking perfectly deterministic from inside any single one. Running the same
        evaluation under two different salts is what makes that visible — and it guards
        every future unstable hash, not just the one that caused it.
        """
        outputs = {seed: _evaluate_in_subprocess(seed) for seed in ("0", "1", "12345")}
        distinct = set(outputs.values())

        assert len(distinct) == 1, (
            "the harness gives different results under different PYTHONHASHSEED values, "
            f"so its numbers cannot be regenerated: {outputs}"
        )


_SCRIPT = """
from recoup.eval.harness import build_world, evaluate
from recoup.policies import TaxonomyAware
from recoup.sim.params import SimParams

result = evaluate(TaxonomyAware(), build_world(SimParams(n_customers=400, seed=0)))
print(result.recovered, result.total_attempts, sorted(result.wasted_by_class.items()))
"""


def _evaluate_in_subprocess(hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=300,
    )
    return out.stdout.strip()
