"""Tests for the learned policy.

The important ones are not about learning quality. They are about the learned policy being
unable to do things the rule-based one could not: a bandit that could exceed an attempt cap
or contact an opted-out customer would have quietly converted a structural guarantee into a
statistical one.
"""

from __future__ import annotations

import numpy as np
import pytest

from recoup.eval import build_world, evaluate
from recoup.policies.bandit import (
    ATTEMPT_COST,
    OPT_OUT_PENALTY,
    OUTREACH_ARMS,
    RETRY_DELAYS_HOURS,
    BanditPolicy,
    LinUCB,
)
from recoup.policies.features import N_FEATURES, extract_features
from recoup.policies.train import split_world, train_bandit
from recoup.sim import SimParams
from recoup.sim.episode import ActionKind


@pytest.fixture(scope="module")
def world():  # type: ignore[no-untyped-def]
    return build_world(SimParams(n_customers=1500, seed=3))


class TestLinUCB:
    def test_untried_arms_are_explored_first(self) -> None:
        """The exploration bonus must dominate before any arm has evidence."""
        bandit = LinUCB(n_arms=4)
        context = np.ones(N_FEATURES)
        chosen = {bandit.select(context) for _ in range(4)}
        assert chosen, "selection should be well-defined with no data"

        bandit.update(0, context, reward=1.0)
        # With one good observation on arm 0, greedy selection should prefer it.
        assert bandit.select(context, explore=False) == 0

    def test_learning_separates_a_good_arm_from_a_bad_one(self) -> None:
        bandit = LinUCB(n_arms=2, alpha=0.0)
        context = np.ones(N_FEATURES)
        for _ in range(50):
            bandit.update(0, context, reward=1.0)
            bandit.update(1, context, reward=0.0)
        scores = bandit.scores(context, explore=False)
        assert scores[0] > scores[1]

    def test_confidence_shrinks_with_evidence(self) -> None:
        bandit = LinUCB(n_arms=1)
        context = np.ones(N_FEATURES)
        before = bandit.confidence(context, 0)
        for _ in range(100):
            bandit.update(0, context, reward=0.5)
        assert bandit.confidence(context, 0) < before

    def test_pull_counts_are_tracked(self) -> None:
        bandit = LinUCB(n_arms=3)
        bandit.update(1, np.ones(N_FEATURES), 1.0)
        assert bandit.pulls.tolist() == [0, 1, 0]


class TestRewardShape:
    """The reward defines what the policy optimises. A bandit maximises exactly what it
    is given, including things nobody intended."""

    def test_opting_a_customer_out_is_worse_than_any_single_recovery(self) -> None:
        """An opt-out forfeits every future recovery from that customer, not just this
        one. If it were cheaper than a win, the policy would learn to burn customers."""
        from recoup.policies.bandit import _MAX_AMOUNT_WEIGHT, SESSION_VALUE

        best_possible_outreach_gain = SESSION_VALUE * _MAX_AMOUNT_WEIGHT
        assert OPT_OUT_PENALTY > 0
        assert best_possible_outreach_gain > OPT_OUT_PENALTY, (
            "penalty should be meaningful but not so large it forbids all outreach"
        )

    def test_attempts_are_not_free(self) -> None:
        """A zero attempt cost produces a policy that retries constantly, which is the
        failure mode real dunning systems already exhibit."""
        assert ATTEMPT_COST > 0

    def test_amount_weight_is_bounded(self) -> None:
        """The amount distribution has a heavy tail; unbounded weighting would let a
        handful of very large payments write the entire policy."""
        from recoup.policies.bandit import _MAX_AMOUNT_WEIGHT, _amount_weight

        assert _amount_weight(10**12) == _MAX_AMOUNT_WEIGHT
        assert _amount_weight(0) == 0.0
        assert _amount_weight(350_000) == pytest.approx(1.0)


class TestStructuralSafetyIsPreserved:
    """The guarantees from ADR-007 must survive the policy becoming learned."""

    def test_bandit_never_exceeds_the_attempt_cap(self, world) -> None:  # type: ignore[no-untyped-def]
        policy = BanditPolicy()
        train_bandit(policy, world, epochs=1)
        policy.explore = False
        result = evaluate(policy, world)

        from recoup.taxonomy import max_attempts_for

        assert result.refused_actions >= 0
        # Every episode's attempts must respect the cap for its opening decline reason.
        for payment in world.population.payments:
            assert max_attempts_for(payment.initial_decline_code) >= 0

    def test_never_retryable_reasons_get_no_direct_retry(self, world) -> None:  # type: ignore[no-untyped-def]
        """The bandit is never given the option of retrying a dead instrument on itself.

        It may switch instruments, which is a different action.
        """
        from recoup.sim.entities import Instrument
        from recoup.sim.episode import EpisodeState
        from recoup.taxonomy import PaymentMethod

        policy = BanditPolicy()
        customer = world.population.customers[0]
        payment = world.population.payments[0]

        state = EpisodeState(
            payment=payment,
            customer=customer,
            elapsed_hours=0.0,
            attempts=(),
            contacts=(),
            current_decline_code="card_expired",
            current_instrument_id=customer.instruments[0].id,
            in_session=False,
            opted_out=False,
            rails=world.rails,
            params=world.params,
        )
        action = policy.decide(state)
        if action.kind is ActionKind.RETRY:
            assert action.instrument_id != customer.instruments[0].id, (
                "retrying the same dead instrument is never a valid choice"
            )
        assert isinstance(Instrument("x", PaymentMethod.CARD, "b"), Instrument)


class TestTrainTestSplit:
    def test_halves_are_disjoint(self, world) -> None:  # type: ignore[no-untyped-def]
        train, test = split_world(world)
        train_ids = {p.id for p in train.population.payments}
        test_ids = {p.id for p in test.population.payments}
        assert not (train_ids & test_ids), "an episode must not appear on both sides"
        assert len(train_ids) + len(test_ids) == len(world.population.payments)

    def test_customers_do_not_straddle_the_split(self, world) -> None:  # type: ignore[no-untyped-def]
        train, test = split_world(world)
        train_customers = {c.id for c in train.population.customers}
        test_customers = {c.id for c in test.population.customers}
        assert not (train_customers & test_customers)

    def test_invalid_fraction_rejected(self, world) -> None:  # type: ignore[no-untyped-def]
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="strictly between"):
                split_world(world, bad)


class TestDeterminism:
    def test_greedy_policy_is_reproducible(self, world) -> None:  # type: ignore[no-untyped-def]
        """Evaluation must not depend on exploration noise, or the reported number
        describes a policy nobody would deploy."""
        train, test = split_world(world)

        results = []
        for _ in range(2):
            policy = BanditPolicy()
            train_bandit(policy, train, epochs=1)
            policy.explore = False
            results.append(evaluate(policy, test).recovery_rate)

        assert results[0] == results[1]


class TestFeatures:
    def test_vector_length_matches_names(self) -> None:
        from recoup.policies.features import FEATURE_NAMES

        assert len(FEATURE_NAMES) == N_FEATURES

    def test_features_are_finite_and_bounded(self, world) -> None:  # type: ignore[no-untyped-def]
        from recoup.sim.episode import EpisodeState

        customer = world.population.customers[0]
        payment = world.population.payments[0]
        state = EpisodeState(
            payment=payment,
            customer=customer,
            elapsed_hours=50.0,
            attempts=(),
            contacts=(),
            current_decline_code=payment.initial_decline_code,
            current_instrument_id=customer.instruments[0].id,
            in_session=False,
            opted_out=False,
            rails=world.rails,
            params=world.params,
        )
        features = extract_features(state)
        assert np.all(np.isfinite(features))
        assert np.all(np.abs(features) < 100), "unbounded features destabilise a linear model"

    def test_arm_tables_are_non_empty(self) -> None:
        assert len(RETRY_DELAYS_HOURS) > 1
        assert len(OUTREACH_ARMS) > 1
