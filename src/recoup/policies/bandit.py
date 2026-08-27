"""Contextual bandit for retry timing and outreach channel.

**What this does and does not decide.** The decline taxonomy keeps every structural
decision: whether a retry can succeed at all, whether outreach is the only viable remedy,
whether an instrument switch is required. Those are knowable from Razorpay's published
semantics and do not need to be learned — learning them would mean rediscovering, at some
cost in data and confidence, that an expired card cannot be charged.

The bandit learns only what is genuinely unknown: *when* within the permitted window, and
*through which channel*. That is a numeric optimisation against a defined reward, which is
exactly the kind of problem a bandit is for and exactly the kind of problem an LLM is not
(docs/DECISIONS.md ADR-003).

**Why LinUCB.** It carries an explicit confidence interval per arm, so the exploration
bonus is inspectable rather than implicit: at any decision the policy can state which arm
it chose, what it expects that arm to return, and how uncertain it is. A policy that cannot
explain a choice is unusable in payments, where every automated action costs a real
customer something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from recoup.policies.features import FEATURE_NAMES, N_FEATURES, extract_features
from recoup.sim.entities import ContactChannel
from recoup.sim.episode import Action, EpisodeState
from recoup.taxonomy import DeclineClass, lookup

__all__ = [
    "OUTREACH_ARMS",
    "RETRY_DELAYS_HOURS",
    "BanditPolicy",
    "LinUCB",
]

RETRY_DELAYS_HOURS: tuple[float, ...] = (1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0, 120.0)
"""Candidate retry delays.

Spaced roughly geometrically. The long end matters because a customer three weeks into
their salary cycle may simply not have the money until the next credit, and no amount of
retrying within a day changes that.
"""

OUTREACH_ARMS: tuple[tuple[float, ContactChannel], ...] = (
    (6.0, ContactChannel.EMAIL),
    (12.0, ContactChannel.EMAIL),
    (24.0, ContactChannel.EMAIL),
    (48.0, ContactChannel.EMAIL),
    (12.0, ContactChannel.SMS),
    (24.0, ContactChannel.SMS),
    (48.0, ContactChannel.SMS),
    (24.0, ContactChannel.WHATSAPP),
)
"""Candidate outreach timings and channels.

Channel is an arm rather than a fixed choice because the trade-off is genuinely
uncertain: SMS reaches people who ignore email, and also annoys them faster. The reward
has to settle it.
"""


# Reward weights. These define what the policy is optimising, which is the single most
# consequential choice in this file — a bandit will faithfully maximise whatever it is
# given, including things nobody wanted.
ATTEMPT_COST = 0.02
"""Cost of one charge attempt. Small but non-zero: attempts consume issuer goodwill and
rail capacity, and a reward that treats them as free produces a policy that retries
constantly, which is the failure mode real dunning systems already have."""

CONTACT_COST = 0.05
"""Cost of one customer contact. Higher than an attempt because it is intrusive."""

OPT_OUT_PENALTY = 1.5
"""Cost of driving a customer to opt out. Deliberately larger than any single recovery is
worth: an opt-out forfeits every future recovery from that customer, not just this one."""

SESSION_VALUE = 0.7
"""Value of successful outreach relative to a completed payment. Below 1.0 because a
re-engaged customer still has to clear the charge."""

_REWARD_REFERENCE_PAISE = 350_000.0
_MAX_AMOUNT_WEIGHT = 5.0


def _amount_weight(amount_paise: int) -> float:
    """Revenue weighting for the reward, bounded.

    Without this the bandit maximises the *count* of recoveries and is indifferent between
    clearing a Rs 200 charge and a Rs 20,000 one. With it unbounded, the heavy tail of the
    amount distribution would let a handful of very large payments dominate learning
    entirely. Bounded weighting keeps revenue in the objective without letting outliers
    write the policy.
    """
    return min(amount_paise / _REWARD_REFERENCE_PAISE, _MAX_AMOUNT_WEIGHT)


class LinUCB:
    """Linear upper-confidence-bound contextual bandit.

    Maintains, per arm, the ridge-regression sufficient statistics ``A = X'X + lambda*I``
    and ``b = X'y``. The score for an arm is its predicted reward plus ``alpha`` standard
    deviations of predictive uncertainty, so arms that are merely untried get explored
    while arms that are reliably poor stop being chosen.

    ``alpha`` is the exploration weight. It is deliberately modest here: in payments, an
    exploratory action is not a cheap information purchase, it is a real charge attempt
    against a real customer.
    """

    def __init__(self, n_arms: int, n_features: int = N_FEATURES, alpha: float = 0.35) -> None:
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self._A = np.array([np.eye(n_features) for _ in range(n_arms)])
        self._b = np.zeros((n_arms, n_features))
        self.pulls = np.zeros(n_arms, dtype=int)

    def _theta(self, arm: int) -> np.ndarray:
        return np.linalg.solve(self._A[arm], self._b[arm])

    def scores(self, context: np.ndarray, explore: bool = True) -> np.ndarray:
        """Predicted reward per arm, optionally with the exploration bonus."""
        out = np.empty(self.n_arms)
        for arm in range(self.n_arms):
            a_inv_x = np.linalg.solve(self._A[arm], context)
            mean = float(self._theta(arm) @ context)
            if explore:
                variance = float(context @ a_inv_x)
                mean += self.alpha * np.sqrt(max(variance, 0.0))
            out[arm] = mean
        return out

    def select(self, context: np.ndarray, explore: bool = True) -> int:
        return int(np.argmax(self.scores(context, explore)))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self._A[arm] += np.outer(context, context)
        self._b[arm] += reward * context
        self.pulls[arm] += 1

    def confidence(self, context: np.ndarray, arm: int) -> float:
        """Predictive standard deviation for one arm. Used for explanation, not selection."""
        variance = float(context @ np.linalg.solve(self._A[arm], context))
        return float(np.sqrt(max(variance, 0.0)))

    def contributions(self, context: np.ndarray, arm: int) -> np.ndarray:
        """Each feature's share of this arm's predicted reward.

        The score is a dot product, so a feature's contribution is exactly
        ``theta[i] * x[i]``. Nothing is being approximated: this is the arithmetic the
        model performed. It is the reason a linear policy was chosen over something that
        would need its reasoning reconstructed afterwards.
        """
        contributions: np.ndarray = self._theta(arm) * context
        return contributions

    def to_dict(self) -> dict[str, Any]:
        """Serialise the learned statistics, so a trained policy can be reloaded.

        Without this, anything that wanted to *show* the policy would have to retrain it,
        and would then be showing a different policy from the one the results describe.
        """
        return {
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "A": self._A.tolist(),
            "b": self._b.tolist(),
            "pulls": self.pulls.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinUCB:
        bandit = cls(int(data["n_arms"]), int(data["n_features"]), float(data["alpha"]))
        bandit._A = np.array(data["A"], dtype=float)
        bandit._b = np.array(data["b"], dtype=float)
        bandit.pulls = np.array(data["pulls"], dtype=int)
        return bandit


@dataclass
class PendingDecision:
    """A decision awaiting its outcome, so the bandit can be credited afterwards."""

    arm: int
    context: np.ndarray
    is_retry: bool
    amount_paise: int


@dataclass
class BanditPolicy:
    """Taxonomy for structure, bandit for timing and channel.

    Structural decisions are taken exactly as ``TaxonomyAware`` takes them. Where that
    policy applies a fixed schedule, this one asks a bandit.
    """

    retry_bandit: LinUCB = field(default_factory=lambda: LinUCB(len(RETRY_DELAYS_HOURS)))
    outreach_bandit: LinUCB = field(default_factory=lambda: LinUCB(len(OUTREACH_ARMS)))
    explore: bool = True
    max_contacts: int = 2
    pending: PendingDecision | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "bandit" if self.explore else "bandit_greedy"

    def decide(self, state: EpisodeState) -> Action:
        reason = lookup(state.current_decline_code)
        self.pending = None

        match reason.decline_class:
            case DeclineClass.NEVER_RETRYABLE:
                # No timing decision exists here: either another instrument is available
                # or the customer has to act. Nothing for a bandit to learn.
                alternatives = state.customer.alternatives_to(state.current_instrument_id)
                if alternatives and state.remaining_attempts > 0:
                    return Action.retry(1.0, instrument_id=alternatives[0].id)
                if state.contact_count == 0 and not state.opted_out:
                    return self._outreach(state)
                return Action.give_up()

            case DeclineClass.SESSION_CONDITIONAL:
                if state.in_session and state.remaining_attempts > 0:
                    return Action.retry(0.0)
                if state.contact_count < self.max_contacts and not state.opted_out:
                    return self._outreach(state)
                return Action.give_up()

            case DeclineClass.RAIL_CONDITIONAL:
                if state.remaining_attempts <= 0:
                    return Action.give_up()
                if state.rail_is_degraded():
                    # The rail is down now. How long to wait is a real learned decision.
                    return self._retry(state)
                return Action.retry(0.5)

            case DeclineClass.TIME_CONDITIONAL:
                if state.remaining_attempts <= 0:
                    return Action.give_up()
                return self._retry(state)

            case _:
                if state.remaining_attempts > 0 and state.attempt_count == 0:
                    return Action.retry(reason.min_backoff_hours)
                return Action.give_up()

    def observe(self, succeeded: bool, opted_out: bool) -> None:
        """Credit the last decision with what actually happened.

        Called by the episode runner once an action resolves. In production this is the
        webhook: the policy learns from the reported outcome, never from anything it could
        not have known when it decided.
        """
        pending = self.pending
        if pending is None:
            return
        self.pending = None

        weight = _amount_weight(pending.amount_paise)

        if pending.is_retry:
            reward = (weight if succeeded else 0.0) - ATTEMPT_COST
            self.retry_bandit.update(pending.arm, pending.context, reward)
        else:
            reward = (SESSION_VALUE * weight if succeeded else 0.0) - CONTACT_COST
            if opted_out:
                reward -= OPT_OUT_PENALTY
            self.outreach_bandit.update(pending.arm, pending.context, reward)

    def _retry(self, state: EpisodeState) -> Action:
        context = extract_features(state)
        arm = self.retry_bandit.select(context, self.explore)
        self.pending = PendingDecision(arm, context, True, state.payment.amount_paise)
        delay = max(RETRY_DELAYS_HOURS[arm], lookup(state.current_decline_code).min_backoff_hours)
        return Action.retry(delay)

    def _outreach(self, state: EpisodeState) -> Action:
        context = extract_features(state)
        arm = self.outreach_bandit.select(context, self.explore)
        self.pending = PendingDecision(arm, context, False, state.payment.amount_paise)
        delay, channel = OUTREACH_ARMS[arm]
        return Action.outreach(delay, channel)

    def explain(self, state: EpisodeState) -> str:
        """Why the policy would act as it does. For the audit trail and the demo.

        The point of a numeric policy over a generative one: this is derived from the
        model, not narrated after the fact.
        """
        reason = lookup(state.current_decline_code)
        context = extract_features(state)

        if reason.decline_class is DeclineClass.SESSION_CONDITIONAL and not state.in_session:
            bandit, arms = self.outreach_bandit, [f"{d:g}h {c.value}" for d, c in OUTREACH_ARMS]
        else:
            bandit, arms = self.retry_bandit, [f"retry +{d:g}h" for d in RETRY_DELAYS_HOURS]

        scores = bandit.scores(context, explore=False)
        best = int(np.argmax(scores))
        return (
            f"{state.current_decline_code} -> {reason.decline_class.value}; "
            f"chose {arms[best]} (expected {scores[best]:.3f}, "
            f"+/- {bandit.confidence(context, best):.3f}, {bandit.pulls[best]} pulls)"
        )

    def explain_dict(self, state: EpisodeState) -> dict[str, Any]:
        """The same explanation as ``explain``, structured for a screen.

        Returns every arm with its predicted reward, not only the winner. Seeing the
        alternatives is what distinguishes a decision from an assertion: the policy is
        choosing between eight options with known scores, and a reader can check that the
        chosen one really is the argmax.
        """
        reason = lookup(state.current_decline_code)
        context = extract_features(state)

        if reason.decline_class is DeclineClass.SESSION_CONDITIONAL and not state.in_session:
            bandit = self.outreach_bandit
            labels = [f"{d:g}h {c.value}" for d, c in OUTREACH_ARMS]
            kind = "outreach"
        else:
            bandit = self.retry_bandit
            labels = [f"+{d:g}h" for d in RETRY_DELAYS_HOURS]
            kind = "retry"

        scores = bandit.scores(context, explore=False)
        best = int(np.argmax(scores))
        contributions = bandit.contributions(context, best)
        ranked = np.argsort(-np.abs(contributions))

        return {
            "kind": kind,
            "trained": int(bandit.pulls.sum()),
            "arms": [
                {
                    "label": labels[arm],
                    "score": float(scores[arm]),
                    "pulls": int(bandit.pulls[arm]),
                    "chosen": arm == best,
                }
                for arm in range(bandit.n_arms)
            ],
            "chosen": {
                "label": labels[best],
                "score": float(scores[best]),
                "confidence": float(bandit.confidence(context, best)),
                "pulls": int(bandit.pulls[best]),
            },
            "features": [
                {
                    "name": FEATURE_NAMES[i],
                    "value": float(context[i]),
                    "contribution": float(contributions[i]),
                }
                for i in ranked
                if context[i] != 0.0
            ][:6],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise a trained policy, arms and feature names included.

        The names are stored so a reload can refuse a stale file rather than quietly
        scoring the wrong columns: the weights are meaningless if the feature vector has
        changed shape or order underneath them.
        """
        return {
            "retry": self.retry_bandit.to_dict(),
            "outreach": self.outreach_bandit.to_dict(),
            "max_contacts": self.max_contacts,
            "retry_delays_hours": list(RETRY_DELAYS_HOURS),
            "outreach_arms": [[delay, channel.value] for delay, channel in OUTREACH_ARMS],
            "feature_names": list(FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BanditPolicy:
        """Reload a trained policy. Exploration is off: this is for serving, not learning."""
        saved = list(data.get("feature_names", ()))
        if saved != list(FEATURE_NAMES):
            raise ValueError(
                "the saved bandit was trained on a different feature vector "
                f"({len(saved)} features) than this build expects ({N_FEATURES}). "
                "Retrain with scripts/run_eval.py rather than scoring on stale weights."
            )
        policy = cls(explore=False)
        policy.retry_bandit = LinUCB.from_dict(data["retry"])
        policy.outreach_bandit = LinUCB.from_dict(data["outreach"])
        policy.max_contacts = int(data.get("max_contacts", policy.max_contacts))
        return policy
