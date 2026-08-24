"""Map a raw gateway failure onto the canonical decline taxonomy.

The important design decision here is **when not to call the model**.

Razorpay's own ``error_reason`` already uses the taxonomy vocabulary, so the overwhelming
majority of live failures resolve by dictionary lookup. Sending those to a language model
would add latency, cost and non-determinism to a question that has already been answered
exactly. The model is reached only when the deterministic path cannot answer — an
unrecognised code, or a payload with no machine-readable reason at all.

That is the whole argument for where an LLM belongs in this system. It handles the long
tail, where a free-text description written by a bank is the only signal available and
language understanding is genuinely the right tool. It does not handle the common path,
where a lookup is both faster and correct.

Every classification is validated against the taxonomy before use. A model that returns a
code we do not recognise is treated as having failed, not as having discovered something.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

from recoup.llm.base import LLMProvider, LLMRequest
from recoup.taxonomy import DeclineClass, all_reasons, is_known, lookup

__all__ = ["Classification", "DeclineNormalizer", "Source"]

REASONING_BUDGET = 1600
"""Output budget per call.

Sized for reasoning models, which spend most of their tokens thinking before they answer.
At 300 tokens the chain of thought consumed the entire budget and every call returned an
empty string with finish_reason "length".
"""


class Source(Enum):
    """How a classification was reached. Recorded so the cheap path stays visible."""

    EXACT = "exact"
    """``error_reason`` was already a known taxonomy code. No model call."""

    HEURISTIC = "heuristic"
    """Matched a known code by normalising case and separators. No model call."""

    MODEL = "model"
    """Classified by the language model from free text."""

    FALLBACK = "fallback"
    """Nothing could classify it. The conservative unknown default applies."""


@dataclass(frozen=True, slots=True)
class Classification:
    """The outcome of normalising one failure."""

    code: str
    source: Source
    confidence: float
    rationale: str = ""

    @property
    def used_model(self) -> bool:
        return self.source is Source.MODEL

    @property
    def is_confident(self) -> bool:
        """Whether the classification should drive an automated action.

        Deterministic sources are certain by construction. A model classification has to
        clear a bar, because acting on a low-confidence guess about *why* a payment failed
        means retrying something that cannot succeed.
        """
        if self.source in (Source.EXACT, Source.HEURISTIC):
            return True
        return self.source is Source.MODEL and self.confidence >= 0.7


def _system_prompt() -> str:
    """Built from the taxonomy itself, so the two cannot drift apart."""
    lines = [
        "You classify failed payment errors from an Indian payment gateway.",
        "",
        "Given a raw gateway error, return the single canonical decline code that best",
        "describes it. Choose only from this list. Each code is followed by what it means.",
        "",
    ]
    for reason in all_reasons():
        lines.append(f"  {reason.code} - {reason.description}")
    lines += [
        "",
        "Rules:",
        "  - Return exactly one code from the list above, copied verbatim.",
        "  - If the error does not clearly match any code, return __unknown__.",
        "    An honest __unknown__ is far better than a confident wrong code: a wrong",
        "    classification causes real retries against a customer's account.",
        "  - confidence is your own estimate between 0 and 1.",
        "",
        'Respond with JSON only: {"code": "...", "confidence": 0.0, "rationale": "..."}',
    ]
    return "\n".join(lines)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _slug(text: str) -> str:
    """Lowercase, collapse separators. Catches trivial formatting differences."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


@dataclass
class DeclineNormalizer:
    """Resolves raw gateway errors to taxonomy codes, cheaply where possible."""

    provider: LLMProvider | None = None
    min_model_confidence: float = 0.7

    exact_hits: int = 0
    heuristic_hits: int = 0
    model_calls: int = 0
    fallbacks: int = 0
    model_errors: int = 0
    last_error: str = ""
    """Falling back on a model failure is correct; doing it invisibly is not.

    A wrong model name produced a 404 on every call, and because the handler returned
    None the whole layer degraded to fallback and looked like it was working. These two
    fields are what made that visible.
    """

    def classify(
        self,
        error_reason: str | None = None,
        error_description: str | None = None,
        error_code: str | None = None,
    ) -> Classification:
        """Classify one failure, escalating to the model only when necessary."""
        # 1. The common case. Razorpay's error_reason is already our vocabulary.
        if error_reason and is_known(error_reason):
            self.exact_hits += 1
            return Classification(error_reason, Source.EXACT, 1.0, "exact taxonomy match")

        # 2. Formatting differences are not a language problem.
        for candidate in (error_reason, error_code):
            if not candidate:
                continue
            slug = _slug(candidate)
            if is_known(slug):
                self.heuristic_hits += 1
                return Classification(slug, Source.HEURISTIC, 0.95, f"normalised {candidate!r}")

        # 3. Only now is this a language problem.
        if self.provider is not None:
            classification = self._ask_model(error_reason, error_description, error_code)
            if classification is not None:
                return classification

        self.fallbacks += 1
        return Classification(
            "__unknown__",
            Source.FALLBACK,
            0.0,
            "no deterministic match and no usable model classification",
        )

    def _ask_model(
        self,
        error_reason: str | None,
        error_description: str | None,
        error_code: str | None,
    ) -> Classification | None:
        assert self.provider is not None

        user = "\n".join(
            f"{label}: {value}"
            for label, value in (
                ("error_reason", error_reason),
                ("error_code", error_code),
                ("error_description", error_description),
            )
            if value
        )
        if not user:
            return None

        self.model_calls += 1
        try:
            raw = self.provider.complete(
                LLMRequest(
                    system=_system_prompt(), user=user, max_tokens=REASONING_BUDGET, temperature=0.0
                )
            )
        except Exception as exc:  # noqa: BLE001
            # A model outage must degrade to the conservative default, never take down
            # the payment path - but it must be visible, not silent.
            self.model_errors += 1
            self.last_error = f"{exc.__class__.__name__}: {str(exc)[:200]}"
            return None

        match = _JSON_BLOCK.search(raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        code = str(parsed.get("code", "")).strip()
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        rationale = str(parsed.get("rationale", ""))[:300]

        # Validation, not trust. A model is free to invent a plausible-looking code, and
        # acting on one would mean retrying a payment for a reason that does not exist.
        if not is_known(code):
            return None
        if confidence < self.min_model_confidence:
            return None

        return Classification(code, Source.MODEL, min(max(confidence, 0.0), 1.0), rationale)

    # --- reporting -------------------------------------------------------------------

    @property
    def total(self) -> int:
        return self.exact_hits + self.heuristic_hits + self.model_calls + self.fallbacks

    @property
    def model_call_rate(self) -> float:
        """Share of classifications that needed a model.

        The number that justifies the architecture. If this is high, the taxonomy is
        missing codes and should be extended rather than papered over with inference.
        """
        return self.model_calls / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"{self.total} classified: {self.exact_hits} exact, "
            f"{self.heuristic_hits} heuristic, {self.model_calls} model "
            f"({self.model_call_rate:.1%}), {self.fallbacks} fallback"
            + (
                f"  [{self.model_errors} model errors: {self.last_error}]"
                if self.model_errors
                else ""
            )
        )


def decline_class_of(code: str) -> DeclineClass:
    """Convenience for callers that only need the class."""
    return lookup(code).decline_class
