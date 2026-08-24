"""Tests for the LLM layer.

All hermetic — they use ``StubProvider`` and never touch a network. The behaviour worth
pinning is not "the model gives good answers"; it is that the system does the right thing
when the model is unavailable, slow, wrong, or confidently inventing something.
"""

from __future__ import annotations

import json

import pytest

from recoup.agent import (
    DeclineNormalizer,
    Language,
    Orchestrator,
    OutreachWriter,
    Source,
    default_tools,
)
from recoup.agent.outreach import MAX_CHARS, sanitise
from recoup.llm import LLMRequest, StubProvider
from recoup.sim.entities import ContactChannel


class _Always:
    """A provider that returns one fixed string."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    @property
    def model(self) -> str:
        return "always"

    def complete(self, request: LLMRequest) -> str:
        self.calls += 1
        return self.text


class _Broken:
    """A provider that is down."""

    @property
    def model(self) -> str:
        return "broken"

    def complete(self, request: LLMRequest) -> str:
        raise ConnectionError("model unavailable")


# =====================================================================================
# Normalisation
# =====================================================================================


class TestTheCheapPathDominates:
    """The architectural claim: most traffic must never reach a model."""

    def test_a_known_reason_never_calls_the_model(self) -> None:
        provider = _Always("{}")
        n = DeclineNormalizer(provider=provider)

        result = n.classify(error_reason="insufficient_funds")

        assert result.source is Source.EXACT
        assert provider.calls == 0, "a dictionary lookup must not become an inference call"

    def test_formatting_differences_are_not_a_language_problem(self) -> None:
        provider = _Always("{}")
        n = DeclineNormalizer(provider=provider)

        result = n.classify(error_reason="Insufficient Funds")

        assert result.source is Source.HEURISTIC
        assert result.code == "insufficient_funds"
        assert provider.calls == 0

    def test_model_call_rate_stays_low_on_realistic_traffic(self) -> None:
        """If this rate is high, the taxonomy is missing codes and should be extended
        rather than papered over with inference."""
        n = DeclineNormalizer(provider=_Always("{}"))
        for _ in range(20):
            n.classify(error_reason="insufficient_funds")
        n.classify(error_description="something nobody has seen")

        assert n.model_call_rate < 0.1


class TestModelOutputIsValidatedNotTrusted:
    def test_an_invented_code_is_rejected(self) -> None:
        """A model is free to return a plausible-looking code that does not exist.

        Acting on one means retrying a payment for a reason that is not real.
        """
        provider = _Always(json.dumps({"code": "card_slightly_tired", "confidence": 0.99}))
        n = DeclineNormalizer(provider=provider)

        result = n.classify(error_description="the card seems unwell")

        assert result.source is Source.FALLBACK
        assert result.code == "__unknown__"

    def test_low_confidence_is_rejected(self) -> None:
        provider = _Always(json.dumps({"code": "card_expired", "confidence": 0.3}))
        n = DeclineNormalizer(provider=provider, min_model_confidence=0.7)

        assert n.classify(error_description="maybe expired?").source is Source.FALLBACK

    def test_a_confident_valid_code_is_accepted(self) -> None:
        provider = _Always(
            json.dumps({"code": "card_expired", "confidence": 0.95, "rationale": "x"})
        )
        n = DeclineNormalizer(provider=provider)

        result = n.classify(error_description="validity date has passed")

        assert result.source is Source.MODEL
        assert result.code == "card_expired"
        assert result.is_confident

    def test_unparseable_output_falls_back(self) -> None:
        n = DeclineNormalizer(provider=_Always("I think it's probably an expired card!"))
        assert n.classify(error_description="???").source is Source.FALLBACK

    def test_a_model_outage_degrades_rather_than_raises(self) -> None:
        """A payment pipeline must not go down because inference did."""
        n = DeclineNormalizer(provider=_Broken())

        result = n.classify(error_description="anything")

        assert result.source is Source.FALLBACK
        assert n.model_errors == 1

    def test_model_failures_are_visible_not_silent(self) -> None:
        """A wrong model name once produced a 404 on every call, and the layer looked
        like it was working because the failure was swallowed."""
        n = DeclineNormalizer(provider=_Broken())
        n.classify(error_description="anything")

        assert "ConnectionError" in n.last_error
        assert "model errors" in n.summary()


# =====================================================================================
# Outreach
# =====================================================================================


class TestSanitisation:
    def test_exotic_spaces_become_plain_ones(self) -> None:
        """A narrow no-break space pushes an SMS from GSM-7 into UCS-2, halving the
        segment limit and doubling the cost. The model emitted one on its first run."""
        assert sanitise("Rs 1,250") == "Rs 1,250"

    def test_typographic_punctuation_is_folded(self) -> None:
        assert sanitise("“don’t” — now…") == '"don\'t" - now...'

    def test_output_is_always_ascii(self) -> None:
        assert sanitise("payment ₹500 – éè 你好").isascii()

    def test_whitespace_is_collapsed(self) -> None:
        assert sanitise("a  \n\n  b") == "a b"


class TestGeneratedMessagesAreValidated:
    def test_an_overlong_sms_is_rejected(self) -> None:
        writer = OutreachWriter(provider=_Always("x" * 500))

        message = writer.write("card_expired", 100_000, ContactChannel.SMS)

        assert not message.generated, "an overlong message must fall back to a template"
        assert message.within_limit
        assert "over the" in writer.last_rejection

    def test_a_placeholder_is_rejected(self) -> None:
        """Every message is sent as written. A [CUSTOMER_NAME] reaches a real person."""
        writer = OutreachWriter(provider=_Always("Hello [CUSTOMER_NAME], your payment failed."))

        message = writer.write("card_expired", 100_000, ContactChannel.SMS)

        assert not message.generated
        assert "placeholder" in writer.last_rejection

    def test_an_invented_number_is_rejected(self) -> None:
        """A message naming a card number the system never supplied is a message that
        lies to a customer."""
        writer = OutreachWriter(
            provider=_Always("Your card ending 4829173 failed. Please update it.")
        )

        message = writer.write("card_expired", 100_000, ContactChannel.SMS)

        assert not message.generated

    def test_a_model_outage_still_produces_a_message(self) -> None:
        """A failed payment still needs a message; it does not need this particular one."""
        writer = OutreachWriter(provider=_Broken())

        message = writer.write("insufficient_funds", 100_000, ContactChannel.SMS)

        assert message.body
        assert not message.generated
        assert writer.model_errors == 1

    def test_templates_respect_channel_limits(self) -> None:
        writer = OutreachWriter()
        for code in (
            "card_expired",
            "insufficient_funds",
            "payment_cancelled",
            "bank_technical_error",
            "payment_risk_check_failed",
        ):
            for channel in ContactChannel:
                message = writer.write(code, 12_500_000, channel)
                assert message.within_limit, f"{code}/{channel.value} exceeds its limit"

    def test_the_message_follows_the_remedy_not_the_code(self) -> None:
        """A customer whose payment will be retried automatically must not be sent a
        link asking them to act."""
        writer = OutreachWriter()
        auto = writer.write("insufficient_funds", 100_000, ContactChannel.SMS)
        manual = writer.write("card_expired", 100_000, ContactChannel.SMS)

        assert "no action needed" in auto.body.lower()
        assert writer.payment_link in manual.body

    def test_language_is_carried_through(self) -> None:
        message = OutreachWriter().write(
            "card_expired", 100_000, ContactChannel.SMS, Language.HINGLISH
        )
        assert message.language is Language.HINGLISH

    def test_sms_has_no_subject(self) -> None:
        assert OutreachWriter().write("card_expired", 1000, ContactChannel.SMS).subject is None
        assert OutreachWriter().write("card_expired", 1000, ContactChannel.EMAIL).subject


# =====================================================================================
# Orchestration
# =====================================================================================


class TestAgentLoop:
    def test_a_direct_decision_ends_the_loop(self) -> None:
        agent = Orchestrator(
            provider=_Always(json.dumps({"decision": "retry_later", "reasoning": "bank down"}))
        )
        result = agent.run("case")

        assert result.decision == "retry_later"
        assert result.completed
        assert result.steps == 1
        assert result.calls == ()

    def test_a_tool_is_called_and_its_result_fed_back(self) -> None:
        provider = _Always(
            json.dumps({"tool": "lookup_decline_code", "args": {"code": "card_expired"}})
        )
        agent = Orchestrator(provider=provider, max_steps=2)

        result = agent.run("case")

        assert "lookup_decline_code" in result.tools_used
        assert "card_expired" in result.calls[0].result

    def test_escalation_ends_the_loop_immediately(self) -> None:
        agent = Orchestrator(
            provider=_Always(
                json.dumps({"tool": "escalate_to_human", "args": {"reason": "ambiguous"}})
            )
        )
        result = agent.run("case")

        assert result.escalated
        assert result.completed
        assert result.reasoning == "ambiguous"

    def test_an_inconclusive_run_escalates_rather_than_guessing(self) -> None:
        """Running out of steps must not mean acting on an unfinished investigation."""
        agent = Orchestrator(
            provider=_Always(json.dumps({"tool": "lookup_decline_code", "args": {"code": "x"}})),
            max_steps=3,
        )
        result = agent.run("case")

        assert result.escalated
        assert not result.completed
        assert result.steps == 3

    def test_a_model_outage_escalates(self) -> None:
        result = Orchestrator(provider=_Broken()).run("case")

        assert result.escalated
        assert not result.completed
        assert "unavailable" in result.reasoning

    def test_an_invented_tool_name_does_not_crash(self) -> None:
        agent = Orchestrator(
            provider=_Always(json.dumps({"tool": "charge_customer_twice", "args": {}})),
            max_steps=2,
        )
        result = agent.run("case")

        assert "No tool named" in result.calls[0].result
        assert result.escalated, "an agent that cannot proceed must escalate"

    def test_the_agent_cannot_move_money(self) -> None:
        """The toolset is read-only plus escalation. An agent with a charge tool would put
        a language model inside the payment path, which ADR-003 exists to prevent."""
        names = {tool.name for tool in default_tools()}
        assert names == {
            "lookup_decline_code",
            "check_bank_health",
            "get_customer_history",
            "escalate_to_human",
        }

    def test_malformed_json_is_survivable(self) -> None:
        agent = Orchestrator(provider=_Always("thinking out loud, no json here"), max_steps=2)
        result = agent.run("case")

        assert result.escalated
        assert not result.completed


class TestProviderContract:
    def test_stub_provider_records_requests(self) -> None:
        stub = StubProvider()
        DeclineNormalizer(provider=stub).classify(error_description="mystery")
        assert len(stub.calls) == 1

    @pytest.mark.parametrize("channel", list(ContactChannel))
    def test_every_channel_has_a_limit(self, channel: ContactChannel) -> None:
        assert MAX_CHARS[channel] > 0
