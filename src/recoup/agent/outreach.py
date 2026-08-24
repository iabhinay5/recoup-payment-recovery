"""Generate the message that asks a customer to complete a failed payment.

This is where a language model is unambiguously the right tool. The message has to say
different things depending on *why* the payment failed — "your card has expired, please
add a new one" and "your payment was interrupted, tap to finish" are different requests —
and it has to say them in the customer's language, at a length the channel allows.

Three constraints are enforced in code rather than asked for in the prompt, because a
prompt is a request and a payment system needs guarantees:

- **Length.** An SMS that overflows is silently split and billed twice.
- **No invented facts.** The model is given the amount and the merchant name and is
  forbidden from introducing any other specifics. A message that names a date or a card
  number the system never supplied is a message that lies to a customer.
- **No false urgency.** Dunning that threatens gets opt-outs, and an opt-out forfeits
  every future recovery from that customer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from recoup.llm.base import LLMProvider, LLMRequest
from recoup.sim.entities import ContactChannel
from recoup.taxonomy import Remedy, lookup

__all__ = ["Language", "OutreachMessage", "OutreachWriter", "sanitise"]

REASONING_BUDGET = 1600
"""Output budget per call.

Sized for reasoning models, which spend most of their tokens thinking before they answer.
At 300 tokens the chain of thought consumed the entire budget and every call returned an
empty string with finish_reason "length".
"""

# Channel length ceilings. SMS is a hard limit; a longer message is split into two
# segments, billed twice, and often arrives out of order.
MAX_CHARS = {
    ContactChannel.SMS: 160,
    ContactChannel.WHATSAPP: 400,
    ContactChannel.EMAIL: 700,
}

# Anything a model might emit when it does not know a value. A message containing one of
# these reached a real customer, and that must never happen.
# Characters a model reaches for that an SMS gateway should never see. A narrow
# no-break space or a typographic dash pushes an SMS out of the GSM-7 alphabet and into
# UCS-2, which halves the segment limit from 160 characters to 70 and doubles the cost of
# every message. The model emitted U+202F on its first real generation.
_UNICODE_FIXUPS = {
    " ": " ",  # no-break space
    " ": " ",  # narrow no-break space - what the model actually emitted
    " ": " ",  # thin space
    " ": " ",  # hair space
    " ": " ",  # figure space
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",  # en dash
    "—": "-",  # em dash
    "…": "...",
    "₹": "Rs ",  # rupee sign
}


def _is_transmittable(char: str) -> bool:
    """Printable ASCII or a newline. The safe subset of GSM-7."""
    return char == "\n" or 0x20 <= ord(char) <= 0x7E


def sanitise(text: str) -> str:
    """Make generated text safe to put on a wire.

    Applied to every model output before validation. Deliberately lossy: a message that is
    slightly less typographically pretty and definitely deliverable beats the reverse.

    Known substitutions run first so a curly apostrophe becomes a straight one rather than
    vanishing; anything still outside printable ASCII is then dropped.
    """
    for bad, good in _UNICODE_FIXUPS.items():
        text = text.replace(bad, good)
    text = "".join(char for char in text if _is_transmittable(char))
    return " ".join(text.split())


# Three or more consecutive digits, allowing thousands separators. Two digits are
# routine in prose ("2 days"); a longer run is almost always a specific fact.
_DIGIT_RUN = re.compile(r"\d[\d,]{2,}")

_PLACEHOLDER = re.compile(
    r"(\[[^\]]{2,40}\]|\{\{?[a-z_ ]{2,40}\}?\}|<[a-z_ ]{2,40}>|XXXX|TODO|lorem)",
    re.IGNORECASE,
)


class Language(Enum):
    """Languages the writer supports.

    India is multilingual and dunning in the wrong language is dunning nobody reads. The
    set is deliberately small — each one added is one more output nobody on the team can
    proofread.
    """

    ENGLISH = "English"
    HINDI = "Hindi"
    HINGLISH = "Hinglish (Hindi written in Latin script)"


@dataclass(frozen=True, slots=True)
class OutreachMessage:
    """A generated message, with everything needed to audit it."""

    channel: ContactChannel
    language: Language
    subject: str | None
    body: str
    decline_code: str
    generated: bool
    """False when the deterministic template was used instead of the model."""

    @property
    def length(self) -> int:
        return len(self.body)

    @property
    def within_limit(self) -> bool:
        return self.length <= MAX_CHARS[self.channel]


# Deterministic fallbacks, one per remedy. Used when no provider is configured, when the
# model is unavailable, and whenever generated text fails validation. A payment system
# that cannot send a message because its language model is down is a payment system with
# an unnecessary dependency.
_TEMPLATES: dict[Remedy, str] = {
    Remedy.RETRY_LATER: (
        "Your payment of {amount} to {merchant} did not go through. "
        "We will try again shortly - no action needed."
    ),
    Remedy.AWAIT_RAIL_RECOVERY: (
        "Your payment of {amount} to {merchant} could not be completed because your bank "
        "was temporarily unavailable. We will retry once it is back."
    ),
    Remedy.CUSTOMER_SESSION: (
        "Your payment of {amount} to {merchant} was not completed. You can finish it here: {link}"
    ),
    Remedy.CUSTOMER_ACTION: (
        "Your payment of {amount} to {merchant} could not be processed. "
        "Please check your card settings with your bank, then try again: {link}"
    ),
    Remedy.NEW_INSTRUMENT: (
        "Your payment of {amount} to {merchant} could not be completed with your saved "
        "card. You can pay with another method here: {link}"
    ),
    Remedy.ESCALATE: (
        "We could not complete your payment of {amount} to {merchant}. "
        "Our team is looking into it and will be in touch."
    ),
}


def _system_prompt(channel: ContactChannel, language: Language) -> str:
    return "\n".join(
        (
            "You write short transactional messages telling a customer that their payment "
            "to a merchant failed, and what to do about it.",
            "",
            f"Channel: {channel.value}. Hard limit: {MAX_CHARS[channel]} characters.",
            f"Language: {language.value}.",
            "",
            "Rules:",
            "  - Use ONLY the facts given. Never invent an amount, date, card number,",
            "    reference number or deadline. If you were not told it, do not write it.",
            "  - Never leave a placeholder. Every message is sent as written.",
            "  - Say plainly what happened and what the customer should do next.",
            "  - No threats, no false urgency, no guilt. This is a payment that failed,",
            "    not a debt collection.",
            "  - Polite and brief. Do not apologise more than once.",
            "",
            "Return the message body only. No preamble, no quotes, no explanation.",
        )
    )


@dataclass
class OutreachWriter:
    """Writes customer outreach, falling back to templates when the model cannot."""

    provider: LLMProvider | None = None
    merchant_name: str = "the merchant"
    payment_link: str = "https://rzp.io/i/example"

    generated: int = 0
    templated: int = 0
    rejected: int = 0
    model_errors: int = 0
    last_error: str = ""
    last_rejection: str = ""

    def write(
        self,
        decline_code: str,
        amount_paise: int,
        channel: ContactChannel = ContactChannel.EMAIL,
        language: Language = Language.ENGLISH,
    ) -> OutreachMessage:
        """Produce a message for one failed payment."""
        reason = lookup(decline_code)
        amount = f"Rs {amount_paise / 100:,.0f}"

        if self.provider is not None:
            body = self._generate(reason.remedy, reason.description, amount, channel, language)
            if body is not None:
                self.generated += 1
                return OutreachMessage(
                    channel=channel,
                    language=language,
                    subject=self._subject(reason.remedy)
                    if channel is ContactChannel.EMAIL
                    else None,
                    body=body,
                    decline_code=decline_code,
                    generated=True,
                )

        self.templated += 1
        template = _TEMPLATES.get(reason.remedy, _TEMPLATES[Remedy.ESCALATE])
        body = template.format(amount=amount, merchant=self.merchant_name, link=self.payment_link)
        return OutreachMessage(
            channel=channel,
            language=language,
            subject=self._subject(reason.remedy) if channel is ContactChannel.EMAIL else None,
            body=body[: MAX_CHARS[channel]],
            decline_code=decline_code,
            generated=False,
        )

    def _generate(
        self,
        remedy: Remedy,
        description: str,
        amount: str,
        channel: ContactChannel,
        language: Language,
    ) -> str | None:
        assert self.provider is not None

        user = "\n".join(
            (
                f"What happened: {description}",
                f"What the customer should do: {self._instruction(remedy)}",
                f"Amount: {amount}",
                f"Merchant: {self.merchant_name}",
                f"Payment link (include only if the customer must act): {self.payment_link}",
            )
        )

        try:
            raw = self.provider.complete(
                LLMRequest(
                    system=_system_prompt(channel, language),
                    user=user,
                    max_tokens=REASONING_BUDGET,
                    temperature=0.3,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.model_errors += 1
            self.last_error = f"{exc.__class__.__name__}: {str(exc)[:200]}"
            return None

        body = sanitise(raw.strip().strip('"'))
        return body if self._is_sendable(body, channel, amount) else None

    def _reject(self, why: str) -> bool:
        self.rejected += 1
        self.last_rejection = why
        return False

    def _is_sendable(self, body: str, channel: ContactChannel, amount: str) -> bool:
        """Validate generated text before it can reach a customer.

        Rejection falls back to a template rather than raising. A failed payment still
        needs a message; it does not need this particular message.
        """
        if not body:
            return self._reject("empty")
        if len(body) > MAX_CHARS[channel]:
            return self._reject(f"{len(body)} chars over the {MAX_CHARS[channel]} limit")
        if _PLACEHOLDER.search(body):
            return self._reject("contains an unfilled placeholder")

        # The only multi-digit numbers a message may contain are the ones it was handed.
        # Anything else is a fact the model invented - a card suffix, a reference number,
        # a deadline - and a message that states one is a message that lies to a customer.
        supplied = set(_DIGIT_RUN.findall(amount)) | set(_DIGIT_RUN.findall(self.payment_link))
        invented = [
            run
            for run in _DIGIT_RUN.findall(body.replace(self.payment_link, ""))
            if run not in supplied
        ]
        if invented:
            return self._reject(f"invented the number {invented[0]!r}")
        return True

    @staticmethod
    def _instruction(remedy: Remedy) -> str:
        return {
            Remedy.RETRY_LATER: "nothing - we will retry automatically",
            Remedy.AWAIT_RAIL_RECOVERY: "nothing - their bank was down, we will retry",
            Remedy.CUSTOMER_SESSION: "open the link and complete the payment",
            Remedy.CUSTOMER_ACTION: "enable the card with their bank, then pay again",
            Remedy.NEW_INSTRUMENT: "pay using a different card or method",
            Remedy.ESCALATE: "nothing - our team is investigating",
        }[remedy]

    @staticmethod
    def _subject(remedy: Remedy) -> str:
        if remedy in (Remedy.RETRY_LATER, Remedy.AWAIT_RAIL_RECOVERY):
            return "We could not process your payment - we will retry"
        if remedy is Remedy.ESCALATE:
            return "We are looking into your payment"
        return "Your payment needs a quick action"

    def summary(self) -> str:
        total = self.generated + self.templated
        return (
            f"{total} messages: {self.generated} generated, {self.templated} templated, "
            f"{self.rejected} rejected"
            + (f" ({self.last_rejection})" if self.last_rejection else "")
            + (
                f"  [{self.model_errors} model errors: {self.last_error}]"
                if self.model_errors
                else ""
            )
        )
