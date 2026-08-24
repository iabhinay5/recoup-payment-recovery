"""The agent that handles what the deterministic path cannot.

**This runs on the exception, not the rule.** A recognised decline code is classified by
lookup and decided by the bandit, and neither needs an agent. What needs an agent is the
residue: an unrecognised code, a description that contradicts its reason field, a case
where the right answer is to stop and ask a human. That is a small share of traffic and
the only share where open-ended tool use beats a decision table.

Running an agent on every payment would be slower, costlier, non-deterministic, and worse
— the deterministic path is not a simplification of the agent, it is a better answer to
the questions it covers.

Tool calling is expressed as structured JSON rather than a vendor's function-calling API,
so the same loop runs against Groq, a local Ollama model, or anything else that returns
text. That is the ADR-004 portability constraint applied to the agent layer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from recoup.llm.base import LLMProvider, LLMRequest
from recoup.taxonomy import is_known, lookup

__all__ = ["AgentResult", "Orchestrator", "Tool", "ToolCall", "default_tools"]

MAX_STEPS = 5
"""Tool calls permitted per case.

A bound rather than a budget. An agent that has not reached a conclusion in five tool
calls is looping, and looping on a payment decision costs latency on something a human
could answer.
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Tool:
    """Something the agent may do. Deliberately few, and all read-only or terminal."""

    name: str
    description: str
    parameters: str
    run: Callable[[dict[str, Any]], str]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation and what it returned. The audit trail."""

    tool: str
    args: dict[str, Any]
    result: str


@dataclass(frozen=True, slots=True)
class AgentResult:
    """What the agent concluded, and how it got there."""

    decision: str
    reasoning: str
    calls: tuple[ToolCall, ...]
    escalated: bool
    steps: int
    completed: bool
    """False when the agent hit the step bound without concluding.

    Surfaced rather than swallowed: an inconclusive run must escalate to a human, not
    quietly return whatever it had.
    """

    @property
    def tools_used(self) -> tuple[str, ...]:
        return tuple(call.tool for call in self.calls)


def default_tools(
    bank_health: Callable[[str], float] | None = None,
    customer_history: Callable[[str], dict[str, Any]] | None = None,
) -> list[Tool]:
    """The standard read-only toolset, plus escalation.

    Nothing here can move money. The agent investigates and recommends; executing an
    action stays with the policy and the guardrail layer, which are testable and
    deterministic. An agent with a ``charge_customer`` tool would put a language model
    inside the payment path, which is exactly what ADR-003 exists to prevent.
    """

    def _taxonomy(args: dict[str, Any]) -> str:
        code = str(args.get("code", ""))
        if not is_known(code):
            return f"'{code}' is not a known decline code."
        reason = lookup(code)
        return json.dumps(
            {
                "code": reason.code,
                "class": reason.decline_class.value,
                "remedy": reason.remedy.value,
                "max_attempts": reason.max_attempts,
                "min_backoff_hours": reason.min_backoff_hours,
                "description": reason.description,
            }
        )

    def _bank(args: dict[str, Any]) -> str:
        bank_id = str(args.get("bank_id", ""))
        if bank_health is None:
            return "Bank health is unavailable in this environment."
        health = bank_health(bank_id)
        state = "degraded" if health < 0.5 else "healthy"
        return json.dumps({"bank_id": bank_id, "health": round(health, 3), "state": state})

    def _history(args: dict[str, Any]) -> str:
        customer_id = str(args.get("customer_id", ""))
        if customer_history is None:
            return "Customer history is unavailable in this environment."
        return json.dumps(customer_history(customer_id))

    def _escalate(args: dict[str, Any]) -> str:
        return f"Escalated to human review: {args.get('reason', 'no reason given')}"

    return [
        Tool(
            "lookup_decline_code",
            "Look up what a decline code means and what remedy applies.",
            '{"code": "string"}',
            _taxonomy,
        ),
        Tool(
            "check_bank_health",
            "Check whether a bank's payment rail is currently degraded.",
            '{"bank_id": "string"}',
            _bank,
        ),
        Tool(
            "get_customer_history",
            "Past payment outcomes for a customer, including which instruments work.",
            '{"customer_id": "string"}',
            _history,
        ),
        Tool(
            "escalate_to_human",
            "Hand the case to a human. Use when the evidence is contradictory or the "
            "right action would risk charging a customer incorrectly.",
            '{"reason": "string"}',
            _escalate,
        ),
    ]


def _system_prompt(tools: list[Tool]) -> str:
    lines = [
        "You are triaging a failed payment that an automated classifier could not resolve.",
        "Your job is to work out what should happen next, using the tools available.",
        "",
        "Tools:",
    ]
    for tool in tools:
        lines.append(f"  {tool.name}{tool.parameters}")
        lines.append(f"      {tool.description}")
    lines += [
        "",
        "Respond with JSON only, one of:",
        '  {"tool": "<name>", "args": {...}}                 to use a tool',
        '  {"decision": "<what to do>", "reasoning": "<why>"} when you have concluded',
        "",
        "Rules:",
        "  - Look up any decline code you are unsure about before deciding.",
        "  - You cannot charge anyone. You recommend; the policy layer executes.",
        "  - If the evidence is contradictory, or acting could charge a customer who",
        "    should not be charged, escalate to a human. That is a correct outcome,",
        "    not a failure.",
        "  - Be brief. This is an operational decision, not an essay.",
    ]
    return "\n".join(lines)


@dataclass
class Orchestrator:
    """Runs the tool-calling loop for one hard case."""

    provider: LLMProvider
    tools: list[Tool] = field(default_factory=default_tools)
    max_steps: int = MAX_STEPS

    def run(self, case: str) -> AgentResult:
        """Triage one case description."""
        by_name = {tool.name: tool for tool in self.tools}
        system = _system_prompt(self.tools)
        transcript = [f"Case:\n{case}"]
        calls: list[ToolCall] = []

        for step in range(1, self.max_steps + 1):
            try:
                raw = self.provider.complete(
                    LLMRequest(
                        system=system,
                        user="\n\n".join(transcript),
                        max_tokens=600,
                        temperature=0.0,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                return AgentResult(
                    decision="escalate_to_human",
                    reasoning=f"the model was unavailable ({exc.__class__.__name__})",
                    calls=tuple(calls),
                    escalated=True,
                    steps=step,
                    completed=False,
                )

            parsed = self._parse(raw)
            if parsed is None:
                transcript.append("Your last reply was not valid JSON. Respond with JSON only.")
                continue

            if "decision" in parsed:
                decision = str(parsed["decision"])
                return AgentResult(
                    decision=decision,
                    reasoning=str(parsed.get("reasoning", "")),
                    calls=tuple(calls),
                    escalated="escalate" in decision.lower(),
                    steps=step,
                    completed=True,
                )

            name = str(parsed.get("tool", ""))
            args = parsed.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            tool = by_name.get(name)
            if tool is None:
                # An invented tool name is a model error, not a crash. Say so and let it
                # correct itself.
                result = f"No tool named '{name}'. Available: {', '.join(by_name)}."
            else:
                try:
                    result = tool.run(args)
                except Exception as exc:  # noqa: BLE001
                    result = f"Tool failed: {exc.__class__.__name__}"

            calls.append(ToolCall(name, args, result))
            transcript.append(json.dumps({"tool": name, "args": args}))
            transcript.append(f"Tool result: {result}")

            if name == "escalate_to_human":
                return AgentResult(
                    decision="escalate_to_human",
                    reasoning=str(args.get("reason", "")),
                    calls=tuple(calls),
                    escalated=True,
                    steps=step,
                    completed=True,
                )

        # Out of steps. Escalating is the only safe conclusion: the alternative is acting
        # on an investigation that never finished.
        return AgentResult(
            decision="escalate_to_human",
            reasoning=f"no conclusion within {self.max_steps} steps",
            calls=tuple(calls),
            escalated=True,
            steps=self.max_steps,
            completed=False,
        )

    @staticmethod
    def _parse(raw: str) -> dict[str, Any] | None:
        match = _JSON_BLOCK.search(raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
