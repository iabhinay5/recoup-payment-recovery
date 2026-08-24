"""The LLM layer: normalisation, outreach, and triage for hard cases.

The model is the escalation path, not the default path. See each module's docstring for
where the deterministic route ends and inference begins.
"""

from recoup.agent.normalizer import Classification, DeclineNormalizer, Source
from recoup.agent.orchestrator import AgentResult, Orchestrator, Tool, ToolCall, default_tools
from recoup.agent.outreach import Language, OutreachMessage, OutreachWriter

__all__ = [
    "AgentResult",
    "Classification",
    "DeclineNormalizer",
    "Language",
    "Orchestrator",
    "OutreachMessage",
    "OutreachWriter",
    "Source",
    "Tool",
    "ToolCall",
    "default_tools",
]
