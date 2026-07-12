"""
emergency_agent.py

Handles the "emergency" intent. This agent does NOT try to solve the
caller's problem — its only job is to:
1. Respond calmly and briefly, telling the caller they're being connected
   to a human right now.
2. Set state["escalate"] = True, which the graph uses to trigger a warm
   transfer to a human agent (wired in later with VAPI's transfer feature).

Deliberately kept simple and deterministic (no LLM call needed for the
response itself) — in a real emergency, the LAST thing you want is an
LLM call adding latency or unpredictability. Speed and reliability matter
more than natural phrasing here.
"""

from agents.state import AgentState

EMERGENCY_RESPONSE = (
    "I understand this is urgent. I'm connecting you to a member of our staff "
    "right now — please stay on the line."
)


def handle_emergency(state: AgentState) -> AgentState:
    """
    LangGraph node function for the emergency sub-agent.
    Deterministic — no LLM call, to keep this path as fast and predictable
    as possible.
    """
    state["agent_response"] = EMERGENCY_RESPONSE
    state["escalate"] = True
    return state
