"""
graph.py

Wires the Router Agent and all 4 sub-agents (Appointment, FAQ, Billing,
Emergency) into a single LangGraph StateGraph.

Structure: router -> conditional edge (based on state["intent"]) -> exactly
one sub-agent -> END.

This graph represents ONE caller turn: VAPI sends the transcribed text in,
the graph classifies + routes + generates a response, and that response goes
back out to be spoken. The conversation_history and collected_data fields in
AgentState are what carry context BETWEEN turns — the caller code (or later,
the VAPI webhook handler) is responsible for passing the updated state back
in on the next turn, and appending this turn to conversation_history.

Note on why sub-agents route straight to END instead of back to "router":
each webhook call from VAPI is one turn already. Looping back to the router
here would just re-classify the same input again with no new caller message,
which does nothing useful. The "can the caller change topic mid-call" case
is naturally handled by ALWAYS starting a fresh turn back at "router" on the
NEXT webhook call — not by looping within a single turn.
"""

from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.extraction import extract_booking_details
from agents.router_agent import classify_intent
from agents.appointment_agent import handle_appointment
from agents.faq_agent import handle_faq
from agents.billing_agent import handle_billing
from agents.emergency_agent import handle_emergency


def route_by_intent(state: AgentState) -> str:
    """Reads the intent the router just set and picks which node runs next."""
    return state["intent"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extract_details", extract_booking_details)
    graph.add_node("router", classify_intent)
    graph.add_node("appointment", handle_appointment)
    graph.add_node("faq", handle_faq)
    graph.add_node("billing", handle_billing)
    graph.add_node("emergency", handle_emergency)

    # extract_details runs FIRST, unconditionally, on every turn — this is
    # what lets a caller mention their name/department/date on a turn that
    # doesn't even get routed to the Appointment Agent, without losing it.
    graph.set_entry_point("extract_details")
    graph.add_edge("extract_details", "router")

    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "appointment": "appointment",
            "faq": "faq",
            "billing": "billing",
            "emergency": "emergency",
        },
    )

    graph.add_edge("appointment", END)
    graph.add_edge("faq", END)
    graph.add_edge("billing", END)
    graph.add_edge("emergency", END)

    return graph.compile()


# Compiled once at import time and reused across turns/calls — compiling the
# graph is cheap but there's no reason to redo it on every webhook request.
compiled_graph = build_graph()


def run_turn(caller_input: str, conversation_history=None, collected_data=None) -> AgentState:
    """
    Convenience wrapper for running a single turn through the graph.
    This is what the VAPI webhook handler (Phase 4) will call on every
    caller utterance.
    """
    state: AgentState = {
        "caller_input": caller_input,
        "conversation_history": conversation_history or [],
        "intent": None,
        "collected_data": collected_data or {},
        "agent_response": None,
        "escalate": False,
    }
    return compiled_graph.invoke(state)
