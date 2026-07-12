"""
state.py

Defines the shared state object that flows through the LangGraph.
Every node (router + sub-agents) reads from and writes to this same state,
which is how context is preserved when the conversation hands off between agents.
"""

from typing import TypedDict, Optional, List, Dict


class AgentState(TypedDict):
    # The raw text from the caller for this turn (comes from VAPI transcription later)
    caller_input: str

    # Full conversation history so far: list of {"role": "caller"/"agent", "text": "..."}
    conversation_history: List[Dict[str, str]]

    # Set by the router agent. One of: "appointment", "faq", "billing", "emergency"
    intent: Optional[str]

    # Any structured data collected during the conversation
    # e.g. {"patient_name": "...", "department": "...", "preferred_date": "..."}
    collected_data: Dict[str, str]

    # The response text to be spoken back to the caller this turn
    agent_response: Optional[str]

    # Set to True by the Emergency agent to signal the call should be transferred to a human
    escalate: bool

    # Transient: set by the extraction step when the caller mentions a department name that
    # doesn't match any known department, so the Appointment Agent can ask for clarification
    # directly instead of silently re-asking "which department" as if nothing was said.
    _unrecognized_department: Optional[str]
