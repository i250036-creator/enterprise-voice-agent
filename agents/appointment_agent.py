"""
appointment_agent.py

Handles the "appointment" intent: booking, rescheduling, and availability
questions. This is the most complex sub-agent because — unlike FAQ/Billing,
which just retrieve-and-answer in one shot — this agent has to collect
structured data ACROSS MULTIPLE TURNS before it can act:

  patient_name -> department -> preferred_date -> (check availability) -> confirm

Note: extraction of these fields from the caller's speech happens BEFORE this
node runs, in agents/extraction.py, which runs on every turn regardless of
intent (so a caller who mentions their name on a non-appointment turn doesn't
lose it). By the time handle_appointment runs, state["collected_data"] already
has whatever's been captured so far — this agent's job is just:

1. Check which of the 3 required fields are still missing and ask ONLY for
   the next missing one (never dump all three questions on the caller at
   once — that's not how a real receptionist talks).
2. Once all fields are present, check the mock schedule (Phase 5 will swap
   this for a real n8n webhook call). If a slot is available, book it and
   confirm. If not, offer the nearest alternate slots for that department
   and ask the caller to pick one.
"""

from agents.state import AgentState
from agents.mock_schedule import (
    find_slot_on_date,
    book_slot,
    get_available_slots,
    DEPARTMENTS,
)

REQUIRED_FIELDS = ["patient_name", "department", "preferred_date"]


def _next_missing_field_prompt(missing_field: str) -> str:
    prompts = {
        "patient_name": "Sure, I can help you book that. Can I get the patient's name, please?",
        "department": "Got it. Which department or type of doctor would you like to see?",
        "preferred_date": "Great — what date were you hoping to come in?",
    }
    return prompts.get(missing_field, "Could you give me a bit more detail on that?")


def handle_appointment(state: AgentState) -> AgentState:
    """
    LangGraph node function for the appointment sub-agent. Assumes
    extract_booking_details (agents/extraction.py) already ran earlier this
    turn and populated state["collected_data"] with anything the caller
    has provided so far.
    """
    collected = state.get("collected_data", {}) or {}

    # If this turn's message mentioned a department the extraction step
    # couldn't recognize, tell the caller directly instead of just silently
    # asking for department again as if nothing was said.
    unrecognized = state.get("_unrecognized_department")
    if unrecognized:
        dept_list = ", ".join(DEPARTMENTS)
        state["agent_response"] = (
            f"I don't have a department called '{unrecognized}'. We have {dept_list}. "
            "Which one would you like?"
        )
        return state

    missing = [f for f in REQUIRED_FIELDS if not collected.get(f)]
    if missing:
        state["agent_response"] = _next_missing_field_prompt(missing[0])
        return state

    # All three fields present — check availability against the mock schedule.
    slot = find_slot_on_date(collected["department"], collected["preferred_date"])

    if slot:
        booked = book_slot(collected["department"], collected["preferred_date"], collected["patient_name"])
        state["agent_response"] = (
            f"You're all set, {collected['patient_name']}. I've booked you with "
            f"{booked['doctor']} in {collected['department']} on {booked['date']} at "
            f"{booked['time']}. Is there anything else I can help with?"
        )
        # Booking complete — clear collected_data so a new request starts fresh
        # if the caller asks for something else in the same call.
        state["collected_data"] = {}
        return state

    # No slot on the requested date — offer real alternatives instead of just
    # saying "no." This is what makes the agent feel helpful instead of a dead end.
    alternates = get_available_slots(collected["department"])[:3]
    if alternates:
        options = "; ".join([f"{s['date']} at {s['time']} with {s['doctor']}" for s in alternates])
        state["agent_response"] = (
            f"I don't see anything open on {collected['preferred_date']} for "
            f"{collected['department']}, but I do have: {options}. Would any of those work?"
        )
    else:
        state["agent_response"] = (
            f"I'm sorry, there's nothing currently available in {collected['department']}. "
            "I can transfer you to the front desk to check further options."
        )

    # Keep patient_name and department, but drop the date so the caller's next
    # answer (picking one of the alternates) gets captured as the new preferred_date.
    collected.pop("preferred_date", None)
    state["collected_data"] = collected
    return state
