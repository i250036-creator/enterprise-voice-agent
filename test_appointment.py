"""
test_appointment.py

Simulates multi-turn callers talking to the Appointment Agent, since this
agent (unlike FAQ/Billing) depends on state carrying over between turns.

As of the extraction refactor, extract_booking_details() runs BEFORE
handle_appointment() on every turn (this is what graph.py does automatically
via the "extract_details" node) — so this test calls both, in that order,
to accurately simulate what actually happens inside the graph.

Three scenarios:
1. Caller gives everything at once, on a date that IS available -> books immediately.
2. Caller gives info piece by piece across turns -> agent asks for each missing
   field in turn, then books once complete.
3. Caller asks for a date that's NOT available -> agent offers real alternates.
"""

from agents.state import AgentState
from agents.extraction import extract_booking_details
from agents.appointment_agent import handle_appointment


def new_state(caller_input, history=None, collected=None) -> AgentState:
    return {
        "caller_input": caller_input,
        "conversation_history": history or [],
        "intent": "appointment",
        "collected_data": collected or {},
        "agent_response": None,
        "escalate": False,
        "_unrecognized_department": None,
    }


def run(caller_input, history=None, collected=None):
    state = new_state(caller_input, history, collected)
    state = extract_booking_details(state)
    state = handle_appointment(state)
    return state


print("=== Scenario 1: all info given at once, available slot ===")
state = run("Hi, this is Ali Raza, I'd like to book with cardiology on the 16th of July")
print("Agent:", state["agent_response"])
print()

print("=== Scenario 2: info given piece by piece ===")
history = []
state = run("Hi, I want to book an appointment", history)
print("Agent:", state["agent_response"])
history.append({"role": "caller", "text": "Hi, I want to book an appointment"})
history.append({"role": "agent", "text": state["agent_response"]})

state = run("My name is Sana Malik", history, state["collected_data"])
print("Agent:", state["agent_response"])
history.append({"role": "caller", "text": "My name is Sana Malik"})
history.append({"role": "agent", "text": state["agent_response"]})

state = run("I need to see someone for my teeth", history, state["collected_data"])
print("Agent:", state["agent_response"])
history.append({"role": "caller", "text": "I need to see someone for my teeth"})
history.append({"role": "agent", "text": state["agent_response"]})

state = run("The 14th of July works for me", history, state["collected_data"])
print("Agent:", state["agent_response"])
print()

print("=== Scenario 3: unavailable date, agent offers alternates ===")
state = run("Book me with Dr for orthopedics, name is Bilal Khan, on July 20th")
print("Agent:", state["agent_response"])
