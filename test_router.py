"""
test_router.py

Quick standalone test — run this to verify the router agent classifies
correctly BEFORE we wire it into the full LangGraph. Debugging one node
in isolation is far easier than debugging it live inside a call.
"""

from agents.router_agent import classify_intent
from agents.state import AgentState

test_cases = [
    "Hi, I'd like to book an appointment with a cardiologist",
    "What insurance providers do you accept?",
    "I think there's a mistake on my last bill, can you check?",
    "My father is having chest pain and can't breathe properly",
    "What time does the dental department open on Saturday?",
    "I need to reschedule my appointment from Tuesday to Thursday",
]

for caller_input in test_cases:
    state: AgentState = {
        "caller_input": caller_input,
        "conversation_history": [],
        "intent": None,
        "collected_data": {},
        "agent_response": None,
        "escalate": False,
    }

    result = classify_intent(state)
    print(f"Caller said: \"{caller_input}\"")
    print(f"  -> Classified as: {result['intent']}\n")
