"""
test_billing_emergency.py

Standalone test for the billing agent and emergency agent.
"""

from agents.billing_agent import handle_billing
from agents.emergency_agent import handle_emergency
from agents.state import AgentState


def make_state(caller_input: str, intent: str) -> AgentState:
    return {
        "caller_input": caller_input,
        "conversation_history": [],
        "intent": intent,
        "collected_data": {},
        "agent_response": None,
        "escalate": False,
    }


print("=== Billing Agent Tests ===\n")

billing_questions = [
    "What payment methods do you accept?",
    "What's your refund policy if I cancel an appointment?",
    "Why is my balance $500, I don't understand this charge",  # account-specific — should offer transfer
]

for q in billing_questions:
    state = make_state(q, "billing")
    result = handle_billing(state)
    print(f"Caller asked: \"{q}\"")
    print(f"Agent response: {result['agent_response']}\n")


print("=== Emergency Agent Test ===\n")

state = make_state("My father is having chest pain and can't breathe", "emergency")
result = handle_emergency(state)
print(f"Caller said: \"{state['caller_input']}\"")
print(f"Agent response: {result['agent_response']}")
print(f"Escalate flag: {result['escalate']}")
