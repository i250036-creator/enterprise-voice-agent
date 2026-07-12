"""
test_faq_agent.py

Standalone test for the FAQ agent. Tests that retrieval + answer
formatting work correctly before this gets wired into the full graph.
"""

from agents.faq_agent import handle_faq
from agents.state import AgentState

test_questions = [
    "What insurance providers do you accept?",
    "What time does the dental department open?",
    "Can I bring my 8 year old to visit a relative in the hospital ward?",
    "Do you have a helipad for medical evacuations?",  # not in knowledge base — tests the fallback
]

for question in test_questions:
    state: AgentState = {
        "caller_input": question,
        "conversation_history": [],
        "intent": "faq",
        "collected_data": {},
        "agent_response": None,
        "escalate": False,
    }

    result = handle_faq(state)
    print(f"Caller asked: \"{question}\"")
    print(f"Agent response: {result['agent_response']}\n")
