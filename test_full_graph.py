"""
test_full_graph.py

End-to-end test of the complete Phase 3 system: Router + all 4 sub-agents,
wired together as one LangGraph graph.

Two things being proven here:
1. Each of the 4 intents gets routed to the correct sub-agent and produces
   a sensible response (basic sanity check, same queries as test_router.py).
2. The "killer feature" from the spec: a caller can switch topics MID-CALL
   (start booking an appointment, then ask a billing question, then come
   back to finish the booking) without the system losing track of what it
   already collected. This only works because conversation_history and
   collected_data are threaded through every turn manually here, exactly
   like the real webhook handler will do in Phase 4.
"""

from agents.graph import run_turn

print("=== Single-turn sanity check across all 4 intents ===")
for text in [
    "Hi, I'd like to book an appointment with a cardiologist",
    "What insurance providers do you accept?",
    "I think there's a mistake on my last bill, can you check?",
    "My father is having chest pain and can't breathe properly",
]:
    result = run_turn(text)
    print(f'Caller: "{text}"')
    print(f"  -> intent: {result['intent']} | escalate: {result['escalate']}")
    print(f"  -> response: {result['agent_response']}")
    print()


print("=== Multi-turn call with a mid-call topic switch ===")
history = []
collected = {}

turns = [
    "Hi, I'd like to book an appointment, my name is Ahmed Raza",
    "Actually wait, what's your billing policy on late payments?",
    "Okay thanks. Let's go back to the booking — I need cardiology",
    "The 15th of July works for me",
]

for text in turns:
    result = run_turn(text, conversation_history=history, collected_data=collected)
    print(f'Caller: "{text}"')
    print(f"  -> intent: {result['intent']}")
    print(f"  -> response: {result['agent_response']}")
    print(f"  -> collected_data so far: {result['collected_data']}")
    print()

    history.append({"role": "caller", "text": text})
    history.append({"role": "agent", "text": result["agent_response"]})
    # Only the appointment agent uses collected_data across turns — billing/faq
    # don't touch it, so carrying it forward as-is is safe and correct here.
    collected = result["collected_data"]
