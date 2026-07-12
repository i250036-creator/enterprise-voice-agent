"""
test_webhook_server.py

Simulates what VAPI actually sends, using FastAPI's TestClient so you don't
need a live deployment to test this. VAPI sends the FULL messages array each
turn (like a normal OpenAI chat request) plus a "call" object with the call's
unique id.

Two things being proven:
1. A single call can carry an appointment booking across several turns
   (same as test_full_graph.py, but now going through the HTTP layer).
2. TWO DIFFERENT calls (different call ids) happening "at the same time"
   don't leak state into each other — call A's patient name shouldn't show
   up in call B's session.
"""

from fastapi.testclient import TestClient
from webhook_server import app

client = TestClient(app)


def send_turn(call_id: str, messages: list):
    body = {
        "model": "horizon-voice-agent",
        "messages": messages,
        "call": {"id": call_id},
    }
    response = client.post("/chat/completions", json=body)
    return response.json()["choices"][0]["message"]["content"]


print("=== Call A: booking an appointment across turns ===")
call_a_id = "call-A-111"
history_a = []

turns_a = [
    "Hi, my name is Zainab Sheikh",
    "I need to see a dentist",
    "The 14th of July please",
]
for text in turns_a:
    history_a.append({"role": "user", "content": text})
    reply = send_turn(call_a_id, history_a)
    print(f'Call A caller: "{text}"')
    print(f"  -> agent: {reply}")
    history_a.append({"role": "assistant", "content": reply})

print()
print("=== Call B: a completely different caller, interleaved with Call A ===")
call_b_id = "call-B-222"
history_b = [{"role": "user", "content": "What time does the dental department open?"}]
reply_b = send_turn(call_b_id, history_b)
print(f'Call B caller: "What time does the dental department open?"')
print(f"  -> agent: {reply_b}")

print()
print("=== Verifying Call A's session wasn't affected by Call B ===")
print("Call A session collected_data:", __import__("webhook_server").call_sessions[call_a_id]["collected_data"])
print("Call B session collected_data:", __import__("webhook_server").call_sessions[call_b_id]["collected_data"])

print()
print("=== Health check ===")
print(client.get("/health").json())
