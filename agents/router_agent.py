"""
router_agent.py

The Router Agent is the entry point of the graph. On every caller turn,
it looks at what the caller just said (plus recent history) and decides
which specialized sub-agent should handle it: appointment, faq, billing,
or emergency.

Why this matters: a naive approach would just keep the caller in whatever
sub-agent they started with. This router re-classifies on EVERY turn,
which is what allows a caller to say "actually, what's your billing policy"
in the middle of booking an appointment, and have the system correctly
hand off — without losing the conversation state.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from agents.state import AgentState

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-4o"

ROUTER_SYSTEM_PROMPT = """You are a call routing classifier for Horizon Health Clinic's phone system.

Given what the caller just said, classify their intent into EXACTLY ONE of these categories:

- "appointment": caller wants to book, reschedule, or cancel an appointment, or ask about doctor availability
- "faq": caller is asking a general question about departments, hours, insurance coverage, or clinic policies
- "billing": caller is asking about their bill, payment, outstanding balance, or refund
- "emergency": caller mentions urgent symptoms (chest pain, difficulty breathing, severe bleeding, loss of consciousness, or says it's an emergency)

Rules:
- If there is ANY ambiguity between emergency and something else, choose "emergency". Safety comes first.
- Base your decision on the CURRENT message primarily, using conversation history only for context.
- Respond with ONLY valid JSON in this exact format, nothing else: {"intent": "appointment"}
"""


def classify_intent(state: AgentState) -> AgentState:
    """
    LangGraph node function. Reads state["caller_input"] and recent history,
    calls GPT-4o via OpenRouter to classify intent, and writes state["intent"].
    """
    history_snippet = ""
    if state.get("conversation_history"):
        recent = state["conversation_history"][-4:]  # last 4 turns for context
        history_snippet = "\n".join([f'{turn["role"]}: {turn["text"]}' for turn in recent])

    user_message = f"""Recent conversation:
{history_snippet}

Caller just said: "{state['caller_input']}"

Classify the intent."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=30,
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw_output)
        intent = parsed.get("intent", "faq")
    except json.JSONDecodeError:
        intent = "faq"

    valid_intents = {"appointment", "faq", "billing", "emergency"}
    if intent not in valid_intents:
        intent = "faq"

    state["intent"] = intent
    return state