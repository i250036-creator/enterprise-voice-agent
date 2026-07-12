"""
extraction.py

Runs on EVERY caller turn, BEFORE the router decides which sub-agent handles
it, to opportunistically capture booking-relevant details (patient name,
department, preferred date) the caller mentions.

Why this can't live only inside the Appointment Agent: callers don't always
state their intent and their details in the same breath. A caller might say
"Hi, my name is Zainab Sheikh" on turn 1 (which the router sends to FAQ,
since nothing there signals booking intent) and only mention wanting an
appointment a few turns later. If extraction only ran inside the Appointment
Agent, that name would be gone forever, since the Appointment Agent never
even runs on that turn. Running extraction unconditionally on every turn
means any detail the caller drops — whenever they drop it — is captured into
collected_data and is still there once they do start booking.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from agents.state import AgentState
from agents.mock_schedule import normalize_department

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-4o"

EXTRACTION_SYSTEM_PROMPT = """You extract appointment-booking details from a caller's speech
for Horizon Health Clinic's phone system. Today's date is 2026-07-13.

Extract these fields ONLY if the caller's CURRENT message ("Caller just said") provides or
changes them:
- "patient_name": the patient's full or first name
- "department": the department or type of doctor requested (e.g. cardiology, dental, pediatrics)
- "preferred_date": a date the caller mentioned, NORMALIZED to strict "YYYY-MM-DD" format.
  Convert whatever the caller said ("the 16th", "Tuesday", "July 16th", "16 July") into that
  format yourself, using today's date (2026-07-13) as reference for resolving day names or
  missing months/years (e.g. "the 16th" with no month said -> assume the current month, 2026-07-16).

Rules:
- IMPORTANT: if a field was already mentioned in an EARLIER turn and the caller's CURRENT
  message does not restate or change it, output null for that field. Only report what THIS
  message adds or changes. Use conversation history only to resolve references like "that one"
  or an incomplete sentence continued from before.
- These details may come up even if the caller isn't actively booking an appointment yet
  (e.g. they might just introduce their name in passing) — extract them whenever mentioned.
- preferred_date MUST always be in "YYYY-MM-DD" format if present, never free text.
- Respond with ONLY valid JSON in this exact format, nothing else:
  {"patient_name": null, "department": null, "preferred_date": null}
"""


def _extract_fields(state: AgentState) -> dict:
    history_snippet = ""
    if state.get("conversation_history"):
        recent = state["conversation_history"][-4:]
        history_snippet = "\n".join([f'{turn["role"]}: {turn["text"]}' for turn in recent])

    user_message = f"""Conversation so far:
{history_snippet}

Caller just said: "{state['caller_input']}"

Extract the booking fields."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=150,
    )

    raw_output = response.choices[0].message.content.strip()
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {}


def extract_booking_details(state: AgentState) -> AgentState:
    """
    LangGraph node function. Always runs, regardless of intent. Silently
    merges whatever it can into collected_data — it never sets
    agent_response itself, since generating the reply is each sub-agent's
    job, not this preprocessing step's.
    """
    collected = state.get("collected_data", {}) or {}
    extracted = _extract_fields(state)

    if extracted.get("patient_name"):
        collected["patient_name"] = extracted["patient_name"]

    if extracted.get("preferred_date"):
        collected["preferred_date"] = extracted["preferred_date"]

    raw_department = extracted.get("department")
    if raw_department:
        normalized = normalize_department(raw_department)
        if normalized:
            collected["department"] = normalized
        else:
            # Don't store an unrecognized department silently — stash the raw
            # attempt in a transient (non-persisted-across-turns-by-design)
            # field so the Appointment Agent can specifically tell the caller
            # "I don't recognize that department" if/when this turn actually
            # gets routed there. If it doesn't (e.g. this was a passing
            # remark during an FAQ turn), it's simply ignored.
            state["_unrecognized_department"] = raw_department

    state["collected_data"] = collected
    return state
