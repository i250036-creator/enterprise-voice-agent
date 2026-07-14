"""
webhook_server.py

This is the bridge between VAPI (the phone call) and the LangGraph backend
(agents/graph.py) built in Phase 3.

How VAPI's "Custom LLM" integration works: instead of using VAPI's built-in
model, you point VAPI at YOUR server's URL. On every caller turn, VAPI sends
an OpenAI-compatible chat completion request (the full message history so
far) to that URL, and expects an OpenAI-compatible response back containing
the text VAPI should speak next.

The tricky part Phase 3 didn't have to deal with: VAPI handles MANY calls at
once, each with its own conversation. So this server has to keep each call's
conversation_history and collected_data SEPARATE, keyed by VAPI's call ID —
otherwise two simultaneous callers would corrupt each other's booking state.

NOTE on state storage: this uses a simple in-memory dict, which is fine for
a demo/single-instance deployment (Render/Railway free tier runs one worker).
It will NOT survive a server restart or work correctly if you scale to
multiple worker processes — for a real production deployment you'd swap
`call_sessions` for Redis or a database keyed the same way. Flagging this
now so it isn't a surprise later.
"""

import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from agents.graph import run_turn
from agents import retrieval

app = FastAPI()


@app.on_event("startup")
async def preload_models():
    """
    Loads the embedding model and Qdrant client once, when the server
    starts (deploy time / wake-from-sleep time) — not during a live call.
    See agents/retrieval.py's preload() docstring for why this matters.
    """
    retrieval.preload()

# call_id -> {"conversation_history": [...], "collected_data": {...}}
call_sessions: dict = {}


def get_session(call_id: str) -> dict:
    if call_id not in call_sessions:
        call_sessions[call_id] = {"conversation_history": [], "collected_data": {}}
    return call_sessions[call_id]


def extract_call_id(body: dict) -> str:
    """
    VAPI includes call metadata in the request body as body["call"]["id"].
    Falling back to a random id if it's ever missing (e.g. local testing
    with curl) so the server doesn't crash — it just won't have continuity
    across turns in that fallback case.
    """
    call_info = body.get("call") or {}
    return call_info.get("id") or "no-call-id-" + str(uuid.uuid4())


def extract_last_user_message(messages: list) -> str:
    """VAPI sends the full message history each turn — we only need the caller's latest line."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    call_id = extract_call_id(body)
    session = get_session(call_id)

    caller_input = extract_last_user_message(body.get("messages", []))

    result = run_turn(
        caller_input=caller_input,
        conversation_history=session["conversation_history"],
        collected_data=session["collected_data"],
    )

    # Persist updated state for this call_id so the NEXT turn picks up
    # where this one left off.
    session["conversation_history"].append({"role": "caller", "text": caller_input})
    session["conversation_history"].append({"role": "agent", "text": result["agent_response"]})
    session["collected_data"] = result["collected_data"]

    # NOTE on escalation: result["escalate"] is True when the Emergency
    # Agent fires. Actually transferring the live call is a VAPI-side
    # "transferCall" tool config, wired in during Phase 6 — for now this
    # response just gets spoken like any other turn, and the escalate flag
    # is available here for logging/monitoring in the meantime.

    return JSONResponse(_build_openai_response(result["agent_response"]))


def _build_openai_response(text: str) -> dict:
    """
    VAPI expects a standard OpenAI chat-completion-shaped response.
    Most fields beyond `choices[0].message.content` are ignored by VAPI,
    but are included here so the response is a valid OpenAI response shape
    (helps if you ever test this endpoint against other OpenAI-compatible
    tooling too).
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "horizon-voice-agent",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/health")
async def health():
    """Simple check so Render/Railway (and you) can confirm the server is up."""
    return {"status": "ok", "active_calls": len(call_sessions)}
