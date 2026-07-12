# VAPI Setup Guide — Connecting Voice to the LangGraph Backend

This covers Phase 4: wiring the LangGraph agent system (Phase 3, already
working) to an actual phone call via VAPI.

---

## 1. Deploy the webhook server first

VAPI needs a public HTTPS URL to call — it can't reach `localhost`. Deploy
`webhook_server.py` to Render or Railway (free tier is enough for a demo):

1. Push your `enterprise-voice-agent` folder to a GitHub repo.
2. On Render: New → Web Service → connect the repo.
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn webhook_server:app --host 0.0.0.0 --port $PORT`
3. Add your environment variables (`OPENROUTER_API_KEY`, `QDRANT_URL`,
   `QDRANT_API_KEY`) in Render's dashboard — not in code.
4. Once deployed, confirm it's alive: visit `https://your-app.onrender.com/health`
   — should return `{"status": "ok", ...}`.
5. Your Custom LLM endpoint is: `https://your-app.onrender.com/chat/completions`

Make sure `requirements.txt` includes `fastapi` and `uvicorn` in addition to
everything from Phases 1-3 (`openai`, `langgraph`, `qdrant-client`,
`sentence-transformers`, `python-dotenv`).

---

## 2. Create the VAPI account + assistant

1. Sign up at vapi.ai, grab your API key from the dashboard.
2. Create a new Assistant.
3. **Model section** → choose **Custom LLM** as the provider.
   - Server URL: your `/chat/completions` endpoint from step 1.
   - Model name field: anything (e.g. `horizon-voice-agent`) — the webhook
     ignores this value, it's just required by the form.
4. **Transcriber (STT)** → Deepgram (VAPI's default, already good).
5. **Voice (TTS)** → ElevenLabs, Flash v2.5 model — pick a calm, professional
   voice from the ElevenLabs voice list VAPI shows you.

---

## 3. Configure the first message and compliance disclaimer

Set the assistant's **First Message** directly in VAPI (not through your
backend — this greeting plays before any caller input exists, so your
webhook never even sees this turn):

> "Thank you for calling Horizon Health Clinic. This call may be recorded
> and handled by an AI assistant. How can I help you today?"

This satisfies the Phase 6 compliance disclaimer requirement from day one.

---

## 4. Turn on interruption handling (barge-in)

In the assistant's **Speech settings**, enable "Allow interruptions" (VAPI
may label this "Voicemail/Interruption" or similar depending on dashboard
version — look for barge-in / interruption sensitivity). Leave sensitivity
at VAPI's default to start; tune it after live testing if callers get cut
off too eagerly or not eagerly enough.

---

## 5. Get a test phone number

1. In VAPI, go to Phone Numbers → Buy a number (uses Twilio under the hood,
   billed through VAPI).
2. Attach your Assistant to that number.
3. Call it from your own phone.

---

## 6. What to test on the first live call

Run through the same scenarios already proven in `test_full_graph.py`,
but out loud:
- Book an appointment (give info across multiple sentences, not all at once)
- Ask an FAQ question
- Ask a billing question
- Say something that should trigger emergency detection
- **The mid-call topic switch** — start a booking, ask something unrelated,
  then come back to finish the booking. This is your strongest differentiator,
  so it's worth deliberately testing on a real call, not just in code.

---

## 7. Latency check

Spec target: under 800ms response time. If responses feel slow on a live call:
- The embedding model (`all-MiniLM-L6-v2`) loading on cold start is the
  likely first culprit — Render's free tier spins down after inactivity,
  so the first call after idle time will be slow to load weights. This is
  a free-tier limitation, not a bug in your code.
- If it's consistently slow (not just first-call), consider caching common
  FAQ answers as mentioned in the original spec.

---

## Known limitation carried over from `webhook_server.py`

Call state is stored in memory (a plain Python dict), which works for a
single-instance demo deployment but won't survive a server restart or scale
correctly across multiple worker processes. Fine for portfolio/demo purposes;
flag this explicitly if a client asks about production-readiness, and the
fix (Redis or a small database table keyed by call id) is a quick follow-up
once someone actually needs it.
