# Deployment Guide — Render

This gets `webhook_server.py` running on a public HTTPS URL so VAPI can reach it.
Render's free tier is enough for a demo/portfolio deployment.

---

## 1. Prepare the repo

Your `enterprise-voice-agent` folder should now look like:

```
enterprise-voice-agent/
  agents/
    __init__.py
    state.py
    router_agent.py
    retrieval.py
    faq_agent.py
    billing_agent.py
    emergency_agent.py
    extraction.py
    appointment_agent.py
    mock_schedule.py
    graph.py
  webhook_server.py
  test_router.py
  test_faq_agent.py
  test_billing_emergency.py
  test_appointment.py
  test_full_graph.py
  test_webhook_server.py
  requirements.txt
  .env.example
  .gitignore
```

Make sure `.env` (your REAL keys) is NOT committed — `.gitignore` already excludes it.
Only `.env.example` (placeholder values) should be in the repo.

```powershell
cd enterprise-voice-agent
git init
git add .
git commit -m "Phase 1-4: multi-agent voice system + webhook server"
```

Create a new GitHub repo (via github.com or `gh repo create`), then:

```powershell
git remote add origin https://github.com/<your-username>/enterprise-voice-agent.git
git branch -M main
git push -u origin main
```

---

## 2. Create the Render Web Service

1. Go to render.com, sign in, click **New +** → **Web Service**.
2. Connect your GitHub account and select the `enterprise-voice-agent` repo.
3. Configure:
   - **Name**: `enterprise-voice-agent` (or anything)
   - **Region**: closest to you
   - **Branch**: `main`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn webhook_server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free

---

## 3. Add environment variables

In the Render service's **Environment** tab, add each one individually (do NOT
upload the .env file itself):

| Key | Value |
|---|---|
| `OPENROUTER_API_KEY` | your real key |
| `QDRANT_URL` | your real Qdrant cluster URL |
| `QDRANT_API_KEY` | your real Qdrant key |

Click **Save Changes** — this triggers a redeploy automatically.

---

## 4. Verify the deployment

Once the build finishes (watch the Logs tab), visit:

```
https://<your-app-name>.onrender.com/health
```

You should see:
```json
{"status": "ok", "active_calls": 0}
```

If it fails, check the Logs tab first — most common issues:
- Missing env var (import error mentioning `OPENROUTER_API_KEY` etc. being `None`)
- Wrong start command (typo in `webhook_server:app`)
- Build failing on `sentence-transformers`/`qdrant-client` install (rare, but if it
  times out on Render's free tier, retry the deploy — first-time model/package
  downloads can be slow)

---

## 5. Free tier behavior to expect

Render's free web services spin down after ~15 minutes of no traffic, and the
next request wakes it back up (can take 30-60 seconds, plus the embedding
model reloading). This means:
- The very first call after idle time will feel slow / might even time out
  on VAPI's side if it's not patient enough.
- For a live demo, "warm up" the server by hitting `/health` a minute or two
  before the call.
- If this becomes a real problem later (not just a demo), Render's paid tier
  keeps the instance always-on.

---

## 6. Your Custom LLM endpoint for VAPI

Once verified, this is the URL to paste into VAPI's Custom LLM Server URL field:

```
https://<your-app-name>.onrender.com/chat/completions
```

Continue with `VAPI_SETUP_GUIDE.md` from here.
