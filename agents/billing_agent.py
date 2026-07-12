"""
billing_agent.py

Handles the "billing" intent: questions about bills, payment methods,
outstanding balances, and refunds.

Similar structure to the FAQ agent (retrieve + answer), but with a
system prompt tuned for billing-specific tone: acknowledging the caller
may be frustrated about a charge, and being precise about numbers/policy
since billing mistakes are costly to get wrong.

Note: in a real deployment, this agent would also call an n8n webhook
to look up the caller's ACTUAL account balance (Phase 5). For now, it
answers general billing policy questions from the knowledge base — the
account-specific lookup gets wired in once n8n automation is built.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from agents.state import AgentState
from agents.retrieval import retrieve_context

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-4o"

BILLING_SYSTEM_PROMPT = """You are a phone receptionist for Horizon Health Clinic handling a billing question.
You are given relevant context retrieved from the clinic's billing policy knowledge base.
Answer using ONLY that context.

Rules:
- Keep the answer short and conversational — this will be SPOKEN aloud. 2-3 sentences maximum.
- Billing questions can be sensitive (the caller may be frustrated about a charge). Be calm,
  precise, and reassuring in tone, without being overly apologetic.
- Do not mention "the knowledge base" or "the context" — answer naturally.
- If the caller is asking about their SPECIFIC account balance or a specific charge (not general
  policy), say you'll need to pull up their account details and offer to transfer them to billing,
  since you don't have access to individual account records yet.
- If the context doesn't contain the answer, say so honestly and offer to transfer the caller.
"""


def handle_billing(state: AgentState) -> AgentState:
    """
    LangGraph node function for the billing sub-agent.
    """
    query = state["caller_input"]
    context = retrieve_context(query, top_k=3)

    user_message = f"""Retrieved context:
{context}

Caller's question: "{query}"

Answer the caller's question naturally, based only on the context above."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": BILLING_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=150,
    )

    answer = response.choices[0].message.content.strip()
    state["agent_response"] = answer
    return state
