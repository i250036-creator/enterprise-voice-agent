"""
faq_agent.py

Handles the "faq" intent: general questions about departments, doctor
availability, insurance policy, and clinic hours.

Flow: retrieve relevant chunks from Qdrant (Phase 2 RAG layer) -> hand
those chunks to GPT-4o along with the caller's question -> GPT-4o writes
a short, natural, SPOKEN-style answer (not a document dump).
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

FAQ_SYSTEM_PROMPT = """You are a friendly phone receptionist for Horizon Health Clinic.
A caller has asked a general question. You are given relevant context retrieved from
the clinic's knowledge base. Answer using ONLY that context.

Rules:
- Keep the answer short and conversational — this will be SPOKEN aloud, not read as text.
  2-3 sentences maximum.
- Do not mention "the knowledge base," "the document," or "the context" — just answer naturally,
  as a receptionist who simply knows this information.
- If the context doesn't contain the answer, say you don't have that information and offer
  to transfer the caller to the front desk. Do not make anything up.
"""


def handle_faq(state: AgentState) -> AgentState:
    """
    LangGraph node function for the FAQ sub-agent.
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
            {"role": "system", "content": FAQ_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # slight flexibility for natural phrasing, still mostly factual
        max_tokens=150,
    )

    answer = response.choices[0].message.content.strip()
    state["agent_response"] = answer
    return state
