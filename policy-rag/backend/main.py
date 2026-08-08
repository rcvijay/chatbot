import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from rag_engine import HybridRAGEngine
from nemoguardrails import RailsConfig, LLMRails

# ---------------------------------------------------------
# Environment Variable Guardrail Check
# ---------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("FATAL ERROR: 'OPENAI_API_KEY' environment variable is missing!")
    sys.exit(1)

# Ensure the key is explicitly set in os.environ for NeMo / OpenAI SDK
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

app = FastAPI(title="Policy RAG Engine")
engine = HybridRAGEngine()

# Load NeMo Guardrails Config
config_path = os.path.join(os.path.dirname(__file__), "config")
guardrails_config = RailsConfig.from_path(config_path)

# Initialize NeMo Guardrails (Pass ONLY config to __init__)
rails = LLMRails(config=guardrails_config)

# Pre-ingest HR Manual on startup
# Pre-ingest HR Document on startup
@app.on_event("startup")
def startup_event():
    sample_pdf = "HR_Document.pdf"
    if os.path.exists(sample_pdf):
        engine.ingest_pdf(sample_pdf, doc_name="HR Document", effective_date="2026")
    else:
        print(f"WARNING: {sample_pdf} not found in the backend root directory.")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    refused: bool

@app.post("/query", response_model=QueryResponse)
async def query_policy(request: QueryRequest):
    # 1. Apply Input Guardrails via NeMo
    res = await rails.generate_async(prompt=request.question)
    if "refuse off topic" in res.response or "I am an internal HR" in res.response:
        return QueryResponse(answer=res.response, citations=[], refused=True)

    # 2. Retrieve & Rerank Chunks (BM25 + FAISS + Recency)
    retrieved_chunks = engine.retrieve_and_rerank(request.question, top_k=3)
    
    if not retrieved_chunks:
        return QueryResponse(
            answer="I could not find any official policy document covering your question.",
            citations=[],
            refused=True
        )

    # 3. Construct Grounded Context Prompt
    context_str = "\n\n".join([
        f"--- DOCUMENT: {c['doc_name']} (Date: {c['effective_date']}, Section: {c['section']}) ---\n{c['text']}"
        for c in retrieved_chunks
    ])
    
    grounded_prompt = f"""You are an internal HR and Policy assistant. Answer the employee question strictly using the provided context below. 

CRITICAL RULES:
1. If the context does NOT contain the answer, reply ONLY: "The official policy documents do not cover this question."
2. Do NOT invent policies, forms, or limits.
3. Cite the exact document and section at the end of your answer.

Context:
{context_str}

Question: {request.question}
Answer:"""

    # 4. LLM Generation via Guardrails
    generation_response = await rails.generate_async(prompt=grounded_prompt)
    answer = generation_response.response

    citations = [
        {"doc_name": c["doc_name"], "section": c["section"], "effective_date": c["effective_date"]}
        for c in retrieved_chunks
    ]

    return QueryResponse(
        answer=answer,
        citations=citations,
        refused="do not cover this question" in answer
    )