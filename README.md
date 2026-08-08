# chatbot folder structure
policy-rag/
├── app.py                  # Streamlit frontend UI
├── backend/
│   ├── main.py             # FastAPI backend server
│   ├── rag_engine.py       # BM25 + FAISS + Reranker pipeline
│   └── config/
│       ├── rails.co        # NeMo Guardrails dialog rules
│       └── config.yml      # NeMo Guardrails configuration
├── Dockerfile.frontend
├── Dockerfile.backend
├── docker-compose.yml
├── nginx.conf
└── requirements.txt

Summery

1. Problem Statement Summary
The core goal is to eliminate workplace reliance on outdated "watercooler chatter" by building a Hybrid RAG Assistant (Vector + Knowledge Graph) that searches official HR, Finance, IT, and Operations documents to give grounded, cited, and recency-aware answers.

Key Rules & Requirements
Recency & Conflict Resolution: When old and new policies conflict (e.g., a 2018 HR manual vs. a 2024 travel policy circular), the system must prefer the newest document.

Zero Hallucination / Honest Refusal: If an unwritten policy or missing form is requested, the bot must explicitly refuse to answer rather than guessing.

Granular Citations: Answers must contain specific document titles and section references so employees can verify details in under a minute.

Contextual Conversations: Must handle multi-turn follow-ups (e.g., "What is my leave quota?" followed by "Can I carry it forward?").

2. Technical Workflow Alignment
Here is how your proposed pipeline maps directly to this problem statement:
[HR/IT/Finance PDFs]
         │
         ▼
 ┌───────────────┐
 │ 1. Ingestion  │ ──► Parse text, metadata (Date, Doc Name, Section)
 └───────┬───────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
 ┌───────────────┐              ┌───────────────┐
 │ 2a. Vector DB │              │  2b. Graph DB │
 │ (Dense Embed) │              │  (Entities &  │
 └───────┬───────┘              │ Hierarchy)    │
         │                      └───────┬───────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
             ┌─────────────────────┐
             │ 3. Hybrid Retrieval │
             └──────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │ 4a. Reranking &     │ ──► Sort by Recency + Relevance Score
             │    Conflict Rules   │
             └──────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │ 4b. Guardrails      │ ──► Filter out ungrounded context
             └──────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │ 4c. LLM Generation  │ ──► Answer with precise citations
             └──────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │ 5. Re-Evaluation    │ ──► Groundedness, Citation accuracy,
             └─────────────────────┘     Refusal compliance


3. Step-by-Step Implementation Map
1.Document Ingestion & Metadata Tagging:Addresses PDF processing and recency rules.Extract text from documents like HR Mannual- Final (20.09.2018).pdf. Extract critical metadata fields: Document Type, Effective Date, and Section Headers.Critical: Include the effective_date metadata field for every chunk to handle policy version conflicts downstream.

2.Vector DB + Knowledge Graph Construction:Addresses fragmented policy mapping.Vector Store (ChromaDB / Qdrant / Pinecone): Stores semantic text embeddings for unstructured search (e.g., travel allowance rules).Knowledge Graph (Neo4j): Connects policy relationships and constraints (e.g., (Leave Policy) -[INCLUDES]-> (Casual Leave) -[HAS_RULE]-> (Carry Forward Limit)).

3.Hybrid Retrieval Engine:Combines semantic search with structural relationships.Query both stores simultaneously upon receiving an employee question:Fetch dense semantic matches from the Vector DB.Traverse graph nodes to retrieve connected rules, forms, and constraints.

4.Reranking, Conflict Resolution & Guardrails:Prevents outdated answers and hallucinations.Recency Reranker (Re-RAG): Rerank retrieved chunks by combined relevance and document release date (giving higher priority to post-2018 circulars over old manuals).Guardrails (NeMo Guardrails / Llama Guard): Check the retrieved context against a strict threshold. If relevance scores are below the cutoff, trigger an Explicit Refusal Response.LLM Generation: Send filtered chunks to the LLM with a system prompt mandating inline citations ([Doc Name, Section X]).

5.Re-Evaluation Suite:Validates system completion criteria.Evaluate the pipeline against the problem statement's primary metrics:Grounded Accuracy: Is the answer 100% supported by the retrieved chunk?Citation Correctness: Does the citation point to the correct document and section?Honest Refusal Rate: Does the system refuse correctly when given a query about unlisted policies?