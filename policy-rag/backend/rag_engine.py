import os
from typing import List, Dict, Any
from datetime import datetime
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

class HybridRAGEngine:
    def __init__(self):
        # Dense Embedding Model & Cross-Encoder Reranker
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.chunks: List[Dict[str, Any]] = []
        self.bm25 = None
        self.faiss_index = None

    def ingest_pdf(self, file_path: str, doc_name: str, effective_date: str):
        """Ingests a PDF, chunks it with section context, and stores date metadata."""
        reader = PdfReader(file_path)
        raw_chunks = []
        
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text.strip():
                continue
            
            # Simple paragraph-based chunking
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
            for p_idx, paragraph in enumerate(paragraphs):
                raw_chunks.append({
                    "id": f"{doc_name}_p{idx+1}_c{p_idx+1}",
                    "doc_name": doc_name,
                    "section": f"Page {idx+1}",
                    "effective_date": effective_date,
                    "text": paragraph
                })

        self.chunks.extend(raw_chunks)
        self._build_indexes()

    def _build_indexes(self):
        """Builds both BM25 and FAISS dense indexes."""
        if not self.chunks:
            return

        # 1. BM25 Index
        corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

        # 2. FAISS Index
        embeddings = self.embedder.encode([c["text"] for c in self.chunks], show_progress_bar=False)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)

    def retrieve_and_rerank(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid retrieval (BM25 + FAISS) with Cross-Encoder & Recency Reranking."""
        if not self.chunks or not self.faiss_index:
            return []

        # 1. Lexical Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]

        # 2. Dense Search (FAISS)
        query_vector = self.embedder.encode([query])
        faiss.normalize_L2(query_vector)
        _, faiss_indices = self.faiss_index.search(query_vector, top_k * 2)
        faiss_top_indices = faiss_indices[0]

        # Merge candidate indices
        candidate_indices = list(set(bm25_top_indices).union(set(faiss_top_indices)))
        candidates = [self.chunks[i] for i in candidate_indices if i < len(self.chunks)]

        if not candidates:
            return []

        # 3. Cross-Encoder Reranking + Recency Weighting
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        current_year = datetime.now().year
        scored_candidates = []
        
        for candidate, relevance_score in zip(candidates, rerank_scores):
            # Parse year for recency boost
            try:
                doc_year = int(candidate["effective_date"].split(".")[-1])
            except (ValueError, IndexError):
                doc_year = 2018  # Default fallback
                
            recency_weight = 1.0 + max(0, (doc_year - 2018) * 0.05) # Priority boost for post-2018 docs
            final_score = relevance_score * recency_weight
            
            scored_candidates.append({
                "chunk": candidate,
                "score": float(final_score)
            })

        # Sort by final score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return [item["chunk"] for item in scored_candidates[:top_k]]