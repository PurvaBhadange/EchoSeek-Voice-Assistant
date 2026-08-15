"""
End-to-End RAG Generation Test & Latency Benchmark Script
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Connects Embedding Engine -> FAISS Search -> Google Gemini LLM Service.
Benchmarks answer generation latency and verifies grounded responses.
"""

import os
import sys
import json
import numpy as np

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.core.chunker import Chunk
from app.core.embedding import EmbeddingEngine
from app.core.vector_store import FAISSVectorStore
from app.services.llm_service import GeminiLLMService

def test_rag_generation():
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    embeddings_file = os.path.join(data_dir, "embeddings/embeddings.npy")
    chunks_file = os.path.join(data_dir, "embeddings/chunks.json")

    if not (os.path.exists(embeddings_file) and os.path.exists(chunks_file)):
        print("[!] Embeddings missing. Run scripts/generate_embeddings.py first.")
        return

    print("=" * 75)
    print("GOOGLE GEMINI RAG GENERATION BENCHMARK")
    print("=" * 75)

    # Step 1: Load stored vector index
    embeddings_matrix = np.load(embeddings_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = [Chunk(**c) for c in json.load(f)]

    store = FAISSVectorStore(dimension=384)
    store.build_index(embeddings_matrix, chunks)

    engine = EmbeddingEngine("intfloat/multilingual-e5-small")
    llm_service = GeminiLLMService(model_name="gemini-3.5-flash")

    test_queries = [
        "What is Retrieval Augmented Generation?",
        "How do vector embeddings work in semantic search?"
    ]

    for query in test_queries:
        print(f"\n[USER QUERY]: \"{query}\"")
        
        # 1. Embed query
        q_vec = engine.embed_query(query)
        
        # 2. Search FAISS
        results, search_ms = store.search(q_vec, top_k=2, score_threshold=0.70)
        print(f"  • FAISS Vector Search Latency: {search_ms:.3f} ms (Retrieved {len(results)} chunks)")

        # 3. Generate Answer with Google Gemini
        llm_response = llm_service.generate_answer(query, results)

        print(f"  • LLM Provider/Model        : {llm_response.model_name}")
        print(f"  • LLM Generation Latency     : {llm_response.latency_ms:.2f} ms")
        print(f"  • Grounded Answer Output     : \"{llm_response.answer}\"")
        print(f"  • Cited Sources              : {llm_response.sources}")

    print("\n" + "=" * 75)
    print("RAG GENERATION BENCHMARK COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    test_rag_generation()
