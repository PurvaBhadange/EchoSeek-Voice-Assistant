"""
Embedding Generation & Persistence Script
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

1. Loads MSMARCO-XI dataset chunks from data/msmarco_sample.json
2. Chunks documents using MetadataAwarePassageChunker
3. Generates 384-dimensional vector embeddings using intfloat/multilingual-e5-small
4. Measures embedding generation latency
5. Persists vectors to data/embeddings/embeddings.npy and chunks to data/embeddings/chunks.json
"""

import os
import sys
import json
import time
import numpy as np

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.core.chunker import MetadataAwarePassageChunker
from app.core.embedding import EmbeddingEngine

def generate_and_save_embeddings():
    data_path = os.path.join(os.path.dirname(__file__), "../data/msmarco_sample.json")
    if not os.path.exists(data_path):
        print(f"[!] Dataset not found at {data_path}. Run scripts/prepare_dataset_subset.py first.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print("=" * 70)
    print("VECTOR EMBEDDING GENERATION & BENCHMARK")
    print("=" * 70)
    print(f"[*] Processing {len(docs)} input documents...")

    # Step 1: Chunk documents using MetadataAwarePassageChunker
    chunker = MetadataAwarePassageChunker()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc))

    print(f"[+] Total Chunks to Embed: {len(all_chunks)}")

    # Step 2: Initialize Embedding Engine
    engine = EmbeddingEngine("intfloat/multilingual-e5-small")
    passage_texts = [c.text for c in all_chunks]

    print("[*] Generating passage vector embeddings...")
    start_time = time.perf_counter()
    embeddings_matrix = engine.embed_passages(passage_texts)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"[+] Generated Embeddings Matrix Shape: {embeddings_matrix.shape}")
    print(f"[+] Passage Batch Embedding Latency : {elapsed_ms:.2f} ms ({elapsed_ms/len(all_chunks):.2f} ms/chunk)")

    # Step 3: Test single query embedding latency
    test_query = "What is retrieval augmented generation?"
    start_q = time.perf_counter()
    query_vector = engine.embed_query(test_query)
    q_elapsed_ms = (time.perf_counter() - start_q) * 1000

    print(f"\n[*] Single Query Embedding Test:")
    print(f"  • Query Text     : '{test_query}'")
    print(f"  • Query Vector Dim: {query_vector.shape[0]}")
    print(f"  • Latency        : {q_elapsed_ms:.2f} ms (Target: <50 ms)")

    # Step 4: Compute similarity scores against stored passages
    similarities = engine.compute_cosine_similarity(query_vector, embeddings_matrix)
    best_idx = np.argmax(similarities)
    print(f"  • Top Match Score: {similarities[best_idx]:.4f}")
    print(f"  • Top Passage    : '{all_chunks[best_idx].text[:90]}...'")

    # Step 5: Save embeddings and chunk metadata to disk
    embeddings_dir = os.path.join(os.path.dirname(__file__), "../data/embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    npy_path = os.path.join(embeddings_dir, "embeddings.npy")
    json_path = os.path.join(embeddings_dir, "chunks.json")

    np.save(npy_path, embeddings_matrix)
    
    chunks_dict_list = [c.model_dump() for c in all_chunks]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict_list, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("[+] PERSISTENCE SUCCESS:")
    print(f"  - Vector Matrix saved to : {npy_path} ({os.path.getsize(npy_path)} bytes)")
    print(f"  - Chunk Metadata saved to: {json_path} ({os.path.getsize(json_path)} bytes)")
    print("=" * 70)

if __name__ == "__main__":
    generate_and_save_embeddings()
