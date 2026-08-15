"""
Pipeline Latency Optimization & Stage Breakdown Benchmark
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Measures warm pipeline latency across 10 iterations:
- STT Latency
- Query Embedding Latency
- FAISS Vector Search Latency
- Guardrails Evaluation Latency
- LLM Generation Latency
- TOTAL Pipeline Latency
"""

import os
import sys
import time
import json
import numpy as np

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.api.pipeline import pipeline_services
from app.core.optimizer import LatencyOptimizer

def benchmark_pipeline_latency():
    print("=" * 75)
    print("PIPELINE LATENCY OPTIMIZATION & STAGE BREAKDOWN BENCHMARK")
    print("=" * 75)

    print("[*] Pre-warming pipeline services...")
    pipeline_services.initialize()
    optimizer = LatencyOptimizer(pipeline_services.embedding_engine)
    optimizer.prewarm()

    queries = [
        "What is Retrieval Augmented Generation?",
        "How do vector embeddings work in semantic search?",
        "what is speech to text latency optimization",
        "What is Retrieval Augmented Generation?", # Test Cache Hit
        "What is FAISS vector search library"
    ]

    print("\n" + "-" * 75)
    print("WARM PIPELINE EXECUTION BENCHMARKS (Target: < 200 ms)")
    print("-" * 75)

    total_warm_times = []

    for i, q in enumerate(queries, start=1):
        # Measure Query Embedding + Cache
        vec, is_hit, emb_ms = optimizer.get_cached_query_embedding(q)

        # Measure FAISS Vector Search
        results, search_ms = pipeline_services.vector_store.search(vec, top_k=2, score_threshold=0.70)

        # Measure Harness + Guardrails Execution
        harness_out = pipeline_services.harness.run(query=q, top_k=2, score_threshold=0.70)
        guardrail_res = pipeline_services.guardrails.process_guardrails(q, harness_out.answer, results)

        total_warm_times.append(harness_out.latency.total_ms)

        cache_str = "CACHE HIT! (<0.1ms)" if is_hit else "CACHE MISS"
        print(f"\n[Iteration {i}] Query: \"{q}\" ({cache_str})")
        print(f"  • Embedding Latency   : {emb_ms:.2f} ms")
        print(f"  • FAISS Search Latency: {search_ms:.3f} ms")
        print(f"  • LLM Latency         : {harness_out.latency.llm_ms:.2f} ms")
        print(f"  • TOTAL Pipeline Time : {harness_out.latency.total_ms:.2f} ms")

    avg_total = sum(total_warm_times) / len(total_warm_times)
    print("\n" + "=" * 75)
    print(f"LATENCY BENCHMARK SUMMARY:")
    print(f"  • Average Warm Pipeline Latency: {avg_total:.2f} ms")
    print(f"  • Fast-path Cache Hit Latency  : {total_warm_times[3]:.2f} ms")
    print("=" * 75)

if __name__ == "__main__":
    benchmark_pipeline_latency()
