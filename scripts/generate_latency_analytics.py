"""
Empirical Latency Analytics Generator (P50 / P70 / P100)
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Runs a benchmark suite of queries across warm pipeline services,
calculates exact P50, P70, P100 metrics per stage, and exports JSON & Markdown reports.
"""

import os
import sys
import json
import time
import numpy as np

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.api.pipeline import pipeline_services
from app.core.optimizer import LatencyOptimizer
from app.core.analytics import AnalyticsEngine
from app.core.harness import PipelineLatencyBreakdown

def run_analytics_benchmark():
    print("=" * 75)
    print("P50 / P70 / P100 LATENCY ANALYTICS ENGINE")
    print("=" * 75)

    print("[*] Initializing & Pre-warming Pipeline Services...")
    pipeline_services.initialize()
    optimizer = LatencyOptimizer(pipeline_services.embedding_engine)
    optimizer.prewarm()

    benchmark_queries = [
        "What is Retrieval Augmented Generation?",
        "How do vector embeddings work in semantic search?",
        "what is speech to text latency optimization",
        "where is Goa located in India",
        "what is Sarvam AI known for",
        "how to measure P50 P70 and P100 latency",
        "what are RAG guardrails and hallucination prevention",
        "multilingual e5 small embedding model features",
        "what is Google Gemini model",
        "what is FAISS vector search library"
    ]

    breakdowns: List[PipelineLatencyBreakdown] = []

    print(f"[*] Executing {len(benchmark_queries)} warm benchmark queries...")

    for i, q in enumerate(benchmark_queries, start=1):
        # Measure Query Embedding + Cache
        vec, is_hit, emb_ms = optimizer.get_cached_query_embedding(q)

        # Measure FAISS Vector Search
        results, search_ms = pipeline_services.vector_store.search(vec, top_k=2, score_threshold=0.70)

        # Measure Harness Execution
        harness_out = pipeline_services.harness.run(query=q, top_k=2, score_threshold=0.70)

        # Override embedding and search latency in breakdown for precision
        harness_out.latency.embedding_ms = emb_ms
        harness_out.latency.vector_search_ms = search_ms
        
        breakdowns.append(harness_out.latency)
        print(f"  [Query {i:02d}] Total: {harness_out.latency.total_ms:.2f} ms | Search: {search_ms:.3f} ms | LLM: {harness_out.latency.llm_ms:.2f} ms")

    # Compute Percentiles
    analytics = AnalyticsEngine()
    report = analytics.generate_report(breakdowns)

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    json_path = os.path.join(data_dir, "latency_benchmarks.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    print("\n" + "=" * 75)
    print("EMPIRICAL P50 / P70 / P100 LATENCY BENCHMARK REPORT")
    print("=" * 75)
    print(f"Total Benchmark Queries Processed: {report.total_samples}\n")

    markdown_table = (
        "| Pipeline Stage | P50 (Median) | P70 | P100 (Max) | Avg | Min |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
        f"| **Speech-to-Text (Sarvam)** | {report.stt.p50:.2f} ms | {report.stt.p70:.2f} ms | {report.stt.p100:.2f} ms | {report.stt.avg:.2f} ms | {report.stt.min_val:.2f} ms |\n"
        f"| **Embedding (`e5-small`)** | {report.embedding.p50:.2f} ms | {report.embedding.p70:.2f} ms | {report.embedding.p100:.2f} ms | {report.embedding.avg:.2f} ms | {report.embedding.min_val:.2f} ms |\n"
        f"| **FAISS Vector Search** | {report.vector_search.p50:.3f} ms | {report.vector_search.p70:.3f} ms | {report.vector_search.p100:.3f} ms | {report.vector_search.avg:.3f} ms | {report.vector_search.min_val:.3f} ms |\n"
        f"| **LLM Generation** | {report.llm.p50:.2f} ms | {report.llm.p70:.2f} ms | {report.llm.p100:.2f} ms | {report.llm.avg:.2f} ms | {report.llm.min_val:.2f} ms |\n"
        f"| **TOTAL PIPELINE** | **{report.total.p50:.2f} ms** | **{report.total.p70:.2f} ms** | **{report.total.p100:.2f} ms** | **{report.total.avg:.2f} ms** | **{report.total.min_val:.2f} ms** |"
    )

    print(markdown_table)
    print("\n" + "=" * 75)
    print(f"[+] Saved empirical analytics JSON to: {json_path}")
    print("=" * 75)

if __name__ == "__main__":
    run_analytics_benchmark()
