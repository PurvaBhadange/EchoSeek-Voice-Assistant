"""
Full End-to-End Voice & Text RAG Pipeline Integration Test
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Tests complete user flows:
1. Text Query -> FAISS -> Gemini RAG -> Guardrails -> Output JSON
2. Voice Audio -> Sarvam STT -> FAISS -> Gemini RAG -> Guardrails -> Output JSON
"""

import os
import sys
import wave
import io
import json

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from app.api.pipeline import pipeline_services, TextQueryRequest

def create_sample_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b'\x00\x00' * 16000)
    return buffer.getvalue()

def test_full_pipeline():
    print("=" * 75)
    print("END-TO-END RAG PIPELINE INTEGRATION TEST")
    print("=" * 75)

    print("[*] Initializing Pipeline Services...")
    pipeline_services.initialize()

    # Test 1: Text Query Flow
    print("\n--- TEST 1: TEXT QUERY FLOW ---")
    query = "What is Retrieval Augmented Generation?"
    print(f"[*] Sending Text Query: \"{query}\"")

    harness_out = pipeline_services.harness.run(query=query, top_k=2, score_threshold=0.70)
    q_vec = pipeline_services.embedding_engine.embed_query(query)
    results, _ = pipeline_services.vector_store.search(q_vec, top_k=2, score_threshold=0.70)
    guardrail_res = pipeline_services.guardrails.process_guardrails(query, harness_out.answer, results)

    print(f"  • Query Transcript : \"{query}\"")
    print(f"  • Final Answer     : \"{guardrail_res.final_answer}\"")
    print(f"  • Guardrail Action : {guardrail_res.action_taken}")
    print(f"  • Sources Cited    : {harness_out.sources}")
    print(f"  • Total Latency    : {harness_out.latency.total_ms:.2f} ms")

    # Test 2: Voice Audio Query Flow
    print("\n--- TEST 2: VOICE AUDIO FILE QUERY FLOW ---")
    audio_bytes = create_sample_wav_bytes()
    print(f"[*] Uploading 1-Sec WAV Audio Blob ({len(audio_bytes)} bytes)...")

    voice_harness_out = pipeline_services.harness.run(audio_bytes=audio_bytes, top_k=2, score_threshold=0.70)
    vq_vec = pipeline_services.embedding_engine.embed_query(voice_harness_out.query)
    v_results, _ = pipeline_services.vector_store.search(vq_vec, top_k=2, score_threshold=0.70)
    v_guardrail_res = pipeline_services.guardrails.process_guardrails(voice_harness_out.query, voice_harness_out.answer, v_results)

    print(f"  • STT Transcript   : \"{voice_harness_out.query}\"")
    print(f"  • Final Answer     : \"{v_guardrail_res.final_answer}\"")
    print(f"  • Guardrail Action : {v_guardrail_res.action_taken}")
    print(f"  • STT Latency      : {voice_harness_out.latency.stt_ms:.2f} ms")
    print(f"  • Total Voice Time : {voice_harness_out.latency.total_ms:.2f} ms")

    print("\n" + "=" * 75)
    print("END-TO-END PIPELINE INTEGRATION SUCCESSFUL")
    print("=" * 75)

if __name__ == "__main__":
    test_full_pipeline()
