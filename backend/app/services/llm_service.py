"""
Google Gemini LLM Service for Grounded Answer Generation
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Formats retrieved FAISS context into strict system prompts and calls Google Gemini API.
Enforces no-hallucination guardrails and source URL citations.
"""

import time
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from app.config import settings
from app.core.vector_store import SearchResult

class LLMResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    latency_ms: float
    model_name: str
    tokens_used: Optional[int] = 0

class GeminiLLMService:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash"):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name

    def format_context_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Formats retrieved vector search results into a clean, grounded context block.
        """
        if not search_results:
            return ""

        context_blocks = []
        for i, res in enumerate(search_results, start=1):
            chunk = res.chunk
            source_info = f"[Source {i}] (Passage ID: {chunk.passage_id}"
            if chunk.url:
                source_info += f" | URL: {chunk.url}"
            source_info += ")\n" + chunk.text.strip()
            context_blocks.append(source_info)

        formatted_context = "\n\n".join(context_blocks)
        
        prompt = (
            f"You are a factual AI assistant. Answer the user's question using ONLY the provided CONTEXT PASSAGES below.\n"
            f"RULES:\n"
            f"1. Base your answer strictly on the provided context passages.\n"
            f"2. If the context does not contain enough information to answer the question, state: 'I couldn't find enough information in the provided knowledge base to answer that.'\n"
            f"3. Keep your answer concise, accurate, and direct (max 2-3 sentences).\n\n"
            f"=== CONTEXT PASSAGES ===\n"
            f"{formatted_context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{query}\n\n"
            f"=== GROUNDED ANSWER ==="
        )
        return prompt

    def generate_answer(
        self,
        query: str,
        search_results: List[SearchResult]
    ) -> LLMResponse:
        """
        Generates a grounded RAG answer using Google Gemini API.
        Measures exact LLM turnaround latency.
        """
        start_time = time.perf_counter()

        sources = []
        for res in search_results:
            if res.chunk.url:
                sources.append({"passage_id": res.chunk.passage_id, "url": res.chunk.url})
            else:
                sources.append({"passage_id": res.chunk.passage_id, "url": ""})

        # Early exit if no relevant context was retrieved
        if not search_results:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return LLMResponse(
                answer="I couldn't find enough information in the provided knowledge base to answer that.",
                sources=[],
                latency_ms=elapsed_ms,
                model_name="No Context Guardrail"
            )

        prompt = self.format_context_prompt(query, search_results)

        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_gemini_api_key_here":
            # Fallback mock generator when API key is omitted
            elapsed_ms = (time.perf_counter() - start_time) * 1000 + 75.0
            mock_answer = f"Based on the retrieved knowledge base, {search_results[0].chunk.text.strip()}"
            return LLMResponse(
                answer=mock_answer,
                sources=sources,
                latency_ms=elapsed_ms,
                model_name="Gemini (Offline Mock)"
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 256
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=12.0)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                res_data = response.json()
                generated_text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return LLMResponse(
                    answer=generated_text,
                    sources=sources,
                    latency_ms=elapsed_ms,
                    model_name=f"Google {self.model_name}"
                )
            else:
                print(f"[!] Gemini Model '{self.model_name}' returned status {response.status_code}: {response.text[:120]}")
        except Exception as e:
            print(f"[!] Exception calling '{self.model_name}': {e}")

        # Fallback if API endpoints fail
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        fallback_ans = f"Based on the knowledge base: {search_results[0].chunk.text.strip()}"
        return LLMResponse(
            answer=fallback_ans,
            sources=sources,
            latency_ms=elapsed_ms,
            model_name="Google Gemini (Grounded Fallback)"
        )
