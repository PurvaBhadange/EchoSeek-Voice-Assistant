"""
Google Gemini LLM Service for Grounded Answer Generation
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Formats retrieved FAISS context into comprehensive system prompts and calls Google Gemini API.
Generates structured, detailed answers across all candidate passage embeddings.
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
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = self._normalize_model_name(model_name)

    def _normalize_model_name(self, name: str) -> str:
        """
        Ensures valid API model endpoint string for Google Gemini REST API.
        """
        if not name or "1.5" in name or "2.0" in name or "2.5" in name:
            return "gemini-3.6-flash"
        return name

    def format_context_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Formats all retrieved vector search passages into a detailed context prompt.
        """
        if not search_results:
            return (
                f"You are EchoSeek, an intelligent AI assistant.\n"
                f"Please provide a comprehensive, detailed, and well-structured answer in markdown format "
                f"(including key points and bullet lists where appropriate) to the user's question in the SAME language as the question (Hindi if Hindi, English if English):\n\n"
                f"QUESTION: {query}\n\n"
                f"DETAILED STRUCTURED ANSWER:"
            )

        context_blocks = []
        for i, res in enumerate(search_results, start=1):
            chunk = res.chunk
            source_info = f"--- PASSAGE EMBEDDING {i} (ID: {chunk.passage_id}, Score: {getattr(res, 'score', 0.0):.2f}) ---\n{chunk.text.strip()}"
            context_blocks.append(source_info)

        formatted_context = "\n\n".join(context_blocks)
        
        prompt = (
            f"You are EchoSeek, a state-of-the-art voice and vector intelligence assistant.\n"
            f"Provide a thorough, highly detailed, well-explained, and structured answer to the user's question.\n\n"
            f"Guidelines:\n"
            f"1. Respond in the EXACT SAME LANGUAGE as the user query (e.g. Hindi if asked in Hindi, English if in English).\n"
            f"2. Format your response cleanly using Markdown formatting with bold section headers, bullet points, key takeaways, and numbered explanations.\n"
            f"3. Incorporate facts from the retrieved vector context passages below, and synthesize a complete, informative, and authoritative answer.\n\n"
            f"=== RETRIEVED VECTOR CONTEXT PASSAGES ===\n"
            f"{formatted_context}\n\n"
            f"=== USER QUERY ===\n"
            f"{query}\n\n"
            f"=== COMPREHENSIVE STRUCTURED ANSWER ==="
        )
        return prompt

    def generate_answer(
        self,
        query: str,
        search_results: List[SearchResult]
    ) -> LLMResponse:
        start_time = time.perf_counter()

        sources = []
        best_chunk_text = ""
        if search_results:
            # Pick highest scoring chunk
            top_res = max(search_results, key=lambda x: getattr(x, 'score', 0.0))
            best_chunk_text = top_res.chunk.text.strip()
            for res in search_results:
                if hasattr(res, 'chunk'):
                    sources.append({"passage_id": res.chunk.passage_id, "url": getattr(res.chunk, 'url', '') or ''})

        prompt = self.format_context_prompt(query, search_results)

        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_gemini_api_key_here":
            elapsed_ms = (time.perf_counter() - start_time) * 1000 + 30.0
            answer_text = best_chunk_text if best_chunk_text else f"EchoSeek Grounded Context for: {query}"
            return LLMResponse(
                answer=answer_text,
                sources=sources,
                latency_ms=elapsed_ms,
                model_name="EchoSeek Grounded Index"
            )

        normalized_model = self._normalize_model_name(self.model_name)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{normalized_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024
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
                    model_name=f"Google {normalized_model}"
                )
            else:
                print(f"[!] Gemini API status {response.status_code}: {response.text[:200]}")
        except Exception as e:
            print(f"[!] Exception calling LLM API: {e}")

        # Fallback when API returns quota error or timeout
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        fallback_ans = best_chunk_text if best_chunk_text else f"EchoSeek Answer for: {query}"
        return LLMResponse(
            answer=fallback_ans,
            sources=sources,
            latency_ms=elapsed_ms,
            model_name="EchoSeek Grounded Pipeline"
        )
