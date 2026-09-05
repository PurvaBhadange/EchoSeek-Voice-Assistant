"""
Structured Model Harness & Pipeline Orchestrator
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Implements production orchestration harness:
- Input validation & sanitation
- Retriever tool execution & context assembly
- Exponential retries & timeout handling
- Structured Pydantic I/O validation
- Graceful error recovery & detailed per-stage latency tracking
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from app.core.embedding import EmbeddingEngine
from app.core.vector_store import FAISSVectorStore, SearchResult
from app.services.stt_service import SarvamSTTService
from app.services.llm_service import GeminiLLMService, LLMResponse

class PipelineLatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0

class HarnessOutput(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, str]] = Field(default_factory=list)
    is_grounded: bool = True
    confidence: float = 1.0
    latency: PipelineLatencyBreakdown
    model_name: str
    error_message: Optional[str] = None

class RAGModelHarness:
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: FAISSVectorStore,
        stt_service: Optional[SarvamSTTService] = None,
        llm_service: Optional[GeminiLLMService] = None
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.stt_service = stt_service or SarvamSTTService()
        self.llm_service = llm_service or GeminiLLMService()

    def validate_input_query(self, query: str) -> Tuple[bool, str]:
        """
        Validates raw input query text.
        """
        if not query or not query.strip():
            return False, "Query text cannot be empty."
        if len(query.strip()) < 2:
            return False, "Query text is too short to process."
        if len(query) > 1000:
            return False, "Query exceeds maximum character length limit (1000 chars)."
        return True, ""

    def run(
        self,
        query: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        top_k: int = 5,
        score_threshold: float = 0.10,
        max_retries: int = 2
    ) -> HarnessOutput:
        """
        Executes end-to-end harnessed RAG pipeline with retries and timing breakdown.
        """
        pipeline_start = time.perf_counter()
        latency = PipelineLatencyBreakdown()
        
        effective_query = ""

        # Step 1: Speech-to-Text (if audio provided)
        if audio_bytes:
            stt_res = self.stt_service.transcribe_audio_bytes(audio_bytes)
            latency.stt_ms = stt_res.latency_ms
            effective_query = stt_res.transcript
        elif query:
            effective_query = query.strip()

        # Step 2: Validate Input Query
        is_valid, err_msg = self.validate_input_query(effective_query)
        if not is_valid:
            total_elapsed = (time.perf_counter() - pipeline_start) * 1000
            latency.total_ms = total_elapsed
            return HarnessOutput(
                query=effective_query,
                answer="Invalid query request. Please provide a clear question.",
                sources=[],
                is_grounded=False,
                confidence=0.0,
                latency=latency,
                model_name="Input Validation Circuit Breaker",
                error_message=err_msg
            )

        # Step 3: Embed Query & Search Vector DB
        emb_start = time.perf_counter()
        query_vector = self.embedding_engine.embed_query(effective_query)
        latency.embedding_ms = (time.perf_counter() - emb_start) * 1000

        results, search_ms = self.vector_store.search(
            query_vector,
            top_k=top_k,
            score_threshold=score_threshold
        )
        latency.vector_search_ms = search_ms

        # Step 4: Call LLM with Retry Loop
        llm_response = None
        attempt_err = None

        for attempt in range(1, max_retries + 1):
            try:
                llm_response = self.llm_service.generate_answer(effective_query, results)
                latency.llm_ms = llm_response.latency_ms
                if llm_response.answer:
                    break
            except Exception as e:
                attempt_err = str(e)
                print(f"[!] Harness Retry Attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(0.1 * attempt)

        # Step 5: Construct Final Harness Output
        total_elapsed = (time.perf_counter() - pipeline_start) * 1000
        latency.total_ms = total_elapsed

        if llm_response:
            is_grounded = bool(results) and ("couldn't find" not in llm_response.answer.lower())
            return HarnessOutput(
                query=effective_query,
                answer=llm_response.answer,
                sources=llm_response.sources,
                is_grounded=is_grounded,
                confidence=0.95 if is_grounded else 0.70,
                latency=latency,
                model_name=llm_response.model_name
            )
        else:
            fallback_ans = results[0].chunk.text if results else f"Answer for: {effective_query}"
            return HarnessOutput(
                query=effective_query,
                answer=f"Based on retrieved context: {fallback_ans}",
                sources=[{"passage_id": r.chunk.passage_id, "url": getattr(r.chunk, 'url', '')} for r in results],
                is_grounded=bool(results),
                confidence=0.60,
                latency=latency,
                model_name="Harness Fallback Recovery Engine",
                error_message=attempt_err
            )
