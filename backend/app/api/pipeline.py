"""
FastAPI Pipeline Routes for Voice & Text Queries
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Exposes unified endpoints:
- POST /api/v1/query (Text Query)
- POST /api/v1/voice-query (Voice Audio File Upload)
"""

import os
import json
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel

from app.config import settings
from app.core.chunker import Chunk
from app.core.embedding import EmbeddingEngine
from app.core.vector_store import FAISSVectorStore
from app.core.harness import RAGModelHarness, HarnessOutput, PipelineLatencyBreakdown
from app.core.guardrails import GuardrailsEngine, GuardrailResult
from app.services.stt_service import SarvamSTTService
from app.services.llm_service import GeminiLLMService

router = APIRouter(prefix=settings.API_V1_STR, tags=["RAG Pipeline"])

class PipelineServices:
    def __init__(self):
        self.embedding_engine = EmbeddingEngine(settings.EMBEDDING_MODEL_NAME)
        self.vector_store = FAISSVectorStore(dimension=384)
        self.stt_service = SarvamSTTService()
        self.llm_service = GeminiLLMService()
        self.guardrails = GuardrailsEngine(min_similarity_threshold=0.65)
        self.harness = None
        self._is_initialized = False

    def initialize(self):
        if self._is_initialized:
            return

        # Correct path to root data directory
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
        embeddings_file = os.path.join(data_dir, "embeddings/embeddings.npy")
        chunks_file = os.path.join(data_dir, "embeddings/chunks.json")
        faiss_dir = os.path.join(data_dir, "faiss_index")

        if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
            self.vector_store.load_index(faiss_dir)
        elif os.path.exists(embeddings_file) and os.path.exists(chunks_file):
            mat = np.load(embeddings_file)
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = [Chunk(**c) for c in json.load(f)]
            self.vector_store.build_index(mat, chunks)
            self.vector_store.save_index(faiss_dir)
        else:
            print(f"[!] Warning: Vector store files not found at {data_dir}. Run scripts/generate_embeddings.py.")

        self.harness = RAGModelHarness(
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
            stt_service=self.stt_service,
            llm_service=self.llm_service
        )
        self._is_initialized = True

pipeline_services = PipelineServices()

class APITextQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 2
    score_threshold: Optional[float] = 0.70

class APIPipelineResponse(BaseModel):
    success: bool
    query: str
    transcript: Optional[str] = None
    answer: str
    sources: List[Dict[str, str]]
    is_grounded: bool
    confidence: float
    guardrail_action: str
    latency: PipelineLatencyBreakdown
    model_name: str
    error: Optional[str] = None

@router.post("/query", response_model=APIPipelineResponse)
async def process_text_query(req: APITextQueryRequest):
    """
    Processes a text search query through the full RAG harness & guardrail engine.
    """
    pipeline_services.initialize()
    
    harness_out = pipeline_services.harness.run(
        query=req.query,
        top_k=req.top_k,
        score_threshold=req.score_threshold
    )
    
    q_vec = pipeline_services.embedding_engine.embed_query(harness_out.query)
    results, _ = pipeline_services.vector_store.search(q_vec, top_k=req.top_k, score_threshold=req.score_threshold)

    guardrail_res = pipeline_services.guardrails.process_guardrails(
        query=harness_out.query,
        candidate_answer=harness_out.answer,
        search_results=results
    )

    return APIPipelineResponse(
        success=guardrail_res.is_safe,
        query=harness_out.query,
        transcript=None,
        answer=guardrail_res.final_answer,
        sources=harness_out.sources if guardrail_res.is_on_topic else [],
        is_grounded=guardrail_res.is_grounded,
        confidence=harness_out.confidence if guardrail_res.is_grounded else 0.50,
        guardrail_action=guardrail_res.action_taken,
        latency=harness_out.latency,
        model_name=harness_out.model_name,
        error=harness_out.error_message or guardrail_res.flagged_reason
    )

@router.post("/voice-query", response_model=APIPipelineResponse)
async def process_voice_query(
    file: UploadFile = File(...),
    top_k: int = Form(2),
    score_threshold: float = Form(0.70)
):
    """
    Processes an uploaded audio file (WAV/WebM) through STT -> FAISS -> Gemini RAG.
    """
    pipeline_services.initialize()
    audio_bytes = await file.read()

    harness_out = pipeline_services.harness.run(
        audio_bytes=audio_bytes,
        top_k=top_k,
        score_threshold=score_threshold
    )

    q_vec = pipeline_services.embedding_engine.embed_query(harness_out.query)
    results, _ = pipeline_services.vector_store.search(q_vec, top_k=top_k, score_threshold=score_threshold)

    guardrail_res = pipeline_services.guardrails.process_guardrails(
        query=harness_out.query,
        candidate_answer=harness_out.answer,
        search_results=results
    )

    return APIPipelineResponse(
        success=guardrail_res.is_safe,
        query=harness_out.query,
        transcript=harness_out.query,
        answer=guardrail_res.final_answer,
        sources=harness_out.sources if guardrail_res.is_on_topic else [],
        is_grounded=guardrail_res.is_grounded,
        confidence=harness_out.confidence if guardrail_res.is_grounded else 0.50,
        guardrail_action=guardrail_res.action_taken,
        latency=harness_out.latency,
        model_name=harness_out.model_name,
        error=harness_out.error_message or guardrail_res.flagged_reason
    )
