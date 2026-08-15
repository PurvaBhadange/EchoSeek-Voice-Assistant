import os
import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.core.chunker import Chunk
from app.core.embedding import EmbeddingEngine
from app.core.vector_store import FAISSVectorStore
from app.core.harness import RAGModelHarness, HarnessOutput
from app.core.guardrails import GuardrailsEngine
from app.services.stt_service import SarvamSTTService
from app.services.llm_service import GeminiLLMService

router = APIRouter(prefix="/api/v1", tags=["RAG Pipeline"])

class TextQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 2
    score_threshold: Optional[float] = 0.30

class PipelineServicesContainer:
    def __init__(self):
        self.embedding_engine = None
        self.vector_store = None
        self.stt_service = None
        self.llm_service = None
        self.harness = None
        self.guardrails = None

    def initialize(self):
        print("[*] Initializing Pipeline Services Container...")
        self.embedding_engine = EmbeddingEngine("intfloat/multilingual-e5-small")
        self.vector_store = FAISSVectorStore(dimension=384)
        
        # Resolve path to root data directory
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
        embeddings_file = os.path.join(data_dir, "embeddings/embeddings.npy")
        chunks_file = os.path.join(data_dir, "embeddings/chunks.json")

        if os.path.exists(embeddings_file) and os.path.exists(chunks_file):
            import json
            import numpy as np
            embeddings_matrix = np.load(embeddings_file)
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = [Chunk(**c) for c in json.load(f)]
            self.vector_store.build_index(embeddings_matrix, chunks)
            print(f"[+] Loaded FAISS index ({len(chunks)} vectors) from {data_dir}")

        self.stt_service = SarvamSTTService()
        self.llm_service = GeminiLLMService()
        self.guardrails = GuardrailsEngine(min_similarity_threshold=0.30)
        self.harness = RAGModelHarness(
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
            stt_service=self.stt_service,
            llm_service=self.llm_service
        )

pipeline_services = PipelineServicesContainer()

@router.on_event("startup")
async def startup_event():
    pipeline_services.initialize()

@router.post("/query")
async def process_text_query(req: TextQueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    harness_output = pipeline_services.harness.run(
        query=req.query,
        top_k=req.top_k or 2,
        score_threshold=req.score_threshold or 0.30
    )

    q_vec = pipeline_services.embedding_engine.embed_query(req.query)
    results, search_ms = pipeline_services.vector_store.search(
        q_vec, top_k=req.top_k or 2, score_threshold=req.score_threshold or 0.30
    )

    guardrail_result = pipeline_services.guardrails.process_guardrails(
        query=req.query,
        llm_answer=harness_output.answer,
        retrieved_chunks=results
    )

    return {
        "query": req.query,
        "answer": guardrail_result.final_answer,
        "is_grounded": guardrail_result.is_grounded,
        "guardrail_action": guardrail_result.action_taken,
        "confidence": guardrail_result.confidence,
        "sources": [s.model_dump() for s in harness_output.sources],
        "latency": harness_output.latency.model_dump()
    }

@router.post("/voice-query")
async def process_voice_query(
    file: UploadFile = File(...),
    top_k: int = Form(2),
    score_threshold: float = Form(0.30)
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded")

    stt_res = pipeline_services.stt_service.transcribe_audio(audio_bytes)
    transcript = stt_res.transcript.strip()

    if not transcript:
        transcript = "What is Retrieval Augmented Generation?"

    harness_output = pipeline_services.harness.run(
        query=transcript,
        top_k=top_k,
        score_threshold=score_threshold
    )
    harness_output.latency.stt_ms = stt_res.latency_ms

    q_vec = pipeline_services.embedding_engine.embed_query(transcript)
    results, search_ms = pipeline_services.vector_store.search(
        q_vec, top_k=top_k, score_threshold=score_threshold
    )

    guardrail_result = pipeline_services.guardrails.process_guardrails(
        query=transcript,
        llm_answer=harness_output.answer,
        retrieved_chunks=results
    )

    return {
        "query": transcript,
        "transcript": transcript,
        "answer": guardrail_result.final_answer,
        "is_grounded": guardrail_result.is_grounded,
        "guardrail_action": guardrail_result.action_taken,
        "confidence": guardrail_result.confidence,
        "sources": [s.model_dump() for s in harness_output.sources],
        "latency": harness_output.latency.model_dump()
    }
