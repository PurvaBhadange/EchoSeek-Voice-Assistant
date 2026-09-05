import os
import time
import json
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from pydantic import BaseModel

from app.core.chunker import Chunk, SentenceRecursiveChunker, MetadataAwarePassageChunker
from app.core.doc_parser import extract_text_from_file_bytes
from app.core.embedding import EmbeddingEngine
from app.core.vector_store import FAISSVectorStore
from app.core.harness import RAGModelHarness, HarnessOutput
from app.core.guardrails import GuardrailsEngine
from app.services.stt_service import SarvamSTTService
from app.services.llm_service import GeminiLLMService

router = APIRouter(prefix="/api/v1", tags=["RAG Pipeline"])

# In-memory query session history store
QUERY_HISTORY: List[Dict[str, Any]] = []

class TextQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    score_threshold: Optional[float] = 0.10

class VectorSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    score_threshold: Optional[float] = 0.10

class IngestTextRequest(BaseModel):
    title: str
    text: str
    source_url: Optional[str] = ""

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
            embeddings_matrix = np.load(embeddings_file)
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = [Chunk(**c) for c in json.load(f)]
            self.vector_store.build_index(embeddings_matrix, chunks)
            print(f"[+] Loaded FAISS index ({len(chunks)} vectors) from {data_dir}")

        self.stt_service = SarvamSTTService()
        self.llm_service = GeminiLLMService()
        self.guardrails = GuardrailsEngine(min_similarity_threshold=0.10)
        self.harness = RAGModelHarness(
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
            stt_service=self.stt_service,
            llm_service=self.llm_service
        )

pipeline_services = PipelineServicesContainer()

def record_history(entry: Dict[str, Any]):
    QUERY_HISTORY.insert(0, entry)
    # Keep last 50 queries
    if len(QUERY_HISTORY) > 50:
        QUERY_HISTORY.pop()

@router.post("/query")
async def process_text_query(req: TextQueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    harness_output = pipeline_services.harness.run(
        query=req.query,
        top_k=req.top_k or 5,
        score_threshold=req.score_threshold if req.score_threshold is not None else 0.10
    )

    q_vec = pipeline_services.embedding_engine.embed_query(req.query)
    results, search_ms = pipeline_services.vector_store.search(
        q_vec, top_k=req.top_k or 5, score_threshold=req.score_threshold if req.score_threshold is not None else 0.10
    )

    guardrail_result = pipeline_services.guardrails.process_guardrails(
        query=req.query,
        llm_answer=harness_output.answer,
        retrieved_chunks=results
    )

    res_payload = {
        "id": f"query_{int(time.time()*1000)}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": req.query,
        "type": "text",
        "answer": guardrail_result.final_answer,
        "is_grounded": guardrail_result.is_grounded,
        "guardrail_action": guardrail_result.action_taken,
        "confidence": guardrail_result.confidence,
        "sources": [s.model_dump() if hasattr(s, "model_dump") else s for s in harness_output.sources],
        "latency": harness_output.latency.model_dump()
    }

    record_history(res_payload)
    return res_payload

@router.post("/voice-query")
async def process_voice_query(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    top_k: int = Form(2),
    score_threshold: float = Form(0.30)
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded")

    stt_res = pipeline_services.stt_service.transcribe_audio(audio_bytes, fallback_transcript=prompt)
    transcript = stt_res.transcript.strip()

    if not transcript:
        transcript = prompt or "Please provide a valid question query."

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

    res_payload = {
        "id": f"voice_{int(time.time()*1000)}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": transcript,
        "transcript": transcript,
        "type": "voice",
        "answer": guardrail_result.final_answer,
        "is_grounded": guardrail_result.is_grounded,
        "guardrail_action": guardrail_result.action_taken,
        "confidence": guardrail_result.confidence,
        "sources": [s.model_dump() if hasattr(s, "model_dump") else s for s in harness_output.sources],
        "latency": harness_output.latency.model_dump()
    }

    record_history(res_payload)
    return res_payload

@router.post("/search")
async def explore_vector_search(req: VectorSearchRequest):
    """
    Dedicated endpoint for Vector Explorer view to inspect raw FAISS hits & similarity scores.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    q_vec = pipeline_services.embedding_engine.embed_query(req.query)
    results, search_ms = pipeline_services.vector_store.search(
        q_vec,
        top_k=req.top_k or 5,
        score_threshold=req.score_threshold or 0.10
    )

    hits = []
    for r in results:
        hits.append({
            "rank": r.rank,
            "score": round(r.score, 4),
            "passage_id": r.chunk.passage_id,
            "text": r.chunk.text,
            "url": r.chunk.url or "",
            "char_length": r.chunk.char_length,
            "word_count": r.chunk.word_count
        })

    return {
        "query": req.query,
        "total_hits": len(hits),
        "search_latency_ms": round(search_ms, 3),
        "hits": hits
    }

@router.get("/history")
async def get_query_history():
    """
    Returns query history logs for History view.
    """
    return {
        "total_queries": len(QUERY_HISTORY),
        "history": QUERY_HISTORY
    }

@router.delete("/history")
async def clear_query_history():
    QUERY_HISTORY.clear()
    return {"status": "success", "message": "Query history cleared"}

@router.get("/datasets")
async def get_datasets_info():
    """
    Returns dataset stats & passage vector count.
    """
    chunks = pipeline_services.vector_store.chunks
    passage_count = len(chunks)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    
    unique_sources = set(c.url for c in chunks if c.url)
    
    samples = []
    for c in chunks[:10]:
        samples.append({
            "passage_id": c.passage_id,
            "text_snippet": c.text[:120] + "..." if len(c.text) > 120 else c.text,
            "url": c.url,
            "word_count": c.word_count
        })

    return {
        "dataset_name": "AI4Bharat MSMARCO-XI Grounded Dataset",
        "vector_store_type": "FAISS IndexFlatIP (Cosine Similarity)",
        "dimension": 384,
        "embedding_model": "intfloat/multilingual-e5-small",
        "total_passages": passage_count,
        "total_sources": len(unique_sources),
        "index_path": data_dir,
        "sample_passages": samples
    }

@router.post("/ingest")
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    source_url: Optional[str] = Form("")
):
    """
    Ingests text or uploaded document (PDF, Word, CSV, Excel, JSON, HTML, Markdown, TXT),
    chunks, embeds via e5-small, adds to active FAISS index, and persists update to disk.
    """
    content_text = ""
    doc_title = title or "User Uploaded Document"

    if file:
        doc_title = title or file.filename
        raw_bytes = await file.read()
        try:
            content_text = extract_text_from_file_bytes(file.filename, raw_bytes)
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"Error parsing uploaded file '{file.filename}': {err}")
    elif text:
        content_text = text
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or text content to ingest.")

    if not content_text or not content_text.strip():
        raise HTTPException(status_code=400, detail=f"Could not extract readable text content from '{doc_title}'.")

    # Create chunks using SentenceRecursiveChunker
    chunker = SentenceRecursiveChunker(target_chunk_size=300, overlap=50)
    doc_obj = {
        "passage": content_text,
        "passage_id": f"usr_{int(time.time())}",
        "url": source_url or "",
        "language": "en"
    }
    new_chunks = chunker.chunk_document(doc_obj)

    if not new_chunks:
        raise HTTPException(status_code=400, detail="Could not create chunks from provided content.")

    # Generate embeddings for new chunks
    passages_text = [c.text for c in new_chunks]
    new_embeddings = pipeline_services.embedding_engine.embed_passages(passages_text)

    # Add to active vector store
    all_chunks = list(pipeline_services.vector_store.chunks) + new_chunks

    if pipeline_services.vector_store.index is None:
        pipeline_services.vector_store.build_index(new_embeddings, new_chunks)
    else:
        # Combine existing matrix with new embeddings
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
        embeddings_file = os.path.join(data_dir, "embeddings/embeddings.npy")
        if os.path.exists(embeddings_file):
            old_matrix = np.load(embeddings_file)
            combined_matrix = np.vstack([old_matrix, new_embeddings])
        else:
            combined_matrix = new_embeddings

        pipeline_services.vector_store.build_index(combined_matrix, all_chunks)

        # Save to disk
        embeddings_dir = os.path.join(data_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        np.save(os.path.join(embeddings_dir, "embeddings.npy"), combined_matrix)
        chunks_json_path = os.path.join(embeddings_dir, "chunks.json")
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in all_chunks], f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "message": f"Successfully ingested '{doc_title}'",
        "chunks_added": len(new_chunks),
        "total_index_passages": len(all_chunks)
    }
