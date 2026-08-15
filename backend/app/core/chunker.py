"""
Chunking Engine for Voice-Enabled RAG Model
Hacker House Goa 2026 — Task 2

Implements non-naive, thoughtful chunking strategies:
1. NaiveFixedChunker (Baseline for benchmarking)
2. SentenceRecursiveChunker (Sentence-boundary aware with overlap)
3. MetadataAwarePassageChunker (Production choice for MSMARCO-XI with full source metadata)
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    chunk_id: str
    text: str
    passage_id: str
    url: Optional[str] = ""
    language: Optional[str] = "en"
    char_length: int
    word_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseChunker:
    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        raise NotImplementedError

class NaiveFixedChunker(BaseChunker):
    """
    Baseline naive chunker: rigid character slicing every chunk_size characters.
    Intentionally cuts words/sentences mid-way to serve as a negative baseline.
    """
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size

    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        text = doc.get("passage", "")
        passage_id = doc.get("passage_id", "unknown")
        chunks = []
        
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i : i + self.chunk_size]
            chunks.append(Chunk(
                chunk_id=f"{passage_id}_naive_{i}",
                text=chunk_text,
                passage_id=passage_id,
                url=doc.get("url", ""),
                language=doc.get("language", "en"),
                char_length=len(chunk_text),
                word_count=len(chunk_text.split()),
                metadata={"strategy": "naive_fixed", "start_idx": i}
            ))
        return chunks

class SentenceRecursiveChunker(BaseChunker):
    """
    Sentence-boundary aware chunker with sliding character overlap.
    Preserves sentence integrity across boundaries.
    """
    def __init__(self, target_chunk_size: int = 300, overlap: int = 50):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        text = doc.get("passage", "")
        passage_id = doc.get("passage_id", "unknown")
        
        # Split on sentence end periods
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
        if not sentences:
            sentences = [text]
            
        chunks = []
        current_chunk_text = ""
        chunk_idx = 0
        
        for sentence in sentences:
            if len(current_chunk_text) + len(sentence) <= self.target_chunk_size:
                current_chunk_text = (current_chunk_text + " " + sentence).strip()
            else:
                if current_chunk_text:
                    chunks.append(Chunk(
                        chunk_id=f"{passage_id}_rec_{chunk_idx}",
                        text=current_chunk_text,
                        passage_id=passage_id,
                        url=doc.get("url", ""),
                        language=doc.get("language", "en"),
                        char_length=len(current_chunk_text),
                        word_count=len(current_chunk_text.split()),
                        metadata={"strategy": "sentence_recursive", "chunk_idx": chunk_idx}
                    ))
                    chunk_idx += 1
                # Start next chunk with overlap from end of previous
                overlap_text = current_chunk_text[-self.overlap:] if len(current_chunk_text) > self.overlap else ""
                current_chunk_text = (overlap_text + " " + sentence).strip()
                
        if current_chunk_text:
            chunks.append(Chunk(
                chunk_id=f"{passage_id}_rec_{chunk_idx}",
                text=current_chunk_text,
                passage_id=passage_id,
                url=doc.get("url", ""),
                language=doc.get("language", "en"),
                char_length=len(current_chunk_text),
                word_count=len(current_chunk_text.split()),
                metadata={"strategy": "sentence_recursive", "chunk_idx": chunk_idx}
            ))
            
        return chunks

class MetadataAwarePassageChunker(BaseChunker):
    """
    Production choice for MSMARCO-XI dataset:
    Keeps dataset passages as atomic, metadata-rich chunks while preserving
    clean sentence boundaries and full source URL attribution.
    """
    def chunk_document(self, doc: Dict[str, Any]) -> List[Chunk]:
        text = doc.get("passage", "").strip()
        passage_id = str(doc.get("passage_id", "unknown"))
        url = doc.get("url", "")
        language = doc.get("language", "en")
        query_id = doc.get("query_id", "")
        
        # Ensure sentence boundaries are clean
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        cleaned_text = ". ".join(sentences) + "." if sentences else text
        
        chunk = Chunk(
            chunk_id=f"chunk_{passage_id}",
            text=cleaned_text,
            passage_id=passage_id,
            url=url,
            language=language,
            char_length=len(cleaned_text),
            word_count=len(cleaned_text.split()),
            metadata={
                "strategy": "metadata_aware_passage",
                "query_id": query_id,
                "sentence_count": len(sentences),
                "source_url": url,
                "has_grounding_url": bool(url)
            }
        )
        return [chunk]
