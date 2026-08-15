"""
FAISS Vector Store & Retrieval Engine
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Provides sub-millisecond similarity search using faiss.IndexFlatIP.
Attaches chunk metadata and enforces minimum similarity score thresholds.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pydantic import BaseModel

from app.core.chunker import Chunk

class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.chunks: List[Chunk] = []

    def _init_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install via: pip install faiss-cpu")

    def build_index(self, embeddings: np.ndarray, chunks: List[Chunk]):
        """
        Builds in-memory FAISS IndexFlatIP (Inner Product / Cosine Similarity).
        """
        faiss = self._init_faiss()
        assert embeddings.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
        
        self.index = faiss.IndexFlatIP(self.dimension)
        # Enforce float32 array type
        embeddings_f32 = embeddings.astype(np.float32)
        self.index.add(embeddings_f32)
        self.chunks = chunks
        print(f"[+] FAISS Index built with {self.index.ntotal} vectors (Dim: {self.dimension})")

    def save_index(self, directory: str):
        """
        Saves index binary and chunk metadata to disk.
        """
        faiss = self._init_faiss()
        os.makedirs(directory, exist_ok=True)
        index_file = os.path.join(directory, "index.faiss")
        chunks_file = os.path.join(directory, "chunks.json")

        if self.index is not None:
            faiss.write_index(self.index, index_file)
            
        chunks_data = [c.model_dump() for c in self.chunks]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
        print(f"[+] Saved FAISS index to {index_file} and chunks to {chunks_file}")

    def load_index(self, directory: str) -> bool:
        """
        Loads index binary and chunk metadata from disk.
        """
        faiss = self._init_faiss()
        index_file = os.path.join(directory, "index.faiss")
        chunks_file = os.path.join(directory, "chunks.json")

        if not (os.path.exists(index_file) and os.path.exists(chunks_file)):
            print(f"[!] FAISS index files not found in {directory}")
            return False

        self.index = faiss.read_index(index_file)
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            self.chunks = [Chunk(**c) for c in chunks_data]
            
        print(f"[+] Loaded FAISS index ({self.index.ntotal} vectors) from {directory}")
        return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        score_threshold: float = 0.30
    ) -> Tuple[List[SearchResult], float]:
        """
        Executes Top-K vector search against FAISS index.
        Returns Tuple[List[SearchResult], search_latency_ms]
        """
        if self.index is None or self.index.ntotal == 0:
            return [], 0.0

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        query_vector_f32 = query_vector.astype(np.float32)

        start_time = time.perf_counter()
        scores, indices = self.index.search(query_vector_f32, min(top_k, self.index.ntotal))
        search_latency_ms = (time.perf_counter() - start_time) * 1000

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            score_val = float(score)
            if score_val >= score_threshold:
                results.append(SearchResult(
                    chunk=self.chunks[idx],
                    score=score_val,
                    rank=rank
                ))

        return results, search_latency_ms
