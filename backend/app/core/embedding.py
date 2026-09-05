"""
Vector Embedding Engine for Voice-Enabled RAG Model
Hacker House Goa 2026 — Task 2

Uses intfloat/multilingual-e5-small to generate 384-dimensional dense vectors.
Enforces E5 asymmetric retrieval prefixes ('query: ' vs 'passage: ').
"""

import time
from typing import List, Union
import numpy as np

class EmbeddingEngine:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._engine_type = "fastembed"

    def _load_model(self):
        if self._model is None:
            import os
            os.environ["TORCH_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            start = time.perf_counter()
            try:
                print(f"[*] Loading ONNX embedding model via fastembed: {self.model_name}...")
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
                self._engine_type = "fastembed"
                elapsed = (time.perf_counter() - start) * 1000
                print(f"[+] Loaded fastembed ONNX model in {elapsed:.2f} ms")
            except Exception as err:
                print(f"[!] fastembed load failed ({err}). Falling back to SentenceTransformer...")
                import torch
                torch.set_num_threads(1)
                from sentence_transformers import SentenceTransformer
                fallback_name = "intfloat/multilingual-e5-small"
                self._model = SentenceTransformer(fallback_name)
                self._engine_type = "sentence_transformers"
                elapsed = (time.perf_counter() - start) * 1000
                print(f"[+] Loaded SentenceTransformer model in {elapsed:.2f} ms")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string. Returns normalized 1D numpy array of shape (384,)
        """
        self._load_model()
        formatted_query = f"query: {query.strip()}"
        if self._engine_type == "fastembed":
            vecs = list(self._model.embed([formatted_query]))
            vec = vecs[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32)
        else:
            vec = self._model.encode(
                formatted_query,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return vec.astype(np.float32)

    def embed_passages(self, passages: List[str]) -> np.ndarray:
        """
        Embeds a list of passage strings. Returns normalized 2D numpy array of shape (N, 384)
        """
        self._load_model()
        formatted_passages = [f"passage: {p.strip()}" for p in passages]
        if self._engine_type == "fastembed":
            vecs = np.array(list(self._model.embed(formatted_passages, batch_size=32)))
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
            return vecs.astype(np.float32)
        else:
            vecs = self._model.encode(
                formatted_passages,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return vecs.astype(np.float32)

    @staticmethod
    def compute_cosine_similarity(query_vector: np.ndarray, passage_vectors: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity scores between 1D query vector and 2D passage vectors matrix.
        Returns 1D array of scores between 0.0 and 1.0.
        """
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        
        # Since vectors are L2-normalized, cosine similarity is matrix multiplication
        scores = np.dot(passage_vectors, query_vector.T).squeeze(axis=1)
        return scores
