"""
Latency Analytics Engine (P50 / P70 / P100 Metrics)
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Computes percentile latency distributions across individual pipeline stages:
- STT (Speech-to-Text)
- Query Vector Embedding
- FAISS Similarity Search
- LLM Response Generation
- Total Pipeline End-to-End
"""

from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel

from app.core.harness import PipelineLatencyBreakdown

class LatencyMetric(BaseModel):
    p50: float
    p70: float
    p100: float
    avg: float
    min_val: float
    max_val: float

class StagePercentileReport(BaseModel):
    stt: LatencyMetric
    embedding: LatencyMetric
    vector_search: LatencyMetric
    llm: LatencyMetric
    total: LatencyMetric
    total_samples: int

class AnalyticsEngine:
    @staticmethod
    def _calculate_metric(values: List[float]) -> LatencyMetric:
        if not values:
            return LatencyMetric(p50=0.0, p70=0.0, p100=0.0, avg=0.0, min_val=0.0, max_val=0.0)
        
        arr = np.array(values, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        p70 = float(np.percentile(arr, 70))
        p100 = float(np.percentile(arr, 100))
        avg = float(np.mean(arr))
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))

        return LatencyMetric(
            p50=round(p50, 2),
            p70=round(p70, 2),
            p100=round(p100, 2),
            avg=round(avg, 2),
            min_val=round(min_v, 2),
            max_val=round(max_v, 2)
        )

    def generate_report(self, breakdowns: List[PipelineLatencyBreakdown]) -> StagePercentileReport:
        stt_vals = [b.stt_ms for b in breakdowns]
        emb_vals = [b.embedding_ms for b in breakdowns]
        search_vals = [b.vector_search_ms for b in breakdowns]
        llm_vals = [b.llm_ms for b in breakdowns]
        total_vals = [b.total_ms for b in breakdowns]

        return StagePercentileReport(
            stt=self._calculate_metric(stt_vals),
            embedding=self._calculate_metric(emb_vals),
            vector_search=self._calculate_metric(search_vals),
            llm=self._calculate_metric(llm_vals),
            total=self._calculate_metric(total_vals),
            total_samples=len(breakdowns)
        )
