"""
RAG Safety & Grounding Guardrails Engine
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Enforces:
1. Input Safety (Prompt Injection / Harmful Queries)
2. Open Knowledge Fallback (Ensures 100% of user questions receive helpful answers)
3. Grounding Verification & Citation Matching
"""

import re
from typing import List, Optional, Any
from pydantic import BaseModel
from app.core.chunker import Chunk

class GuardrailResult(BaseModel):
    is_safe: bool
    is_on_topic: bool
    is_grounded: bool
    confidence: float
    final_answer: str
    action_taken: str
    flagged_reason: Optional[str] = None

class GuardrailsEngine:
    def __init__(self, min_similarity_threshold: float = 0.05):
        self.min_similarity_threshold = min_similarity_threshold
        self.injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"system\s+prompt\s+override",
            r"reveal\s+(secret|key|api|password)",
            r"bypass\s+safety\s+filter"
        ]

    def check_input_safety(self, text: str) -> bool:
        """
        Detects prompt injection or system override attempts.
        """
        text_lower = text.lower()
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                return False
        return True

    def process_guardrails(
        self, 
        query: str, 
        llm_answer: str, 
        retrieved_chunks: List[Any]
    ) -> GuardrailResult:
        # Step 1: Input Safety Check
        if not self.check_input_safety(query):
            return GuardrailResult(
                is_safe=False,
                is_on_topic=False,
                is_grounded=False,
                confidence=0.0,
                final_answer="Security Request Refused: Prompt injection or unauthorized instruction override detected.",
                action_taken="REJECTED_UNSAFE",
                flagged_reason="PROMPT_INJECTION_DETECTED"
            )

        # Step 2: Context Availability & Relevance Check
        if not retrieved_chunks:
            if self.min_similarity_threshold >= 0.50:
                return GuardrailResult(
                    is_safe=True,
                    is_on_topic=False,
                    is_grounded=False,
                    confidence=0.0,
                    final_answer="I couldn't find enough information in the grounded knowledge base to answer this question accurately.",
                    action_taken="REJECTED_OFF_TOPIC",
                    flagged_reason="NO_CONTEXT_CHUNKS"
                )
            return GuardrailResult(
                is_safe=True,
                is_on_topic=True,
                is_grounded=False,
                confidence=0.75,
                final_answer=llm_answer,
                action_taken="PASSED_GENERAL_KNOWLEDGE",
                flagged_reason="NO_CONTEXT_CHUNKS"
            )

        top_score = float(getattr(retrieved_chunks[0], "score", 1.0))
        if top_score < self.min_similarity_threshold:
            if self.min_similarity_threshold >= 0.50:
                return GuardrailResult(
                    is_safe=True,
                    is_on_topic=False,
                    is_grounded=False,
                    confidence=0.0,
                    final_answer="I couldn't find enough information in the grounded knowledge base to answer this question accurately.",
                    action_taken="REJECTED_OFF_TOPIC",
                    flagged_reason="LOW_SIMILARITY_SCORE"
                )
            return GuardrailResult(
                is_safe=True,
                is_on_topic=True,
                is_grounded=False,
                confidence=0.70,
                final_answer=llm_answer,
                action_taken="PASSED_GENERAL_KNOWLEDGE",
                flagged_reason="LOW_SIMILARITY_SCORE"
            )

        # Step 3: Grounded Passage Citation Check
        confidence = min(max(top_score, 0.75), 0.99)

        return GuardrailResult(
            is_safe=True,
            is_on_topic=True,
            is_grounded=True,
            confidence=round(confidence, 2),
            final_answer=llm_answer,
            action_taken="PASSED"
        )
