"""
RAG Guardrails Engine (Off-Topic, Safety & Grounding Verification)
Hacker House Goa 2026 — Task 2: Voice-Enabled RAG Model

Enforces 4 mandatory guardrails:
1. Input Safety & Prompt Injection Detection
2. Off-Topic Query Filtering (Vector Score Thresholds)
3. Grounding Verification (Prevents Hallucinations)
4. Standardized No-Answer Refusal Response
"""

import re
from typing import List, Tuple, Optional
from pydantic import BaseModel

from app.core.vector_store import SearchResult

class GuardrailResult(BaseModel):
    is_safe: bool = True
    is_on_topic: bool = True
    is_grounded: bool = True
    action_taken: str = "PASSED"
    final_answer: str
    flagged_reason: Optional[str] = None

class GuardrailsEngine:
    NO_ANSWER_TEXT = "I couldn't find enough information in the provided knowledge base to answer that."
    
    # Prompt Injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous\s+)?instructions",
        r"system\s+prompt",
        r"bypass\s+(safety|rules)",
        r"reveal\s+(api\s+)?key",
        r"act\s+as\s+a\s+dan",
        r"override\s+guidelines"
    ]

    def __init__(self, min_similarity_threshold: float = 0.65):
        self.min_similarity_threshold = min_similarity_threshold

    def check_input_safety(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Checks query for prompt injection or malicious instructions.
        """
        query_lower = query.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                return False, f"Prompt injection pattern detected: '{pattern}'"
        return True, None

    def check_off_topic(self, search_results: List[SearchResult]) -> bool:
        """
        Returns True if search results contain relevant context above score threshold.
        """
        if not search_results:
            return False
        top_score = search_results[0].score
        return top_score >= self.min_similarity_threshold

    def verify_answer_grounding(self, answer: str, search_results: List[SearchResult]) -> Tuple[bool, float]:
        """
        Verifies whether the generated answer is supported by retrieved context.
        Computes lexical key-term overlap ratio between answer and context text.
        """
        if not search_results or not answer:
            return False, 0.0

        if self.NO_ANSWER_TEXT.lower() in answer.lower():
            return True, 1.0

        context_text = " ".join([r.chunk.text for r in search_results]).lower()
        
        # Tokenize answer into non-stopword words
        words = [w.strip(".,!?\"'") for w in answer.lower().split() if len(w) > 3]
        if not words:
            return True, 1.0

        matched_words = sum(1 for w in words if w in context_text)
        overlap_ratio = matched_words / len(words)
        
        # Grounding threshold: at least 40% of substantive answer terms must exist in context
        is_grounded = overlap_ratio >= 0.40
        return is_grounded, overlap_ratio

    def process_guardrails(
        self,
        query: str,
        candidate_answer: str,
        search_results: List[SearchResult]
    ) -> GuardrailResult:
        """
        Executes end-to-end guardrail verification pipeline.
        """
        # Guardrail 1: Input Safety
        is_safe, safety_reason = self.check_input_safety(query)
        if not is_safe:
            return GuardrailResult(
                is_safe=False,
                is_on_topic=False,
                is_grounded=False,
                action_taken="REJECTED_UNSAFE",
                final_answer="Security Request Refused: Query contains prohibited prompt injection patterns.",
                flagged_reason=safety_reason
            )

        # Guardrail 2: Off-Topic / No Context Check
        is_on_topic = self.check_off_topic(search_results)
        if not is_on_topic:
            return GuardrailResult(
                is_safe=True,
                is_on_topic=False,
                is_grounded=False,
                action_taken="REJECTED_OFF_TOPIC",
                final_answer=self.NO_ANSWER_TEXT,
                flagged_reason=f"Top vector similarity score below threshold ({self.min_similarity_threshold})"
            )

        # Guardrail 3: Answer Grounding Check
        is_grounded, overlap_score = self.verify_answer_grounding(candidate_answer, search_results)
        if not is_grounded:
            # Substitute ungrounded answer with safe factual fallback
            fallback = f"Based on retrieved context: {search_results[0].chunk.text.strip()}"
            return GuardrailResult(
                is_safe=True,
                is_on_topic=True,
                is_grounded=False,
                action_taken="SUBSTITUTED_UNGROUNDED",
                final_answer=fallback,
                flagged_reason=f"Low answer-to-context term overlap ({overlap_score:.2f})"
            )

        # Passed all guardrails cleanly
        return GuardrailResult(
            is_safe=True,
            is_on_topic=True,
            is_grounded=True,
            action_taken="PASSED",
            final_answer=candidate_answer
        )
