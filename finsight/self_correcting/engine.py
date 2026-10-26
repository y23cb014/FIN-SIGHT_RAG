"""
Self-Correcting Retrieval and Evidence Validation Engine for FinSight-RAG
Evaluates evidence sufficiency; autonomously reformulates queries and expands search horizons.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import re
from finsight.ingestion.chunker import DocumentChunk

@dataclass
class CorrectionAttempt:
    attempt: int
    query_used: str
    evidence_count: int
    sufficient: bool
    reason: str

class SelfCorrectingEngine:
    """Autonomously audits and retries retrieval if initial evidence is sparse or ambiguous."""

    @classmethod
    def evaluate_sufficiency(cls, 
                              chunks: List[Tuple[DocumentChunk, float]], 
                              target_metrics: List[str]) -> Tuple[bool, str]:
        if not chunks:
            return False, "No chunks returned from primary retrieval."

        top_score = chunks[0][1]
        if top_score < 0.01:
            return False, f"Relevance score ({top_score:.4f}) is below confidence threshold."

        # Check if any target metric is mentioned in the retrieved text
        if target_metrics:
            combined_text = " ".join([c.text.lower() for c, _ in chunks[:3]])
            found = any(m.lower() in combined_text for m in target_metrics)
            if not found:
                return False, f"Target metrics {target_metrics} not adequately represented in top chunks."

        return True, "Evidence is sufficient and above confidence threshold."

    @classmethod
    def reformulate_query(cls, query: str, attempt: int, failed_reason: str) -> str:
        """Expands or reformulates query using financial synonyms and canonical accounting terms."""
        q_lower = query.lower()
        replacements = [
            (r'\brevenue\b', 'total net sales revenue operations'),
            (r'\bnet income\b', 'net income net earnings consolidated'),
            (r'\boperating income\b', 'operating income income from operations operating profit'),
            (r'\bmargin\b', 'margin percentage gross operating net'),
            (r'\bdebt\b', 'total debt short-term debt long-term debt liabilities'),
            (r'\bsegment\b', 'business segment reportable segments net sales')
        ]

        reformulated = query
        for pattern, replacement in replacements:
            if re.search(pattern, reformulated, re.IGNORECASE):
                reformulated = re.sub(pattern, replacement, reformulated, flags=re.IGNORECASE)
                break

        if reformulated == query:
            # Append general financial statement terminology
            reformulated = f"{query} consolidated financial statements Form 10-K Item 8"

        return reformulated
