"""
27 Research Evaluation Metrics for FinSight-RAG
Covers Retrieval, Generation, Financial-Specific, and System/Cost Metrics.
"""

import math
from typing import List, Dict, Any, Optional

class FinancialMetrics:
    """Calculates retrieval, generation, financial numerical, and cost metrics."""

    # --- RETRIEVAL METRICS ---
    @staticmethod
    def hit_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
        top_k = retrieved_ids[:k]
        return 1.0 if any(gt in top_k for gt in ground_truth_ids) else 0.0

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
        if not ground_truth_ids:
            return 1.0
        top_k = retrieved_ids[:k]
        hits = len(set(top_k).intersection(set(ground_truth_ids)))
        return hits / len(ground_truth_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        hits = len(set(top_k).intersection(set(ground_truth_ids)))
        return hits / len(top_k)

    @staticmethod
    def mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1.0 if doc_id in ground_truth_ids else 0.0
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
        # Ideal DCG
        idcg = sum((2.0 ** 1.0 - 1.0) / math.log2(i + 2) for i in range(min(len(ground_truth_ids), k)))
        return (dcg / idcg) if idcg > 0 else 0.0

    # --- FINANCIAL SPECIFIC METRICS ---
    @staticmethod
    def numerical_accuracy(predicted_val: Optional[float], actual_val: float, tolerance: float = 0.015) -> float:
        if predicted_val is None:
            return 0.0
        if actual_val == 0:
            return 1.0 if abs(predicted_val) < 1e-4 else 0.0
        rel_error = abs(predicted_val - actual_val) / abs(actual_val)
        return 1.0 if rel_error <= tolerance else 0.0

    @staticmethod
    def citation_accuracy(citations: List[Dict[str, Any]], expected_doc: str, expected_page: Optional[int] = None) -> float:
        if not citations:
            return 0.0
        for c in citations:
            src = c.get("source_doc") or c.get("source_file") or ""
            pg = c.get("page")
            if expected_doc.lower() in src.lower():
                if expected_page is None or pg == expected_page:
                    return 1.0
        return 0.0

    @staticmethod
    def calculate_cost(input_tokens: int, output_tokens: int, model: str = "fast") -> float:
        # Gemini 2.5 Flash pricing: ~$0.15 / 1M input, $0.60 / 1M output
        # Gemini 2.5 Pro: ~$1.25 / 1M input, $5.00 / 1M output
        if model == "fast":
            cost = (input_tokens * 0.00000015) + (output_tokens * 0.00000060)
        else:
            cost = (input_tokens * 0.00000125) + (output_tokens * 0.00000500)
        return round(cost, 6)
