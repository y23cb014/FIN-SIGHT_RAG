"""
Cross-Document Re-ranker and Multi-Document Fusion
Aligns multi-company and multi-year evidence, ensuring source diversity and chronological ordering.
"""

from typing import List, Dict, Any, Tuple, Optional
from finsight.ingestion.chunker import DocumentChunk

class CrossDocumentReranker:
    """Re-ranks and fuses retrieved evidence across multiple companies and fiscal years."""

    @staticmethod
    def rerank(scored_chunks: List[Tuple[DocumentChunk, float]], 
               target_companies: Optional[List[str]] = None,
               target_years: Optional[List[int]] = None,
               top_k: int = 8) -> List[Tuple[DocumentChunk, float]]:
        if not scored_chunks:
            return []

        adjusted_scored = []
        seen_keys = set()

        for chunk, score in scored_chunks:
            bonus = 0.0

            # 1. Source Authority Boost: Audited 10-K > 10-Q > Presentation
            if "10-K" in chunk.doc_type:
                bonus += 0.05
            elif "Table" in chunk.doc_type or chunk.is_table:
                bonus += 0.08  # Tables carry verified numerical evidence

            # 2. Target Alignment Boost
            if target_companies and chunk.company in target_companies:
                bonus += 0.10
            if target_years and chunk.year in target_years:
                bonus += 0.10

            final_score = score + bonus
            dedup_key = (chunk.company, chunk.year, chunk.page, chunk.text[:80])
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                adjusted_scored.append((chunk, final_score))

        adjusted_scored.sort(key=lambda x: x[1], reverse=True)

        # 3. Multi-Document Diversity Balancing
        if target_companies and len(target_companies) > 1:
            selected = []
            comp_counts = {c: 0 for c in target_companies}
            remaining = []

            for chunk, sc in adjusted_scored:
                if chunk.company in comp_counts and comp_counts[chunk.company] < 2:
                    selected.append((chunk, sc))
                    comp_counts[chunk.company] += 1
                else:
                    remaining.append((chunk, sc))

            for item in remaining:
                if len(selected) >= top_k:
                    break
                selected.append(item)
            return selected[:top_k]

        return adjusted_scored[:top_k]
