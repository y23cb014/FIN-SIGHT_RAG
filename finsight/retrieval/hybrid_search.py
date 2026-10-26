"""
Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF) and Metadata Filtering
"""

from typing import List, Dict, Any, Tuple, Optional
from finsight.ingestion.chunker import DocumentChunk
from finsight.retrieval.bm25_retriever import BM25Retriever
from finsight.retrieval.vector_retriever import VectorRetriever
from finsight.retrieval.reranker import CrossDocumentReranker

class HybridSearchEngine:
    """Multi-stage hybrid retrieval: Metadata filter -> BM25 + Vector -> RRF Fusion -> Re-ranking."""

    def __init__(self, chunks: List[DocumentChunk]):
        self.all_chunks = chunks
        self.bm25_retriever = BM25Retriever(chunks)
        self.vector_retriever = VectorRetriever(chunks)
        self.reranker = CrossDocumentReranker()

    def search(self, 
               query: str, 
               company: Optional[str] = None,
               target_companies: Optional[List[str]] = None,
               year: Optional[int] = None,
               target_years: Optional[List[int]] = None,
               doc_type: Optional[str] = None,
               top_k: int = 6,
               rrf_k: int = 60) -> List[Tuple[DocumentChunk, float]]:
        
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 3)
        vector_results = self.vector_retriever.search(query, top_k=top_k * 3)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (chunk, _) in enumerate(vector_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        filtered_scored = []
        for c_id, score in rrf_scores.items():
            c = chunk_map[c_id]
            if company and c.company.lower() != company.lower():
                continue
            if target_companies and c.company not in target_companies:
                continue
            if year and c.year != year:
                continue
            if target_years and c.year not in target_years:
                continue
            if doc_type and doc_type.lower() not in c.doc_type.lower():
                continue
            filtered_scored.append((c, score))

        if not filtered_scored and rrf_scores:
            filtered_scored = [(chunk_map[c_id], sc) for c_id, sc in rrf_scores.items()]

        filtered_scored.sort(key=lambda x: x[1], reverse=True)

        effective_comps = target_companies or ([company] if company else None)
        effective_years = target_years or ([year] if year else None)
        final_ranked = self.reranker.rerank(
            filtered_scored,
            target_companies=effective_comps,
            target_years=effective_years,
            top_k=top_k
        )
        return final_ranked
