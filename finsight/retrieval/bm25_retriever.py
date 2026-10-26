"""
BM25 Lexical Retriever for FinSight-RAG
Performs exact and term-frequency based keyword search over financial document chunks.
"""

import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from finsight.ingestion.chunker import DocumentChunk

class BM25Retriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'[a-zA-Z0-9$%\.-]+', text.lower())
        return tokens

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        scored_chunks = list(zip(self.chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
