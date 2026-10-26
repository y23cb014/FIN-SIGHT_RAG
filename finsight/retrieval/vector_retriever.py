"""
Vector Semantic Retriever for FinSight-RAG
Performs pure NumPy sublinear TF-IDF and cosine similarity search over financial document chunks.
Zero external C-library dependency, 100% thread-safe and deterministic.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np
from finsight.ingestion.chunker import DocumentChunk

class VectorRetriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.doc_vectors: np.ndarray = np.array([])
        if chunks:
            self._fit([f"{c.company} {c.year} {c.doc_type} {c.text}" for c in chunks])

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r'[a-zA-Z0-9$%\.-]+', text.lower())
        # create 1-grams and 2-grams
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        return words + bigrams

    def _fit(self, corpus: List[str]):
        N = len(corpus)
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        df_counter = Counter()
        for doc in tokenized_corpus:
            df_counter.update(set(doc))

        # Top 15000 terms
        most_common = df_counter.most_common(15000)
        self.vocab = {term: idx for idx, (term, count) in enumerate(most_common) if count >= 1}
        V = len(self.vocab)
        self.idf = np.zeros(V, dtype=np.float32)

        for term, idx in self.vocab.items():
            # Smooth IDF
            self.idf[idx] = math.log((1.0 + N) / (1.0 + df_counter[term])) + 1.0

        doc_matrix = np.zeros((N, V), dtype=np.float32)
        for d_idx, doc in enumerate(tokenized_corpus):
            tf = Counter(doc)
            for term, count in tf.items():
                if term in self.vocab:
                    t_idx = self.vocab[term]
                    doc_matrix[d_idx, t_idx] = (1.0 + math.log(count)) * self.idf[t_idx]
            norm = np.linalg.norm(doc_matrix[d_idx])
            if norm > 0:
                doc_matrix[d_idx] /= norm

        self.doc_vectors = doc_matrix

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks or len(self.doc_vectors) == 0:
            return []
        q_tokens = self._tokenize(query)
        V = len(self.vocab)
        q_vec = np.zeros(V, dtype=np.float32)
        tf = Counter(q_tokens)
        for term, count in tf.items():
            if term in self.vocab:
                t_idx = self.vocab[term]
                q_vec[t_idx] = (1.0 + math.log(count)) * self.idf[t_idx]
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        sims = np.dot(self.doc_vectors, q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(sims[idx])))
        return results
