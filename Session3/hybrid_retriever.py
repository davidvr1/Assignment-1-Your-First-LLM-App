"""
Task 6, experiment 2 - hybrid (dense + BM25) retrieval.

Motivated directly by experiment 1's id-24 regression (assignment_03_exp3_topk10.xlsx):
raising top-K to 10 fixed 3 questions but broke the one exact-identifier ("plan code")
question, because more chunks in context means more near-miss codes for the model to
confuse -- exactly the "embeddings are weak at exact identifiers" trap the assignment
calls out. BM25 is a lexical/exact-match retriever, so it should surface the chunk
containing the literal code string even when its semantic embedding is unremarkable.

HybridRetriever fuses FAISS (dense) and BM25 (lexical) rankings with Reciprocal Rank
Fusion (RRF) -- simple, weight-free, and standard for combining heterogeneous rankers.
It exposes similarity_search(query, k), the same duck-typed interface FAISS exposes, so
it's a drop-in replacement for vectorstore everywhere in rag_pipeline.py / evaluate.py.
"""

from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, chunks: list, vectorstore, dense_k: int = 20, bm25_k: int = 20, rrf_k: int = 60):
        """chunks: the same Document list the FAISS index was built from (needed
        because BM25 has no persistent index of its own -- it's rebuilt in memory
        from the corpus each run, which is cheap: no embeddings involved).
        dense_k/bm25_k: how many candidates each retriever contributes before fusion
        (wider than the final k so RRF has real signal to work with).
        rrf_k: RRF's smoothing constant (60 is the standard default from the
        original RRF paper -- dampens the impact of any single rank-1 hit).
        """
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        tokenized = [c.page_content.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def similarity_search(self, query: str, k: int = 5) -> list:
        dense_hits = self.vectorstore.similarity_search(query, k=self.dense_k)

        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: self.bm25_k]
        bm25_hits = [self.chunks[i] for i in bm25_ranked_idx]

        # Reciprocal Rank Fusion: score = sum over rankers of 1 / (rrf_k + rank).
        # Identify a chunk by (doc_name, chunk_id) since Document objects from the
        # two retrievers are different instances even for the "same" chunk.
        def key(doc):
            return (doc.metadata.get("doc_name"), doc.metadata.get("chunk_id"))

        fused_scores: dict = {}
        chunk_by_key: dict = {}
        for rank, doc in enumerate(dense_hits):
            k_ = key(doc)
            chunk_by_key[k_] = doc
            fused_scores[k_] = fused_scores.get(k_, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, doc in enumerate(bm25_hits):
            k_ = key(doc)
            chunk_by_key[k_] = doc
            fused_scores[k_] = fused_scores.get(k_, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked_keys = sorted(fused_scores, key=lambda k_: fused_scores[k_], reverse=True)
        return [chunk_by_key[k_] for k_ in ranked_keys[:k]]
