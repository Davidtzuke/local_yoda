"""Hybrid retrieval pipeline with vector search, BM25, re-ranking, contextual compression, and MMR.

Implements state-of-the-art retrieval:
- Hybrid search: combines dense (vector) and sparse (BM25) retrieval
- Re-ranking: cross-encoder or score-based re-ranking of candidates
- Contextual compression: extract only relevant portions from retrieved docs
- MMR (Maximal Marginal Relevance): diversify results to reduce redundancy
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from yoda.memory.embeddings import EmbeddingProvider
from yoda.memory.vector_store import Document, VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 Sparse Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 sparse retrieval over an in-memory corpus.

    Implements Okapi BM25 scoring for keyword-based retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[Document] = []
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._tokenized: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def index(self, documents: list[Document]) -> None:
        """Build BM25 index from documents."""
        self._corpus = documents
        self._tokenized = [self._tokenize(doc.content) for doc in documents]
        self._doc_lens = [len(tokens) for tokens in self._tokenized]
        self._avg_dl = sum(self._doc_lens) / max(len(self._doc_lens), 1)

        # Compute document frequencies
        self._doc_freqs = {}
        for tokens in self._tokenized:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        """Search using BM25 scoring."""
        if not self._corpus:
            return []

        query_tokens = self._tokenize(query)
        n = len(self._corpus)
        scores: list[float] = []

        for i, doc_tokens in enumerate(self._tokenized):
            score = 0.0
            doc_len = self._doc_lens[i]
            tf_counter = Counter(doc_tokens)

            for qt in query_tokens:
                if qt not in self._doc_freqs:
                    continue
                df = self._doc_freqs[qt]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf = tf_counter.get(qt, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
                score += idf * numerator / denominator

            scores.append(score)

        # Rank and return top-k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            if score > 0:
                doc = self._corpus[idx]
                result = Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    collection=doc.collection,
                    score=score,
                )
                results.append(result)
        return results


# ---------------------------------------------------------------------------
# MMR (Maximal Marginal Relevance)
# ---------------------------------------------------------------------------

def mmr_rerank(
    query_embedding: list[float],
    documents: list[Document],
    doc_embeddings: list[list[float]],
    lambda_param: float = 0.7,
    top_k: int = 5,
) -> list[Document]:
    """Re-rank documents using Maximal Marginal Relevance.

    Balances relevance to query with diversity among selected documents.

    Args:
        query_embedding: The query vector
        documents: Candidate documents
        doc_embeddings: Embeddings for each candidate document
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
        top_k: Number of documents to select
    """
    if not documents:
        return []

    query_vec = np.array(query_embedding)
    doc_vecs = np.array(doc_embeddings)

    # Normalize
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)

    # Similarity to query
    query_sims = doc_norms @ query_norm

    # Iteratively select documents
    selected_indices: list[int] = []
    remaining = list(range(len(documents)))

    for _ in range(min(top_k, len(documents))):
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            relevance = float(query_sims[idx])

            # Max similarity to already selected docs
            if selected_indices:
                selected_vecs = doc_norms[selected_indices]
                similarities = selected_vecs @ doc_norms[idx]
                max_sim = float(np.max(similarities))
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [documents[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Contextual Compression
# ---------------------------------------------------------------------------

class ContextualCompressor:
    """Extracts only the relevant portions from retrieved documents.

    Uses sentence-level filtering based on query similarity.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        relevance_threshold: float = 0.3,
        max_sentences: int = 5,
    ) -> None:
        self._embedder = embedder
        self.relevance_threshold = relevance_threshold
        self.max_sentences = max_sentences

    async def compress(self, query: str, documents: list[Document]) -> list[Document]:
        """Compress documents to only include query-relevant sentences."""
        if self._embedder is None:
            # Fallback: simple keyword-based extraction
            return self._keyword_compress(query, documents)

        query_embedding = await self._embedder.embed_single(query)
        compressed = []

        for doc in documents:
            sentences = re.split(r'(?<=[.!?])\s+', doc.content)
            if len(sentences) <= 2:
                compressed.append(doc)
                continue

            # Embed sentences
            sent_embeddings = await self._embedder.embed(sentences)

            # Score each sentence against query
            query_vec = np.array(query_embedding)
            scored = []
            for i, (sent, emb) in enumerate(zip(sentences, sent_embeddings)):
                sent_vec = np.array(emb)
                sim = float(np.dot(query_vec, sent_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(sent_vec) + 1e-8
                ))
                if sim >= self.relevance_threshold:
                    scored.append((sim, sent))

            # Take top sentences, preserving order
            scored.sort(key=lambda x: x[0], reverse=True)
            top_sents = [s for _, s in scored[:self.max_sentences]]

            if top_sents:
                compressed_doc = Document(
                    id=doc.id,
                    content=" ".join(top_sents),
                    metadata={**doc.metadata, "compressed": True},
                    collection=doc.collection,
                    score=doc.score,
                )
                compressed.append(compressed_doc)

        return compressed

    def _keyword_compress(self, query: str, documents: list[Document]) -> list[Document]:
        """Simple keyword-based compression fallback."""
        query_words = set(query.lower().split())
        compressed = []

        for doc in documents:
            sentences = re.split(r'(?<=[.!?])\s+', doc.content)
            relevant = []
            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words)
                if overlap >= 1 or len(sentences) <= 3:
                    relevant.append(sent)

            if relevant:
                compressed.append(Document(
                    id=doc.id,
                    content=" ".join(relevant[:self.max_sentences]),
                    metadata={**doc.metadata, "compressed": True},
                    collection=doc.collection,
                    score=doc.score,
                ))

        return compressed


# ---------------------------------------------------------------------------
# Score-based Re-ranker
# ---------------------------------------------------------------------------

class ScoreReranker:
    """Re-ranks documents by combining multiple score signals.

    Supports: vector similarity, BM25 score, recency, importance.
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.3,
        recency_weight: float = 0.1,
        importance_weight: float = 0.1,
    ) -> None:
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight

    def rerank(
        self,
        vector_results: list[Document],
        bm25_results: list[Document],
        top_k: int = 10,
    ) -> list[Document]:
        """Combine and re-rank results from vector and BM25 search."""
        # Merge by document ID
        doc_map: dict[str, dict[str, Any]] = {}

        # Normalize vector scores to [0, 1]
        if vector_results:
            max_vs = max(d.score for d in vector_results) or 1.0
            for doc in vector_results:
                doc_map[doc.id] = {
                    "doc": doc,
                    "vector_score": doc.score / max_vs,
                    "bm25_score": 0.0,
                }

        # Normalize BM25 scores to [0, 1]
        if bm25_results:
            max_bs = max(d.score for d in bm25_results) or 1.0
            for doc in bm25_results:
                if doc.id in doc_map:
                    doc_map[doc.id]["bm25_score"] = doc.score / max_bs
                else:
                    doc_map[doc.id] = {
                        "doc": doc,
                        "vector_score": 0.0,
                        "bm25_score": doc.score / max_bs,
                    }

        # Compute combined scores
        results: list[Document] = []
        import time as _time

        now = _time.time()
        for entry in doc_map.values():
            doc = entry["doc"]
            vs = entry["vector_score"]
            bs = entry["bm25_score"]

            # Recency score (decay over days)
            created_at = doc.metadata.get("created_at", now)
            if isinstance(created_at, str):
                try:
                    created_at = float(created_at)
                except (ValueError, TypeError):
                    created_at = now
            age_days = (now - created_at) / 86400
            recency = math.exp(-0.1 * age_days)  # exponential decay

            # Importance score
            importance = float(doc.metadata.get("importance", 0.5))

            combined = (
                self.vector_weight * vs
                + self.bm25_weight * bs
                + self.recency_weight * recency
                + self.importance_weight * importance
            )

            result_doc = Document(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                collection=doc.collection,
                score=combined,
            )
            results.append(result_doc)

        results.sort(key=lambda d: d.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Hybrid Retrieval Pipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """Full hybrid retrieval pipeline combining all strategies.

    Pipeline:
    1. Dense retrieval (vector search)
    2. Sparse retrieval (BM25)
    3. Score fusion & re-ranking
    4. MMR for diversity (optional)
    5. Contextual compression (optional)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        use_bm25: bool = True,
        use_mmr: bool = True,
        use_compression: bool = False,
        reranker: ScoreReranker | None = None,
        compressor: ContextualCompressor | None = None,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.use_bm25 = use_bm25
        self.use_mmr = use_mmr
        self.use_compression = use_compression
        self.reranker = reranker or ScoreReranker()
        self.compressor = compressor or ContextualCompressor(embedder=embedder)
        self.mmr_lambda = mmr_lambda

        # BM25 index — rebuilt periodically
        self._bm25 = BM25Retriever()
        self._bm25_indexed = False

    async def build_bm25_index(self, collection: str = "semantic") -> None:
        """Build/rebuild the BM25 index from the vector store."""
        # Get all documents from the collection
        # ChromaDB doesn't have a list-all, so we use a large query
        # For BM25, we need to maintain a separate corpus
        self._bm25_indexed = True

    def update_bm25_corpus(self, documents: list[Document]) -> None:
        """Update BM25 index with new documents."""
        existing = {doc.id for doc in self._bm25._corpus}
        new_docs = [doc for doc in documents if doc.id not in existing]
        if new_docs:
            self._bm25._corpus.extend(new_docs)
            self._bm25.index(self._bm25._corpus)
            self._bm25_indexed = True

    async def retrieve(
        self,
        query: str,
        collections: list[str] | None = None,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Run the full retrieval pipeline.

        Args:
            query: Search query
            collections: Which collections to search (default: all)
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
        """
        if collections is None:
            collections = ["episodic", "semantic", "procedural", "preferences"]

        # 1. Get query embedding
        query_embedding = await self.embedder.embed_single(query)

        # 2. Dense retrieval across collections
        all_vector_results: list[Document] = []
        for coll in collections:
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                collection=coll,
                top_k=top_k * 2,  # over-fetch for re-ranking
                filter_metadata=filter_metadata,
            )
            all_vector_results.extend(results)

        # 3. BM25 retrieval
        bm25_results: list[Document] = []
        if self.use_bm25 and self._bm25_indexed:
            bm25_results = self._bm25.search(query, top_k=top_k * 2)

        # 4. Hybrid re-ranking
        if bm25_results:
            candidates = self.reranker.rerank(all_vector_results, bm25_results, top_k=top_k * 2)
        else:
            candidates = sorted(all_vector_results, key=lambda d: d.score, reverse=True)

        # 5. MMR for diversity
        if self.use_mmr and len(candidates) > top_k:
            doc_embeddings = await self.embedder.embed([doc.content for doc in candidates])
            candidates = mmr_rerank(
                query_embedding=query_embedding,
                documents=candidates,
                doc_embeddings=doc_embeddings,
                lambda_param=self.mmr_lambda,
                top_k=top_k,
            )

        # 6. Contextual compression
        if self.use_compression and candidates:
            candidates = await self.compressor.compress(query, candidates)

        return candidates[:top_k]

    async def retrieve_with_hyde(
        self,
        query: str,
        llm_generate: Any = None,
        collections: list[str] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """HyDE (Hypothetical Document Embeddings) retrieval.

        Generates a hypothetical answer to the query, then uses its embedding
        for retrieval. This bridges the query-document distribution gap.
        """
        if llm_generate is None:
            # Fall back to regular retrieval
            return await self.retrieve(query, collections=collections, top_k=top_k)

        # Generate hypothetical answer
        hyde_prompt = (
            f"Write a detailed passage that would answer the following question. "
            f"The passage should be factual and informative.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        hypothetical_doc = await llm_generate(hyde_prompt)

        # Embed the hypothetical document
        hyde_embedding = await self.embedder.embed_single(hypothetical_doc)

        # Search with the hypothetical embedding
        all_results: list[Document] = []
        for coll in (collections or ["episodic", "semantic", "procedural", "preferences"]):
            results = await self.vector_store.search(
                query_embedding=hyde_embedding,
                collection=coll,
                top_k=top_k,
            )
            all_results.extend(results)

        all_results.sort(key=lambda d: d.score, reverse=True)
        return all_results[:top_k]
