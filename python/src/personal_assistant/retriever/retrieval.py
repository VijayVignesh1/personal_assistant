import re
from typing import Any, Literal, get_args

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from personal_assistant.models.base import BaseEmbeddingModel
from personal_assistant.objects.memory import Memory

RetrievalMethod = Literal["bm25", "embedding", "hybrid"]


class Retriever:
    def __init__(self,
                 memories: list[Memory],
                 embedding_model: BaseEmbeddingModel | None = None,
                 method: RetrievalMethod = "bm25",
                 bm25_weight: float = 0.5,
                 rrf_k: int = 60) -> None:
        """Initialize the Retriever class.

        Args:
            memories: The memories to search.
            embedding_model: Model used to embed the query. Memory embeddings are read
                from the stored ``Memory.embedding`` bytes, never recomputed.
            method: "bm25" (keyword), "embedding" (cosine similarity) or "hybrid"
                (weighted Reciprocal Rank Fusion of both rankings).
            bm25_weight: Weight of the BM25 ranking in hybrid mode, between 0 and 1.
                The embedding ranking gets ``1 - bm25_weight``.
            rrf_k: Rank offset of Reciprocal Rank Fusion.
        """
        if method not in get_args(RetrievalMethod):
            raise ValueError(f"Unknown retrieval method: {method!r}")
        if method != "bm25" and embedding_model is None:
            raise ValueError(f"Retrieval method {method!r} requires an embedding model.")
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be between 0 and 1.")

        self.memories = memories
        self.embedding_model = embedding_model
        self.method = method
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

        self.tokenized_memories = [self._tokenize(memory.content) for memory in memories]

        self.bm25 = BM25Okapi(self.tokenized_memories) if self.tokenized_memories else None

        # (memory index, stored embedding) for memories embedded with the same model
        self.stored_embeddings: list[tuple[int, Any]] = []
        if embedding_model is not None:
            for i, memory in enumerate(memories):
                vector = self._decode_embedding(memory, embedding_model.model_name)
                if vector is not None:
                    self.stored_embeddings.append((i, vector))

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize the given text into lowercase words, ignoring punctuation."""
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _decode_embedding(memory: Memory, model_name: str) -> Any:
        """Decode a stored float32 embedding, or return None if it is missing or from another model."""
        if memory.model_name != model_name or not memory.embedding:
            return None
        if len(memory.embedding) % np.dtype(np.float32).itemsize != 0:
            return None
        return np.frombuffer(memory.embedding, dtype=np.float32)

    def _bm25_ranking(self, query: str, positive_only: bool = False) -> list[int]:
        """Memory indices ordered by BM25 score, best first.

        With ``positive_only``, memories with a score of 0 or less are left out.
        """
        scores = self.bm25.get_scores(self._tokenize(query)) if self.bm25 else []
        ranking = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        if positive_only:
            ranking = [i for i in ranking if scores[i] > 0]
        return ranking

    async def _embedding_ranking(self, query: str) -> list[int]:
        """Indices of memories with a usable embedding, ordered by cosine similarity, best first."""
        assert self.embedding_model is not None, "Embedding model is not set."
        query_vector = np.frombuffer(await self.embedding_model(query), dtype=np.float32)

        candidates = [(i, vector) for i, vector in self.stored_embeddings
                      if vector.shape == query_vector.shape]
        if not candidates:
            return []

        # embeddings are normalized, so the dot product is the cosine similarity
        similarities = np.stack([vector for _, vector in candidates]) @ query_vector
        order = np.argsort(-similarities, kind="stable")
        return [candidates[j][0] for j in order]

    async def _hybrid_ranking(self, query: str) -> list[int]:
        """Memory indices ordered by weighted Reciprocal Rank Fusion of BM25 and embedding rankings."""
        bm25_ranking = self._bm25_ranking(query, positive_only=True)
        embedding_ranking = await self._embedding_ranking(query)

        fused: dict[int, float] = {}
        for rank, i in enumerate(bm25_ranking, start=1):
            fused[i] = fused.get(i, 0.0) + self.bm25_weight / (self.rrf_k + rank)
        for rank, i in enumerate(embedding_ranking, start=1):
            fused[i] = fused.get(i, 0.0) + (1 - self.bm25_weight) / (self.rrf_k + rank)

        return sorted(fused, key=fused.__getitem__, reverse=True)

    async def retrieve_relevant_memories(self, query: str, top_n: int = 5) -> list[Memory]:
        """Retrieve relevant memories based on the given query."""
        if self.method == "embedding":
            ranked_memories = await self._embedding_ranking(query)
        elif self.method == "hybrid":
            ranked_memories = await self._hybrid_ranking(query)
        else:
            ranked_memories = self._bm25_ranking(query)

        results = [self.memories[i] for i in ranked_memories[:top_n]]

        return results
