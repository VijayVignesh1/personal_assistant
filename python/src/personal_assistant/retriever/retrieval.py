import re

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from personal_assistant.objects.memory import Memory


class Retriever:
    def __init__(self, memories: list[Memory]) -> None:
        """Initialize the Retriever class."""
        self.memories = memories

        self.tokenized_memories = [self._tokenize(memory.content) for memory in memories]

        self.bm25 = BM25Okapi(self.tokenized_memories) if self.tokenized_memories else None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize the given text into lowercase words, ignoring punctuation."""
        return re.findall(r"\w+", text.lower())

    def retrieve_relevant_memories(self, query: str, top_n: int = 5) -> list[Memory]:
        """Retrieve relevant memories based on the given query."""
        tokenized_query = self._tokenize(query)
        relevant_memories = self.bm25.get_scores(tokenized_query) if self.bm25 else []

        ranked_memories = sorted(range(len(relevant_memories)), key = relevant_memories.__getitem__, reverse=True)

        results = [self.memories[i] for i in ranked_memories[:top_n]]

        return results
