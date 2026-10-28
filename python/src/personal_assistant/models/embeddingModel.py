import asyncio
import warnings

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from personal_assistant.models.base import BaseEmbeddingModel


class embeddingModel(BaseEmbeddingModel):
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_query_prompt: bool = True) -> None:
        """Initialize the embeddingModel class.

        SentenceTransformer reads the pooling configuration of each model, so both
        mean-pooled models (e.g. all-MiniLM-L6-v2) and last-token-pooled models
        (e.g. Qwen3-Embedding) produce correct embeddings.

        Args:
            model_name: The SentenceTransformer model to load.
            use_query_prompt: Encode queries with the model's "query" prompt, if its
                config defines one (e.g. Qwen3-Embedding). Documents never get a prompt.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            warnings.warn("Running on CPU, performance may be slower.")
        self.model_name = model_name

        self.model = SentenceTransformer(model_name, device=self.device)

        # name of the prompt used for queries, or None if queries are encoded like documents.
        # sentence-transformers fills in an empty "query" prompt for models whose config
        # defines none, so only a non-empty prompt counts.
        self.query_prompt_name: str | None = (
            "query" if use_query_prompt and self.model.prompts.get("query") else None
        )

    def _generate_embedding(self, input: str, prompt_name: str | None = None) -> bytes:
        """Generate a normalized float32 embedding for the given input."""
        embedding = self.model.encode(input, prompt_name=prompt_name,
                                      normalize_embeddings=True, convert_to_numpy=True)
        data: bytes = np.asarray(embedding, dtype=np.float32).tobytes()
        return data

    async def __call__(self, 
                 input: str) -> bytes:
        """Generate an embedding for the given input."""
        return await asyncio.to_thread(self._generate_embedding, input)

    async def embed_query(self, input: str) -> bytes:
        """Generate an embedding for a search query, with the query prompt if the model has one."""
        return await asyncio.to_thread(self._generate_embedding, input, self.query_prompt_name)
