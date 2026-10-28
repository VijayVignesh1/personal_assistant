import asyncio
import warnings

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from personal_assistant.models.base import BaseEmbeddingModel


class embeddingModel(BaseEmbeddingModel):
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the embeddingModel class.

        SentenceTransformer reads the pooling configuration of each model, so both
        mean-pooled models (e.g. all-MiniLM-L6-v2) and last-token-pooled models
        (e.g. Qwen3-Embedding) produce correct embeddings.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            warnings.warn("Running on CPU, performance may be slower.")
        self.model_name = model_name

        self.model = SentenceTransformer(model_name, device=self.device)

    def _generate_embedding(self, input: str) -> bytes:
        """Generate a normalized float32 embedding for the given input."""
        embedding = self.model.encode(input, normalize_embeddings=True, convert_to_numpy=True)
        data: bytes = np.asarray(embedding, dtype=np.float32).tobytes()
        return data

    async def __call__(self, 
                 input: str) -> bytes:
        """Generate an embedding for the given input."""
        return await asyncio.to_thread(self._generate_embedding, input)
