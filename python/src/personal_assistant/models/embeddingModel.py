from sentence_transformers import SentenceTransformer
import torch

from personal_assistant.models.base import BaseLLM

class embeddingModel(BaseLLM):
    def __init__(self,
                 model_name = "Qwen/Qwen3-Embedding-0.6B"):
        """Initialize the embeddingModel class."""
        self.model = SentenceTransformer(model_name)

    def embed(self, input: str) -> list[float]:
        """Generate an embedding for the given input."""
        return self.model.encode(input)