import asyncio
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from personal_assistant.models.base import BaseEmbeddingModel


class embeddingModel(BaseEmbeddingModel):
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
        """Initialize the embeddingModel class."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            warnings.warn("Running on CPU, performance may be slower.")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name).to(self.device)


    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _generate_embedding(self, input: str) -> bytes:
        """Generate an embedding synchronously."""
        texts = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**texts)

        pooled_output = self._last_token_pool(
            outputs.last_hidden_state,
            texts["attention_mask"],
        )

        embedding = F.normalize(pooled_output.float(), p=2, dim=1)

        return embedding.squeeze().cpu().numpy().tobytes()  # type: ignore[no-any-return]

    async def __call__(self, 
                 input: str) -> bytes:
        """Generate an embedding for the given input."""
        return await asyncio.to_thread(self._generate_embedding, input)