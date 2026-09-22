import torch
from dataclasses import dataclass

@dataclass
class Memory:
    content: str
    timestamp: str
    importance: float
    episode_id: str
    embedding: torch.Tensor = None
    model_name: str = "text-embedding-3-small"