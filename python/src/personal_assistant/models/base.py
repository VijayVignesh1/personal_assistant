from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class BaseLLM(ABC):
    """Base class for language models."""
    def __init__(self, model_name: str) -> None:
        """Initialize the BaseLLM class."""
        self.model_name = model_name

    @abstractmethod
    def __call__(self, query: list[dict[str, str]], *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate a response based on the given messages.

        Args:
            query (list[dict[str, str]]): A list of messages, where each message is a 
                                            dictionary with 'role' and 'content' keys.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class BaseEmbeddingModel(ABC):
    """Base class for embedding models."""
    def __init__(self, model_name: str) -> None:
        """Initialize the BaseEmbeddingModel class."""
        self.model_name = model_name
    @abstractmethod
    async def __call__(self, input: str) -> bytes:
        """Generate an embedding for the given input."""
        raise NotImplementedError("Subclasses must implement this method.")