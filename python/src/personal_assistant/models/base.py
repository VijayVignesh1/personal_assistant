from typing import Any


class BaseLLM:
    """Base class for language models."""
    def __init__(self, model_name: str) -> None:
        """Initialize the BaseLLM class."""
        self.model_name = model_name

    def __call__(self, query: list[dict[str, str]], *args: Any, **kwargs: Any) -> str:
        """Generate a response based on the given messages.

        Args:
            query (list[dict[str, str]]): A list of messages, where each message is a 
                                            dictionary with 'role' and 'content' keys.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    