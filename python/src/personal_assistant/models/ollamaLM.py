import ollama

from personal_assistant.models.base import BaseLLM


class ollamaLM(BaseLLM):
    def __init__(self, model_name: str = "gemma4:e2b", 
                 ):
        """Initialize the ollamaLM class."""
        self.model_name = model_name

        models = ollama.list()
        available = [m.model for m in models.models]
        if self.model_name not in available:
            print(f"Model '{self.model_name}' is not available. Downloading the model")
            ollama.pull(model=self.model_name)
            print(f"Model '{self.model_name}' downloaded successfully.")

    def __call__(self, query: list[dict[str, str]]
                 ) -> str:
        """Generate a response based on the given messages using the Ollama model.

        Args:
            query (list[dict[str, str]]): A list of messages, where each message is a 
                                            dictionary with 'role' and 'content' keys.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The sampling temperature.
            top_p (float): The cumulative probability for nucleus sampling.
        """
        if not isinstance(query, list) or not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in query):
            raise ValueError("Query must be a list of dictionaries with 'role' and 'content' keys.")
        
        response = ollama.chat(model=self.model_name, messages=query)
        return response.message.content