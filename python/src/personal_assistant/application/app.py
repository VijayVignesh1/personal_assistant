from personal_assistant.models.smolLM2 import SmolLM2
from personal_assistant.response.response_generator import ResponseGenerator


class Application:
    def __init__(self, 
                 ) -> None:
        """Initialize the Application class."""
        self.llm_model = SmolLM2()
        self.response_generator = ResponseGenerator(model=self.llm_model)

    def generate_response(self, query: str) -> str:
        """Generate a response based on the given query."""

        # placeholder for memory retrieval, history retrieval and query generation
        context = [{'role': 'user', 'content': query}]
        return self.response_generator.generate_response(context)