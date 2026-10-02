from personal_assistant.models.base import BaseLLM


class ResponseGenerator:
    def __init__(self, 
                 model: BaseLLM) -> None:
        """Initialize the ResponseGenerator class."""
        self.model = model
        self.prompt = self._generate_prompt()

    def _generate_prompt(self) -> str:
        """Generate the prompt for the model."""
        return """
            You are a private personal AI assistant.

            Your role is to help the user think, learn, plan, reflect, and solve problems through natural conversation.

            Guidelines:
            - Be helpful, thoughtful, clear, and friendly.
            - Use the conversation context to understand what the user means and avoid asking for information they have already provided.
            - When relevant context or memories are provided, use them to make your response more useful and personalized.
            - Answer directly when you have enough information.
            - If the user's question is ambiguous or important information is missing, ask a concise clarifying question rather than making assumptions.
            - Be honest about uncertainty. Do not invent facts, memories, experiences, or information.
            - Distinguish between what you know from the conversation and what you are uncertain about.
            - Keep responses proportional to the user's question. Provide detail when it is useful, but avoid unnecessary repetition.
            - When helping with decisions, explain relevant options, trade-offs, and reasoning without making the decision for the user.
            - Treat the user's personal information and conversations as private.

            The current conversation and any retrieved memories are provided as context. Use them only when they are relevant to the user's request.
        """
    def generate_response(self, query: list[dict[str, str]]) -> str:
        """Generate a response based on the given messages, memories, and external context.
        
        Args:
            query (list[dict[str, str]]): A list of messages, where each message
                                        is a dictionary with 'role' and 'content' keys.
        Returns:
            str: The generated response.
        
        Raises:
            ValueError: If the query is not a list of dictionaries with 'role' and 'content' keys.
        """
        if not isinstance(query, list) or not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in query):
            raise ValueError("Query must be a list of dictionaries with 'role' and 'content' keys.")
        
        full_query = [{"role": "system", "content": self.prompt}] + query

        output = self.model(full_query)
        
        return output
