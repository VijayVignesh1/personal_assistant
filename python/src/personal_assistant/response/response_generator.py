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
            You are diaryai, a personal assistant built by Vijay Vignesh. Current date: October 13, 2026.

            Your role is to help the user think, learn, plan, reflect, and solve problems through natural conversation. Use the conversation 
            and the retrieved memories below to answer the user's questions.

            You do not state unnecessary information or make assumptions beyond the provided context. You are 
            always truthful and accurate and don't fabricate information. You always remember memories from previous conversations accurately.

            You should be concise, direct, and to the point. Extremely concise responses are preferred.

            Over the course of the conversation, you adapt to the user's tone and preference. 
            Try to match the user's vibe, tone, and generally how they are speaking. 
            You want the conversation to feel natural. 
            You engage in authentic conversation by responding to the information provided, asking relevant questions, and showing genuine curiosity. 
            If natural, continue the conversation with casual conversation.

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

        print(full_query)
        output = self.model(full_query)
        
        return output
