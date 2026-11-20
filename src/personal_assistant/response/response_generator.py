import datetime
from collections.abc import AsyncGenerator
from typing import Any

from personal_assistant.models.base import BaseLLM


class ResponseGenerator:
    def __init__(self, 
                 model: BaseLLM,
                 ) -> None:
        """Initialize the ResponseGenerator class."""
        self.model = model
        self.prompt = self._generate_prompt(current_date=datetime.datetime.now().strftime("%B %d, %Y"))

    def _generate_prompt(self, current_date: str = "October 13, 2026") -> str:
        """Generate the prompt for the model."""
        return f"""
            You are diaryai, a personal assistant built by Vijay Vignesh. Current date: {current_date}.

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

            Use the tools provided below to assist the user when necessary.
            web_search: Use this tool to search the web for relevant information.
            
            
        """

    def additional_context_to_final_response(self) -> None:
        self.prompt += """
        For the update memory, make sure to map to the correct memory ID from the context. 
        If multiple memories need update, handle each one separately with the appropriate memory ID.

        When multiple memories contain identical, overlapping, or semantically equivalent information, identify and remove redundant memories while preserving the most useful and complete version.

        Follow these rules:

        **Identify duplicates**: Compare memories based on their meaning, not just exact text matches.

        **Preserve the best memory**: Retain the memory with the most complete, accurate, specific, and useful information.

        **Delete redundant memories**: Delete other memories that convey the same information and provide no additional useful context.

        **Preserve unique details**: Never delete memories that contain additional context, distinctions, preferences, or information not captured by the retained memory.

        **Use the correct memory ID**: Every DELETE operation must reference the exact ID of the redundant memory identified in the search results. Never use the ID of the memory being retained.

        **Avoid unnecessary updates**: Do not update the retained memory unless its content genuinely needs correction or consolidation.

        **Verify consistency**: Before returning the result, ensure that the retained memory still exists and that no useful information has been lost.

        CRITICAL OUTPUT REQUIREMENTS:

        You MUST ALWAYS follow the exact output format specified below.

        1. NEVER generate plain text outside the specified delimiters.
        2. NEVER omit <|response_start|>, <|response_end|>,
        <|memory_updates_start|>, or <|memory_updates_end|>.
        3. ALWAYS generate both sections, even when there are no memory updates.
        4. The response section must contain only user-facing text.
        5. The memory updates section must contain valid JSON.
        6. If there are no memory updates, return an empty array [].
        7. NEVER use Markdown code fences.
        8. NEVER explain or describe the output format.
        9. Do not stop generation until all four delimiters have been generated.
        10. Treat these formatting requirements as mandatory.

        Your entire output MUST follow this structure. DO NOT CHANGE THE FORMAT:
        <|response_start|>
        <|response|>
        <|response_end|>

        <|memory_updates_start|>
        [
            {
                "operation": "<ADD/UPDATE/DELETE>",
                "memory_id": "<|memory_id|>",
                "content": "<|content|>",
                "reason": "<|reason|>",
                "importance": "<|confidence|>"
            }
        ]
        <|memory_updates_end|>

        **Example**:

        Input:

        Memory ID: 001 — The user's full name is Shiva Narayanan.

        Memory ID: 002 — The user's name is Shiva Narayanan.

        Expected action:
        UPDATE 001 because it contains the more specific information.
        DELETE 002 because it is redundant.

        Example output:

        <|response_start|>
        Removed the redundant memory while preserving the more complete existing memory.
        <|response_end|>

        <|memory_updates_start|>
        [
            {
                "operation": "DELETE",
                "memory_id": "002",
                "content": "The user's name is Shiva Narayanan.",
                "reason": "This memory duplicates the information already stored in mem_001, which preserves the more specific full-name information.",
                "importance": 0.99
            },
            {
                "operation": "UPDATE",
                "memory_id": "001",
                "content": "The user's full name is Shiva Narayanan.",
                "reason": "This memory contains more specific information than the redundant one.",
                "importance": 0.90
            }
        ]
        <|memory_updates_end|>

        """

    async def generate_response(self, query: list[dict[str, str]], final_response: bool = False, **kwargs: Any) -> AsyncGenerator[Any, None]:
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
        if final_response:
            self.additional_context_to_final_response()
        full_query = [{"role": "system", "content": self.prompt}] + query

        async for token in self.model(full_query, **kwargs):
            yield token
        
