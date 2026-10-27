import datetime
import json
import uuid

from personal_assistant.models.base import BaseEmbeddingModel, BaseLLM
from personal_assistant.objects.memory import Memory
from personal_assistant.objects.message import Message


class MemoryBuilder:
    """A builder class for creating and managing memories."""
    def __init__(self, llm_model: BaseLLM, embedding_model: BaseEmbeddingModel) -> None:
        """Initialize the MemoryBuilder class."""
        self.memories: list[Memory] = []
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.token_size = 512

    def _chunk_input(self, combined_message: str) -> list[str]:
        """Chunk the input message into smaller parts based on the specified token size.
        
        Args:
            combined_message (str): The input messages to be chunked. All the messages from the episode 
                            are combined into a single string before chunking.   
        Note:
            The chunking is done based on the specified token size, which may not align perfectly with sentence boundaries.
        Returns:
            List[str]: A list of chunked input messages.
        """
        return [combined_message[i:i+self.token_size] for i in range(0, len(combined_message), self.token_size)]

    def _generate_memory_prompt(self) -> str:
        """Generate the prompt for the model."""
        return """
            You are a experienced memory builder.

            Your role is to help build and manage memories based on the importance of the information for
            long term storage and retrieval. You always give the output in the specified format and do not
            deviate from it at all.

            Guidelines:
            - Focus on identifying relavant parts of the message.
            - Names and relationships are extremely important.
            - Focus on the significance and relevance of the information for long-term memory.
            - Do not store trivial information.
            - Dates and events are important for contextualizing memories.
            - DO NOT GIVE MORE THAN 8 MEMORIES AT A TIME.
            - Ensure that each memory is concise and captures a single piece of important information.
            
            Give the output exactly in the format given below as a list of dictionaries of memories.
            DO NOT CHANGE THE FORMAT OR ADD EXTRA FIELDS/TEXTS.
            [
                {
                    "content": "<content of the memory>",
                    "importance": "<importance float score of the memory from 0 to 1>",
                    "entities": "<list of entities mentioned in the memory>"
                }
            ]

            Example:
            [
                {
                    "content": "Alice went to the park with Bob.",
                    "importance": "0.9",
                    "entities": "Alice, Bob, park"
                },
                {
                    "content": "Bob bought a new car.",
                    "importance": "0.8",
                    "entities": "Bob, car"
                }
            ]
        """
    
    async def create_memories(self, messages: list[Message], episode_id: str) -> list[Memory]:
        """Create memories for the given episode ID."""
        # step 1: chunk the input in pieces
        message_chunks = self._chunk_input(" ".join(f"{msg.role};;{msg.content}" for msg in messages))
        # step 2: generate memories from the chunked inputs
        system_context = self._generate_memory_prompt() 
        try:
            for chunk in message_chunks:
                input_context = [{'role': 'system', 'content': system_context}, {'role': 'user', 'content': chunk}]
                memory_response = ""
                async for token in self.llm_model(input_context):
                    memory_response += token
                prospect_memories = json.loads(memory_response)
                # step 3: narrow down the important memories
                filtered_memories = [Memory(content=mem["content"], 
                                            timestamp=str(datetime.datetime.now(tz=datetime.UTC)), 
                                            importance=float(mem["importance"]), 
                                            episode_id=episode_id, 
                                            memory_id=str(uuid.uuid4()),
                                            embedding=await self.embedding_model(mem["content"]),
                                            model_name=self.embedding_model.model_name) for mem in prospect_memories if float(mem["importance"]) > 0.5]
                self.memories.extend(filtered_memories)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error generating memories: {e}")

        # step 4: return the created memories
        return self.memories
