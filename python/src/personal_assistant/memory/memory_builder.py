from typing import List, Dict
from personal_assistant.models.base import BaseLLM
from personal_assistant.objects.memory import Memory
from personal_assistant.objects.message import Message
import json
import uuid
import datetime

class MemoryBuilder:
    """A builder class for creating and managing memories."""
    def __init__(self, llm_model: BaseLLM, embedding_model: BaseLLM) -> None:
        """Initialize the MemoryBuilder class."""
        self.memories: List[Memory] = []
        self.model = llm_model
        self.embedding_model = embedding_model
        self.token_size = 512

    def _chunk_input(self, combined_message: str) -> None:
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
            long term storage and retrieval.

            Guidelines:
            - Focus on identifying relavant parts of the message.
            - Names and relationships are extremely important.
            - Focus on the significance and relevance of the information for long-term memory.
            - Do not store trivial information.
            - Dates and events are important for contextualizing memories.
            - DO NOT GIVE MORE THAN 8 MEMORIES AT A TIME.
            
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
                }
            ]
        """
    
    def create_memories(self, messages: List[Message], episode_id: str) -> List[Memory]:
        """Create memories for the given episode ID."""
        # step 1: chunk the input in pieces
        message_chunks = self._chunk_input(" ".join([";;".join([msg.role, msg.content]) for msg in messages]))
        # step 2: generate memories from the chunked inputs
        system_context = self._generate_memory_prompt() 
        try:
            for chunk in message_chunks:
                input_context = [{'role': 'system', 'content': system_context}, {'role': 'user', 'content': chunk}]
                prospect_memories = json.loads(self.model(input_context))
                # step 3: narrow down the important memories
                filtered_memories = [Memory(content=mem["content"], 
                                            timestamp=str(datetime.datetime.now(tz=datetime.UTC)), 
                                            importance=float(mem["importance"]), 
                                            episode_id=episode_id, 
                                            memory_id=str(uuid.uuid4()),
                                            embedding = self.embedding_model.embed(mem["content"]),
                                            model_name=self.embedding_model.model_name) for mem in prospect_memories if float(mem["importance"]) > 0.5]
                self.memories.extend(filtered_memories)
        except Exception as e:
            raise Exception(f"Error generating memories: {e}")

        # step 4: return the created memories
        return self.memories
