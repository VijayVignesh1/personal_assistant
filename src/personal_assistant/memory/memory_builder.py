import datetime
import json
import uuid

from tqdm import tqdm

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
        self.token_size = 8192

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

    def _prompt_role(self) -> str:
        return """
            # Role

                Acts as an efficient memory manager. Organises the memory into structured, retrievable units based on importance and relevance.

            ##Scope

                Structures memory into respective buckets.

                Identifies users' preferences, interests, relationships, and behavioral tendencies.


            ## Boundaries

                Never fabricates information as memories.

                Grounds factual claims in available evidence.

                Never presents assumptions as established facts.

            ## Handling Uncertainty

                Matches confidence levels to the strength of available evidence.

                Flags ambiguities, inconsistencies, and knowledge gaps.

        """

    def _prompt_episodic_memory(self) -> str:
        return """
        ## Episodic Memory
        
        ### Objective

            Organise the user's conversations into structured episodic memories that capture meaningful experiences, events, interactions, and ideas.

            Episodic memory focuses on what happened, what the user experienced, and the context in which it occurred.

            It includes:

                Personal experiences, such as attending conferences, travelling, or visiting places.

                Interactions with people, including meetings, conversations, and significant encounters.

                Professional experiences, such as work events, projects, and discussions.

                Idea dumps related to technology, finance, home decor, and other interests.

                Emotional experiences, including the situations that triggered them and the user's reflections.

                Significant decisions, challenges, achievements, and lessons learned.
        
        ### Execution Steps

        1. Understand the Interaction

            Understand the conversation and identify meaningful experiences, events, and ideas.

            Distinguish between actual experiences, hypothetical scenarios, plans, and future intentions.

            Identify which information is relevant for episodic memory.

            Exclude routine conversation and information that does not contribute meaningful context.

        2. Identify Entities and Context

            Identify the people, places, organisations, tools, projects, and other entities involved.

            Identify the date, time, location, and surrounding circumstances when available.

            Establish the relationships between the user, the entities, and the event.

            Identify the user's role, actions, observations, and feelings where relevant.

        3. Extract Independent Memories

            Break the interaction into multiple independent memories.

            Each memory must represent exactly one event or one distinct idea.

            Separate different events even when they occur in the same conversation.

            Separate distinct ideas even when they belong to the same topic.

            Preserve important context that makes each memory understandable on its own.

            Avoid creating multiple memories for the same event or idea.
        
        """

    def _prompt_semantic_memory(self) -> str:
        return """
        ## Semantic Memory
            Organise the user's conversations into structured semantic memories that capture stable facts, personal knowledge, preferences, relationships, and recurring patterns.

            Semantic memory focuses on what is known about the user rather than what happened during a particular event.

            It may include:

                Personal preferences, likes, dislikes, and habits.

                Interests, hobbies, and areas of expertise.

                Relationships with family members, friends, colleagues, and other significant people.

                Important dates, including birthdays, anniversaries, wedding days, and recurring commitments.

                Long-term goals, values, priorities, and personal principles.

                Stable work-related information, responsibilities, and professional preferences.

                Recurring behavioural patterns and established preferences.

                Persistent facts about the user's environment, lifestyle, and circumstances.
        
        ## Execution Steps

        1. Understand the Interaction

            Understand the conversation and identify factual information about the user.

            Distinguish stable facts from temporary circumstances, individual experiences, and future intentions.

            Identify information that may be useful across future conversations.

            Determine whether the information is new, an update, a correction, or a repetition of an existing fact.

        2. Identify People and Relationships

            Identify the people mentioned and their relationships with the user.

            Capture relationship types, such as family member, friend, colleague, partner, or professional contact.

            Identify relevant relationship context when explicitly provided.

            Distinguish confirmed relationships from assumptions.

            Preserve the user's preferred names or ways of referring to people when available.

        3. Extract Semantic Facts

            Identify the user's preferences, interests, likes, dislikes, and habits.

            Identify important dates, recurring commitments, and significant personal milestones.

            Identify stable personal facts, values, goals, and priorities.

            Identify recurring behavioural patterns supported by multiple interactions.

            Convert relevant information into concise, independent facts.

            Separate unrelated facts into distinct memory entries.
        
        """
    def _prompt_memory_format(self) -> str:
        return """
        ## CRITICAL OUTPUT REQUIREMENTS:

        You MUST ALWAYS follow the exact output format specified below.

        Memory Format:
        {
            "episodic" : [{
                "content": "<content of the memory>",
                "importance": "<importance float score of the memory from 0 to 1>",
                "entities": "<list of entities mentioned in the memory>"
            }],
            "semantic" : [{
                "content": "<content of the memory>",
                "importance": "<importance float score of the memory from 0 to 1>",
                "entities": "<list of entities mentioned in the memory>"
            }]
        }

        ## Example:
        [
        {"role": "user", "content": "I attended the ICCV conference in Nashville last week."},
        {"role": "assistant", "content": "That's great! Can you tell me more about the sessions you attended? Who did you meet during the conference and what did you learn?"},
        {"role": "user", "content": "I met my friend Laxminarayen there and attended several interesting sessions together. I was impressed with the paper \"Attention is all you need\" paper."}
        ]

        {
            "episodic" : [
            {
                "content": "user attended the ICCV conference in Nashville on the week of Nov 20, 2026.",
                "importance": "0.9",
                "entities": ["ICCV conference", "Nashville", "Nov 20, 2026"]
            },
            {
                "content": "user met their friend Laxminarayen at the ICCV conference.",
                "importance": "0.8",
                "entities": ["Laxminarayen", "ICCV conference"]
            },
            {
                "content": "user attended several interesting sessions at the ICCV conference with Laxminarayen.",
                "importance": "0.7",
                "entities": ["Laxminarayen", "ICCV conference"]
            }
            ],
            "semantic" : [
            {
                "content": "user likes \"Attention is all you need\" paper.",
                "importance": "0.8",
                "entities": ["Attention is all you need"]
            },
            {
                "content": "user knows Laxminarayen.",
                "importance": "0.9",
                "entities": ["Laxminarayen"]
            },
            ]
        } 
        
    """

    def _generate_memory_prompt(self) -> str:
        return f"""
        {self._prompt_role()}
        {self._prompt_episodic_memory()}
        {self._prompt_semantic_memory()}
        {self._prompt_memory_format()}
        """
    
    def deduplicate_memories(self, memories: list[Memory]) -> list[Memory]:
        """Deduplicate the given list of memories based on their content."""
        seen_contents = set()
        deduplicated = []
        for mem in memories:
            if mem.content not in seen_contents:
                seen_contents.add(mem.content)
                deduplicated.append(mem)
        return deduplicated

    async def create_updated_memory(self, memory_id: str, new_content: str, importance: float, episode_id: str) -> Memory:
        """Update the content of a memory with the given memory ID."""
        return Memory(content=new_content,
                      timestamp=str(datetime.datetime.now(tz=datetime.UTC)),
                      importance=importance,
                      episode_id=episode_id,
                      memory_id=memory_id,
                      embedding=await self.embedding_model(new_content),
                      model_name=self.embedding_model.model_name)

    async def create_new_memory(self, content: str, importance: float, episode_id: str) -> Memory:
        """Create a new memory with the given content and importance."""
        return Memory(content=content,
                      timestamp=str(datetime.datetime.now(tz=datetime.UTC)),
                      importance=importance,
                      episode_id=episode_id,
                      memory_id=str(uuid.uuid4()),
                      embedding=await self.embedding_model(content),
                      model_name=self.embedding_model.model_name)
    
    async def create_memories(self, messages: list[Message], episode_id: str) -> list[Memory]:
        """Create memories for the given episode ID."""
        # step 1: chunk the input in pieces
        message_chunks = self._chunk_input(" ".join(f"{msg.role};;{msg.content}" for msg in messages))
        # step 2: generate memories from the chunked inputs
        system_context = self._generate_memory_prompt() 
        try:
            for chunk in tqdm(message_chunks, desc="Processing message chunks"):
                input_context = [{'role': 'system', 'content': system_context}, {'role': 'user', 'content': chunk}]
                memory_response = ""
                async for token in self.llm_model(input_context):
                    memory_response += token.message.content
                memory_response = memory_response.strip().removeprefix("```json").removesuffix("```").strip()
                prospect_memories = json.loads(memory_response)

                prospect_memories = prospect_memories["episodic"] + prospect_memories["semantic"]

                # add date to every memory content
                for mem in prospect_memories:
                    mem["content"] = f"{datetime.datetime.now(tz=datetime.UTC)!s} : {mem['content']}"

                # step 3: narrow down the important memories
                filtered_memories = [Memory(content=mem["content"], 
                                            timestamp=str(datetime.datetime.now(tz=datetime.UTC)), 
                                            importance=float(mem["importance"]), 
                                            episode_id=episode_id, 
                                            memory_id=str(uuid.uuid4()),
                                            embedding=await self.embedding_model(mem["content"]),
                                            model_name=self.embedding_model.model_name) for mem in prospect_memories if float(mem["importance"]) > 0.5]
                self.memories.extend(filtered_memories)

            # step 4: deduplicate the filtered memories
            self.memories = self.deduplicate_memories(self.memories)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error generating memories: {e}")

        # step 4: return the created memories
        return self.memories
