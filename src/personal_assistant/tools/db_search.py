
import datetime
import uuid
from typing import Any

from personal_assistant.db.db import Database
from personal_assistant.models.embeddingModel import embeddingModel
from personal_assistant.objects.retrieved_context import RetrievedContext
from personal_assistant.retriever.retrieval import Retriever


async def db_search(query: str, 
                    message_id: str,
                    episode_id: str,
                    db: Database,
                    embedding_model: embeddingModel,
                    retrieval_method: Any) -> str:

    # step 1: retrieve relevant memories for context augmentation
    retriever = Retriever(memories=db.get_all_memories(),
                            embedding_model=embedding_model,
                            method=retrieval_method)
    relevant_memories = await retriever.retrieve_relevant_memories(query=query, top_n=5)

    # step 2: save retrieved contexts to the database for logs
    retrieved_contexts = [
        RetrievedContext(
            id=str(uuid.uuid4()),
            episode_id=episode_id,
            message_id=message_id,
            memory_id=memory.memory_id,
            retrieved_content=memory.content,
            retrieval_timestamp=str(datetime.datetime.now(tz=datetime.UTC))
        )
        for memory in relevant_memories
    ]
    db.save_retrieved_context(retrieved_contexts)

    memories = ""
    # step 3: append relevant memories
    for memory in relevant_memories:
        memories += memory.content + "\n"

    return memories

db_search_tool = {
        "type": "function",
        "function": {
            "name": "db_search",
            "description": "Search the database and retrieve relevant memories about the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }