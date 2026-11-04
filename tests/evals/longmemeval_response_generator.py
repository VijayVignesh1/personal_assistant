import asyncio
import datetime
import json
import uuid
from dataclasses import asdict
from typing import Any

import numpy as np

from personal_assistant.memory.memory_builder import MemoryBuilder
from personal_assistant.models.causalLM import causalLM
from personal_assistant.models.embeddingModel import embeddingModel
from personal_assistant.objects.memory import Memory
from personal_assistant.objects.message import Message
from personal_assistant.response.response_generator import ResponseGenerator
from personal_assistant.retriever.retrieval import Retriever

# Configuration
MODEL_NAME = "gemma4:e2b"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

INPUT_FILE = "longmemeval_oracle.json"
MEMORY_FILE = "longmemeval_memory.json"
OUTPUT_FILE = "longmemeval_response.json"

TOP_N_MEMORIES = 10
NUM_CONVERSATIONS = 5


# Model initialization
llm_model = causalLM(model_name=MODEL_NAME)

embedding_model = embeddingModel(
    model_name=EMBEDDING_MODEL_NAME
)


# JSON serialization
def json_serializer(obj: Any) -> Any:
    """Serialize NumPy arrays and byte objects into JSON-compatible values."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, bytes):
        return list(np.frombuffer(obj, dtype=np.float32))

    return str(obj)


def load_json(file_path: str) -> Any:
    """Load data from a JSON file."""

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, file_path: str) -> None:
    """Save data to a JSON file."""

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            default=json_serializer,
            indent=4
        )


# Memory creation
async def create_episode_memory(
    episode_id: str,
    messages: list[dict]
) -> dict:
    """Create messages and memories for a conversation episode."""

    memory_builder = MemoryBuilder(
        llm_model=llm_model,
        embedding_model=embedding_model
    )

    all_messages = [
        Message(
            message_id=str(uuid.uuid4()),
            episode_id=episode_id,
            timestamp=datetime.datetime.now(),
            role=message.get("role", "user"),
            content=message.get("content", "")
        )
        for message in messages
    ]

    memories = await memory_builder.create_memories(
        all_messages,
        episode_id=episode_id
    )

    return {
        "episode_id": episode_id,
        "messages": [asdict(message) for message in all_messages],
        "memories": [asdict(memory) for memory in memories]
    }


async def save_conversations(conversations: list[dict]) -> None:
    """Generate and save memories for the selected conversations."""

    memory_data = []

    for conversation in conversations[:NUM_CONVERSATIONS]:

        for episode_id, session in zip(
            conversation["haystack_session_ids"],
            conversation["haystack_sessions"]
        ):
            episode_memory = await create_episode_memory(
                episode_id,
                session
            )

            memory_data.append(episode_memory)

    save_json(memory_data, MEMORY_FILE)


# Memory retrieval
async def retrieve_memories(
    question: str,
    episode_ids: list[str],
    memory_data: list[dict]
) -> list[Memory]:
    """Retrieve the most relevant memories for a given question."""

    memories = [
        Memory(**memory)
        for episode in memory_data
        for memory in episode["memories"]
    ]

    # Convert embedding lists back to bytes.
    for memory in memories:
        memory.embedding = np.asarray(
            memory.embedding,
            dtype=np.float32
        ).tobytes()

    retriever = Retriever(
        memories=memories,
        embedding_model=embedding_model,
        method="embedding"
    )

    return await retriever.retrieve_relevant_memories(
        query=question,
        top_n=TOP_N_MEMORIES
    )


# Response generation
async def generate_response(
    question: str,
    relevant_memories: list[Memory]
) -> str:
    """Generate a response using the retrieved memories."""

    response_generator = ResponseGenerator(
        model=llm_model
    )

    context = [
        {
            "role": "user",
            "content": question
        }
    ]

    context.extend(
        {
            "role": "system",
            "content": memory.content
        }
        for memory in relevant_memories
    )

    response = ""

    async for chunk in response_generator.generate_response(context):
        response += chunk
        print(chunk, end="", flush=True)

    return response


# Evaluation pipeline
async def save_responses(
    conversations: list[dict],
    memory_data: list[dict]
) -> None:
    """Retrieve memories, generate responses, and save evaluation results."""

    results = []

    for conversation in conversations[:NUM_CONVERSATIONS]:

        question = conversation.get("question", "")
        expected_answer = conversation.get("answer", "")

        episode_ids = conversation["haystack_session_ids"]

        # Retrieve relevant memories.
        relevant_memories = await retrieve_memories(
            question=question,
            episode_ids=episode_ids,
            memory_data=memory_data
        )

        # Generate response.
        response = await generate_response(
            question=question,
            relevant_memories=relevant_memories
        )

        print("\n")

        # Store evaluation results.
        results.append({
            "question": question,
            "output": response,
            "expected_output": expected_answer,
            "relevant_memories": [
                asdict(memory)
                for memory in relevant_memories
            ],
            "answer_episode_ids": [
                episode["episode_id"]
                for episode in memory_data
                if episode["episode_id"] in episode_ids
            ]
        })

    save_json(results, OUTPUT_FILE)


# Main execution
async def main() -> None:
    """Execute the complete memory and response generation pipeline."""

    conversations = load_json(INPUT_FILE)

    # Step 1: Create and save memories.
    await save_conversations(conversations)

    # Step 2: Load generated memories.
    memory_data = load_json(MEMORY_FILE)

    # Step 3: Retrieve memories and generate responses.
    await save_responses(
        conversations,
        memory_data
    )


if __name__ == "__main__":
    asyncio.run(main())
