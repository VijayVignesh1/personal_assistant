import datetime
import inspect
import json
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from functools import partial
from pathlib import Path
from typing import Any

from personal_assistant.db.db import Database
from personal_assistant.memory.memory_builder import MemoryBuilder
from personal_assistant.models.causalLM import causalLM
from personal_assistant.models.embeddingModel import embeddingModel
from personal_assistant.objects.episode import Episode
from personal_assistant.objects.message import Message
from personal_assistant.response.response_generator import ResponseGenerator
from personal_assistant.retriever.retrieval import RetrievalMethod
from personal_assistant.tools.db_search import db_search
from personal_assistant.tools.web_search import web_search, web_search_tool  # noqa: F401

DATA_DIR = Path.home() / ".local" / "share" / "diary.AI"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "personal_assistant.db"

ToolFunction = (
    Callable[..., str]
    | Callable[..., Coroutine[Any, Any, str]]
)

class Application:
    def __init__(self, 
                 db_path: str | Path = DB_PATH,
                 llm_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 retrieval_method: RetrievalMethod = "embedding",
                 ) -> None:
        """Initialize the Application class."""
        self.llm_model = causalLM(model_name=llm_model_name)
        self.embedding_model = embeddingModel(model_name=embedding_model_name)
        self.retrieval_method: RetrievalMethod = retrieval_method
        self.response_generator = ResponseGenerator(model=self.llm_model)
        self.tools=[web_search_tool]
        self.memory_builder = MemoryBuilder(llm_model=self.llm_model, embedding_model=self.embedding_model)
        self.db = Database(db_path=db_path)
        self.episode_id = str(uuid.uuid4())
        self.started_at = str(datetime.datetime.now(tz=datetime.UTC))
        self.ended_at: str | None = None

        self.db.create_episode(Episode(episode_id=self.episode_id, started_at=self.started_at))

    async def split_response_and_memory(self, stream: AsyncGenerator[Any]) -> AsyncGenerator[dict[str, str], None]:
        buffer = ""
        response_mode = False
        memory_mode = False
        response_found = False
        memory_found = False

        START_RESPONSE = "<|response_start|>"
        END_RESPONSE = "<|response_end|>"
        START_MEMORY = "<|memory_updates_start|>"
        END_MEMORY = "<|memory_updates_end|>"

        full_response = ""
        while True:
            # Get next chunk
            try:
                chunk = await anext(stream)
            except StopAsyncIteration:
                break

            content = chunk.message.content
            buffer += content
            full_response += content

            while buffer:
                if not response_mode and not memory_mode:
                    if START_RESPONSE in buffer:
                        buffer = buffer.split(START_RESPONSE, 1)[1]
                        response_mode = True
                        response_found = True
                    elif START_MEMORY in buffer:
                        buffer = buffer.split(START_MEMORY, 1)[1]
                        memory_mode = True
                        memory_found = True
                    else:
                        break

                elif response_mode:
                    if END_RESPONSE in buffer:
                        text, buffer = buffer.split(END_RESPONSE, 1)

                        if text:
                            yield {
                                "type": "response",
                                "content": text
                            }

                        response_mode = False

                    else:
                        safe_length = len(buffer) - max(
                            len(END_RESPONSE) - 1,
                            len(START_MEMORY) - 1
                        )

                        if safe_length > 0:
                            yield {
                                "type": "response",
                                "content": buffer[:safe_length]
                            }
                            buffer = buffer[safe_length:]
                        break

                elif memory_mode:
                    if END_MEMORY in buffer:
                        memory_text, buffer = buffer.split(END_MEMORY, 1)

                        try:
                            updates = json.loads(memory_text.strip())

                            yield {
                                "type": "memory_updates",
                                "content": updates
                            }

                        except json.JSONDecodeError:
                            yield {
                                "type": "error",
                                "content": "Invalid memory update JSON"
                            }

                        memory_mode = False

                    else:
                        break  

        # 1. No response or memory found
        if not response_found and not memory_found:
            print("No response or memory found")
            yield {
                "type": "response",
                "content": full_response
            }

        # 2. Memory started but didn't end
        elif memory_mode:
            print("Memory started but didn't end")
            try:
                updates = json.loads(buffer.strip())

                yield {
                    "type": "memory_updates",
                    "content": updates
                }

            except json.JSONDecodeError:
                yield {
                    "type": "error",
                    "content": "Incomplete or invalid memory JSON"
                }

    async def generate_response(self, query: str) -> AsyncGenerator[str, None]:
        """Generate a response based on the given query."""

        # step 1: add message to the episode
        message_id = str(uuid.uuid4())
        message_time = str(datetime.datetime.now(tz=datetime.UTC))
        self.db.add_message(Message(message_id=message_id, 
                                               episode_id=self.episode_id, 
                                               content=query, 
                                               timestamp=message_time,
                                               role='user'))

        last_messages = self.db.get_messages_by_episode(episode_id=self.episode_id)[-10:]
        context = []
        for message in last_messages:
            context.append({'role': message.role, 'content': message.content})

        # Tool calling LLM
        tools_result = ""
        for _ in range(3):
            async for chunk in self.response_generator.generate_response([context[-1]], tools=self.tools, think=False):
                if chunk.message.tool_calls:
                    for tool in chunk.message.tool_calls:        
                        func: ToolFunction = globals()[tool.function.name]
                        # Execute the function
                        result = func(**tool.function.arguments)
                        tools_result += f"Tool: {tool.function.name}, Arguments: {tool.function.arguments}, Result: {result}\n"

        # mandate db_search call
        # if "db_search" not in tools_result:
        func = partial(
            db_search,
            message_id=message_id,
            db=self.db,
            episode_id=self.episode_id,
            embedding_model=self.embedding_model,
            retrieval_method=self.retrieval_method,
            query=query
        )
        if inspect.iscoroutinefunction(func):
            result = await func(query=query)
        else:
            result = func(query=query)
        tools_result += f"Tool: db_search, Arguments: {{'query': query}}, Result: {result}\n"
        print(f"Tools result: {tools_result}")
        context.append({'role': 'user', 'content': tools_result})
        context.append({'role': 'user', 'content': last_messages[-1].content})

        stream = self.response_generator.generate_response(
            context,
            final_response=True,
            tools=self.tools,
            think=True
        )
        response = ""
        update: dict[str, str]
        memories = []
        async for item in self.split_response_and_memory(stream):
            if item["type"] == "response":
                response += item["content"]
                yield item["content"]

            elif item["type"] == "memory_updates":
                memory_updates = item["content"]

                for update in memory_updates:  # type: ignore[assignment]
                    if update["operation"] == "UPDATE":
                        try:
                            memory_update = await self.memory_builder.create_updated_memory(
                                memory_id=update["memory_id"],
                                new_content=update["content"], 
                                importance=float(update["importance"]),
                                episode_id=self.episode_id
                            )
                            print(f"Updating memory: {memory_update.memory_id} - {memory_update.content}")
                            self.db.update_memory(memory_update)
                        except ValueError as e:
                            print(f"Failed to update memory: {update['memory_id']}, Error: {e}")
                    if update["operation"] == "DELETE":
                        try:
                            print(f"Deleting memory: {update['memory_id']} - {update['content']}")
                            self.db.delete_memory(update["memory_id"])
                        except ValueError as e:
                            print(f"Failed to delete memory: {update['memory_id']}, Error: {e}")
                    if update["operation"] == "ADD":
                        try:
                            memory = await self.memory_builder.create_new_memory(
                                content=update["content"], 
                                importance=float(update["importance"]),
                                episode_id=self.episode_id
                            )
                            print(f"Adding new memory: {memory.memory_id} - {memory.content}")
                            memories.append(memory)
                        except ValueError as e:
                            print(f"Failed to add memory: {update['content']}, Error: {e}")
        self.db.save_memory(memories)

        # step 3: add the generated response to the episode
        response_id = str(uuid.uuid4())
        response_time = str(datetime.datetime.now(tz=datetime.UTC))
        self.db.add_message(Message(message_id=response_id, 
                                    episode_id=self.episode_id, 
                                    content=response, 
                                    timestamp=response_time,
                                    role='assistant'))

    async def close_episode(self) -> None:
        """Ensure the episode is ended when closing the episode and create memories."""
        print(f"Closing episode {self.episode_id} at {datetime.datetime.now(tz=datetime.UTC)!s}")
        # create memories for the episode
        # use only user's messages for creating memories
        all_user_messages = [msg for msg in self.db.get_messages_by_episode(episode_id=self.episode_id) if msg.role == "user"]
        memories = await self.memory_builder.create_memories(all_user_messages, episode_id=self.episode_id)
        self.db.save_memory(memories)
        print(f"Saved memories for episode {[memory.content for memory in memories]}")
        self.ended_at = str(datetime.datetime.now(tz=datetime.UTC))
        self.db.close_episode(self.episode_id)

async def main() -> None:
    app = Application(llm_model_name="gemma4:e2b", 
                      embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
                      retrieval_method="hybrid")
    response = ""
    async for chunk in app.generate_response("What are the best places to see in the cities I lived in the past? Search the web"):
        response += chunk
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())