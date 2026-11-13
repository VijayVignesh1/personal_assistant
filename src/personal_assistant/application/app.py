import datetime
import inspect
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
from personal_assistant.tools.db_search import db_search, db_search_tool
from personal_assistant.tools.web_search import web_search_tool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "database" / "personal_assistant.db"

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
        self.tools=[web_search_tool, db_search_tool]
        self.memory_builder = MemoryBuilder(llm_model=self.llm_model, embedding_model=self.embedding_model)
        self.db = Database(db_path=db_path)
        self.episode_id = str(uuid.uuid4())
        self.started_at = str(datetime.datetime.now(tz=datetime.UTC))
        self.ended_at: str | None = None

        self.db.create_episode(Episode(episode_id=self.episode_id, started_at=self.started_at))
        
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
            async for chunk in self.response_generator.generate_response(context, tools=self.tools, think=True):
                if chunk.message.tool_calls:
                    print(chunk.message.tool_calls)
                    for tool in chunk.message.tool_calls:        
                        if tool.function.name == "db_search":
                            func: ToolFunction = partial(
                                db_search,
                                message_id=message_id,
                                db=self.db,
                                episode_id=self.episode_id,
                                embedding_model=self.embedding_model,
                                retrieval_method=self.retrieval_method
                            )

                        else:
                            # TODO: Use dict mapping instead of globals()
                            func = globals()[tool.function.name]

                        # Execute the function
                        if inspect.iscoroutinefunction(func):
                            result = await func(**tool.function.arguments)
                        else:
                            result = func(**tool.function.arguments)

                        tools_result += f"Tool: {tool.function.name}, Arguments: {tool.function.arguments}, Result: {result}\n"

        context.append({'role': 'system', 'content': tools_result})

        context.append({'role': 'user', 'content': last_messages[-1].content})

        response = ""
        async for chunk in self.response_generator.generate_response(context, tools=self.tools, think=True):#, think=False):
            response += chunk.message.content
            yield chunk.message.content

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
        all_messages = self.db.get_messages_by_episode(episode_id=self.episode_id)
        memories = await self.memory_builder.create_memories(all_messages, episode_id=self.episode_id)
        self.db.save_memory(memories)
        print(f"Saved memories for episode {[memory.content for memory in memories]}")
        self.ended_at = str(datetime.datetime.now(tz=datetime.UTC))
        self.db.close_episode(self.episode_id)

async def main() -> None:
    app = Application(llm_model_name="gemma4:e2b", 
                      embedding_model_name="Qwen/Qwen3-Embedding-0.6B")
    response = ""
    async for chunk in app.generate_response("What was the last india vs nz ODI score card? Call the tool_call web_search  and tell me."):
        response += chunk
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())