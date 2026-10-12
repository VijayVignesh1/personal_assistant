import datetime
import uuid
from unittest.mock import patch

import pytest

from personal_assistant.memory.memory_builder import MemoryBuilder
from personal_assistant.models.causalLM import causalLM
from personal_assistant.models.embeddingModel import embeddingModel
from personal_assistant.objects.memory import Memory
from personal_assistant.objects.message import Message


@pytest.fixture
def memory_builder():
    llm_model = causalLM()
    embedding_model = embeddingModel()
    return MemoryBuilder(llm_model=llm_model, embedding_model=embedding_model)

def test_create_memories_with_correct_response(memory_builder):
    fake_response = """
    [
        {
            "content": "Alice went to the park with Bob.",
            "importance": "0.9",
            "entities": "Alice, Bob, park"
        },
        {
            "content": "Alice met Charlie at the park.",
            "importance": "0.8",
            "entities": "Alice, Charlie, park"
        },
        {
            "content": "Bob and Charlie decided to have a picnic.",
            "importance": "0.7",
            "entities": "Bob, Charlie, picnic"
        }
    ]
    """

    fake_embedding_response = [0.1, 0.2, 0.3] 

    messages = [
        Message(message_id=str(uuid.uuid4()), episode_id="test_episode", content="Alice went to the park with Bob.", timestamp=str(datetime.datetime.now(tz=datetime.UTC)), role="user"),
        Message(message_id=str(uuid.uuid4()), episode_id="test_episode", content="Alice met Charlie at the park.", timestamp=str(datetime.datetime.now(tz=datetime.UTC)), role="user"),
        Message(message_id=str(uuid.uuid4()), episode_id="test_episode", content="Bob and Charlie decided to have a picnic.", timestamp=str(datetime.datetime.now(tz=datetime.UTC)), role="user"),
    ]
    with patch.object(memory_builder, "llm_model", return_value=fake_response), \
        patch.object(memory_builder, "embedding_model", return_value=fake_embedding_response):
        memories = memory_builder.create_memories(messages, episode_id="test_episode")
        assert all(isinstance(mem, Memory) for mem in memories)
        assert all(mem.episode_id == "test_episode" for mem in memories)
        assert all(0.5 < mem.importance <= 1.0 for mem in memories)
        assert all(isinstance(mem.content, str) and mem.content for mem in memories)


def test_create_memories_with_wrong_response(memory_builder):
    fake_response = "This is a wrong response with incorrect format."
    fake_embedding_response = [0.1, 0.2, 0.3] 
    messages = [
        Message(message_id=str(uuid.uuid4()), episode_id="test_episode", content="Alice went to the park with Bob.", timestamp=str(datetime.datetime.now(tz=datetime.UTC)), role="user"),
        Message(message_id=str(uuid.uuid4()), episode_id="test_episode", content="Alice met Charlie at the park.", timestamp=str(datetime.datetime.now(tz=datetime.UTC)), role="user"),
        Message(message_id=str(uuid.uuid4()), episode_id="test_episode", content="Bob and Charlie decided to have a picnic.", timestamp=str(datetime.datetime.now(tz=datetime.UTC)), role="user"),
    ]
    with patch.object(memory_builder, "llm_model", return_value=fake_response), \
        patch.object(memory_builder, "embedding_model", return_value=fake_embedding_response), \
        pytest.raises(ValueError):
        _ = memory_builder.create_memories(messages, episode_id="test_episode")