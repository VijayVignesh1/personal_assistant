import pytest

from personal_assistant.objects.memory import Memory
from personal_assistant.retriever.retrieval import Retriever


@pytest.fixture
def memories():
    return [Memory(memory_id="test_memory", content="This is a test memory.", timestamp="2024-01-01T00:00:00Z", importance=0.9, episode_id="test_episode", embedding=b"", model_name="text-embedding-3-small")]

def test_retrieval_instance(memories):
    retrieval = Retriever(memories)
    assert retrieval is not None

def test_retrieval_methods(memories):
    retrieval = Retriever(memories)
    assert hasattr(retrieval, "retrieve_relevant_memories")
    assert callable(retrieval.retrieve_relevant_memories)

def test_retrieve_relevant_memories_method(memories):
    retrieval = Retriever(memories)
    relevant_memories = retrieval.retrieve_relevant_memories(query="test memory")
    assert isinstance(relevant_memories, list)
    assert all(isinstance(memory, Memory) for memory in relevant_memories)
    assert len(relevant_memories) > 0
