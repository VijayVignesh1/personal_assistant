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


def test_tokenize_ignores_punctuation(memories):
    retrieval = Retriever(memories)
    assert retrieval._tokenize("Who is my manager?") == ["who", "is", "my", "manager"]
    assert retrieval._tokenize("Tell me about Sofia.") == ["tell", "me", "about", "sofia"]
    assert retrieval._tokenize("The user's laptop, a ThinkPad!") == [
        "the", "user", "s", "laptop", "a", "thinkpad"
    ]


def test_query_with_punctuation_matches_memory():
    memories = [
        Memory(memory_id="m1", content="The user's manager is Priya.", timestamp="t",
               importance=1.0, episode_id="e"),
        Memory(memory_id="m2", content="The user likes tea.", timestamp="t",
               importance=1.0, episode_id="e"),
        Memory(memory_id="m3", content="The user plays chess.", timestamp="t",
               importance=1.0, episode_id="e"),
    ]
    retrieval = Retriever(memories)
    results = retrieval.retrieve_relevant_memories(query="Who is my manager?", top_n=1)
    assert results[0].memory_id == "m1"
