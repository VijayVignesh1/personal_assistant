import numpy as np
import pytest
import pytest_asyncio

from personal_assistant.models.base import BaseEmbeddingModel
from personal_assistant.objects.memory import Memory
from personal_assistant.retriever.retrieval import Retriever


@pytest.fixture
def memories():
    return [Memory(memory_id="test_memory", content="This is a test memory.", timestamp="2024-01-01T00:00:00Z", importance=0.9, episode_id="test_episode", embedding=b"", model_name="text-embedding-3-small")]

@pytest.mark.asyncio
async def test_retrieval_instance(memories):
    retrieval = Retriever(memories, method="bm25")
    assert retrieval is not None

@pytest.mark.asyncio
async def test_retrieval_methods(memories):
    retrieval = Retriever(memories, method="bm25")
    assert hasattr(retrieval, "retrieve_relevant_memories")
    assert callable(retrieval.retrieve_relevant_memories)

@pytest.mark.asyncio
async def test_retrieve_relevant_memories_method(memories):
    retrieval = Retriever(memories, method="bm25")
    relevant_memories = await retrieval.retrieve_relevant_memories(query="test memory")
    assert isinstance(relevant_memories, list)
    assert all(isinstance(memory, Memory) for memory in relevant_memories)
    assert len(relevant_memories) > 0


def test_tokenize_ignores_punctuation(memories):
    retrieval = Retriever(memories, method="bm25")
    assert retrieval._tokenize("Who is my manager?") == ["who", "is", "my", "manager"]
    assert retrieval._tokenize("Tell me about Sofia.") == ["tell", "me", "about", "sofia"]
    assert retrieval._tokenize("The user's laptop, a ThinkPad!") == [
        "the", "user", "s", "laptop", "a", "thinkpad"
    ]


@pytest.mark.asyncio
async def test_query_with_punctuation_matches_memory():
    memories = [
        Memory(memory_id="m1", content="The user's manager is Priya.", timestamp="t",
               importance=1.0, episode_id="e"),
        Memory(memory_id="m2", content="The user likes tea.", timestamp="t",
               importance=1.0, episode_id="e"),
        Memory(memory_id="m3", content="The user plays chess.", timestamp="t",
               importance=1.0, episode_id="e"),
    ]
    retrieval = Retriever(memories, method="bm25")
    results = await retrieval.retrieve_relevant_memories(query="Who is my manager?", top_n=1)
    assert results[0].memory_id == "m1"


# ---------------------------------------------------------------- embedding / hybrid

FAKE_MODEL = "fake-embedding-model"

# Hand-made unit vectors: each memory points along its own axis and each query
# along the axis (or mix of axes) of the memories it should match.
FAKE_VECTORS = {
    "The user's manager is Priya.": [1.0, 0.0, 0.0],
    "The user likes tea.": [0.0, 1.0, 0.0],
    "The user plays chess.": [0.0, 0.0, 1.0],
    "Who is my boss?": [0.9, 0.1, 0.0],
    "What hot drink do I enjoy?": [0.1, 0.9, 0.0],
    "manager drink": [-0.3, 0.9, 0.3],
}


class FakeEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = FAKE_MODEL) -> None:
        super().__init__(model_name)
        self.calls: list[str] = []

    async def __call__(self, input: str) -> bytes:
        self.calls.append(input)
        vector = np.array(FAKE_VECTORS[input], dtype=np.float32)
        return (vector / np.linalg.norm(vector)).tobytes()


async def make_memory(memory_id, content, model=None, embedding=None):
    model = model or FakeEmbeddingModel()
    return Memory(memory_id=memory_id, content=content, timestamp="t", importance=1.0,
                  episode_id="e",
                  embedding=await model(content) if embedding is None else embedding,
                  model_name=model.model_name)


@pytest_asyncio.fixture
async def embedded_memories():
    return [
        await make_memory("m1", "The user's manager is Priya."),
        await make_memory("m2", "The user likes tea."),
        await make_memory("m3", "The user plays chess."),
    ]


@pytest.mark.asyncio
async def test_embedding_method_ranks_by_cosine_similarity(embedded_memories):
    model = FakeEmbeddingModel()
    retrieval = Retriever(embedded_memories, embedding_model=model, method="embedding")

    results = await retrieval.retrieve_relevant_memories(query="Who is my boss?", top_n=3)
    assert [m.memory_id for m in results] == ["m1", "m2", "m3"]

    results = await retrieval.retrieve_relevant_memories(query="What hot drink do I enjoy?", top_n=2)
    assert [m.memory_id for m in results] == ["m2", "m1"]


@pytest.mark.asyncio
async def test_embedding_method_embeds_only_the_query(embedded_memories):
    model = FakeEmbeddingModel()
    retrieval = Retriever(embedded_memories, embedding_model=model, method="hybrid")
    await retrieval.retrieve_relevant_memories(query="Who is my boss?")
    assert model.calls == ["Who is my boss?"]


class FakeQueryPromptModel(FakeEmbeddingModel):
    """Encodes queries differently from documents, like a model with a query prompt."""

    def __init__(self, model_name: str = FAKE_MODEL) -> None:
        super().__init__(model_name)
        self.query_calls: list[str] = []

    async def embed_query(self, input: str) -> bytes:
        self.query_calls.append(input)
        # points at m3, while the plain embedding of the same text points at m1
        return np.array([0.0, 0.0, 1.0], dtype=np.float32).tobytes()


@pytest.mark.asyncio
async def test_base_embed_query_defaults_to_call():
    model = FakeEmbeddingModel()
    assert await model.embed_query("Who is my boss?") == await model("Who is my boss?")


@pytest.mark.parametrize("method", ["embedding", "hybrid"])
@pytest.mark.asyncio
async def test_retriever_embeds_query_with_embed_query(embedded_memories, method):
    model = FakeQueryPromptModel()
    retrieval = Retriever(embedded_memories, embedding_model=model, method=method)

    results = await retrieval.retrieve_relevant_memories(query="Who is my boss?", top_n=1)

    if method == "embedding":
        assert results[0].memory_id == "m3"  # ranked with the embed_query vector
    assert model.query_calls == ["Who is my boss?"]
    assert model.calls == []  # stored memory embeddings are not recomputed


@pytest.mark.asyncio
async def test_hybrid_method_combines_rank_scores(embedded_memories):
    # BM25 only matches "manager" in m1; m2 and m3 score 0 and are left out of the
    # BM25 list. The embedding ranking is m2, m3, m1.
    # fused (k=60, w=0.5): m1 = .5/61 + .5/63, m2 = .5/61, m3 = .5/62
    query = "manager drink"
    retrieval = Retriever(embedded_memories, embedding_model=FakeEmbeddingModel(),
                          method="hybrid")
    results = await retrieval.retrieve_relevant_memories(query, top_n=3)
    assert [m.memory_id for m in results] == ["m1", "m2", "m3"]


@pytest.mark.parametrize(("bm25_weight", "expected_first"), [(0.9, "m1"), (0.1, "m2")])
@pytest.mark.asyncio
async def test_hybrid_bm25_weight_controls_fusion(embedded_memories, bm25_weight, expected_first):
    # With rrf_k=1: m1 = w/2 + (1-w)/4 (BM25 rank 1, embedding rank 3), m2 = (1-w)/2
    retrieval = Retriever(embedded_memories, embedding_model=FakeEmbeddingModel(),
                          method="hybrid", bm25_weight=bm25_weight, rrf_k=1)
    results = await retrieval.retrieve_relevant_memories("manager drink", top_n=1)
    assert results[0].memory_id == expected_first


@pytest.mark.asyncio
async def test_embedding_method_skips_unusable_embeddings():
    model = FakeEmbeddingModel()
    other_model = FakeEmbeddingModel(model_name="other-model")
    memories = [
        await make_memory("same_model", "The user likes tea."),
        await make_memory("other_model", "The user's manager is Priya.", model=other_model),
        Memory(memory_id="no_embedding", content="The user's manager is Priya.",
               timestamp="t", importance=1.0, episode_id="e", model_name=FAKE_MODEL),
        await make_memory("empty_embedding", "The user's manager is Priya.", embedding=b""),
        await make_memory("wrong_dimension", "The user's manager is Priya.",
                    embedding=np.array([1.0, 0.0], dtype=np.float32).tobytes()),
    ]
    retrieval = Retriever(memories, embedding_model=model, method="embedding")
    results = await retrieval.retrieve_relevant_memories(query="Who is my boss?", top_n=5)
    assert [m.memory_id for m in results] == ["same_model"]


@pytest.mark.parametrize("method", ["embedding", "hybrid"])
def test_embedding_methods_require_embedding_model(memories, method):
    with pytest.raises(ValueError):
        Retriever(memories, method=method)


def test_unknown_method_raises(memories):
    with pytest.raises(ValueError):
        Retriever(memories, embedding_model=FakeEmbeddingModel(), method="unknown")


def test_embedding_is_the_default_method(memories):
    retrieval = Retriever(memories, embedding_model=FakeEmbeddingModel())
    assert retrieval.method == "embedding"


def test_default_method_requires_embedding_model(memories):
    with pytest.raises(ValueError):
        Retriever(memories)
