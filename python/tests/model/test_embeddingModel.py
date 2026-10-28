
import pytest

from personal_assistant.models.embeddingModel import embeddingModel


def test_embeddingModel_instance():
    model = embeddingModel()
    assert model is not None
    assert callable(model)
    assert isinstance(model, embeddingModel)


@pytest.mark.asyncio
async def test_embeddingModel_call():
    model = embeddingModel()
    input_text = "This is a test sentence."
    embedding = await model(input_text)
    assert isinstance(embedding, bytes)
    assert len(embedding) > 0


@pytest.mark.asyncio
async def test_embed_query_without_query_prompt_equals_call():
    model = embeddingModel()  # all-MiniLM-L6-v2 defines no "query" prompt
    assert model.query_prompt_name is None
    query = "Who is my boss?"
    assert await model.embed_query(query) == await model(query)


