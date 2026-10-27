
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


