import uuid

import pytest

from personal_assistant.db.db import Database
from personal_assistant.models.embeddingModel import embeddingModel
from personal_assistant.objects.episode import Episode
from personal_assistant.objects.message import Message
from personal_assistant.tools.db_search import db_search


@pytest.fixture
def db_embedding(tmp_path):
    db_path = tmp_path / "test.db"
    db = Database(db_path=str(db_path))
    message_id = str(uuid.uuid4())
    episode_id = str(uuid.uuid4())
    episode = Episode(
        episode_id=episode_id,
        started_at="2024-06-01T00:00:00Z"
    )
    db.create_episode(episode)
    message = Message(
        message_id=message_id,
        episode_id=episode_id,
        content="Test message content",
        timestamp="2024-06-01T00:00:00Z",
        role="user"
    )
    db.add_message(message)
    embedding_model = embeddingModel()
    return db, embedding_model, message_id, episode_id

@pytest.mark.asyncio
async def test_db_search(db_embedding):
    """Test the db_search function."""
    db, embedding_model, message_id, episode_id = db_embedding
    retrieval_method = "embedding"

    result = await db_search(
        message_id=message_id,
        db=db,
        episode_id=episode_id,
        embedding_model=embedding_model,
        retrieval_method=retrieval_method,
        query="Retrieve test message content"
    )
    assert result is not None
    assert isinstance(result, str)

@pytest.mark.asyncio
async def test_db_search_no_results(db_embedding):
    """Test the db_search function when no results are found."""
    db, embedding_model, message_id, episode_id = db_embedding
    retrieval_method = "embedding"

    result = await db_search(
        message_id=message_id,
        db=db,
        episode_id=episode_id,
        embedding_model=embedding_model,
        retrieval_method=retrieval_method,
        query="Non-existent content"
    )
    assert result == ''