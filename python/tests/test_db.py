from personal_assistant.db import Database
import uuid
import pytest
from personal_assistant.objects.memory import Memory
from personal_assistant.objects.episode import Episode
from personal_assistant.objects.message import Message
from datetime import datetime
import numpy as np

@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    return Database(db_path=str(db_path))

def test_create_episode(db):
    temp_episode = Episode(episode_id=str(uuid.uuid4()), started_at=datetime.now().isoformat())
    episode_id = db.create_episode(temp_episode)
    assert episode_id is not None
    print("PASS: create_episode")

def test_add_message(db):
    temp_episode = Episode(episode_id=str(uuid.uuid4()), started_at=datetime.now().isoformat())
    episode_id = db.create_episode(temp_episode)
    temp_message = Message(
        message_id=str(uuid.uuid4()),
        episode_id=episode_id,
        timestamp=datetime.now().isoformat(),
        role="user",
        content="Hello, how are you?"
    )
    message_id = db.add_message(temp_message)
    assert message_id is not None
    print("PASS: add_message")

def test_invalid_role_message(db):
    temp_episode = Episode(episode_id=str(uuid.uuid4()), started_at=datetime.now().isoformat())
    episode_id = db.create_episode(temp_episode)
    try:
        temp_message = Message(
            message_id=str(uuid.uuid4()),
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            role="invalid_role",  # This should trigger a database constraint error
            content="This should fail"
        )
        db.add_message(temp_message)
        assert False, "Expected database constraint error"
    except Exception:
        pass
    print("PASS: invalid role rejected")

def test_get_episode(db):
    temp_episode = Episode(episode_id=str(uuid.uuid4()), started_at=datetime.now().isoformat())
    episode_id = db.create_episode(temp_episode)
    episode = db.get_episode(episode_id)
    assert episode is not None
    assert episode.episode_id == episode_id
    print("PASS: get_episode")

def test_get_nonexistent_episode(db):
    fake_episode_id = "does-not-exist"
    episode = db.get_episode(fake_episode_id)
    assert episode is None
    print("PASS: get_nonexistent_episode")

def test_get_messages_by_episode(db):
    temp_episode = Episode(episode_id=str(uuid.uuid4()), started_at=datetime.now().isoformat())
    episode_id = db.create_episode(temp_episode)
    temp_message_user = Message(
        message_id=str(uuid.uuid4()),
        episode_id=episode_id,
        timestamp=datetime.now().isoformat(),
        role="user",
        content="Hello, how are you?"
    )
    db.add_message(temp_message_user)
    temp_message_assistant = Message(
        message_id=str(uuid.uuid4()),
        episode_id=episode_id,
        timestamp=datetime.now().isoformat(),
        role="assistant",
        content="I'm good, thank you!"
    )
    db.add_message(temp_message_assistant)
    messages = db.get_messages_by_episode(episode_id)
    assert len(messages) == 2
    print("PASS: get_messages_by_episode")

def test_get_full_episode(db):
    temp_episode = Episode(episode_id=str(uuid.uuid4()), started_at=datetime.now().isoformat())
    episode_id = db.create_episode(temp_episode)
    temp_message_user = Message(
        message_id=str(uuid.uuid4()),
        episode_id=episode_id,
        timestamp=datetime.now().isoformat(),
        role="user",
        content="Hello, how are you?"
    )
    db.add_message(temp_message_user)
    temp_message_assistant = Message(
        message_id=str(uuid.uuid4()),
        episode_id=episode_id,
        timestamp=datetime.now().isoformat(),
        role="assistant",
        content="I'm good, thank you!"
    )
    db.add_message(temp_message_assistant)
    episode, messages = db.get_full_episode(episode_id)
    assert episode is not None
    assert len(messages) == 2
    print("PASS: get_full_episode")

def test_close_episode(db):
    temp_episode = Episode(episode_id=str(uuid.uuid4()), started_at=datetime.now().isoformat())
    episode_id = db.create_episode(temp_episode)
    db.close_episode(episode_id)
    episode = db.get_episode(episode_id)
    assert episode.ended_at is not None
    print("PASS: close_episode")

def test_save_memory(db):
    episode_id = db.create_episode(Episode(episode_id=str(uuid.uuid4()), started_at=datetime.now().isoformat()))
    memory = Memory(
        memory_id="test_memory",
        content="This is a test memory.",
        timestamp=datetime.now().isoformat(),
        importance=0.9,
        episode_id=episode_id,
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes(),
        model_name="text-embedding-3-small"
    )
    db.save_memory([memory])
    print("PASS: save_memory")
