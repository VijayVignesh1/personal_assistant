
import pytest

from personal_assistant.application.app import Application


@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "test.db"
    return  Application(db_path=str(db_path))

def test_generate_response(app):
    """Test the generate_response method of the Application class."""
    query = "Hello"
    response = app.generate_response(query)
    assert isinstance(response, str)

def test_episode_creation(app):
    """Test that an episode is created when the Application is initialized."""
    assert app.episode_id is not None
    assert app.started_at is not None
    assert app.ended_at is None

def test_episode_completion(app):
    """Test that an episode is marked as ended correctly."""
    db = app.db
    episode_id = app.episode_id
    app.close_episode()
    
    episode = db.get_episode(episode_id) 

    assert episode is not None
    assert episode.ended_at is not None
    assert isinstance(episode.ended_at, str)

# Placeholder for additional tests related to message history
def test_generate_response_using_last_n_messages(app):
    """Test generating a response using the last few messages."""
    first_query = "The secret code is BLUE1234"
    _ = app.generate_response(first_query)
    second_query = "What is the secret code I gave you?"
    response = app.generate_response(second_query)
    assert isinstance(response, str)