import pytest

from personal_assistant.application.app import Application
from personal_assistant.frontend.ui import UI


@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "test.db"
    return Application(db_path=str(db_path))

def test_display_message_with_history(app):
    """Test the _chat method of the UI class with a message and history."""
    ui = UI(app)

    message = "Hello"
    history = ["my previous message"]

    response = ui._chat(message, history)

    assert isinstance(response, str)