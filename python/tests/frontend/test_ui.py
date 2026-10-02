from personal_assistant.application.app import Application
from personal_assistant.frontend.ui import UI


def test_display_message_with_history():
    """Test the _chat method of the UI class with a message and history."""
    ui = UI(Application())

    message = "Hello"
    history = ["my previous message"]

    response = ui._chat(message, history)

    assert isinstance(response, str)