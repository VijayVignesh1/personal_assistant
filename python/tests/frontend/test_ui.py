from personal_assistant.application.app import Application
from personal_assistant.frontend.ui import UI


def test_display_message_with_history():
    ui = UI(Application())

    message = "Hello"
    history = ["my previous message"]

    response = ui._chat(message, history)

    assert response == f"Bot: {message} (This is a placeholder response. Implement the actual response generation logic.)"