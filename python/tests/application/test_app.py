from personal_assistant.application.app import Application


def test_generate_response():
    """Test the generate_response method of the Application class."""
    app = Application()
    query = "Hello"
    response = app.generate_response(query)
    assert response == f"Bot: {query} (This is a placeholder response. Implement the actual response generation logic.)"