from personal_assistant.tools.web_search import web_search


def test_web_search():
    """Test the web_search function."""
    query = "OpenAI"
    result = web_search(query=query)
    assert result is not None
    assert isinstance(result, str)

