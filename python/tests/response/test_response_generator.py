import pytest

from personal_assistant.models.smolLM2 import SmolLM2
from personal_assistant.response.response_generator import ResponseGenerator


def test_generate_response():
    """Test the response generation functionality."""
    res = ResponseGenerator(model = SmolLM2())
    query = [{'role': 'user', 'content': 'What is the capital of France?'}]
    response = res.generate_response(query)

    assert isinstance(response, str)

def test_wrong_format_query():
    """Test that a ValueError is raised for a query that is not a list of dictionaries."""
    res = ResponseGenerator(model = SmolLM2())
    query = "This is not a list of dictionaries"
    
    with pytest.raises(ValueError):
        res.generate_response(query)

def test_missing_keys_in_query():  
    """Test that a ValueError is raised for a query with missing keys."""
    res = ResponseGenerator(model = SmolLM2())
    query = [{'role': 'user'}]
    
    with pytest.raises(ValueError):
        res.generate_response(query)

