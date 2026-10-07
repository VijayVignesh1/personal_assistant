import pytest

from personal_assistant.models.causalLM import causalLM


def test_causalLM_initialization():
    """Test the initialization of the causalLM model."""
    model = causalLM()
    assert isinstance(model, causalLM)

def test_causalLM_call():
    """Test the call method of the causalLM model."""
    model = causalLM()
    query = [{"role": "user", "content": "Hello, how are you?"}]
    response = model(query)
    
    assert isinstance(response, str)

def test_causalLM_invalid_query():
    """Test that a ValueError is raised for an invalid query format."""
    model = causalLM()
    invalid_query = "This is not a list of dictionaries"
    
    with pytest.raises(ValueError):
        model(invalid_query)

def test_causalLM_missing_keys():
    """Test that a ValueError is raised for a query with missing keys."""
    model = causalLM()
    invalid_query = [{"role": "user"}]
    
    with pytest.raises(ValueError):
        model(invalid_query)
