import pytest

from personal_assistant.models.smolLM2 import SmolLM2


def test_smolLM2_initialization():
    """Test the initialization of the SmolLM2 model."""
    model = SmolLM2()
    assert isinstance(model, SmolLM2)

def test_smolLM2_call():
    """Test the call method of the SmolLM2 model."""
    model = SmolLM2()
    query = [{"role": "user", "content": "Hello, how are you?"}]
    response = model(query)
    
    assert isinstance(response, str)

def test_smolLM2_invalid_query():
    """Test that a ValueError is raised for an invalid query format."""
    model = SmolLM2()
    invalid_query = "This is not a list of dictionaries"
    
    with pytest.raises(ValueError):
        model(invalid_query)

def test_smolLM2_missing_keys():
    """Test that a ValueError is raised for a query with missing keys."""
    model = SmolLM2()
    invalid_query = [{"role": "user"}]
    
    with pytest.raises(ValueError):
        model(invalid_query)
