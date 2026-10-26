from collections.abc import Generator

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
    
    assert isinstance(response, Generator)

def test_causalLM_invalid_query():
    """Test that a ValueError is raised for an invalid query format."""
    model = causalLM()
    invalid_query = "This is not a list of dictionaries"
    
    with pytest.raises(ValueError):
        for _ in model(invalid_query): 
            pass

def test_causalLM_missing_keys():
    """Test that a ValueError is raised for a query with missing keys."""
    model = causalLM()
    invalid_query = [{"role": "user"}]
    
    with pytest.raises(ValueError):
        for _ in model(invalid_query):
            pass


def test_hf_call():
    """Test the HuggingFace model call specifically."""
    model = causalLM()
    query = [{"role": "user", "content": "Hello, how are you?"}]
    response = model._call_huggingface(query)
    
    assert isinstance(response, Generator)

@pytest.mark.ollama
def test_ollama_call():
    """Test the Ollama model call specifically."""
    model = causalLM(model_name = "gemma4:e2b")
    query = [{"role": "user", "content": "Hello, how are you?"}]
    response = model._call_ollama(query)
    
    assert isinstance(response, Generator)