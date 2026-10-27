from collections.abc import AsyncGenerator

import pytest

from personal_assistant.models.causalLM import causalLM
from personal_assistant.response.response_generator import ResponseGenerator

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

@pytest.fixture
def model():
    llm_model = causalLM(model_name=MODEL_NAME)
    response_generator = ResponseGenerator(model=llm_model)
    return response_generator


@pytest.mark.asyncio
async def test_generate_response(model):
    """Test the response generation functionality."""
    query = [{'role': 'user', 'content': 'What is the capital of France?'}]
    response = model.generate_response(query)
    assert isinstance(response, AsyncGenerator)

@pytest.mark.asyncio
async def test_wrong_format_query(model):
    """Test that a ValueError is raised for a query that is not a list of dictionaries."""
    query = "This is not a list of dictionaries"
    
    with pytest.raises(ValueError):
        async for _ in model.generate_response(query):
            pass

@pytest.mark.asyncio
async def test_missing_keys_in_query(model):  
    """Test that a ValueError is raised for a query with missing keys."""
    query = [{'role': 'user'}]
    
    with pytest.raises(ValueError):
        async for _ in model.generate_response(query):
            pass