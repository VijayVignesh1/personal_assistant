import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv

from personal_assistant.models.causalLM import causalLM
from personal_assistant.response.response_generator import ResponseGenerator

MODEL_NAME = "gemma4:e2b"

@pytest.fixture
def api_key():
    env_path = Path(__file__).resolve().parent / ".env"
    print(f"Loading environment variables from: {env_path}")
    # Load environment variables.
    load_dotenv(dotenv_path=env_path, override=True)
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    return api_key

goldens = [Golden(input="What's my name?", context=["The user's name is Alex and your name is GPT."], expected_output="Your name is Alex")]

@pytest.mark.ollama
@pytest.mark.parametrize("golden", goldens)
@pytest.mark.asyncio
async def test_agent_name_recall(golden: Golden, api_key: str):
    llm_model = causalLM(model_name=MODEL_NAME)
    response_generator = ResponseGenerator(model=llm_model)
    messages = [{"role": "user", "content": golden.input},] + [{"role": "system", "content": ctx} for ctx in golden.context]
    response = ""
    async for resp in response_generator.generate_response(messages):
        response += resp
        
    model = GeminiModel(
        model="models/gemini-3.5-flash-lite",
        api_key=api_key,
        temperature=0,
        cost_per_input_token=0.00000125,
        cost_per_output_token=0.00000500
    )
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=response,
        expected_output=golden.expected_output,
        retrieval_context=golden.context,
    )
    geval = GEval(
        name="Name Correctness",
        model=model,
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.5
        )   
    assert_test(test_case, [geval])
