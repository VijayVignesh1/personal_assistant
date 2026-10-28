import json
import os
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv

from personal_assistant.models.causalLM import causalLM
from personal_assistant.response.response_generator import ResponseGenerator

MODEL_NAME = "gemma4:e2b"


env_path = Path(__file__).resolve().parent / ".env"
print(f"Loading environment variables from: {env_path}")
# Load environment variables.
load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

MODEL_NAME = "gemma4:e2b"
llm_model = causalLM(model_name=MODEL_NAME)
response_generator = ResponseGenerator(model=llm_model)

with open("response_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)[:10]
messages = [[{"role": "user", "content": golden["input"]},] + [{"role": "system", "content": ctx} for ctx in golden["context"]] for golden in data]

test_cases = [
    LLMTestCase(
        input=entry["input"],
        context=entry["context"],
        actual_output=response_generator.generate_response(msg),
        expected_output=entry["expected_output"]
    )
    for entry, msg in zip(data, messages)
]

model = GeminiModel(
        model="models/gemini-3.5-flash-lite",
        api_key=api_key,
        temperature=0,
        cost_per_input_token=0.00000125,
        cost_per_output_token=0.00000500
    )

geval = GEval(
    name="Correctness",
    model=model,
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5
    ) 

print(evaluate(test_cases, [geval]))
