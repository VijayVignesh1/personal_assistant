import os
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from personal_assistant.models.causalLM import causalLM
from personal_assistant.response.response_generator import ResponseGenerator

env_path = Path(__file__).resolve().parent / ".env"
print(f"Loading environment variables from: {env_path}")
# Load environment variables.
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
llm_model = causalLM(model_name=MODEL_NAME)
context = "The user's name is Alex and your name is GPT."
response_generator = ResponseGenerator(model=llm_model)
test_case = LLMTestCase(
    input="What's my name?",
    actual_output=response_generator.generate_response([{"role":"user", "content": "What's my name?"}, {"role":"system", "content": context}]),
    expected_output="Your name is Alex",
    retrieval_context=[context],
)
model = GeminiModel(
    model="models/gemini-3.5-flash-lite",
    api_key=api_key,
    temperature=0,
    cost_per_input_token=0.00000125,
    cost_per_output_token=0.00000500
)

relevancy = AnswerRelevancyMetric(
    threshold=0.8,
    model=model
)
evaluate(
    test_cases=[test_case],
    metrics=[relevancy],
)
