import json
import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase, Turn
from metrics import ContainsMetric

from personal_assistant.models.causalLM import causalLM
from personal_assistant.response.response_generator import ResponseGenerator


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return  json.load(f)

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dataset.json")
DATASET = load_dataset(DATA_PATH)
CONTAINS_DATASET = DATASET[0]["contains_testcases"]
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"

@pytest.fixture
def model():
    llm_model = causalLM(model_name=MODEL_NAME)
    response_generator = ResponseGenerator(model=llm_model)
    return response_generator

@pytest.mark.gpu
@pytest.mark.parametrize(
    "data",
    CONTAINS_DATASET,
)    
def test_contains(data, model):
    query = data["turns"]

    actual_output = model.generate_response(
        query
    )

    turns = [Turn(role=turn["role"], content=turn["content"]) for turn in query] + [Turn(role="assistant", content=actual_output)]

    test_case = ConversationalTestCase(
        turns=turns,
    )

    contains_metric = ContainsMetric(contains=data["contains"], verbose_mode=True)
    assert_test(
        test_case=test_case,
        metrics=[contains_metric],
    )