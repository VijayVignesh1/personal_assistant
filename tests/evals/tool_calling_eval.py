import logging

import mlflow
import pyarrow.parquet as pq
from mlflow.genai.scorers import ToolCallCorrectness, ToolCallEfficiency

from personal_assistant.models.causalLM import causalLM
from personal_assistant.response.response_generator import ResponseGenerator
from personal_assistant.tools.web_search import web_search, web_search_tool

logging.getLogger("mlflow.genai.utils.trace_utils").setLevel(logging.ERROR)

PARQUET_FILE = "websearch.parquet"  # https://huggingface.co/datasets/kd13/WebSearchChat-v1/blob/main/websearch.parquet

MODEL_NAME = "gemma4:e2b"

# 1. MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tool-calling-evaluation")

# 2. Local Ollama judge
judge_model = "ollama:/gemma3:4b"

# Model initialization
llm_model = causalLM(model_name=MODEL_NAME)

response_generator = ResponseGenerator(
        model=llm_model
    )

# Define the tool schema for the LLM
tools = [web_search_tool]


# Trace function for web search tool calls
@mlflow.trace(span_type="TOOL")
def web_search_trace(query: str) -> str:
    return web_search(query)


@mlflow.trace
async def agent(query: str):
    tool_calls = []
    context = [{"role": "user", "content": query}]
    async for chunk in response_generator.generate_response(context, tools=tools):
        if chunk.message.tool_calls:
            tool_calls.extend(chunk.message.tool_calls)


    responses = []
    if tool_calls:
        for tool_call in tool_calls:
            args = tool_call.function.arguments
            result = web_search_trace(args["query"])
            responses.append({
                "response": f"Web search result: {result}"
            })

    return {"response": responses if responses else "No tool calls made."}


# open parquet file and create evaluation dataset
table = pq.read_table(PARQUET_FILE)
eval_dataset = []
for messages in table['messages'][:100]:  # Limit to first 100 messages for evaluation
    user_message = messages[1].as_py()["content"]  # only content of the user message as query
    tool_message = messages[2].as_py()['tool_call']
    eval_dataset.append({"inputs": {"query": user_message}, "expectations": {"expected_tool_calls": [tool_message]}})

# Run evaluation with tool call judges
eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=agent,
    scorers=[
        ToolCallCorrectness(model=judge_model),
        ToolCallEfficiency(model=judge_model),
    ],
)
