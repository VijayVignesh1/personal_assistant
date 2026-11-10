import asyncio
import time
import warnings
from collections.abc import AsyncGenerator
from threading import Thread
from typing import Any

import ollama
import torch
from ollama import AsyncClient, ResponseError
from tabulate import tabulate
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer

from personal_assistant.models.base import BaseLLM
from personal_assistant.models.system_monitor import SystemMonitor


class causalLM(BaseLLM):
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", 
                 ):
        """Initialize the causalLM class."""
        self.ollama = None
        self.model = None
        self.model_name = model_name
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                warnings.warn("Running on CPU, performance may be slower.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)  # type: ignore[arg-type]
            
        except OSError as e:
            warnings.warn(f"Failed to initialize the model: {e}")

        if self.model is None:
            try:
                models = ollama.list()
                available = [m.model for m in models.models]
                if self.model_name not in available:
                    print(f"Model '{self.model_name}' is not available. Downloading the model")
                    ollama.pull(model=self.model_name)
                    print(f"Model '{self.model_name}' downloaded successfully.")
                self.ollama = "ollama"
            except ResponseError as e:
                warnings.warn(f"Failed to initialize Ollama model: {e}")

        if self.model is None and self.ollama is None:
            raise RuntimeError("Failed to initialize both HuggingFace and Ollama models. Give a valid model name.")

    async def __call__(self, query: list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate a response based on the given query using the initialized model."""
        if self.model is not None:
            async for chunk in self._call_huggingface(query, **kwargs):
                yield chunk
        elif self.ollama is not None:
            async for chunk in self._call_ollama(query, **kwargs):
                yield chunk
        else:
            raise RuntimeError("No model is initialized.")
        
    async def _call_huggingface(self, query: list[dict[str, str]], 
                 **kwargs: Any
                 ) -> AsyncGenerator[str, None]:
        """Generate a response based on the given messages.

        Args:
            query (list[dict[str, str]]): A list of messages, where each message is a 
                                            dictionary with 'role' and 'content' keys.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The sampling temperature.
            top_p (float): The cumulative probability for nucleus sampling.
        """
        if not isinstance(query, list) or not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in query):
            raise ValueError("Query must be a list of dictionaries with 'role' and 'content' keys.")
        
        model_inputs = self.tokenizer.apply_chat_template(
            query,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)  # type: ignore[union-attr]

        self.streamer = AsyncTextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True
                )
        generation_kwargs = dict(
            **model_inputs,
            streamer=self.streamer,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            max_length=None,
            do_sample=True,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
        )
        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)  # type: ignore[union-attr]
            thread.start()

        async for chunk in self.streamer:
            yield chunk

        await asyncio.to_thread(thread.join)

    async def _call_ollama(self, query: list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        """Generate a response using the Ollama model."""
        if not isinstance(query, list) or not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in query):
            raise ValueError("Query must be a list of dictionaries with 'role' and 'content' keys.")

        response = await AsyncClient().chat(model=self.model_name, messages=query, stream=True, **kwargs)

        async for chunk in response:
            yield chunk.message.content  # type: ignore[misc]


async def consume(model: causalLM, query: list[dict[str, str]], **kwargs: Any) -> dict[str, float]:
    start_time = time.perf_counter()

    first_token_time = None
    token_count = 0

    async for _ in model(query, **kwargs):
        token_count += 1

        if first_token_time is None:
            first_token_time = time.perf_counter()

    end_time = time.perf_counter()

    if first_token_time is None:
        return {"ttft": 0.0, "e2e": 0.0, "tokens": 0, "itl": 0.0}

    ttft = first_token_time - start_time
    e2e = end_time - start_time

    itl = (
        (e2e - ttft) / (token_count - 1)
        if token_count > 1 else 0.0
    )

    return {
        "ttft": ttft,
        "e2e": e2e,
        "tokens": token_count,
        "itl": itl
    }


async def concurrent_benchmark(
    model: causalLM,
    query: list[dict[str, str]],
    concurrency: int = 5,
    **kwargs: Any
) -> dict[str, float]:


    monitor = SystemMonitor(
        gpu_id=0,
        interval=0.1
    )

    monitor.start()

    start_time = time.perf_counter()

    try:
        results = await asyncio.gather(*[
            consume(model, query, **kwargs)
            for _ in range(concurrency)
        ])
    finally:
        end_time = time.perf_counter()
        system_metrics = await monitor.stop()

    elapsed = end_time - start_time

    results = [r for r in results if r is not None]

    total_requests = len(results)
    total_tokens = sum(r["tokens"] for r in results)

    benchmarks = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "total_time": elapsed,

        "requests_per_sec": total_requests / elapsed,
        "tokens_per_sec": total_tokens / elapsed,

        "avg_ttft": sum(r["ttft"] for r in results) / total_requests,
        "avg_e2e": sum(r["e2e"] for r in results) / total_requests,
        "avg_itl": sum(r["itl"] for r in results) / total_requests,
    } | system_metrics

    return benchmarks

async def main() -> None:

    huggingface_model = causalLM(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )

    ollama_model = causalLM(
        model_name="deepseek-r1:1.5b"
    )

    query = [
        {
            "role": "user",
            "content": "Explain LLM. Answer in exactly 10 words."
        }
    ]
    results = {}
    
    for req in range(5, 25, 5):

        hf_result = await concurrent_benchmark(
            huggingface_model,
            query,
            concurrency=req,
            max_new_tokens=8
        )

        ollama_result = await concurrent_benchmark(
            ollama_model,
            query,
            concurrency=req,
            options={
                "num_predict": 10,
                "temperature": 0
            }
        )

        results[req] = {
            "huggingface": hf_result,
            "ollama": ollama_result
        }
       

    # print the results as a table
    table = []
    for concurrency_level, result in results.items():
        for model_name, model_result in result.items():
            
            row = [f"{model_name} (c={concurrency_level})"] + [model_result[k] for k in model_result]
            table.append(row)

    headers = ["Model"] + list(next(iter(results.values()))["huggingface"].keys())
    
    print(tabulate(table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    asyncio.run(main())