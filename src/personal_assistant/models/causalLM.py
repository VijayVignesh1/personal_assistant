import asyncio
import warnings
from collections.abc import AsyncGenerator
from threading import Thread
from typing import Any

import ollama
import torch
from ollama import AsyncClient, ResponseError
from transformers import AsyncTextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer

from personal_assistant.models.base import BaseLLM


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
        