import warnings

import torch
import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        except Exception as e:
            warnings.warn(f"Failed to initialize the model: {e}")
        
        try:
            models = ollama.list()
            available = [m.model for m in models.models]
            if self.model_name not in available:
                print(f"Model '{self.model_name}' is not available. Downloading the model")
                ollama.pull(model=self.model_name)
                print(f"Model '{self.model_name}' downloaded successfully.")
            self.ollama = "ollama"
        except Exception as e:
            warnings.warn(f"Failed to initialize Ollama model: {e}")

        if self.model is None and self.ollama is None:
            raise RuntimeError("Failed to initialize both HuggingFace and Ollama models. Give a valid model name.")

    def __call__(self, query: list[dict[str, str]], **kwargs) -> str:
        if self.model is not None:
            return self._call_huggingface(query, **kwargs)
        elif self.ollama is not None:
            return self._call_ollama(query, **kwargs)
        else:
            raise RuntimeError("No model is initialized.")
        
    def _call_huggingface(self, query: list[dict[str, str]], 
                 **kwargs
                 ) -> str:
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
            return_tensors="pt",
        ).to(self.device)  # type: ignore[union-attr]

        with torch.inference_mode():
            response = self.model.generate(**model_inputs, 
                                       max_new_tokens=kwargs.get("max_new_tokens", 256), 
                                       max_length=None,
                                       do_sample=True, 
                                       temperature=kwargs.get("temperature", 0.7), 
                                       top_p=kwargs.get("top_p", 0.9))  

        output_ids = response[0][len(model_inputs["input_ids"][0]):]  # Get only the generated part

        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return output if isinstance(output, str) else output[0]  # skip batch decode for now

    def _call_ollama(self, query: list[dict[str, str]], **kwargs) -> str:
        if not isinstance(query, list) or not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in query):
            raise ValueError("Query must be a list of dictionaries with 'role' and 'content' keys.")

        response = ollama.chat(model=self.model_name, messages=query, **kwargs)
        return response.message.content