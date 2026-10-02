from transformers import AutoModelForCausalLM, AutoTokenizer

from personal_assistant.models.base import BaseLLM


class SmolLM2(BaseLLM):
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", 
                 device: str = "cpu",
                 ):
        """Initialize the SmolLM2 class."""
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)  # type: ignore[arg-type]

    def __call__(self, query: list[dict[str, str]], 
                 max_new_tokens: int = 32768, 
                 temperature: float = 0.7, 
                 top_p: float = 0.9
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
        
        input_t = self.tokenizer.apply_chat_template(query, 
                                                     tokenize=False,
                                                     add_generation_prompt=True)
        
        model_inputs = self.tokenizer([input_t], return_tensors="pt").to(self.device)

        response = self.model.generate(**model_inputs, 
                                       max_new_tokens=max_new_tokens, 
                                       do_sample=True, 
                                       temperature=temperature, 
                                       top_p=top_p)

        output_ids = response[0][len(model_inputs["input_ids"][0]):]  # Get only the generated part

        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return output if isinstance(output, str) else output[0]  # skip batch decode for now