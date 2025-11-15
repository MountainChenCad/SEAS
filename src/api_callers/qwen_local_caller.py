# src/api_callers/qwen_local_caller.py
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from .base_caller import BaseCaller


class QwenLocalCaller(BaseCaller):
    """
    Local Qwen2 model caller for HRRP target recognition.
    Uses transformers library to load and run inference with local Qwen2 model.
    """

    def __init__(self, model_name, api_key="local", model_path=None, adapter_path=None, **kwargs):
        """
        Initialize the local Qwen2 model.

        Args:
            model_name: Name identifier for the model (used for logging)
            api_key: Not used for local model, defaults to "local"
            model_path: Path to the local model directory
            adapter_path: Optional path to LoRA adapter (PEFT)
            **kwargs: Additional parameters from BaseCaller
        """
        # Override api_key check for local model
        if not api_key or api_key == "YOUR_DEFAULT_API_KEY_HERE":
            api_key = "local"

        # Temporarily store model_path before calling super().__init__
        self.model_path = model_path or "/root/autodl-tmp/Qwen"
        self.adapter_path = adapter_path

        # Call parent __init__ but we'll handle the API key validation differently
        super().__init__(model_name, api_key, **kwargs)

        self.model_path = model_path or "/root/autodl-tmp/Qwen"
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading local Qwen2 model from: {self.model_path}")
        if self.adapter_path:
            print(f"Loading LoRA adapter from: {self.adapter_path}")
        print(f"Using device: {self.device}")

        try:
            # Load model with adapter if provided
            if self.adapter_path:
                # Load model with PEFT adapter
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    self.adapter_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                # Load tokenizer from adapter directory
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.adapter_path,
                    trust_remote_code=True
                )
            else:
                # Load base model without adapter
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )

                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )

                if self.device == "cpu":
                    self.model = self.model.to(self.device)

            self.model.eval()
            print(f"Successfully loaded Qwen2 model: {model_name}")

        except Exception as e:
            raise ValueError(f"Failed to initialize local Qwen2 model: {e}")

    def _validate_api_key(self):
        """Override to skip API key validation for local model."""
        pass

    def _call_impl(self, prompt_content):
        """Abstract method implementation - calls get_completion."""
        return self.get_completion(prompt_content)

    def get_completion(self, prompt_content):
        """
        Generate completion from local Qwen2 model.

        Args:
            prompt_content: The prompt text to send to the model

        Returns:
            str: Generated text response or None on failure
        """
        retries = 0

        while retries < self.max_retries:
            try:
                # Format the prompt as a chat message
                messages = [
                    {"role": "system", "content": "You are a helpful assistant specialized in radar target recognition."},
                    {"role": "user", "content": prompt_content}
                ]

                # Apply chat template if available
                try:
                    if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        # Fallback: simple concatenation for models without chat template (like Llama2)
                        text = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
                except Exception:
                    # If apply_chat_template fails, use fallback
                    text = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"

                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length
                )
                inputs = inputs.to(self.device)

                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens_completion,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=self.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                # Decode the generated tokens
                # Skip the input tokens to get only the generated response
                generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                if response:
                    return response.strip()
                else:
                    print(f"Warning: Empty response from local Qwen2 model")
                    return None

            except torch.cuda.OutOfMemoryError as e:
                self._handle_error(e, retries)
                print("CUDA out of memory. Try reducing max_tokens_completion or input length.")
                # Clear cache and retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.api_retry_delay} seconds...")
                    time.sleep(self.api_retry_delay)
                else:
                    print("Max retries reached for local model inference.")
                    return None

            except Exception as e:
                self._handle_error(e, retries)
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.api_retry_delay} seconds...")
                    time.sleep(self.api_retry_delay)
                else:
                    print("Max retries reached for local model inference.")
                    return None

        return None
