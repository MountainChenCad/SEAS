# src/api_callers/anthropic_caller.py
from anthropic import Anthropic
from .base_caller import BaseCaller


class AnthropicCaller(BaseCaller):
    """
    Anthropic Claude API caller.
    Uses unified retry logic from BaseCaller.
    """

    def __init__(self, model_name, api_key, base_url=None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.base_url = base_url

        client_args = {"api_key": self.api_key}
        if self.base_url:
            client_args["base_url"] = self.base_url

        try:
            self.client = Anthropic(**client_args)
        except Exception as e:
            raise ValueError(f"Failed to initialize Anthropic client: {e}")

    def _call_impl(self, prompt_content):
        """
        Implements Anthropic API call (without retry logic).
        Retry logic is handled by base class get_completion_with_retry().
        """
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens_completion,
            messages=[
                {
                    "role": "user",
                    "content": prompt_content,
                }
            ],
            temperature=self.temperature,
            # top_p=self.top_p, # Anthropic also supports top_p
        )

        if message.content and isinstance(message.content, list) and \
                len(message.content) > 0 and hasattr(message.content[0], 'text'):
            return message.content[0].text.strip()
        else:
            raise RuntimeError(f"Unexpected content structure: {message.content}")