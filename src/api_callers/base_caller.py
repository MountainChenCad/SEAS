# src/api_callers/base_caller.py
import time
from abc import ABC, abstractmethod


class BaseCaller(ABC):
    """
    Base class for all LLM API callers.
    Provides unified retry logic and error handling.
    """

    def __init__(self, model_name, api_key, temperature, top_p, max_tokens_completion,
                 frequency_penalty, presence_penalty, api_retry_delay, max_retries, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens_completion = max_tokens_completion
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_retry_delay = api_retry_delay
        self.max_retries = max_retries
        # Store any additional kwargs that might be specific to subclasses
        self.kwargs = kwargs

        # Skip API key validation for local models
        if self.api_key != "local" and (not self.api_key or "YOUR_API_KEY" in self.api_key): # Basic check
            raise ValueError(f"API key for {self.__class__.__name__} is not configured or is a placeholder.")

    @abstractmethod
    def _call_impl(self, prompt_content):
        """
        Subclass-specific API call implementation.
        Should return the completion text or None on failure.
        """
        pass

    def get_completion(self, prompt_content):
        """
        Public method for getting completions with unified retry logic.
        Default implementation uses _call_impl with generic retry wrapper.
        Override in subclass if custom retry logic is needed (e.g., OpenAI).
        """
        return self.get_completion_with_retry(prompt_content, self._call_impl)

    def get_completion_with_retry(self, prompt_content, call_fn):
        """
        Unified retry wrapper for API calls.
        Handles retries with exponential backoff and error logging.

        Args:
            prompt_content: The prompt to send to the API
            call_fn: Function to call that takes prompt_content and returns response or raises exception

        Returns:
            Response text or None if all retries exhausted
        """
        for attempt in range(self.max_retries):
            try:
                return call_fn(prompt_content)
            except Exception as e:
                self._handle_error(e, attempt)

                # Check for terminal errors (don't retry)
                error_str = str(e).lower()
                if "maximum context length" in error_str:
                    print(f"  Context length exceeded. Not retrying.")
                    return None
                if "not in v1/chat/completions" in error_str:
                    print(f"  Endpoint issue detected. Requires special handling.")
                    return None

                # Retry with backoff
                if attempt < self.max_retries - 1:
                    wait_time = self.api_retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"  Retrying in {wait_time} seconds... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"  Max retries ({self.max_retries}) reached. Giving up.")
                    return None

        return None

    def _handle_error(self, e, current_retry):
        """Log detailed error information"""
        error_message = f"LLM API Error ({self.__class__.__name__}, model: {self.model_name}, attempt {current_retry + 1}/{self.max_retries}): {str(e)}"
        # Attempt to get more details from the exception if available (specific to SDKs)
        try:
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'json'):
                error_message += f" | API Error Details: {e.response.json()}"
            elif hasattr(e, 'message'):  # For Anthropic or other SDKs
                error_message += f" | SDK Message: {e.message}"
        except Exception:  # noqa
            pass  # Avoid errors within error handling
        print(error_message)