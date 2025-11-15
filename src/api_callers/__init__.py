# src/api_callers/__init__.py
"""
API Caller implementations and factory for unified access.
Supports OpenAI, Anthropic, Google, and local Qwen models.
"""

# 使用 try-except 以支持部分依赖缺失的环境

try:
    from .openai_caller import OpenAICaller
except ImportError:
    OpenAICaller = None

try:
    from .anthropic_caller import AnthropicCaller
except ImportError:
    AnthropicCaller = None

try:
    from .google_caller import GoogleCaller
except ImportError:
    GoogleCaller = None

try:
    from .qwen_local_caller import QwenLocalCaller
except ImportError:
    QwenLocalCaller = None


class APICallerFactory:
    """
    Factory for creating API callers.
    Replaces hardcoded if-elif chains in main_experiment.py.
    Makes it easy to add new providers without modifying core code.
    """

    _registry = {}

    @classmethod
    def register(cls, provider_name: str, caller_class):
        """Register an API caller class"""
        if caller_class is not None:
            cls._registry[provider_name.lower()] = caller_class

    @classmethod
    def get(cls, provider_name: str, **kwargs):
        """
        Get an API caller instance.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic', 'google', 'qwen_local')
            **kwargs: Arguments to pass to the caller constructor

        Returns:
            API caller instance

        Raises:
            ValueError: If provider not found or not available
        """
        provider_key = provider_name.lower().strip()

        if provider_key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown API provider: '{provider_name}'\n"
                f"Available providers: {available}"
            )

        caller_class = cls._registry[provider_key]
        if caller_class is None:
            raise ValueError(
                f"API provider '{provider_name}' is available but dependencies not installed"
            )

        return caller_class(**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """List all registered API providers"""
        return [name for name, cls_inst in cls._registry.items() if cls_inst is not None]


# Auto-register all available callers
APICallerFactory.register("openai", OpenAICaller)
APICallerFactory.register("deepseek_platform", OpenAICaller)  # Compatible with OpenAI API
APICallerFactory.register("zhipuai_glm", OpenAICaller)  # Compatible with OpenAI API
APICallerFactory.register("anthropic", AnthropicCaller)
APICallerFactory.register("google", GoogleCaller)
APICallerFactory.register("qwen_local", QwenLocalCaller)


__all__ = [
    "OpenAICaller",
    "AnthropicCaller",
    "GoogleCaller",
    "QwenLocalCaller",
    "APICallerFactory",
]