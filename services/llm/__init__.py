"""
CodeIntel Platform — LLM Abstraction Layer
Provider-agnostic interface for language model access.

Supports local models (Ollama, llama.cpp) now, and cloud providers
(Amazon Bedrock, Azure OpenAI) in the future — without refactoring
any business logic.

Usage:
    from services.llm import get_provider
    llm = get_provider()
    answer = llm.complete("Explain this function...")
"""

from services.llm.base import LLMProvider, LLMResponse
from services.llm.config import LLMConfig
from services.llm.factory import create_provider, create_fallback_provider, get_provider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMConfig",
    "create_provider",
    "create_fallback_provider",
    "get_provider",
]
