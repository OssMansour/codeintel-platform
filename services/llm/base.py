"""
CodeIntel Platform — LLM Provider Base Interface
Abstract base class that every LLM provider must implement.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Standardised return type from any LLM provider."""

    content: str
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Message type
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


# ---------------------------------------------------------------------------
# Abstract Provider
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """
    Provider-agnostic interface for language model access.

    Every concrete provider (Ollama, llama.cpp, Bedrock, Azure) implements
    this contract so the rest of the codebase never depends on a specific
    vendor SDK.

    Design decisions
    ----------------
    * ``complete()`` — simple prompt-in / text-out for doc generation.
    * ``chat()`` — multi-turn messages for richer interactions.
    * ``stream()`` — generator for SSE / streaming use-cases.
    * ``get_langchain_chat_model()`` — returns a LangChain-compatible
      ``BaseChatModel`` so the LangGraph agent can use the same provider
      without a second configuration path.
    """

    # -- synchronous ----------------------------------------------------------

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Single-prompt completion (e.g. docstring generation)."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Multi-turn chat completion."""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """Stream tokens one-by-one (for SSE endpoints)."""

    # -- asynchronous ---------------------------------------------------------

    @abstractmethod
    async def acomplete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Async single-prompt completion."""

    @abstractmethod
    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Async multi-turn chat completion."""

    # -- LangChain bridge -----------------------------------------------------

    @abstractmethod
    def get_langchain_chat_model(self, **kwargs: Any) -> Any:
        """
        Return a LangChain ``BaseChatModel`` that uses this provider's
        backend.  Kwargs are forwarded to the LangChain model constructor
        so callers can set ``temperature``, ``num_ctx``, etc.
        """

    # -- metadata -------------------------------------------------------------

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'ollama')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g. 'qwen2.5-coder:14b-instruct-q5_K_M')."""

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _timed(fn, *args, **kwargs) -> tuple[Any, float]:
        """Call *fn* and return (result, elapsed_ms)."""
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return result, elapsed

    @staticmethod
    async def _atimed(coro) -> tuple[Any, float]:
        """Await *coro* and return (result, elapsed_ms)."""
        t0 = time.perf_counter()
        result = await coro
        elapsed = (time.perf_counter() - t0) * 1000.0
        return result, elapsed

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name!r} model={self.model_name!r}>"
