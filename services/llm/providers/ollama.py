"""
CodeIntel Platform — Ollama LLM Provider
Wraps Ollama's OpenAI-compatible API.
"""

from __future__ import annotations

from typing import Any, Generator

import structlog
from openai import AsyncOpenAI, OpenAI

from services.llm.base import ChatMessage, LLMProvider, LLMResponse
from services.llm.config import LLMConfig

log = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """
    LLM provider backed by a local Ollama server.

    Ollama exposes an OpenAI-compatible ``/v1`` endpoint, so we reuse the
    ``openai`` Python SDK — no extra dependency required.
    """

    def __init__(self, cfg: LLMConfig, *, use_batch_model: bool = False) -> None:
        self._cfg = cfg
        self._model = cfg.ollama_batch_model if use_batch_model else cfg.ollama_model
        self._client = OpenAI(
            base_url=f"{cfg.ollama_base_url}/v1",
            api_key="ollama",  # Ollama doesn't require a real key
        )
        self._async_client = AsyncOpenAI(
            base_url=f"{cfg.ollama_base_url}/v1",
            api_key="ollama",
        )

    # -- contract -------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return self._call(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        openai_msgs = [{"role": m.role, "content": m.content} for m in messages]
        return self._call(openai_msgs, temperature=temperature, max_tokens=max_tokens, stop=stop)

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        openai_msgs = [{"role": m.role, "content": m.content} for m in messages]
        temp = temperature if temperature is not None else self._cfg.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._cfg.llm_max_tokens
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=openai_msgs,
                temperature=temp,
                max_tokens=tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            log.error("ollama_stream_failed", model=self._model, error=str(exc))
            yield f"[stream error: {exc}]"

    # -- async contract -------------------------------------------------------

    async def acomplete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self._acall(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        openai_msgs = [{"role": m.role, "content": m.content} for m in messages]
        return await self._acall(openai_msgs, temperature=temperature, max_tokens=max_tokens, stop=stop)

    def get_langchain_chat_model(self, **kwargs: Any) -> Any:
        """Return a ``ChatOllama`` instance compatible with LangGraph."""
        from langchain_ollama import ChatOllama

        defaults = {
            "base_url": self._cfg.ollama_base_url,
            "model": self._model,
            "num_ctx": self._cfg.llm_num_ctx,
            "num_thread": self._cfg.llm_num_threads,
            "temperature": self._cfg.llm_temperature,
        }
        defaults.update(kwargs)
        return ChatOllama(**defaults)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    # -- internal -------------------------------------------------------------

    def _call(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self._cfg.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._cfg.llm_max_tokens

        try:
            raw, latency = self._timed(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stop=stop,
            )
            usage = raw.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()
            return LLMResponse(
                content=raw.choices[0].message.content.strip(),
                model=self._model,
                provider="ollama",
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                latency_ms=latency,
                raw=raw,
            )
        except Exception as exc:
            log.error("ollama_call_failed", model=self._model, error=str(exc))
            return LLMResponse(
                content=f"LLM call failed ({self.provider_name}): {exc}",
                model=self._model,
                provider="ollama",
                latency_ms=0.0,
            )

    async def _acall(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self._cfg.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._cfg.llm_max_tokens

        try:
            raw, latency = await self._atimed(
                self._async_client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    stop=stop,
                )
            )
            usage = raw.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()
            return LLMResponse(
                content=raw.choices[0].message.content.strip(),
                model=self._model,
                provider="ollama",
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                latency_ms=latency,
                raw=raw,
            )
        except Exception as exc:
            log.error("ollama_acall_failed", model=self._model, error=str(exc))
            return LLMResponse(
                content=f"LLM call failed ({self.provider_name}): {exc}",
                model=self._model,
                provider="ollama",
                latency_ms=0.0,
            )
