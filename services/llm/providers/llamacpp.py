"""
CodeIntel Platform — llama.cpp LLM Provider
Wraps a llama.cpp server's OpenAI-compatible API.

Start the server with:
    llama-server -m model.gguf --port 8080
"""

from __future__ import annotations

from typing import Any, Generator

import structlog
from openai import AsyncOpenAI, OpenAI

from services.llm.base import ChatMessage, LLMProvider, LLMResponse
from services.llm.config import LLMConfig

log = structlog.get_logger(__name__)


class LlamaCppProvider(LLMProvider):
    """
    LLM provider backed by a local llama.cpp HTTP server.

    llama.cpp's ``llama-server`` exposes an OpenAI-compatible endpoint
    at ``/v1``, so we reuse the ``openai`` SDK — zero extra dependencies.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._model = cfg.llamacpp_model
        base = cfg.llamacpp_base_url.rstrip("/")
        # llama.cpp server already serves at /v1 by default
        self._client = OpenAI(
            base_url=f"{base}/v1",
            api_key="not-needed",
        )
        self._async_client = AsyncOpenAI(
            base_url=f"{base}/v1",
            api_key="not-needed",
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
            log.error("llamacpp_stream_failed", error=str(exc))
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
        """
        Return a LangChain ``ChatOpenAI`` pointing at the llama.cpp server.

        Falls back to ``ChatOllama`` pattern if ``langchain-openai`` is not
        installed, using the generic community ``ChatOpenAI``.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI

        base = self._cfg.llamacpp_base_url.rstrip("/")
        defaults = {
            "base_url": f"{base}/v1",
            "api_key": "not-needed",
            "model": self._model,
            "temperature": self._cfg.llm_temperature,
            "max_tokens": self._cfg.llm_max_tokens,
        }
        defaults.update(kwargs)
        return ChatOpenAI(**defaults)

    @property
    def provider_name(self) -> str:
        return "llamacpp"

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
                provider="llamacpp",
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                latency_ms=latency,
                raw=raw,
            )
        except Exception as exc:
            log.error("llamacpp_call_failed", error=str(exc))
            return LLMResponse(
                content=f"LLM call failed ({self.provider_name}): {exc}",
                model=self._model,
                provider="llamacpp",
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
                provider="llamacpp",
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                latency_ms=latency,
                raw=raw,
            )
        except Exception as exc:
            log.error("llamacpp_acall_failed", error=str(exc))
            return LLMResponse(
                content=f"LLM call failed ({self.provider_name}): {exc}",
                model=self._model,
                provider="llamacpp",
                latency_ms=0.0,
            )
