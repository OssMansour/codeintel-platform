"""
CodeIntel Platform — Azure OpenAI LLM Provider
Ready-to-activate provider for Azure-hosted OpenAI models.

Requires:
    pip install openai>=1.30  (already a dependency)
    (and optionally: pip install langchain-openai)

Configure via environment:
    LLM_PROVIDER=azure
    AZURE_ENDPOINT=https://my-resource.openai.azure.com/
    AZURE_DEPLOYMENT=gpt-4o
    AZURE_API_KEY=sk-...
    AZURE_API_VERSION=2024-06-01
"""

from __future__ import annotations

from typing import Any, Generator

import structlog
from openai import AsyncAzureOpenAI, AzureOpenAI

from services.llm.base import ChatMessage, LLMProvider, LLMResponse
from services.llm.config import LLMConfig

log = structlog.get_logger(__name__)


class AzureOpenAIProvider(LLMProvider):
    """
    LLM provider backed by Azure OpenAI Service.

    Uses the openai SDK AzureOpenAI client - no extra dependency
    beyond what CodeIntel already has.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._deployment = cfg.azure_deployment

        if not cfg.azure_endpoint:
            raise ValueError(
                "Azure OpenAI provider requires AZURE_ENDPOINT. "
                "Set it in your .env file."
            )

        self._client = AzureOpenAI(
            azure_endpoint=cfg.azure_endpoint,
            api_key=cfg.azure_api_key,
            api_version=cfg.azure_api_version,
        )
        self._async_client = AsyncAzureOpenAI(
            azure_endpoint=cfg.azure_endpoint,
            api_key=cfg.azure_api_key,
            api_version=cfg.azure_api_version,
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
                model=self._deployment,
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
            log.error("azure_stream_failed", error=str(exc))
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
        """Return a LangChain ``AzureChatOpenAI`` instance."""
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "Azure LangChain integration requires langchain-openai. "
                "Install it with: pip install langchain-openai"
            )

        defaults = {
            "azure_endpoint": self._cfg.azure_endpoint,
            "azure_deployment": self._deployment,
            "api_key": self._cfg.azure_api_key,
            "api_version": self._cfg.azure_api_version,
            "temperature": self._cfg.llm_temperature,
            "max_tokens": self._cfg.llm_max_tokens,
        }
        defaults.update(kwargs)
        return AzureChatOpenAI(**defaults)

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def model_name(self) -> str:
        return self._deployment

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
                model=self._deployment,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                stop=stop,
            )
            usage = raw.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()
            return LLMResponse(
                content=raw.choices[0].message.content.strip(),
                model=self._deployment,
                provider="azure",
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                latency_ms=latency,
                raw=raw,
            )
        except Exception as exc:
            log.error("azure_call_failed", deployment=self._deployment, error=str(exc))
            return LLMResponse(
                content=f"LLM call failed ({self.provider_name}): {exc}",
                model=self._deployment,
                provider="azure",
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
                    model=self._deployment,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    stop=stop,
                )
            )
            usage = raw.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()
            return LLMResponse(
                content=raw.choices[0].message.content.strip(),
                model=self._deployment,
                provider="azure",
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
                latency_ms=latency,
                raw=raw,
            )
        except Exception as exc:
            log.error("azure_acall_failed", deployment=self._deployment, error=str(exc))
            return LLMResponse(
                content=f"LLM call failed ({self.provider_name}): {exc}",
                model=self._deployment,
                provider="azure",
                latency_ms=0.0,
            )
