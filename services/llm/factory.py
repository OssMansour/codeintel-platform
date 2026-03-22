"""
CodeIntel Platform -- LLM Provider Factory
Creates and caches provider instances based on configuration.
"""

from __future__ import annotations

import structlog

from services.llm.base import LLMProvider, LLMResponse, ChatMessage
from services.llm.config import LLMConfig

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(
    provider_name: str | None = None,
    cfg: LLMConfig | None = None,
    **kwargs,
) -> LLMProvider:
    """
    Instantiate an LLM provider by name.

    Args:
        provider_name: One of 'ollama', 'llamacpp', 'bedrock', 'azure'.
                       Defaults to ``cfg.llm_provider``.
        cfg: LLMConfig instance.  Created from env if not supplied.
        **kwargs: Forwarded to the provider constructor.

    Returns:
        A ready-to-use ``LLMProvider`` instance.

    Raises:
        ValueError: If the provider name is unknown.
    """
    if cfg is None:
        cfg = LLMConfig()

    name = (provider_name or cfg.llm_provider).lower().strip()

    if name == "ollama":
        from services.llm.providers.ollama import OllamaProvider
        return OllamaProvider(cfg, **kwargs)

    if name == "llamacpp":
        from services.llm.providers.llamacpp import LlamaCppProvider
        return LlamaCppProvider(cfg, **kwargs)

    if name == "bedrock":
        from services.llm.providers.bedrock import BedrockProvider
        return BedrockProvider(cfg, **kwargs)

    if name == "azure":
        from services.llm.providers.azure import AzureOpenAIProvider
        return AzureOpenAIProvider(cfg, **kwargs)

    raise ValueError(
        f"Unknown LLM provider: {name!r}.  "
        f"Choose from: ollama, llamacpp, bedrock, azure"
    )


# ---------------------------------------------------------------------------
# Fallback wrapper
# ---------------------------------------------------------------------------


class FallbackProvider(LLMProvider):
    """
    Tries providers in order; returns the first successful response.

    If the primary provider fails, it automatically falls through to the
    fallback(s).  This is useful for:
    * Ollama -> llama.cpp fallback when one server is busy
    * local -> cloud escalation (future)
    """

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self._providers = providers

    def complete(self, prompt, **kw) -> LLMResponse:
        return self._try("complete", prompt=prompt, **kw)

    def chat(self, messages, **kw) -> LLMResponse:
        return self._try("chat", messages=messages, **kw)

    async def acomplete(self, prompt, **kw) -> LLMResponse:
        return await self._atry("acomplete", prompt=prompt, **kw)

    async def achat(self, messages, **kw) -> LLMResponse:
        return await self._atry("achat", messages=messages, **kw)

    def stream(self, messages, **kw):
        # Streaming uses only the primary provider (no fallback mid-stream)
        return self._providers[0].stream(messages, **kw)

    def get_langchain_chat_model(self, **kw):
        return self._providers[0].get_langchain_chat_model(**kw)

    @property
    def provider_name(self) -> str:
        names = [p.provider_name for p in self._providers]
        return " -> ".join(names)

    @property
    def model_name(self) -> str:
        return self._providers[0].model_name

    def _try(self, method: str, **kw) -> LLMResponse:
        last_exc: Exception | None = None
        for provider in self._providers:
            try:
                resp: LLMResponse = getattr(provider, method)(**kw)
                if not resp.content.startswith("LLM call failed"):
                    return resp
                # Treat an error-content response as a soft failure
                log.warning(
                    "provider_soft_fail",
                    provider=provider.provider_name,
                    content_preview=resp.content[:120],
                )
            except Exception as exc:
                log.warning(
                    "provider_hard_fail",
                    provider=provider.provider_name,
                    error=str(exc),
                )
                last_exc = exc

        # All providers failed -- return last error
        return LLMResponse(
            content=f"All LLM providers failed. Last error: {last_exc}",
            provider=self.provider_name,
        )

    async def _atry(self, method: str, **kw) -> LLMResponse:
        last_exc: Exception | None = None
        for provider in self._providers:
            try:
                resp: LLMResponse = await getattr(provider, method)(**kw)
                if not resp.content.startswith("LLM call failed"):
                    return resp
                log.warning(
                    "provider_async_soft_fail",
                    provider=provider.provider_name,
                    content_preview=resp.content[:120],
                )
            except Exception as exc:
                log.warning(
                    "provider_async_hard_fail",
                    provider=provider.provider_name,
                    error=str(exc),
                )
                last_exc = exc

        return LLMResponse(
            content=f"All LLM providers failed. Last error: {last_exc}",
            provider=self.provider_name,
        )


def create_fallback_provider(cfg: LLMConfig | None = None) -> LLMProvider:
    """
    Build a provider chain: primary, then optional fallback.

    Reads ``LLM_PROVIDER`` and ``LLM_FALLBACK_PROVIDER`` from config.
    If no fallback is configured, returns the primary alone.
    """
    if cfg is None:
        cfg = LLMConfig()

    primary = create_provider(cfg.llm_provider, cfg)

    if cfg.llm_fallback_provider:
        fallback = create_provider(cfg.llm_fallback_provider, cfg)
        log.info(
            "llm_fallback_chain",
            primary=primary.provider_name,
            fallback=fallback.provider_name,
        )
        return FallbackProvider([primary, fallback])

    return primary


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_provider_instance: LLMProvider | None = None


def get_provider(cfg: LLMConfig | None = None) -> LLMProvider:
    """Return a module-level cached LLM provider (with optional fallback)."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = create_fallback_provider(cfg)
        log.info(
            "llm_provider_initialized",
            provider=_provider_instance.provider_name,
            model=_provider_instance.model_name,
        )
    return _provider_instance
