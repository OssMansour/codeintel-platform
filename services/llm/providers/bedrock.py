"""
CodeIntel Platform — Amazon Bedrock LLM Provider
Ready-to-activate provider for AWS Bedrock models.

Requires:
    pip install boto3
    (and optionally: pip install langchain-aws)

Configure via environment:
    LLM_PROVIDER=bedrock
    BEDROCK_REGION=us-east-1
    BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
    BEDROCK_PROFILE=my-aws-profile  (optional)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Generator

import structlog

from services.llm.base import ChatMessage, LLMProvider, LLMResponse
from services.llm.config import LLMConfig

log = structlog.get_logger(__name__)


class BedrockProvider(LLMProvider):
    """
    LLM provider backed by Amazon Bedrock.

    Uses ``boto3`` to call the Bedrock Converse API, which provides a
    unified interface across all Bedrock-hosted models (Claude, Titan,
    Mistral, Llama, etc.).

    This provider is **production-ready** — just install ``boto3`` and
    configure AWS credentials.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._model_id = cfg.bedrock_model_id
        self._client = self._create_client(cfg)

    @staticmethod
    def _create_client(cfg: LLMConfig) -> Any:
        """Create a boto3 Bedrock Runtime client."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "Amazon Bedrock provider requires boto3. "
                "Install it with: pip install boto3"
            )

        session_kwargs: dict[str, Any] = {"region_name": cfg.bedrock_region}
        if cfg.bedrock_profile:
            session_kwargs["profile_name"] = cfg.bedrock_profile

        session = boto3.Session(**session_kwargs)
        return session.client("bedrock-runtime")

    # -- contract -------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        messages = [ChatMessage(role="user", content=prompt)]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self._cfg.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._cfg.llm_max_tokens

        # Build Bedrock Converse API payload
        bedrock_messages = []
        system_prompts = []

        for msg in messages:
            if msg.role == "system":
                system_prompts.append({"text": msg.content})
            else:
                bedrock_messages.append({
                    "role": msg.role,
                    "content": [{"text": msg.content}],
                })

        inference_config: dict[str, Any] = {
            "temperature": temp,
            "maxTokens": tokens,
        }
        if stop:
            inference_config["stopSequences"] = stop

        try:
            kwargs: dict[str, Any] = {
                "modelId": self._model_id,
                "messages": bedrock_messages,
                "inferenceConfig": inference_config,
            }
            if system_prompts:
                kwargs["system"] = system_prompts

            raw, latency = self._timed(self._client.converse, **kwargs)

            content = ""
            for block in raw.get("output", {}).get("message", {}).get("content", []):
                if "text" in block:
                    content += block["text"]

            usage = raw.get("usage", {})
            return LLMResponse(
                content=content.strip(),
                model=self._model_id,
                provider="bedrock",
                prompt_tokens=usage.get("inputTokens", 0),
                completion_tokens=usage.get("outputTokens", 0),
                total_tokens=usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
                latency_ms=latency,
                raw=raw,
            )
        except Exception as exc:
            log.error("bedrock_call_failed", model=self._model_id, error=str(exc))
            return LLMResponse(
                content=f"LLM call failed ({self.provider_name}): {exc}",
                model=self._model_id,
                provider="bedrock",
                latency_ms=0.0,
            )

    # -- async contract -------------------------------------------------------

    async def acomplete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        messages = [ChatMessage(role="user", content=prompt)]
        return await self.achat(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Async chat via asyncio.to_thread (boto3 is not natively async)."""
        return await asyncio.to_thread(
            self.chat,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        temp = temperature if temperature is not None else self._cfg.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._cfg.llm_max_tokens

        bedrock_messages = []
        system_prompts = []
        for msg in messages:
            if msg.role == "system":
                system_prompts.append({"text": msg.content})
            else:
                bedrock_messages.append({
                    "role": msg.role,
                    "content": [{"text": msg.content}],
                })

        kwargs: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {"temperature": temp, "maxTokens": tokens},
        }
        if system_prompts:
            kwargs["system"] = system_prompts

        try:
            response = self._client.converse_stream(**kwargs)
            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        yield delta["text"]
        except Exception as exc:
            log.error("bedrock_stream_failed", error=str(exc))
            yield f"[stream error: {exc}]"

    def get_langchain_chat_model(self, **kwargs: Any) -> Any:
        """Return a LangChain ``ChatBedrockConverse`` instance."""
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError:
            raise ImportError(
                "Bedrock LangChain integration requires langchain-aws. "
                "Install it with: pip install langchain-aws"
            )

        defaults = {
            "model": self._model_id,
            "region_name": self._cfg.bedrock_region,
            "temperature": self._cfg.llm_temperature,
            "max_tokens": self._cfg.llm_max_tokens,
        }
        if self._cfg.bedrock_profile:
            defaults["credentials_profile_name"] = self._cfg.bedrock_profile
        defaults.update(kwargs)
        return ChatBedrockConverse(**defaults)

    @property
    def provider_name(self) -> str:
        return "bedrock"

    @property
    def model_name(self) -> str:
        return self._model_id
