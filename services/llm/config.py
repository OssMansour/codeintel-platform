"""
CodeIntel Platform — LLM Configuration
Single pydantic-settings class for all LLM provider configuration.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """
    Unified LLM configuration loaded from environment / ``.env``.

    Switch providers by setting ``LLM_PROVIDER``.
    The provider-specific sections below are only read when the matching
    provider is active, so unused keys can remain in ``.env`` harmlessly.
    """

    # ── Provider selection ───────────────────────────────────────────────
    llm_provider: Literal["ollama", "llamacpp", "bedrock", "azure"] = "ollama"
    llm_fallback_provider: str | None = None  # e.g. "llamacpp"

    # ── Shared ───────────────────────────────────────────────────────────
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_num_ctx: int = 16384
    llm_num_threads: int = 16

    # ── Ollama ───────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:14b-instruct-q5_K_M"
    ollama_fallback_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    ollama_batch_model: str = "qwen2.5-coder:32b-instruct-q4_K_M"

    # ── llama.cpp  (OpenAI-compatible server) ────────────────────────────
    llamacpp_base_url: str = "http://localhost:8080"
    llamacpp_model: str = "default"  # llama.cpp usually has one model loaded

    # ── Amazon Bedrock (future) ──────────────────────────────────────────
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_profile: str | None = None  # AWS CLI profile name

    # ── Azure OpenAI (future) ────────────────────────────────────────────
    azure_endpoint: str = ""
    azure_deployment: str = ""
    azure_api_key: str = ""
    azure_api_version: str = "2024-06-01"

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )
