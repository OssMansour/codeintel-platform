"""
Unit tests for services/llm/config.py

LLMConfig is a pure pydantic-settings class — no external deps needed.
Tests use monkeypatch.setenv to inject config values.
"""
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from services.llm.config import LLMConfig
    cfg = LLMConfig()
    assert cfg.llm_provider == "ollama"


def test_accepts_all_valid_providers(monkeypatch):
    from services.llm.config import LLMConfig
    for provider in ("ollama", "llamacpp", "bedrock", "azure"):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        cfg = LLMConfig()
        assert cfg.llm_provider == provider


def test_rejects_invalid_provider(monkeypatch):
    from pydantic import ValidationError
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    from services.llm.config import LLMConfig
    with pytest.raises(ValidationError):
        LLMConfig()


def test_temperature_default(monkeypatch):
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    from services.llm.config import LLMConfig
    cfg = LLMConfig()
    assert cfg.llm_temperature == pytest.approx(0.2)
