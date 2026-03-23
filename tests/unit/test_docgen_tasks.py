"""
Unit tests for services/docgen/tasks.py

Only the pure helper functions are tested (no Celery task execution, no Redis).
Celery is stubbed in sys.modules before import; os.walk and os.path.getsize
are patched for filesystem isolation.
"""
import sys
import os
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Stub celery before importing tasks
sys.modules.setdefault("celery", mock.MagicMock())
sys.modules.setdefault("celery.utils.log", mock.MagicMock())

import pytest


# ---------------------------------------------------------------------------
# Lazy import helpers so the module is loaded AFTER sys.modules stubs are in place
# ---------------------------------------------------------------------------


def _get_helpers():
    from services.docgen.tasks import (
        CelerySettings,
        LANGUAGE_EXTENSIONS,
        _is_supported_file,
        _get_supported_extensions,
        _walk_repo,
    )
    return CelerySettings, LANGUAGE_EXTENSIONS, _is_supported_file, _get_supported_extensions, _walk_repo


# ===========================================================================
# _is_supported_file tests
# ===========================================================================


def test_is_supported_file_python_recognized():
    CelerySettings, _, _is_supported_file, _, _ = _get_helpers()
    cfg = CelerySettings()
    assert _is_supported_file("services/agent/tools.py", cfg) is True


def test_is_supported_file_markdown_not_supported():
    CelerySettings, _, _is_supported_file, _, _ = _get_helpers()
    cfg = CelerySettings()
    assert _is_supported_file("README.md", cfg) is False


def test_is_supported_file_typescript_recognized():
    CelerySettings, _, _is_supported_file, _, _ = _get_helpers()
    cfg = CelerySettings()
    assert _is_supported_file("frontend/app.tsx", cfg) is True


# ===========================================================================
# _get_supported_extensions tests
# ===========================================================================


def test_get_supported_extensions_contains_py_for_python():
    CelerySettings, _, _, _get_supported_extensions, _ = _get_helpers()
    cfg = CelerySettings()
    exts = _get_supported_extensions(cfg)
    assert ".py" in exts


def test_get_supported_extensions_excludes_ts_when_not_configured(monkeypatch):
    CelerySettings, _, _, _get_supported_extensions, _ = _get_helpers()
    # Override parse_languages to python only
    cfg = CelerySettings()
    monkeypatch.setattr(cfg, "parse_languages", "python")
    exts = _get_supported_extensions(cfg)
    assert ".ts" not in exts


# ===========================================================================
# _walk_repo tests
# ===========================================================================


def test_walk_repo_skips_skip_dirs():
    CelerySettings, _, _, _, _walk_repo = _get_helpers()
    cfg = CelerySettings()

    # Simulate os.walk yielding node_modules (skip) and src (keep)
    walk_data = [
        ("/repo", ["node_modules", "src"], []),
        ("/repo/src", [], ["app.py"]),
    ]

    with (
        mock.patch("os.walk", return_value=iter(walk_data)),
        mock.patch("os.path.getsize", return_value=100),
    ):
        results = _walk_repo("/repo", cfg)

    assert any("src/app.py" in r or r.endswith("app.py") for r in results)
    assert not any("node_modules" in r for r in results)


def test_walk_repo_skips_oversized_files():
    CelerySettings, _, _, _, _walk_repo = _get_helpers()
    cfg = CelerySettings()
    max_size = cfg.parse_max_file_size_bytes

    walk_data = [
        ("/repo", [], ["large_file.py"]),
    ]

    with (
        mock.patch("os.walk", return_value=iter(walk_data)),
        mock.patch("os.path.getsize", return_value=max_size + 1),
    ):
        results = _walk_repo("/repo", cfg)

    assert results == []


# ===========================================================================
# LANGUAGE_EXTENSIONS coverage test
# ===========================================================================


def test_language_extensions_dict_has_all_configured_keys():
    _, LANGUAGE_EXTENSIONS, _, _, _ = _get_helpers()
    expected_languages = {
        "python", "javascript", "typescript", "go", "java",
        "c", "cpp", "c_sharp", "kotlin", "php",
    }
    assert expected_languages.issubset(set(LANGUAGE_EXTENSIONS.keys()))
