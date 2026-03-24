"""
Unit tests for services/parsing/parser.py

Tests tree-sitter language loading, language detection, file parsing,
symbol extraction, module docstring extraction, import extraction,
fallback chunking for unsupported extensions, and the get_parser() singleton.

All tests are fully offline — source files are created in pytest tmp_path;
no hardcoded absolute paths or live repo dependencies.

Converted from scripts/test_parser.py into a repeatable pytest suite.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from services.parsing.parser import (
    _load_language,
    parse_file,
    get_parser,
    FileSymbolTable,
    EXTENSION_TO_LANGUAGE,
)

# ===========================================================================
# _load_language — grammar availability
# ===========================================================================


def test_load_language_python_returns_non_none():
    lang = _load_language("python")
    assert lang is not None, "Python tree-sitter grammar must be installed"


def test_load_language_javascript_returns_non_none():
    lang = _load_language("javascript")
    assert lang is not None, "JavaScript tree-sitter grammar must be installed"


def test_load_language_typescript_returns_non_none():
    lang = _load_language("typescript")
    assert lang is not None, "TypeScript tree-sitter grammar must be installed"


def test_load_language_unknown_lang_returns_none():
    lang = _load_language("brainfudge_999")
    assert lang is None


# ===========================================================================
# Language detection by file extension
# ===========================================================================


def test_detect_language_py_is_python():
    assert get_parser().detect_language("module.py") == "python"


def test_detect_language_js_is_javascript():
    assert get_parser().detect_language("app.js") == "javascript"


def test_detect_language_jsx_is_javascript():
    assert get_parser().detect_language("component.jsx") == "javascript"


def test_detect_language_ts_is_typescript():
    assert get_parser().detect_language("index.ts") == "typescript"


def test_detect_language_tsx_is_typescript():
    assert get_parser().detect_language("page.tsx") == "typescript"


def test_detect_language_go_is_go():
    assert get_parser().detect_language("main.go") == "go"


def test_detect_language_java_is_java():
    assert get_parser().detect_language("Main.java") == "java"


def test_detect_language_c_header_is_c():
    assert get_parser().detect_language("utils.h") == "c"


def test_detect_language_cpp_is_cpp():
    assert get_parser().detect_language("engine.cpp") == "cpp"


def test_detect_language_unknown_extension_returns_none():
    assert get_parser().detect_language("script.brainfuck") is None


def test_detect_language_no_extension_returns_none():
    assert get_parser().detect_language("Makefile") is None


def test_extension_map_covers_required_languages():
    required = {"python", "javascript", "typescript", "go", "java", "c", "cpp"}
    found = set(EXTENSION_TO_LANGUAGE.values())
    assert required.issubset(found), f"Extension map missing languages: {required - found}"


# ===========================================================================
# Sample Python source fixture
# ===========================================================================

_PYTHON_SAMPLE = '''\
"""Module-level docstring for testing."""

import os
import sys
from pathlib import Path


class MyClass:
    """A sample class with two methods."""

    def method_a(self, x: int) -> int:
        """Return double x."""
        return x * 2

    def method_b(self) -> None:
        """Do nothing."""
        pass


def standalone_func(a, b):
    """Add two numbers and return the result."""
    return a + b


def another_func():
    pass
'''


@pytest.fixture
def python_tmpfile(tmp_path):
    """Write the sample Python source to a temp file and return its path."""
    f = tmp_path / "sample.py"
    f.write_text(_PYTHON_SAMPLE, encoding="utf-8")
    return str(f)


# ===========================================================================
# parse_file — Python: result type and basic structure
# ===========================================================================


def test_parse_file_returns_file_symbol_table(python_tmpfile):
    result = parse_file(python_tmpfile)
    assert isinstance(result, FileSymbolTable)


def test_parse_file_sets_file_path(python_tmpfile):
    result = parse_file(python_tmpfile)
    assert result.file_path == python_tmpfile


def test_parse_file_detects_python_language(python_tmpfile):
    result = parse_file(python_tmpfile)
    assert result.language == "python"


# ===========================================================================
# parse_file — Python: module docstring
# ===========================================================================


def test_parse_file_extracts_module_docstring(python_tmpfile):
    result = parse_file(python_tmpfile)
    assert "Module-level docstring" in result.module_docstring


# ===========================================================================
# parse_file — Python: imports
# ===========================================================================


def test_parse_file_extracts_at_least_two_imports(python_tmpfile):
    result = parse_file(python_tmpfile)
    assert len(result.imports) >= 2


def test_parse_file_imports_include_os(python_tmpfile):
    result = parse_file(python_tmpfile)
    joined = " ".join(result.imports)
    assert "os" in joined


def test_parse_file_imports_include_sys(python_tmpfile):
    result = parse_file(python_tmpfile)
    joined = " ".join(result.imports)
    assert "sys" in joined


# ===========================================================================
# parse_file — Python: class extraction
# ===========================================================================


def test_parse_file_extracts_myclass(python_tmpfile):
    result = parse_file(python_tmpfile)
    class_names = [c.name for c in result.classes]
    assert "MyClass" in class_names


def test_parse_file_class_has_positive_line_numbers(python_tmpfile):
    result = parse_file(python_tmpfile)
    for cls in result.classes:
        assert cls.start_line >= 1
        assert cls.end_line >= cls.start_line


# ===========================================================================
# parse_file — Python: function extraction
# ===========================================================================


def test_parse_file_extracts_standalone_func(python_tmpfile):
    result = parse_file(python_tmpfile)
    func_names = [f.name for f in result.functions]
    assert "standalone_func" in func_names


def test_parse_file_extracts_another_func(python_tmpfile):
    result = parse_file(python_tmpfile)
    func_names = [f.name for f in result.functions]
    assert "another_func" in func_names


def test_parse_file_standalone_func_has_docstring(python_tmpfile):
    result = parse_file(python_tmpfile)
    func = next((f for f in result.functions if f.name == "standalone_func"), None)
    assert func is not None, "standalone_func not found in parsed functions"
    assert "Add two numbers" in func.docstring


def test_parse_file_function_line_numbers_are_valid(python_tmpfile):
    result = parse_file(python_tmpfile)
    for func in result.functions:
        assert func.start_line >= 1
        assert func.end_line >= func.start_line


# ===========================================================================
# parse_file — Python: all_symbols ordering
# ===========================================================================


def test_parse_file_all_symbols_sorted_by_start_line(python_tmpfile):
    result = parse_file(python_tmpfile)
    symbols = result.all_symbols()
    lines = [s.start_line for s in symbols]
    assert lines == sorted(lines), "all_symbols() must be sorted by start_line"


def test_parse_file_all_symbols_contains_class_and_functions(python_tmpfile):
    result = parse_file(python_tmpfile)
    symbols = result.all_symbols()
    kinds = {s.symbol_type for s in symbols}
    assert "class" in kinds
    assert "function" in kinds


# ===========================================================================
# parse_file — Python: source_lines populated
# ===========================================================================


def test_parse_file_source_lines_populated(python_tmpfile):
    result = parse_file(python_tmpfile)
    assert len(result.source_lines) > 0


# ===========================================================================
# parse_file — fallback for unsupported extension
# ===========================================================================


def test_parse_file_fallback_returns_file_symbol_table(tmp_path):
    """Files with unknown extensions should fall back, not raise."""
    f = tmp_path / "script.sh"
    f.write_text("#!/bin/bash\necho hello\n", encoding="utf-8")
    result = parse_file(str(f))
    assert isinstance(result, FileSymbolTable)
    assert result.file_path == str(f)


def test_parse_file_fallback_language_is_not_python(tmp_path):
    f = tmp_path / "script.sh"
    f.write_text("#!/bin/bash\necho hello\n", encoding="utf-8")
    result = parse_file(str(f))
    assert result.language != "python"


# ===========================================================================
# parse_file — non-existent file returns graceful empty table
# ===========================================================================


def test_parse_file_nonexistent_returns_empty_symbol_table():
    result = parse_file("/nonexistent/__does_not_exist__.py")
    assert isinstance(result, FileSymbolTable)
    assert result.classes == []
    assert result.functions == []


# ===========================================================================
# get_parser() — singleton guarantee
# ===========================================================================


def test_get_parser_returns_same_instance():
    p1 = get_parser()
    p2 = get_parser()
    assert p1 is p2, "get_parser() must return the same singleton instance"
