"""
CodeIntel Platform — tree-sitter Multi-Language Parser
Parses source files into structured symbol tables using tree-sitter.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Language Extension Map
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".php": "php",
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SymbolInfo:
    """
    Represents a single extracted symbol (function, class, or method).

    Attributes:
        name: Simple name (e.g., 'process_payment').
        qualified_name: Fully qualified name (e.g., 'PaymentService.process_payment').
        symbol_type: One of 'function', 'class', 'method'.
        docstring: Extracted docstring if present.
        start_line: 1-indexed first line of the symbol.
        end_line: 1-indexed last line of the symbol.
        parent: Name of the enclosing class (None for top-level).
        calls: List of function/method names called within this symbol.
        params: List of parameter names (for functions/methods).
        body: Full source text of the symbol.
    """

    name: str
    qualified_name: str
    symbol_type: str  # 'function', 'class', 'method'
    docstring: str
    start_line: int
    end_line: int
    parent: str | None = None
    calls: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    body: str = ""


@dataclass
class FileSymbolTable:
    """
    Complete symbol table for a single parsed source file.

    Attributes:
        file_path: Absolute or relative path to the source file.
        language: Detected programming language.
        classes: List of class symbols.
        functions: List of function/method symbols.
        imports: List of import statements as strings.
        module_docstring: Module-level docstring if present.
        source_lines: All source lines (for body extraction).
    """

    file_path: str
    language: str
    classes: list[SymbolInfo] = field(default_factory=list)
    functions: list[SymbolInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    module_docstring: str = ""
    source_lines: list[str] = field(default_factory=list)

    def all_symbols(self) -> list[SymbolInfo]:
        """Return all symbols (classes + functions) sorted by start_line."""
        return sorted(self.classes + self.functions, key=lambda s: s.start_line)


# ---------------------------------------------------------------------------
# Tree-sitter Language Loader
# ---------------------------------------------------------------------------


def _load_language(language: str) -> Any | None:
    """
    Load a tree-sitter language grammar.

    Attempts to import tree-sitter-{language} package first,
    then falls back to tree_sitter_languages if available.

    Returns None if the language cannot be loaded.
    """
    try:
        # Primary: tree-sitter v0.22+ individual packages
        import importlib
        module_name = f"tree_sitter_{language.replace('-', '_')}"
        mod = importlib.import_module(module_name)
        from tree_sitter import Language
        return Language(mod.language())
    except (ImportError, AttributeError):
        pass

    try:
        # Fallback: tree-sitter-languages bundled package
        from tree_sitter_languages import get_language
        return get_language(language)
    except (ImportError, Exception):
        pass

    log.warning("tree_sitter_language_unavailable", language=language)
    return None


# ---------------------------------------------------------------------------
# Language-specific Query Strings
# ---------------------------------------------------------------------------

QUERIES: dict[str, dict[str, str]] = {
    "python": {
        "functions": """
            (function_definition
              name: (identifier) @name
              parameters: (parameters) @params
              body: (block
                (expression_statement
                  (string) @docstring)?
              )
            ) @function
        """,
        "classes": """
            (class_definition
              name: (identifier) @name
              body: (block
                (expression_statement
                  (string) @docstring)?
              )
            ) @class
        """,
        "imports": """
            [(import_statement) (import_from_statement)] @import
        """,
    },
    "javascript": {
        "functions": """
            [
              (function_declaration name: (identifier) @name) @function
              (method_definition name: (property_identifier) @name) @function
              (arrow_function) @function
            ]
        """,
        "classes": """
            (class_declaration name: (identifier) @name) @class
        """,
        "imports": """
            (import_statement) @import
        """,
    },
    "typescript": {
        "functions": """
            [
              (function_declaration name: (identifier) @name) @function
              (method_definition name: (property_identifier) @name) @function
            ]
        """,
        "classes": """
            (class_declaration name: (type_identifier) @name) @class
        """,
        "imports": """
            (import_statement) @import
        """,
    },
    "go": {
        "functions": """
            (function_declaration name: (identifier) @name) @function
        """,
        "classes": """
            (type_declaration
              (type_spec name: (type_identifier) @name)) @class
        """,
        "imports": """
            (import_declaration) @import
        """,
    },
    "java": {
        "functions": """
            (method_declaration name: (identifier) @name) @function
        """,
        "classes": """
            (class_declaration name: (identifier) @name) @class
        """,
        "imports": """
            (import_declaration) @import
        """,
    },
    "c": {
        "functions": """
            (function_definition declarator: (function_declarator
              declarator: (identifier) @name)) @function
        """,
        "classes": """
            (struct_specifier name: (type_identifier) @name) @class
        """,
        "imports": """
            (preproc_include) @import
        """,
    },
    "cpp": {
        "functions": """
            (function_definition declarator: (function_declarator
              declarator: (identifier) @name)) @function
        """,
        "classes": """
            (class_specifier name: (type_identifier) @name) @class
        """,
        "imports": """
            (preproc_include) @import
        """,
    },
    "c_sharp": {
        "functions": """
            (method_declaration name: (identifier) @name) @function
        """,
        "classes": """
            (class_declaration name: (identifier) @name) @class
        """,
        "imports": """
            (using_directive) @import
        """,
    },
    "kotlin": {
        "functions": """
            (function_declaration (simple_identifier) @name) @function
        """,
        "classes": """
            (class_declaration (type_identifier) @name) @class
        """,
        "imports": """
            (import_header) @import
        """,
    },
    "php": {
        "functions": """
            (function_definition name: (name) @name) @function
        """,
        "classes": """
            (class_declaration name: (name) @name) @class
        """,
        "imports": """
            (namespace_use_declaration) @import
        """,
    },
}


# ---------------------------------------------------------------------------
# Parser Implementation
# ---------------------------------------------------------------------------


class MultiLanguageParser:
    """
    tree-sitter based multi-language source file parser.

    Parses Python, JavaScript, TypeScript, Go, Java, C, C++, C#, Kotlin,
    and PHP files into structured FileSymbolTable objects containing all
    functions, classes, and metadata.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Any] = {}
        self._languages: dict[str, Any] = {}

    def _get_parser(self, language: str) -> Any | None:
        """Return a cached tree-sitter parser for the given language."""
        if language in self._parsers:
            return self._parsers[language]

        lang = _load_language(language)
        if lang is None:
            return None

        try:
            from tree_sitter import Parser
            parser = Parser()
            parser.set_language(lang)
            self._parsers[language] = parser
            self._languages[language] = lang
            return parser
        except Exception as exc:
            log.error("parser_init_failed", language=language, error=str(exc))
            return None

    def detect_language(self, file_path: str) -> str | None:
        """
        Detect the programming language from a file extension.

        Args:
            file_path: Path to the source file.

        Returns:
            Language string (e.g., 'python') or None if unsupported.
        """
        ext = Path(file_path).suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(ext)

    def parse_file(self, file_path: str) -> FileSymbolTable:
        """
        Parse a source file and return a complete FileSymbolTable.

        Uses tree-sitter for supported languages. Falls back to
        line-window chunking for unsupported languages.

        Args:
            file_path: Path to the source file to parse.

        Returns:
            FileSymbolTable with all extracted symbols.
        """
        language = self.detect_language(file_path)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except OSError as exc:
            log.error("file_read_failed", file_path=file_path, error=str(exc))
            return FileSymbolTable(
                file_path=file_path,
                language=language or "unknown",
            )

        source_lines = source.splitlines(keepends=True)

        if language is None:
            log.info("unsupported_language_fallback", file_path=file_path)
            return self._fallback_parse(file_path, source, source_lines)

        parser = self._get_parser(language)
        if parser is None:
            log.warning(
                "parser_unavailable_fallback",
                file_path=file_path,
                language=language,
            )
            return self._fallback_parse(file_path, source, source_lines)

        try:
            return self._tree_sitter_parse(
                file_path=file_path,
                language=language,
                source=source,
                source_lines=source_lines,
                parser=parser,
            )
        except Exception as exc:
            log.error(
                "tree_sitter_parse_failed",
                file_path=file_path,
                language=language,
                error=str(exc),
            )
            return self._fallback_parse(file_path, source, source_lines)

    def _tree_sitter_parse(
        self,
        file_path: str,
        language: str,
        source: str,
        source_lines: list[str],
        parser: Any,
    ) -> FileSymbolTable:
        """Run tree-sitter parsing and populate a FileSymbolTable."""
        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node
        lang_obj = self._languages[language]

        table = FileSymbolTable(
            file_path=file_path,
            language=language,
            source_lines=source_lines,
        )

        # Extract module docstring (Python only)
        if language == "python":
            table.module_docstring = self._extract_python_module_docstring(source)

        # Extract imports
        table.imports = self._extract_imports(root, lang_obj, language, source)

        # Extract classes
        class_nodes = self._run_query(root, lang_obj, language, "classes", source)
        for node, name in class_nodes:
            symbol = SymbolInfo(
                name=name,
                qualified_name=name,
                symbol_type="class",
                docstring=self._extract_docstring(node, language, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                body=self._extract_body(source_lines, node.start_point[0], node.end_point[0]),
            )
            table.classes.append(symbol)

        # Extract functions/methods
        func_nodes = self._run_query(root, lang_obj, language, "functions", source)
        for node, name in func_nodes:
            parent = self._find_parent_class(node, table.classes)
            symbol_type = "method" if parent else "function"
            qualified = f"{parent}.{name}" if parent else name
            params = self._extract_params(node, language, source)
            calls = self._extract_calls(node, source)
            body = self._extract_body(source_lines, node.start_point[0], node.end_point[0])

            symbol = SymbolInfo(
                name=name,
                qualified_name=qualified,
                symbol_type=symbol_type,
                docstring=self._extract_docstring(node, language, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent=parent,
                calls=calls,
                params=params,
                body=body,
            )
            table.functions.append(symbol)

        log.debug(
            "file_parsed",
            file_path=file_path,
            language=language,
            functions=len(table.functions),
            classes=len(table.classes),
        )
        return table

    def _run_query(
        self,
        root: Any,
        lang_obj: Any,
        language: str,
        query_type: str,
        source: str,
    ) -> list[tuple[Any, str]]:
        """
        Run a tree-sitter query and return (node, name) pairs.

        Returns empty list if query is not defined for this language.
        """
        from tree_sitter import Language, Query
        query_str = QUERIES.get(language, {}).get(query_type, "")
        if not query_str.strip():
            return []

        try:
            query = lang_obj.query(query_str)
            matches = query.matches(root)
            results = []
            for _pattern_index, capture_dict in matches:
                main_node = capture_dict.get(
                    query_type[:-1] if query_type.endswith("s") else query_type,
                    [None]
                )
                if isinstance(main_node, list):
                    main_node = main_node[0] if main_node else None
                name_node = capture_dict.get("name", [None])
                if isinstance(name_node, list):
                    name_node = name_node[0] if name_node else None

                if main_node and name_node:
                    name = source[name_node.start_byte:name_node.end_byte]
                    results.append((main_node, name))
            return results
        except Exception as exc:
            log.debug("query_failed", language=language, query_type=query_type, error=str(exc))
            return []

    def _extract_imports(
        self, root: Any, lang_obj: Any, language: str, source: str
    ) -> list[str]:
        """Extract import statements as raw strings."""
        query_str = QUERIES.get(language, {}).get("imports", "")
        if not query_str.strip():
            return []
        try:
            query = lang_obj.query(query_str)
            matches = query.matches(root)
            imports = []
            for _, capture_dict in matches:
                node = capture_dict.get("import", [None])
                if isinstance(node, list):
                    node = node[0] if node else None
                if node:
                    imports.append(source[node.start_byte:node.end_byte].strip())
            return imports
        except Exception:
            return []

    def _extract_docstring(self, node: Any, language: str, source: str) -> str:
        """Extract the first docstring from a function or class node."""
        if language == "python":
            # Look for the first expression_statement child with a string
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "expression_statement":
                            for sub in stmt.children:
                                if sub.type == "string":
                                    raw = source[sub.start_byte:sub.end_byte]
                                    return raw.strip("\"'").strip('"""').strip("'''").strip()
        return ""

    def _extract_python_module_docstring(self, source: str) -> str:
        """Extract the module-level docstring from Python source."""
        stripped = source.strip()
        for quote in ('"""', "'''", '"', "'"):
            if stripped.startswith(quote):
                end = stripped.find(quote, len(quote))
                if end != -1:
                    return stripped[len(quote):end].strip()
        return ""

    def _find_parent_class(self, node: Any, classes: list[SymbolInfo]) -> str | None:
        """Find which class (if any) contains the given function node."""
        func_start = node.start_point[0] + 1
        func_end = node.end_point[0] + 1
        for cls in classes:
            if cls.start_line <= func_start and cls.end_line >= func_end:
                return cls.name
        return None

    def _extract_params(self, node: Any, language: str, source: str) -> list[str]:
        """Extract parameter names from a function/method node."""
        params: list[str] = []
        try:
            for child in node.children:
                if child.type in ("parameters", "formal_parameters", "parameter_list"):
                    param_text = source[child.start_byte:child.end_byte]
                    # Simple regex extraction of identifiers
                    raw_params = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", param_text)
                    # Filter out keywords
                    keywords = {"self", "cls", "def", "class", "int", "str", "float", "bool", "None"}
                    params = [p for p in raw_params if p not in keywords]
        except Exception:
            pass
        return params

    def _extract_calls(self, node: Any, source: str) -> list[str]:
        """
        Extract names of functions called within a node's body.

        Uses a simple traversal to find call expressions.
        """
        calls: set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type == "call":
                # Get function name from call expression
                for child in current.children:
                    if child.type in ("identifier", "attribute"):
                        call_text = source[child.start_byte:child.end_byte]
                        # For attribute calls like obj.method, take just method name
                        if "." in call_text:
                            call_text = call_text.split(".")[-1]
                        calls.add(call_text)
                        break
            stack.extend(current.children)
        return list(calls)

    def _extract_body(
        self, source_lines: list[str], start_row: int, end_row: int
    ) -> str:
        """Extract the source body text for a given line range (0-indexed)."""
        return "".join(source_lines[start_row : end_row + 1])

    def _fallback_parse(
        self,
        file_path: str,
        source: str,
        source_lines: list[str],
    ) -> FileSymbolTable:
        """
        Fallback parser for unsupported languages.

        Splits the file into 50-line windows and creates synthetic 'chunk'
        symbols for each window.
        """
        language = self.detect_language(file_path) or "unknown"
        table = FileSymbolTable(
            file_path=file_path,
            language=language,
            source_lines=source_lines,
        )

        chunk_size = int(os.getenv("PARSE_FALLBACK_CHUNK_LINES", "50"))
        total_lines = len(source_lines)

        for i, start in enumerate(range(0, total_lines, chunk_size)):
            end = min(start + chunk_size - 1, total_lines - 1)
            body = "".join(source_lines[start : end + 1])
            symbol = SymbolInfo(
                name=f"chunk_{i}",
                qualified_name=f"chunk_{i}",
                symbol_type="function",  # treated as function for chunking purposes
                docstring="",
                start_line=start + 1,
                end_line=end + 1,
                body=body,
            )
            table.functions.append(symbol)

        return table


def extract_function_body(file_path: str, start_line: int, end_line: int) -> str:
    """
    Extract the source text of a symbol by line range.

    Args:
        file_path: Path to the source file.
        start_line: First line (1-indexed, inclusive).
        end_line: Last line (1-indexed, inclusive).

    Returns:
        Source text of the specified line range.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[start_line - 1 : end_line])
    except OSError as exc:
        log.error(
            "extract_body_failed",
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            error=str(exc),
        )
        return ""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_parser_instance: MultiLanguageParser | None = None


def get_parser() -> MultiLanguageParser:
    """Return the module-level parser singleton."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = MultiLanguageParser()
    return _parser_instance


def parse_file(file_path: str) -> FileSymbolTable:
    """
    Module-level convenience function for parsing a single file.

    Args:
        file_path: Path to the source file to parse.

    Returns:
        FileSymbolTable with all extracted symbols.
    """
    return get_parser().parse_file(file_path)
