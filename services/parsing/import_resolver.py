"""
CodeIntel Platform -- Cross-File Import Resolution
Resolves unqualified call names (e.g. ``process_payment``) to fully
qualified component IDs (e.g. ``services.payment.PaymentService.process_payment``)
by analysing import statements extracted during parsing.

This bridges the gap between tree-sitter's local call extraction (which
only captures *names*) and the dependency graph (which needs *qualified IDs*).

Algorithm:
    1. Build an import index from all FileSymbolTables:
       - Maps (file_path, imported_name) -> qualified module path
    2. Build a symbol index from all ComponentNodes:
       - Maps simple_name -> [list of qualified IDs]
    3. For each component, resolve its ``calls`` list:
       - Try import-based resolution first (most accurate)
       - Fall back to same-file resolution (methods in same class)
       - Last resort: global name match (ambiguous, but better than nothing)
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

from services.parsing.dependency_graph import ComponentNode

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ImportEntry:
    """A single resolved import in a source file."""

    source_file: str       # file_path (relative) containing the import
    imported_name: str      # the local name used in code  e.g. 'process_payment'
    module_path: str        # the module being imported from e.g. 'services.payment'
    qualified_name: str     # full dotted name e.g. 'services.payment.process_payment'
    is_wildcard: bool = False


# ---------------------------------------------------------------------------
# Import parser (per-language)
# ---------------------------------------------------------------------------


def extract_imports_from_statements(
    raw_imports: list[str],
    source_file: str,
    language: str,
) -> list[ImportEntry]:
    """
    Parse raw import statement strings into structured ImportEntry objects.

    Args:
        raw_imports: List of raw import strings as extracted by tree-sitter.
        source_file: Relative path of the file containing these imports.
        language: Programming language ('python', 'javascript', etc.).

    Returns:
        List of ImportEntry objects.
    """
    entries: list[ImportEntry] = []

    for stmt in raw_imports:
        if language == "python":
            entries.extend(_parse_python_import(stmt, source_file))
        elif language in ("javascript", "typescript"):
            entries.extend(_parse_js_ts_import(stmt, source_file))
        elif language == "java":
            entries.extend(_parse_java_import(stmt, source_file))
        elif language in ("c_sharp",):
            entries.extend(_parse_csharp_using(stmt, source_file))
        elif language == "go":
            entries.extend(_parse_go_import(stmt, source_file))
        # C/C++: #include doesn't map to callable symbols in the same way
        # Kotlin/PHP: minimal import resolution for now

    return entries


def _parse_python_import(stmt: str, source_file: str) -> list[ImportEntry]:
    """Parse Python import statements.

    Handles:
        import foo
        import foo.bar
        from foo import bar
        from foo.bar import baz, qux
        from foo import *
    """
    entries: list[ImportEntry] = []
    stmt = stmt.strip()

    # from X import Y, Z
    m = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", stmt)
    if m:
        module_path = m.group(1)
        names_str = m.group(2).strip()
        if names_str == "*":
            entries.append(ImportEntry(
                source_file=source_file,
                imported_name="*",
                module_path=module_path,
                qualified_name=f"{module_path}.*",
                is_wildcard=True,
            ))
        else:
            # Handle "bar, baz" and "bar as b, baz as z"
            for name_part in names_str.split(","):
                name_part = name_part.strip()
                if not name_part:
                    continue
                # "bar as b" -> imported_name = "b", original = "bar"
                as_match = re.match(r"(\w+)\s+as\s+(\w+)", name_part)
                if as_match:
                    original = as_match.group(1)
                    alias = as_match.group(2)
                    entries.append(ImportEntry(
                        source_file=source_file,
                        imported_name=alias,
                        module_path=module_path,
                        qualified_name=f"{module_path}.{original}",
                    ))
                else:
                    name = name_part.split()[0]  # safety
                    entries.append(ImportEntry(
                        source_file=source_file,
                        imported_name=name,
                        module_path=module_path,
                        qualified_name=f"{module_path}.{name}",
                    ))
        return entries

    # import X, import X.Y
    m = re.match(r"import\s+(.+)", stmt)
    if m:
        for mod_part in m.group(1).split(","):
            mod_part = mod_part.strip()
            as_match = re.match(r"([\w.]+)\s+as\s+(\w+)", mod_part)
            if as_match:
                module_path = as_match.group(1)
                alias = as_match.group(2)
                entries.append(ImportEntry(
                    source_file=source_file,
                    imported_name=alias,
                    module_path=module_path,
                    qualified_name=module_path,
                ))
            else:
                module_path = mod_part
                # "import foo.bar" makes "foo" available, but "bar" accessible via foo.bar
                short_name = module_path.split(".")[0]
                entries.append(ImportEntry(
                    source_file=source_file,
                    imported_name=short_name,
                    module_path=module_path,
                    qualified_name=module_path,
                ))

    return entries


def _parse_js_ts_import(stmt: str, source_file: str) -> list[ImportEntry]:
    """Parse JS/TS import statements.

    Handles:
        import { foo, bar } from './module'
        import foo from './module'
        import * as foo from './module'
    """
    entries: list[ImportEntry] = []
    stmt = stmt.strip().rstrip(";")

    # import { foo, bar } from '...'
    m = re.match(r"import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", stmt)
    if m:
        names_str = m.group(1)
        module_path = m.group(2)
        for name_part in names_str.split(","):
            name_part = name_part.strip()
            if not name_part:
                continue
            as_match = re.match(r"(\w+)\s+as\s+(\w+)", name_part)
            if as_match:
                original, alias = as_match.group(1), as_match.group(2)
                entries.append(ImportEntry(
                    source_file=source_file,
                    imported_name=alias,
                    module_path=module_path,
                    qualified_name=f"{module_path}.{original}",
                ))
            else:
                name = name_part.strip()
                entries.append(ImportEntry(
                    source_file=source_file,
                    imported_name=name,
                    module_path=module_path,
                    qualified_name=f"{module_path}.{name}",
                ))
        return entries

    # import default from '...'
    m = re.match(r"import\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]", stmt)
    if m:
        entries.append(ImportEntry(
            source_file=source_file,
            imported_name=m.group(1),
            module_path=m.group(2),
            qualified_name=f"{m.group(2)}.default",
        ))

    return entries


def _parse_java_import(stmt: str, source_file: str) -> list[ImportEntry]:
    """Parse Java import statements: import com.foo.Bar;"""
    m = re.match(r"import\s+(?:static\s+)?([\w.]+)\s*;?", stmt.strip())
    if m:
        fqn = m.group(1)
        short = fqn.split(".")[-1]
        module_path = ".".join(fqn.split(".")[:-1])
        return [ImportEntry(
            source_file=source_file,
            imported_name=short,
            module_path=module_path,
            qualified_name=fqn,
            is_wildcard=(short == "*"),
        )]
    return []


def _parse_csharp_using(stmt: str, source_file: str) -> list[ImportEntry]:
    """Parse C# using directives: using System.Linq;"""
    m = re.match(r"using\s+(?:static\s+)?([\w.]+)\s*;?", stmt.strip())
    if m:
        ns = m.group(1)
        short = ns.split(".")[-1]
        return [ImportEntry(
            source_file=source_file,
            imported_name=short,
            module_path=ns,
            qualified_name=ns,
        )]
    return []


def _parse_go_import(stmt: str, source_file: str) -> list[ImportEntry]:
    """Parse Go import statements."""
    entries: list[ImportEntry] = []
    # Single: import "fmt"
    # Multi: import (\n "fmt"\n "os"\n)
    for m in re.finditer(r'"([^"]+)"', stmt):
        pkg = m.group(1)
        short = pkg.split("/")[-1]
        entries.append(ImportEntry(
            source_file=source_file,
            imported_name=short,
            module_path=pkg,
            qualified_name=pkg,
        ))
    return entries


# ---------------------------------------------------------------------------
# Import Index
# ---------------------------------------------------------------------------


class ImportIndex:
    """
    Maps (file_path, local_name) -> possible qualified component IDs.

    Built once from all parsed FileSymbolTables and ComponentNodes,
    then used to resolve calls across files.
    """

    def __init__(self) -> None:
        # file_path -> { local_name -> list of qualified_names from imports }
        self._import_map: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # simple_name -> list of all qualified component IDs with that name
        self._global_name_index: dict[str, list[str]] = defaultdict(list)
        # file_path -> set of qualified component IDs defined in that file
        self._file_symbols: dict[str, set[str]] = defaultdict(set)

    def add_imports(self, entries: list[ImportEntry]) -> None:
        """Register import entries."""
        for entry in entries:
            if not entry.is_wildcard:
                self._import_map[entry.source_file][entry.imported_name].append(
                    entry.qualified_name
                )

    def add_component(self, component: ComponentNode) -> None:
        """Register a component in the global + file indexes."""
        self._global_name_index[component.name].append(component.id)
        self._file_symbols[component.file_path].add(component.id)

    def resolve(
        self,
        call_name: str,
        caller_file: str,
        components: dict[str, ComponentNode],
    ) -> str | None:
        """
        Resolve an unqualified call name to a qualified component ID.

        Resolution priority:
        1. Direct match in components dict (already qualified)
        2. Import-based resolution (most accurate)
        3. Same-file resolution (method in same file)
        4. Global name match (ambiguous -- pick first, log warning)

        Args:
            call_name: The unqualified function/method name.
            caller_file: Relative file path of the calling component.
            components: Full component dict for validation.

        Returns:
            Qualified component ID, or None if unresolvable.
        """
        # 1. Already a valid qualified name
        if call_name in components:
            return call_name

        # 2. Import-based: check imports of the caller's file
        file_imports = self._import_map.get(caller_file, {})
        candidates = file_imports.get(call_name, [])
        for candidate in candidates:
            # Try exact match
            if candidate in components:
                return candidate
            # Try suffix match (import path might differ from component ID)
            for cid in components:
                if cid.endswith(f".{call_name}") and candidate in cid:
                    return cid

        # 3. Same-file resolution
        file_syms = self._file_symbols.get(caller_file, set())
        for sym_id in file_syms:
            if sym_id.endswith(f".{call_name}"):
                return sym_id

        # 4. Global name match (ambiguous)
        global_matches = self._global_name_index.get(call_name, [])
        valid_matches = [m for m in global_matches if m in components]
        if len(valid_matches) == 1:
            return valid_matches[0]
        if len(valid_matches) > 1:
            # Prefer match in the same directory
            caller_dir = os.path.dirname(caller_file)
            for m in valid_matches:
                comp = components[m]
                if os.path.dirname(comp.file_path) == caller_dir:
                    return m
            # Give up and return first match
            return valid_matches[0]

        return None


# ---------------------------------------------------------------------------
# Public API: resolve calls across all components
# ---------------------------------------------------------------------------


def build_import_index(
    components: dict[str, ComponentNode],
    file_imports: dict[str, list[str]] | None = None,
) -> ImportIndex:
    """
    Build an ImportIndex from components and (optionally) raw import
    statements keyed by file path.

    Args:
        components: Dict of component_id -> ComponentNode.
        file_imports: Dict of file_path -> list of raw import strings.
                      If provided, import statements are parsed and added
                      to the index.

    Returns:
        Populated ImportIndex.
    """
    index = ImportIndex()

    # Register all components
    for comp in components.values():
        index.add_component(comp)

    # Parse and register imports
    if file_imports:
        for file_path, raw_stmts in file_imports.items():
            # Detect language from file extension
            lang = _detect_language(file_path)
            entries = extract_imports_from_statements(raw_stmts, file_path, lang)
            index.add_imports(entries)

    return index


def resolve_cross_file_calls(
    components: dict[str, ComponentNode],
    file_imports: dict[str, list[str]] | None = None,
) -> dict[str, ComponentNode]:
    """
    Resolve unqualified call names in all components to qualified IDs
    where possible, and populate ``called_by`` reverse edges.

    This mutates the components in-place and returns the same dict.

    Args:
        components: All parsed components.
        file_imports: Optional raw imports per file.

    Returns:
        The same components dict with resolved calls and populated called_by.
    """
    index = build_import_index(components, file_imports)

    resolved_count = 0
    unresolved_count = 0

    for comp in components.values():
        new_calls: list[str] = []
        for call_name in comp.calls:
            resolved = index.resolve(call_name, comp.file_path, components)
            if resolved:
                new_calls.append(resolved)
                resolved_count += 1
            else:
                # Keep the original unqualified name -- it won't match in
                # build_dependency_graph() but is still useful for docs
                new_calls.append(call_name)
                unresolved_count += 1

        comp.calls = new_calls

    # Populate called_by (reverse edges)
    for comp in components.values():
        for call in comp.calls:
            if call in components:
                target = components[call]
                if comp.id not in target.called_by:
                    target.called_by.append(comp.id)

    log.info(
        "cross_file_call_resolution_complete",
        resolved=resolved_count,
        unresolved=unresolved_count,
        total_components=len(components),
    )

    return components


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_EXT_TO_LANG = {
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
    ".cs": "c_sharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".php": "php",
}


def _detect_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return _EXT_TO_LANG.get(ext, "unknown")
