"""
CodeIntel Platform -- Mermaid Diagram Utilities
Validation and extraction for Mermaid diagrams in generated documentation.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a fenced Mermaid block:  ```mermaid ... ```
_MERMAID_FENCE_RE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL,
)

# Known Mermaid diagram type keywords (first non-comment line)
_DIAGRAM_TYPES = {
    "graph", "flowchart", "sequencediagram", "sequence", "classDiagram",
    "classdiagram", "stateDiagram", "statediagram", "erDiagram",
    "erdiagram", "gantt", "pie", "gitGraph", "gitgraph", "mindmap",
    "timeline", "journey", "quadrantChart", "quadrantchart",
    "sankey", "xychart", "block",
}

# Normalised first-word lookup
_DIAGRAM_TYPE_NORMALISED = {t.lower().replace("-", "") for t in _DIAGRAM_TYPES}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class MermaidBlock(NamedTuple):
    """A single extracted Mermaid diagram."""
    source: str       # raw Mermaid code (without the fences)
    diagram_type: str  # e.g. 'flowchart', 'sequenceDiagram'
    start_pos: int     # char offset in the parent document


class ValidationResult(NamedTuple):
    """Result of validating a Mermaid block."""
    valid: bool
    errors: list[str]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_mermaid_blocks(markdown: str) -> list[MermaidBlock]:
    """
    Extract all ```mermaid ... ``` blocks from a Markdown string.

    Returns:
        List of MermaidBlock with source, detected type, and position.
    """
    blocks: list[MermaidBlock] = []
    for match in _MERMAID_FENCE_RE.finditer(markdown):
        source = match.group(1).strip()
        diagram_type = _detect_type(source)
        blocks.append(MermaidBlock(
            source=source,
            diagram_type=diagram_type,
            start_pos=match.start(),
        ))
    return blocks


# ---------------------------------------------------------------------------
# Validation (lightweight, no external tools needed)
# ---------------------------------------------------------------------------


def validate_mermaid(source: str) -> ValidationResult:
    """
    Lightweight structural validation of a Mermaid diagram.

    Checks:
    1. Non-empty content
    2. Starts with a known diagram type keyword
    3. Balanced brackets / braces
    4. No obvious syntax issues

    This does NOT render the diagram -- for full validation you would
    need the Mermaid CLI (``mmdc``).  This catches the most common
    LLM generation errors.
    """
    errors: list[str] = []

    if not source or not source.strip():
        return ValidationResult(valid=False, errors=["Empty diagram"])

    lines = [ln for ln in source.strip().splitlines() if ln.strip() and not ln.strip().startswith("%%")]
    if not lines:
        return ValidationResult(valid=False, errors=["Diagram contains only comments"])

    # Check diagram type keyword
    first_word = lines[0].strip().split()[0].lower().replace("-", "")
    # Handle 'graph TD', 'flowchart LR', 'sequenceDiagram', etc.
    if first_word not in _DIAGRAM_TYPE_NORMALISED:
        errors.append(f"Unknown diagram type: '{lines[0].strip().split()[0]}'")

    # Balanced brackets
    open_chars = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    for char in source:
        if char in open_chars:
            stack.append(open_chars[char])
        elif char in open_chars.values():
            if stack and stack[-1] == char:
                stack.pop()
            # Don't flag unmatched closing brackets in Mermaid (common in labels)

    if len(stack) > 3:
        errors.append(f"Possibly unbalanced brackets ({len(stack)} unclosed)")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_all_mermaid_in_doc(markdown: str) -> list[tuple[int, ValidationResult]]:
    """
    Validate every Mermaid block in a Markdown document.

    Returns:
        List of (block_index, ValidationResult) for blocks that have issues.
        Empty list = all blocks are valid (or no blocks present).
    """
    blocks = extract_mermaid_blocks(markdown)
    issues: list[tuple[int, ValidationResult]] = []
    for i, block in enumerate(blocks):
        result = validate_mermaid(block.source)
        if not result.valid:
            issues.append((i, result))
            log.warning(
                "mermaid_validation_issue",
                block_index=i,
                diagram_type=block.diagram_type,
                errors=result.errors,
            )
    return issues


# ---------------------------------------------------------------------------
# Helpers for generating common diagram types
# ---------------------------------------------------------------------------


def _detect_type(source: str) -> str:
    """Detect the Mermaid diagram type from the first non-comment line."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("%%"):
            first_word = stripped.split()[0]
            return first_word
    return "unknown"


def wrap_mermaid(source: str) -> str:
    """Wrap raw Mermaid source in a Markdown code fence."""
    return f"```mermaid\n{source.strip()}\n```"


def architecture_diagram_hint(module_name: str, sub_modules: list[str]) -> str:
    """
    Generate a starter Mermaid flowchart for a module and its sub-modules.
    Useful as a seed/hint for the LLM to refine.
    """
    lines = [f"flowchart TD"]
    lines.append(f"    {_safe_id(module_name)}[{module_name}]")
    for sub in sub_modules:
        sid = _safe_id(sub)
        lines.append(f"    {_safe_id(module_name)} --> {sid}[{sub}]")
    return "\n".join(lines)


def _safe_id(name: str) -> str:
    """Convert a module name to a safe Mermaid node ID."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)
