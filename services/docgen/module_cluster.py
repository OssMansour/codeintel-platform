"""
CodeIntel Platform -- LLM-based Module Clustering
Groups code components into semantic modules using the LLM, inspired by
CodeWiki's cluster_modules approach but using safe JSON parsing instead
of eval().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from services.llm.base import LLMProvider
from services.docgen.prompts import CLUSTER_REPO_PROMPT, CLUSTER_SUB_MODULE_PROMPT
from services.parsing.dependency_graph import ComponentNode

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_COMPONENTS_PER_MODULE = 8
DEFAULT_MAX_TOKEN_PER_MODULE = 12000  # rough char estimate / 4


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ModuleSpec:
    """Specification for a clustered module."""

    name: str
    path: str
    component_ids: list[str]
    description: str = ""
    children: dict[str, "ModuleSpec"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_components(
    llm: LLMProvider,
    components: dict[str, ComponentNode],
    *,
    max_components_per_module: int = DEFAULT_MAX_COMPONENTS_PER_MODULE,
    max_token_per_module: int = DEFAULT_MAX_TOKEN_PER_MODULE,
    module_tree_context: str = "",
    parent_module_name: str | None = None,
) -> dict[str, ModuleSpec]:
    """
    Use the LLM to cluster code components into logical modules.

    If a resulting module is too large (exceeds max_token_per_module),
    recursively sub-cluster it.

    Args:
        llm: LLM provider to use for clustering.
        components: All code components to cluster.
        max_components_per_module: Soft limit before triggering sub-clustering.
        max_token_per_module: Token budget per module (rough).
        module_tree_context: Pretty-printed tree for context in sub-clustering.
        parent_module_name: Name of parent module (for sub-clustering prompt).

    Returns:
        Dict mapping module_name -> ModuleSpec.
    """
    if len(components) <= 2:
        # Too few components to cluster -- return a single module
        name = parent_module_name or "root"
        return {
            name: ModuleSpec(
                name=name,
                path="",
                component_ids=list(components.keys()),
            )
        }

    # Build component listing for the prompt
    comp_list = _format_component_list(components)

    # Choose the right prompt
    if parent_module_name:
        prompt = CLUSTER_SUB_MODULE_PROMPT.format(
            module_name=parent_module_name,
            module_tree=module_tree_context or "(top level)",
            component_list=comp_list,
        )
    else:
        prompt = CLUSTER_REPO_PROMPT.format(component_list=comp_list)

    # Call LLM
    response = llm.complete(prompt, temperature=0.1, max_tokens=2048)
    raw_text = response.content

    # Parse response -- extract JSON from <MODULES> tags
    modules = _parse_cluster_response(raw_text)

    if not modules:
        log.warning("cluster_llm_returned_no_modules", parent=parent_module_name)
        # Fallback: file-based grouping
        modules = _fallback_file_grouping(components)

    # Build ModuleSpec instances
    result: dict[str, ModuleSpec] = {}
    for mod_name, mod_info in modules.items():
        comp_ids = mod_info.get("components", [])
        # Filter to only valid component IDs
        valid_ids = [c for c in comp_ids if c in components]
        if not valid_ids:
            continue

        spec = ModuleSpec(
            name=mod_name,
            path=mod_info.get("path", ""),
            component_ids=valid_ids,
            description=mod_info.get("description", ""),
        )

        # Check if this module needs sub-clustering
        module_token_count = sum(
            len(getattr(components[cid], "source_code", ""))
            for cid in valid_ids
        ) // 4  # rough tokens

        if (
            len(valid_ids) > max_components_per_module
            or module_token_count > max_token_per_module
        ):
            log.info(
                "sub_clustering_module",
                module=mod_name,
                components=len(valid_ids),
                est_tokens=module_token_count,
            )
            sub_components = {cid: components[cid] for cid in valid_ids}
            spec.children = cluster_components(
                llm=llm,
                components=sub_components,
                max_components_per_module=max_components_per_module,
                max_token_per_module=max_token_per_module,
                module_tree_context=module_tree_context,
                parent_module_name=mod_name,
            )

        result[mod_name] = spec

    return result


# ---------------------------------------------------------------------------
# Response parsing (safe -- no eval())
# ---------------------------------------------------------------------------


def _parse_cluster_response(text: str) -> dict[str, Any]:
    """
    Extract module grouping JSON from the LLM response.

    Looks for content between <MODULES>...</MODULES> tags first,
    then falls back to finding any JSON object in the text.
    """
    # Try <MODULES> tags
    tag_match = re.search(r"<MODULES>\s*(.*?)\s*</MODULES>", text, re.DOTALL)
    if tag_match:
        json_str = tag_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            log.warning("modules_tag_json_parse_failed")

    # Try <GROUPED_COMPONENTS> tags (CodeWiki compatibility)
    tag_match = re.search(r"<GROUPED_COMPONENTS>\s*(.*?)\s*</GROUPED_COMPONENTS>", text, re.DOTALL)
    if tag_match:
        json_str = tag_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            log.warning("grouped_components_tag_json_parse_failed")

    # Fallback: find the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            log.warning("freeform_json_parse_failed")

    return {}


# ---------------------------------------------------------------------------
# Fallback grouping (by file path)
# ---------------------------------------------------------------------------


def _fallback_file_grouping(
    components: dict[str, ComponentNode],
) -> dict[str, dict[str, Any]]:
    """
    Group components by their file path when LLM clustering fails.
    """
    from collections import defaultdict

    by_file: dict[str, list[str]] = defaultdict(list)
    for cid, comp in components.items():
        # Use the directory or file stem as module name
        parts = comp.file_path.replace("\\", "/").split("/")
        module = parts[-2] if len(parts) >= 2 else parts[-1].rsplit(".", 1)[0]
        by_file[module].append(cid)

    return {
        name: {"path": name, "components": cids, "description": f"Components from {name}"}
        for name, cids in by_file.items()
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_component_list(components: dict[str, ComponentNode]) -> str:
    """Format components for the clustering prompt."""
    lines: list[str] = []
    for cid, comp in components.items():
        doc_preview = comp.docstring[:80] if comp.docstring else "no docstring"
        lines.append(
            f"- **{cid}** ({comp.component_type}) in `{comp.file_path}` "
            f"[lines {comp.start_line}-{comp.end_line}]: {doc_preview}"
        )
    return "\n".join(lines)


def module_tree_to_dict(modules: dict[str, ModuleSpec]) -> dict[str, Any]:
    """Convert ModuleSpec tree to a plain dict (JSON-serialisable)."""
    result: dict[str, Any] = {}
    for name, spec in modules.items():
        result[name] = {
            "path": spec.path,
            "components": spec.component_ids,
            "description": spec.description,
            "children": module_tree_to_dict(spec.children) if spec.children else {},
        }
    return result
