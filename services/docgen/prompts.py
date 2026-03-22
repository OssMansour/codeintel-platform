"""
CodeIntel Platform -- Documentation Prompt Templates
All prompts for the hierarchical wiki generation pipeline.

Inspired by CodeWiki's prompt architecture but adapted for local models
with tighter token budgets and explicit structure enforcement.
"""

# ---------------------------------------------------------------------------
# Complex module prompt (can delegate to sub-agents)
# ---------------------------------------------------------------------------

COMPLEX_MODULE_SYSTEM_PROMPT = """\
You are an expert documentation writer for the {module_name} module.
Your task is to generate comprehensive system documentation based on a given module name and its core code components.

Goals:
1. Explain the module's purpose and core functionality
2. Describe architecture and component relationships
3. Show how the module fits into the overall system

Documentation structure to produce:
1. Brief introduction and purpose
2. Architecture overview with Mermaid diagrams
3. High-level functionality of each sub-module
4. Cross-references to related module documentation (use relative links like [ModuleName](ModuleName.md))

Diagram requirements:
- Include at least ONE Mermaid diagram (architecture, data flow, or component interaction)
- Use ```mermaid code fences
- Keep diagrams readable (max ~15 nodes)

Output format: Return ONLY valid Markdown. No preamble, no commentary.
{custom_instructions}
"""

# ---------------------------------------------------------------------------
# Leaf module prompt (generates docs directly, no sub-delegation)
# ---------------------------------------------------------------------------

LEAF_MODULE_SYSTEM_PROMPT = """\
You are an expert documentation writer for the {module_name} module.
Your task is to generate comprehensive documentation for this leaf-level module.

Goals:
1. Explain the module's purpose and core functionality
2. Describe key classes, functions, and their relationships
3. Show how the module fits into the overall system

Documentation requirements:
1. Brief introduction and purpose
2. Key components with descriptions
3. At least ONE Mermaid diagram (class diagram, sequence diagram, or data flow)
4. Usage patterns or examples where relevant
5. Cross-references to dependencies (use relative links)

Diagram requirements:
- Use ```mermaid code fences
- Keep diagrams focused and readable

Output format: Return ONLY valid Markdown. No preamble, no commentary.
{custom_instructions}
"""

# ---------------------------------------------------------------------------
# User prompt (provides context to the agent)
# ---------------------------------------------------------------------------

MODULE_USER_PROMPT = """\
Generate comprehensive documentation for the **{module_name}** module.

## Module Tree (project structure)
{module_tree}

## Core Components
{component_codes}
"""

# ---------------------------------------------------------------------------
# Module overview (for parent modules summarizing children)
# ---------------------------------------------------------------------------

MODULE_OVERVIEW_PROMPT = """\
You are an expert documentation writer. Generate a module overview for **{module_name}** based on the documentation of its sub-modules.

The overview should include:
- The purpose of the module
- Architecture visualized with a Mermaid diagram
- Brief summary of each sub-module with links to their docs

Sub-module documentation:
{sub_module_docs}

Module tree context:
{module_tree}

Output format: Return ONLY valid Markdown with at least one Mermaid diagram.
"""

# ---------------------------------------------------------------------------
# Repository overview (top-level summary)
# ---------------------------------------------------------------------------

REPO_OVERVIEW_PROMPT = """\
You are an expert documentation writer. Generate a repository overview for **{repo_name}**.

The overview should include:
- The purpose of the repository
- End-to-end architecture visualized with Mermaid diagrams
- Brief description of each core module with links to their documentation

Repository structure:
{repo_structure}

Core module documentation:
{module_summaries}

Output format: Return ONLY valid Markdown with at least one architecture Mermaid diagram.
"""

# ---------------------------------------------------------------------------
# Module clustering prompt (groups components into semantic modules via LLM)
# ---------------------------------------------------------------------------

CLUSTER_REPO_PROMPT = """\
Here are the core code components of this repository:

<COMPONENTS>
{component_list}
</COMPONENTS>

Group these components into logical modules. Each module should contain
closely related components that together form a coherent unit.

Rules:
- Each module should have a clear, descriptive name (snake_case)
- Omit trivial components (e.g. __init__.py boilerplate)
- A module should have 2-8 components; split larger groups

Return your answer as JSON inside <MODULES> tags:
<MODULES>
{{
    "module_name_1": {{
        "path": "path/to/module",
        "components": ["component.id.1", "component.id.2"],
        "description": "Brief purpose"
    }},
    "module_name_2": {{
        "path": "path/to/module",
        "components": ["component.id.3"],
        "description": "Brief purpose"
    }}
}}
</MODULES>
"""

CLUSTER_SUB_MODULE_PROMPT = """\
The module **{module_name}** contains the following components which need
to be organized into sub-modules:

Module tree so far:
{module_tree}

<COMPONENTS>
{component_list}
</COMPONENTS>

Group these components into smaller sub-modules (2-6 components each).

Return your answer as JSON inside <MODULES> tags:
<MODULES>
{{
    "sub_module_name": {{
        "path": "path/to/sub_module",
        "components": ["component.id.1", "component.id.2"],
        "description": "Brief purpose"
    }}
}}
</MODULES>
"""

# ---------------------------------------------------------------------------
# Function-level docstring (kept from original DocGenerator)
# ---------------------------------------------------------------------------

FUNCTION_DOC_PROMPT = """\
You are an expert software documentation writer. Generate a clear, precise docstring for the following function.

## Function Source Code
```{language}
{source_code}
```

## Called Functions (context)
{callee_docs}

## Developer Context
{developer_context}

## Instructions
Write a Google-style docstring:
1. One-line summary
2. Args section (name, type, description)
3. Returns section
4. Raises section (if applicable)

Return ONLY the docstring text (no triple-quotes). Under 200 words.
"""

# ---------------------------------------------------------------------------
# Module-level summary (kept from original DocGenerator)
# ---------------------------------------------------------------------------

FILE_MODULE_DOC_PROMPT = """\
You are an expert software architect documenting a codebase module.

## Module: {file_path}
## Language: {language}

## Functions and Classes
{symbol_summaries}

Write a Markdown module document that includes:
1. **Purpose**: What this module does
2. **Key Components**: Brief description of each class/function
3. **Dependencies**: What this module depends on
4. **Architecture**: A Mermaid diagram showing component relationships
5. **Important Notes**: Gotchas, performance, design decisions

Keep it under 500 words. Return ONLY Markdown.
"""

# ---------------------------------------------------------------------------
# Subsystem-level summary (kept from original DocGenerator)
# ---------------------------------------------------------------------------

SUBSYSTEM_DOC_PROMPT = """\
You are an expert software architect documenting a subsystem.

## Subsystem: {subsystem_name}

## Module Documentation
{module_docs}

Write a Markdown subsystem document:
1. **Overview**: Purpose and role in the larger system
2. **Architecture**: Mermaid diagram showing module relationships and data flow
3. **Key Interfaces**: Public APIs and entry points
4. **Configuration**: Key settings
5. **Known Issues**: Limitations, tech debt

Keep it under 800 words. Return ONLY Markdown.
"""

# ---------------------------------------------------------------------------
# Iterative doc refinement (Phase 3)
# ---------------------------------------------------------------------------

DOC_CRITIQUE_PROMPT = """\
You are a senior technical documentation reviewer. Evaluate the following \
generated documentation for the **{module_name}** module.

<DOCUMENT>
{document}
</DOCUMENT>

Rate the document on each criterion (1-5) and list specific issues:

1. **Completeness** — Are all key components explained?
2. **Accuracy** — Does the text match the source code provided?
3. **Diagrams** — Is there at least one correct Mermaid diagram?
4. **Clarity** — Is the writing concise and well-structured?
5. **Cross-references** — Are related modules linked properly?

Return your review in this exact JSON format inside <REVIEW> tags:
<REVIEW>
{{
    "scores": {{
        "completeness": <int>,
        "accuracy": <int>,
        "diagrams": <int>,
        "clarity": <int>,
        "cross_references": <int>
    }},
    "overall": <float>,
    "issues": [
        "Issue 1 description",
        "Issue 2 description"
    ],
    "pass": <bool>
}}
</REVIEW>

Set "pass" to true if overall >= 4.0 and every individual score >= 3.
"""

DOC_REFINE_PROMPT = """\
You are an expert documentation writer. Improve the following documentation \
for the **{module_name}** module based on the reviewer's feedback.

<ORIGINAL_DOCUMENT>
{document}
</ORIGINAL_DOCUMENT>

<REVIEW_ISSUES>
{issues}
</REVIEW_ISSUES>

{source_context}

Fix ALL listed issues while preserving the overall structure. \
Ensure Mermaid diagrams use valid syntax and are wrapped in \
```mermaid fences. Keep existing good content — only improve what \
the review flagged.

Output format: Return ONLY the improved Markdown document. No preamble.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE = {
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
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".php": "php",
    ".md": "markdown",
    ".sh": "bash",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def format_component_codes(
    component_ids: list[str],
    components: dict,
) -> str:
    """
    Build a formatted code listing for the given components,
    grouped by source file.
    """
    from collections import defaultdict

    by_file: dict[str, list[str]] = defaultdict(list)
    for cid in component_ids:
        comp = components.get(cid)
        if comp is None:
            continue
        fpath = getattr(comp, "file_path", "") or cid.rsplit(".", 1)[0]
        by_file[fpath].append(cid)

    parts: list[str] = []
    for fpath, cids in by_file.items():
        ext = "." + fpath.rsplit(".", 1)[-1] if "." in fpath else ".py"
        lang = EXTENSION_TO_LANGUAGE.get(ext, "text")
        parts.append(f"### File: {fpath}")
        for cid in cids:
            comp = components.get(cid)
            if comp is None:
                parts.append(f"  - {cid}: (not found)")
                continue
            src = getattr(comp, "source_code", "") or getattr(comp, "body", "")
            parts.append(f"#### {cid}")
            parts.append(f"```{lang}")
            parts.append(src.strip())
            parts.append("```")
        parts.append("")

    return "\n".join(parts)


def format_module_tree(tree: dict, indent: int = 0) -> str:
    """Pretty-print a module tree dict for inclusion in prompts."""
    lines: list[str] = []
    for name, info in tree.items():
        prefix = "  " * indent
        comp_list = info.get("components", [])
        lines.append(f"{prefix}- **{name}** ({len(comp_list)} components)")
        children = info.get("children", {})
        if children:
            lines.append(format_module_tree(children, indent + 1))
    return "\n".join(lines)
