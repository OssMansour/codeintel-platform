"""
CodeIntel Platform -- Hierarchical Wiki Generator
Recursive multi-level documentation generation inspired by CodeWiki's
approach, adapted for on-premises models with tight token budgets.

Pipeline:
    1. Parse repository into ComponentNodes
    2. Build dependency graph (with Tarjan cycle resolution)
    3. Cluster components into semantic modules via LLM
    4. Process modules in topological order (leaves first)
        - Leaf modules: generate docs directly
        - Complex modules: recursively process sub-modules, then summarise
    5. Generate repository overview from all module docs
    6. Validate all Mermaid diagrams
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from services.llm.base import ChatMessage, LLMProvider
from services.docgen.mermaid import validate_all_mermaid_in_doc
from services.docgen.module_cluster import (
    ModuleSpec,
    cluster_components,
    module_tree_to_dict,
)
from services.docgen.prompts import (
    COMPLEX_MODULE_SYSTEM_PROMPT,
    LEAF_MODULE_SYSTEM_PROMPT,
    MODULE_OVERVIEW_PROMPT,
    MODULE_USER_PROMPT,
    REPO_OVERVIEW_PROMPT,
    format_component_codes,
    format_module_tree,
)
from services.parsing.dependency_graph import (
    ComponentNode,
    build_dependency_graph,
    get_leaf_nodes,
    symbols_to_components,
    topological_sort,
)
from services.parsing.import_resolver import resolve_cross_file_calls

log = structlog.get_logger(__name__)

# Lazy import to avoid circular deps at module load
_DocRefiner = None

def _get_refiner_class():
    global _DocRefiner
    if _DocRefiner is None:
        from services.docgen.doc_refiner import DocRefiner
        _DocRefiner = DocRefiner
    return _DocRefiner


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WikiGenConfig:
    """Configuration for the wiki generation pipeline."""

    # Output
    docs_dir: str = "/data/wiki"

    # Recursion
    max_depth: int = 3                        # Max sub-module nesting
    max_components_per_module: int = 8
    max_token_per_module: int = 12000

    # LLM budget
    max_doc_tokens: int = 2048               # Max tokens per doc generation call
    context_char_budget: int = 48000          # ~12k tokens of context per call

    # Iterative refinement (Phase 3)
    refine_iterations: int = 0               # 0 = disabled, 1-3 recommended
    refine_pass_threshold: float = 4.0       # Min overall score to accept

    # Behaviour
    validate_mermaid: bool = True
    custom_instructions: str = ""
    skip_existing: bool = False               # Resume interrupted generation


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ModuleDoc:
    """Generated documentation for a single module."""

    module_name: str
    content: str                              # Markdown content
    file_path: str                            # Where it was saved
    component_ids: list[str] = field(default_factory=list)
    mermaid_count: int = 0
    mermaid_issues: int = 0
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class WikiResult:
    """Full result of a wiki generation run."""

    repo_name: str
    overview: str = ""
    modules: list[ModuleDoc] = field(default_factory=list)
    module_tree: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wiki Generator
# ---------------------------------------------------------------------------


class WikiGenerator:
    """
    Hierarchical documentation generator.

    Inspired by CodeWiki's DocumentationGenerator + AgentOrchestrator,
    adapted to work with any LLMProvider (local or cloud).

    Instead of spawning pydantic-ai sub-agents, we use recursive function
    calls with different prompts -- same logical pattern, simpler execution.
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: WikiGenConfig | None = None,
    ) -> None:
        self._llm = llm
        self._cfg = config or WikiGenConfig()
        self._generated_docs: dict[str, ModuleDoc] = {}  # module_name -> doc

        # Initialise refiner if iterations > 0
        self._refiner = None
        if self._cfg.refine_iterations > 0:
            RefinerCls = _get_refiner_class()
            self._refiner = RefinerCls(
                llm=self._llm,
                max_iterations=self._cfg.refine_iterations,
                pass_threshold=self._cfg.refine_pass_threshold,
                max_refine_tokens=self._cfg.max_doc_tokens,
            )

    # ── public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        repo_path: str,
        project_id: str,
        *,
        repo_name: str | None = None,
    ) -> WikiResult:
        """
        Run the full wiki generation pipeline.

        Args:
            repo_path: Absolute path to the cloned repository.
            project_id: GitLab project identifier.
            repo_name: Human-readable repo name (defaults to dir name).

        Returns:
            WikiResult with all generated documentation.
        """
        t0 = time.perf_counter()
        repo_name = repo_name or Path(repo_path).name

        # Prepare output directory
        os.makedirs(self._cfg.docs_dir, exist_ok=True)

        # Step 1: Parse repository into components
        log.info("wiki_step_1_parsing", repo=repo_path)
        components = self._parse_repository(repo_path, project_id)
        if not components:
            log.warning("wiki_no_components_found")
            return WikiResult(repo_name=repo_name, stats={"error": "no components found"})

        # Step 2: Build dependency graph
        log.info("wiki_step_2_dependency_graph", components=len(components))
        dep_graph = build_dependency_graph(components)

        # Step 3: Cluster into modules via LLM
        log.info("wiki_step_3_clustering")
        module_tree = cluster_components(
            llm=self._llm,
            components=components,
            max_components_per_module=self._cfg.max_components_per_module,
            max_token_per_module=self._cfg.max_token_per_module,
        )

        tree_dict = module_tree_to_dict(module_tree)
        tree_path = os.path.join(self._cfg.docs_dir, "module_tree.json")
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(tree_dict, f, indent=2)

        # Step 4: Process modules in topological order (leaves first)
        log.info("wiki_step_4_generating_docs", modules=len(module_tree))
        processing_order = self._get_processing_order(module_tree)

        for module_path, module_name, spec in processing_order:
            self._process_module(
                spec=spec,
                components=components,
                module_tree=tree_dict,
                depth=0,
            )

        # Step 5: Generate repository overview
        log.info("wiki_step_5_repo_overview")
        overview = self._generate_repo_overview(repo_name, tree_dict)
        overview_path = os.path.join(self._cfg.docs_dir, "overview.md")
        self._save_doc(overview_path, overview)

        # Step 6: Collect results
        elapsed = (time.perf_counter() - t0) * 1000.0
        result = WikiResult(
            repo_name=repo_name,
            overview=overview,
            modules=list(self._generated_docs.values()),
            module_tree=tree_dict,
            stats={
                "total_components": len(components),
                "total_modules": len(self._generated_docs),
                "total_latency_ms": elapsed,
                "provider": self._llm.provider_name,
                "model": self._llm.model_name,
            },
        )

        # Save metadata
        meta_path = os.path.join(self._cfg.docs_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(result.stats, f, indent=2)

        log.info(
            "wiki_generation_complete",
            modules=len(self._generated_docs),
            elapsed_ms=round(elapsed),
        )
        return result

    # ── module processing ───────────────────────────────────────────────────

    def _process_module(
        self,
        spec: ModuleSpec,
        components: dict[str, ComponentNode],
        module_tree: dict[str, Any],
        depth: int,
    ) -> ModuleDoc:
        """
        Generate documentation for a single module.

        If the module has children (complex), process children first,
        then generate a parent overview.  Otherwise, generate leaf docs.
        """
        module_name = spec.name

        # Skip if already generated (or resuming)
        if module_name in self._generated_docs:
            return self._generated_docs[module_name]

        if self._cfg.skip_existing:
            existing_path = os.path.join(self._cfg.docs_dir, f"{module_name}.md")
            if os.path.exists(existing_path):
                log.info("wiki_skipping_existing", module=module_name)
                content = Path(existing_path).read_text(encoding="utf-8")
                doc = ModuleDoc(module_name=module_name, content=content, file_path=existing_path)
                self._generated_docs[module_name] = doc
                return doc

        indent = "  " * depth
        arrow = "+-" if depth > 0 else ">"

        if spec.children and depth < self._cfg.max_depth:
            # ── Complex module: process children first ──────────────────
            log.info(f"{indent}{arrow} Processing complex module: {module_name}")

            child_docs: list[ModuleDoc] = []
            for child_name, child_spec in spec.children.items():
                child_doc = self._process_module(
                    spec=child_spec,
                    components=components,
                    module_tree=module_tree,
                    depth=depth + 1,
                )
                child_docs.append(child_doc)

            # Generate parent overview from child docs
            doc = self._generate_parent_doc(module_name, child_docs, module_tree)

        else:
            # ── Leaf module: generate docs directly ─────────────────────
            log.info(f"{indent}{arrow} Generating leaf docs: {module_name}")
            doc = self._generate_leaf_doc(module_name, spec, components, module_tree)

        self._generated_docs[module_name] = doc

        # Save to disk
        out_path = os.path.join(self._cfg.docs_dir, f"{module_name}.md")
        self._save_doc(out_path, doc.content)
        doc.file_path = out_path

        return doc

    def _generate_leaf_doc(
        self,
        module_name: str,
        spec: ModuleSpec,
        components: dict[str, ComponentNode],
        module_tree: dict[str, Any],
    ) -> ModuleDoc:
        """Generate documentation for a leaf module (no children)."""
        # Build context
        comp_codes = format_component_codes(spec.component_ids, components)
        tree_str = format_module_tree(module_tree)

        # Truncate to budget
        comp_codes = self._truncate(comp_codes, self._cfg.context_char_budget)
        tree_str = self._truncate(tree_str, 4000)

        system_prompt = LEAF_MODULE_SYSTEM_PROMPT.format(
            module_name=module_name,
            custom_instructions=self._cfg.custom_instructions,
        )
        user_prompt = MODULE_USER_PROMPT.format(
            module_name=module_name,
            module_tree=tree_str,
            component_codes=comp_codes,
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        resp = self._llm.chat(messages, max_tokens=self._cfg.max_doc_tokens)

        content = resp.content
        total_tokens = resp.total_tokens
        total_latency = resp.latency_ms

        # Iterative refinement (Phase 3)
        if self._refiner is not None:
            content, _reviews = self._refiner.refine(
                module_name=module_name,
                document=content,
                source_context=self._truncate(comp_codes, 8000),
            )

        # Validate Mermaid
        mermaid_issues = 0
        mermaid_count = 0
        if self._cfg.validate_mermaid:
            issues = validate_all_mermaid_in_doc(content)
            from services.docgen.mermaid import extract_mermaid_blocks
            mermaid_count = len(extract_mermaid_blocks(content))
            mermaid_issues = len(issues)

        return ModuleDoc(
            module_name=module_name,
            content=content,
            file_path="",
            component_ids=spec.component_ids,
            mermaid_count=mermaid_count,
            mermaid_issues=mermaid_issues,
            tokens_used=total_tokens,
            latency_ms=total_latency,
        )

    def _generate_parent_doc(
        self,
        module_name: str,
        child_docs: list[ModuleDoc],
        module_tree: dict[str, Any],
    ) -> ModuleDoc:
        """Generate an overview doc for a parent module from its child docs."""
        # Collect child doc summaries (truncated)
        child_summaries: list[str] = []
        for cd in child_docs:
            preview = cd.content[:1500] if cd.content else "(no content)"
            child_summaries.append(f"### {cd.module_name}\n{preview}")

        sub_module_docs = "\n\n---\n\n".join(child_summaries)
        tree_str = format_module_tree(module_tree)

        prompt = MODULE_OVERVIEW_PROMPT.format(
            module_name=module_name,
            sub_module_docs=self._truncate(sub_module_docs, self._cfg.context_char_budget),
            module_tree=self._truncate(tree_str, 4000),
        )

        resp = self._llm.complete(prompt, max_tokens=self._cfg.max_doc_tokens)

        content = resp.content

        # Iterative refinement (Phase 3)
        if self._refiner is not None:
            content, _reviews = self._refiner.refine(
                module_name=module_name,
                document=content,
            )

        return ModuleDoc(
            module_name=module_name,
            content=content,
            file_path="",
            component_ids=[],
            tokens_used=resp.total_tokens,
            latency_ms=resp.latency_ms,
        )

    def _generate_repo_overview(
        self,
        repo_name: str,
        module_tree: dict[str, Any],
    ) -> str:
        """Generate a top-level repository overview document."""
        # Collect summaries of all top-level modules
        module_summaries: list[str] = []
        for mod_name in module_tree:
            doc = self._generated_docs.get(mod_name)
            if doc:
                preview = doc.content[:800] if doc.content else ""
                module_summaries.append(f"### [{mod_name}]({mod_name}.md)\n{preview}")

        repo_structure = format_module_tree(module_tree)

        prompt = REPO_OVERVIEW_PROMPT.format(
            repo_name=repo_name,
            repo_structure=self._truncate(repo_structure, 4000),
            module_summaries=self._truncate(
                "\n\n".join(module_summaries), self._cfg.context_char_budget
            ),
        )

        resp = self._llm.complete(prompt, max_tokens=self._cfg.max_doc_tokens)
        return resp.content

    # ── repository parsing ──────────────────────────────────────────────────

    def _parse_repository(
        self,
        repo_path: str,
        project_id: str,
    ) -> dict[str, ComponentNode]:
        """
        Walk the repository and parse all supported source files into
        ComponentNode instances, then resolve cross-file calls.
        """
        from services.parsing.parser import parse_file

        SKIP_DIRS = {
            ".git", ".svn", "node_modules", "vendor", "__pycache__",
            ".venv", "venv", "env", ".env", "dist", "build",
            ".cache", ".pytest_cache", "target",
        }
        SUPPORTED_EXTS = {
            ".py", ".js", ".jsx", ".ts", ".tsx", ".go",
            ".java", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
            ".cs", ".kt", ".kts", ".php",
        }

        all_symbols: list[dict[str, Any]] = []
        file_imports: dict[str, list[str]] = {}  # file_path -> import stmts

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in SUPPORTED_EXTS:
                    continue

                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, repo_path).replace("\\", "/")

                try:
                    if os.path.getsize(abs_path) > 1_048_576:
                        continue
                    symbol_table = parse_file(abs_path)

                    # Collect imports for cross-file resolution
                    if symbol_table.imports:
                        file_imports[rel_path] = symbol_table.imports

                    for sym in symbol_table.all_symbols():
                        all_symbols.append({
                            "name": sym.name,
                            "qualified_name": sym.qualified_name,
                            "symbol_type": sym.symbol_type,
                            "file_path": rel_path,
                            "body": sym.body,
                            "start_line": sym.start_line,
                            "end_line": sym.end_line,
                            "docstring": sym.docstring,
                            "params": sym.params,
                            "calls": sym.calls,
                            "language": symbol_table.language,
                            "project_id": project_id,
                        })
                except Exception as exc:
                    log.debug("parse_failed", file=rel_path, error=str(exc))

        components = symbols_to_components(all_symbols)

        # Resolve cross-file calls using import analysis
        if file_imports:
            resolve_cross_file_calls(components, file_imports)

        return components

    # ── ordering helpers ────────────────────────────────────────────────────

    def _get_processing_order(
        self,
        modules: dict[str, ModuleSpec],
    ) -> list[tuple[list[str], str, ModuleSpec]]:
        """
        Build a leaf-first processing order from the module tree.
        Returns (path, name, spec) tuples.
        """
        order: list[tuple[list[str], str, ModuleSpec]] = []

        def _collect(tree: dict[str, ModuleSpec], path: list[str]) -> None:
            for name, spec in tree.items():
                current_path = path + [name]
                if spec.children:
                    _collect(spec.children, current_path)
                order.append((current_path, name, spec))

        _collect(modules, [])
        return order

    # ── utility ─────────────────────────────────────────────────────────────

    @staticmethod
    def _save_doc(path: str, content: str) -> None:
        """Write Markdown content to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _truncate(text: str, char_budget: int) -> str:
        """Truncate text to a character budget."""
        if len(text) > char_budget:
            return text[:char_budget] + "\n\n... [truncated for token budget]"
        return text
