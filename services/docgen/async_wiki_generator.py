"""
CodeIntel Platform -- Async Hierarchical Wiki Generator
Async variant of WikiGenerator that processes independent sibling
modules concurrently via asyncio.Semaphore, dramatically reducing
wall-clock time for large repositories.

Usage:
    from services.docgen.async_wiki_generator import AsyncWikiGenerator
    gen = AsyncWikiGenerator(llm=provider, config=cfg)
    result = await gen.generate(repo_path, project_id)

    # From synchronous code (e.g. Celery task):
    import asyncio
    result = asyncio.run(gen.generate(repo_path, project_id))
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import structlog

from services.llm.base import ChatMessage, LLMProvider
from services.docgen.mermaid import extract_mermaid_blocks, validate_all_mermaid_in_doc
from services.docgen.module_cluster import (
    ModuleSpec,
    cluster_components,
    module_tree_to_dict,
)
from services.docgen.prompts import (
    LEAF_MODULE_SYSTEM_PROMPT,
    MODULE_OVERVIEW_PROMPT,
    MODULE_USER_PROMPT,
    REPO_OVERVIEW_PROMPT,
    format_component_codes,
    format_module_tree,
)
from services.docgen.wiki_generator import ModuleDoc, WikiGenConfig, WikiResult
from services.parsing.dependency_graph import (
    ComponentNode,
    build_dependency_graph,
    symbols_to_components,
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

# Default concurrency: how many LLM calls can run at the same time
DEFAULT_MAX_CONCURRENT_LLM = 4


class AsyncWikiGenerator:
    """
    Async hierarchical documentation generator.

    Key differences from the sync ``WikiGenerator``:

    * Sibling modules at the same tree level are processed concurrently
      using ``asyncio.gather()`` with a semaphore-based concurrency limit.
    * All LLM calls go through ``acomplete()`` / ``achat()`` -- the async
      provider methods added in Phase 2.
    * Repository parsing (CPU-bound, tree-sitter) stays synchronous and
      runs in ``asyncio.to_thread()`` so it doesn't block the event loop.
    * The semaphore ``max_concurrent_llm`` prevents overwhelming local
      models (Ollama / llama.cpp) while still allowing meaningful speedup.
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: WikiGenConfig | None = None,
        *,
        max_concurrent_llm: int = DEFAULT_MAX_CONCURRENT_LLM,
    ) -> None:
        self._llm = llm
        self._cfg = config or WikiGenConfig()
        self._generated_docs: dict[str, ModuleDoc] = {}
        self._sem = asyncio.Semaphore(max_concurrent_llm)
        self._lock = asyncio.Lock()  # protects _generated_docs

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

    async def generate(
        self,
        repo_path: str,
        project_id: str,
        *,
        repo_name: str | None = None,
    ) -> WikiResult:
        """
        Run the full async wiki generation pipeline.

        Steps 1-3 (parse, graph, cluster) are CPU-bound and run in a
        thread.  Steps 4-5 (doc generation) are IO-bound and run with
        async concurrency.
        """
        t0 = time.perf_counter()
        repo_name = repo_name or Path(repo_path).name
        os.makedirs(self._cfg.docs_dir, exist_ok=True)

        # Step 1: Parse repository (CPU-bound -- run in thread)
        log.info("async_wiki_step_1_parsing", repo=repo_path)
        components = await asyncio.to_thread(
            self._parse_repository, repo_path, project_id
        )
        if not components:
            log.warning("async_wiki_no_components_found")
            return WikiResult(repo_name=repo_name, stats={"error": "no components found"})

        # Step 2: Build dependency graph
        log.info("async_wiki_step_2_dependency_graph", components=len(components))
        dep_graph = build_dependency_graph(components)

        # Step 3: Cluster into modules via LLM (single call, keep sync for consistency)
        log.info("async_wiki_step_3_clustering")
        module_tree = await asyncio.to_thread(
            cluster_components,
            self._llm,
            components,
            max_components_per_module=self._cfg.max_components_per_module,
            max_token_per_module=self._cfg.max_token_per_module,
        )

        tree_dict = module_tree_to_dict(module_tree)
        tree_path = os.path.join(self._cfg.docs_dir, "module_tree.json")
        await asyncio.to_thread(self._write_json, tree_path, tree_dict)

        # Step 4: Process modules concurrently (leaves first, siblings in parallel)
        log.info("async_wiki_step_4_generating_docs", modules=len(module_tree))
        tasks = [
            self._process_module(
                spec=spec,
                components=components,
                module_tree=tree_dict,
                depth=0,
            )
            for spec in module_tree.values()
        ]
        await asyncio.gather(*tasks)

        # Step 5: Generate repository overview
        log.info("async_wiki_step_5_repo_overview")
        overview = await self._generate_repo_overview(repo_name, tree_dict)
        overview_path = os.path.join(self._cfg.docs_dir, "overview.md")
        await asyncio.to_thread(self._save_doc, overview_path, overview)

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
                "async": True,
                "max_concurrent_llm": self._sem._value,
            },
        )

        meta_path = os.path.join(self._cfg.docs_dir, "metadata.json")
        await asyncio.to_thread(self._write_json, meta_path, result.stats)

        log.info(
            "async_wiki_generation_complete",
            modules=len(self._generated_docs),
            elapsed_ms=round(elapsed),
        )
        return result

    # ── module processing ───────────────────────────────────────────────────

    async def _process_module(
        self,
        spec: ModuleSpec,
        components: dict[str, ComponentNode],
        module_tree: dict[str, Any],
        depth: int,
    ) -> ModuleDoc:
        """
        Generate documentation for a single module (async).

        Complex modules with children: process all children concurrently
        first, then generate the parent overview.
        """
        module_name = spec.name

        # Check if already generated (thread-safe)
        async with self._lock:
            if module_name in self._generated_docs:
                return self._generated_docs[module_name]

        if self._cfg.skip_existing:
            existing_path = os.path.join(self._cfg.docs_dir, f"{module_name}.md")
            if os.path.exists(existing_path):
                log.info("async_wiki_skipping_existing", module=module_name)
                content = await asyncio.to_thread(
                    Path(existing_path).read_text, "utf-8"
                )
                doc = ModuleDoc(module_name=module_name, content=content, file_path=existing_path)
                async with self._lock:
                    self._generated_docs[module_name] = doc
                return doc

        indent = "  " * depth
        arrow = "+-" if depth > 0 else ">"

        if spec.children and depth < self._cfg.max_depth:
            # ── Complex: process children concurrently ──────────────────
            log.info(f"{indent}{arrow} Processing complex module: {module_name}")

            child_tasks = [
                self._process_module(
                    spec=child_spec,
                    components=components,
                    module_tree=module_tree,
                    depth=depth + 1,
                )
                for child_spec in spec.children.values()
            ]
            child_docs = await asyncio.gather(*child_tasks)

            # Generate parent overview (needs children to be done)
            doc = await self._generate_parent_doc(module_name, list(child_docs), module_tree)
        else:
            # ── Leaf: generate docs directly ────────────────────────────
            log.info(f"{indent}{arrow} Generating leaf docs: {module_name}")
            doc = await self._generate_leaf_doc(module_name, spec, components, module_tree)

        async with self._lock:
            self._generated_docs[module_name] = doc

        # Save to disk
        out_path = os.path.join(self._cfg.docs_dir, f"{module_name}.md")
        await asyncio.to_thread(self._save_doc, out_path, doc.content)
        doc.file_path = out_path

        return doc

    # ── leaf / parent doc generation ────────────────────────────────────────

    async def _generate_leaf_doc(
        self,
        module_name: str,
        spec: ModuleSpec,
        components: dict[str, ComponentNode],
        module_tree: dict[str, Any],
    ) -> ModuleDoc:
        """Generate docs for a leaf module using async LLM call."""
        comp_codes = format_component_codes(spec.component_ids, components)
        tree_str = format_module_tree(module_tree)

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

        # Semaphore-guarded LLM call
        async with self._sem:
            resp = await self._llm.achat(messages, max_tokens=self._cfg.max_doc_tokens)

        content = resp.content

        # Iterative refinement (Phase 3)
        if self._refiner is not None:
            async with self._sem:
                content, _reviews = await self._refiner.arefine(
                    module_name=module_name,
                    document=content,
                    source_context=self._truncate(comp_codes, 8000),
                )

        mermaid_count = 0
        mermaid_issues = 0
        if self._cfg.validate_mermaid:
            issues = validate_all_mermaid_in_doc(content)
            mermaid_count = len(extract_mermaid_blocks(content))
            mermaid_issues = len(issues)

        return ModuleDoc(
            module_name=module_name,
            content=content,
            file_path="",
            component_ids=spec.component_ids,
            mermaid_count=mermaid_count,
            mermaid_issues=mermaid_issues,
            tokens_used=resp.total_tokens,
            latency_ms=resp.latency_ms,
        )

    async def _generate_parent_doc(
        self,
        module_name: str,
        child_docs: list[ModuleDoc],
        module_tree: dict[str, Any],
    ) -> ModuleDoc:
        """Generate overview for a parent module from child docs."""
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

        async with self._sem:
            resp = await self._llm.acomplete(prompt, max_tokens=self._cfg.max_doc_tokens)

        content = resp.content

        # Iterative refinement (Phase 3)
        if self._refiner is not None:
            async with self._sem:
                content, _reviews = await self._refiner.arefine(
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

    async def _generate_repo_overview(
        self,
        repo_name: str,
        module_tree: dict[str, Any],
    ) -> str:
        """Generate top-level repository overview."""
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

        async with self._sem:
            resp = await self._llm.acomplete(prompt, max_tokens=self._cfg.max_doc_tokens)
        return resp.content

    # ── repository parsing (sync, runs in thread) ──────────────────────────

    def _parse_repository(
        self,
        repo_path: str,
        project_id: str,
    ) -> dict[str, ComponentNode]:
        """
        Walk the repository and parse all supported source files,
        then resolve cross-file calls.
        This is CPU-bound and must be called via asyncio.to_thread().
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
        file_imports: dict[str, list[str]] = {}

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

        if file_imports:
            resolve_cross_file_calls(components, file_imports)

        return components

    # ── utility ─────────────────────────────────────────────────────────────

    @staticmethod
    def _save_doc(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _truncate(text: str, char_budget: int) -> str:
        if len(text) > char_budget:
            return text[:char_budget] + "\n\n... [truncated for token budget]"
        return text
