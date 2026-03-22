"""
CodeIntel Platform -- Incremental Wiki Generator (Phase 3)

Re-generates documentation only for modules affected by file changes,
instead of re-running the entire pipeline.

Algorithm:
    1. Load previous ``module_tree.json`` from the docs directory
    2. Re-parse the full repository for a fresh component map (fast, CPU-bound)
    3. Map changed/removed files to affected component IDs
    4. Map affected component IDs to module names in the tree
    5. Propagate upward: mark parent modules of any affected module
    6. Re-generate ONLY marked modules (leaf first, parent after)
    7. Load existing Markdown for untouched modules
    8. Regenerate overview if any top-level module was touched

This dramatically reduces wall-clock time for incremental pushes
(e.g. 3 changed files in a 500-file repo → ~5 modules instead of ~40).
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
from services.docgen.module_cluster import ModuleSpec, module_tree_to_dict
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

DEFAULT_MAX_CONCURRENT_LLM = 4


class IncrementalWikiGenerator:
    """
    Regenerates wiki docs only for modules affected by a set of file
    changes.  Reuses the async LLM path for concurrency.

    Parameters
    ----------
    llm : LLMProvider
        LLM backend for doc generation and (optionally) refinement.
    config : WikiGenConfig
        Wiki generation configuration (``docs_dir`` must point to the
        directory containing a previous generation's output).
    max_concurrent_llm : int
        Concurrency cap for async LLM calls.
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
        self._sem = asyncio.Semaphore(max_concurrent_llm)
        self._lock = asyncio.Lock()
        self._generated_docs: dict[str, ModuleDoc] = {}

        # Optional refiner
        self._refiner = None
        if self._cfg.refine_iterations > 0:
            from services.docgen.doc_refiner import DocRefiner
            self._refiner = DocRefiner(
                llm=self._llm,
                max_iterations=self._cfg.refine_iterations,
                pass_threshold=self._cfg.refine_pass_threshold,
                max_refine_tokens=self._cfg.max_doc_tokens,
            )

    # ── public API ──────────────────────────────────────────────────────────

    async def regenerate_affected(
        self,
        repo_path: str,
        project_id: str,
        changed_files: list[str],
        removed_files: list[str] | None = None,
        *,
        repo_name: str | None = None,
    ) -> WikiResult:
        """
        Re-generate documentation only for modules that contain at least
        one component from a changed file.

        Parameters
        ----------
        repo_path : str
            Absolute path to the repository clone.
        project_id : str
            GitLab project identifier.
        changed_files : list[str]
            Relative paths of added/modified files.
        removed_files : list[str], optional
            Relative paths of deleted files.
        repo_name : str, optional
            Human-readable repo name (defaults to dir name).

        Returns
        -------
        WikiResult
            Result containing only the regenerated modules (plus loaded
            existing docs for untouched modules).
        """
        t0 = time.perf_counter()
        repo_name = repo_name or Path(repo_path).name
        removed_files = removed_files or []

        # Step 1: Load previous module tree
        prev_tree = self._load_module_tree()
        if not prev_tree:
            log.warning(
                "incremental_no_previous_tree",
                docs_dir=self._cfg.docs_dir,
            )
            # Fall back to full generation
            from services.docgen.async_wiki_generator import AsyncWikiGenerator
            full_gen = AsyncWikiGenerator(
                llm=self._llm, config=self._cfg,
            )
            return await full_gen.generate(repo_path, project_id, repo_name=repo_name)

        # Step 2: Re-parse repository for fresh components (CPU-bound)
        log.info("incremental_step_2_parsing", repo=repo_path)
        components = await asyncio.to_thread(
            self._parse_repository, repo_path, project_id,
        )
        if not components:
            log.warning("incremental_no_components")
            return WikiResult(repo_name=repo_name, stats={"error": "no components"})

        # Step 3: Determine affected modules
        all_changed = set(changed_files) | set(removed_files)
        affected_modules = self._find_affected_modules(prev_tree, components, all_changed)
        log.info(
            "incremental_affected_modules",
            changed_files=len(all_changed),
            affected=list(affected_modules),
        )

        if not affected_modules:
            log.info("incremental_no_affected_modules")
            # Load everything from disk
            self._load_all_existing(prev_tree)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return WikiResult(
                repo_name=repo_name,
                overview=self._load_existing_doc("overview") or "",
                modules=list(self._generated_docs.values()),
                module_tree=prev_tree,
                stats={
                    "total_components": len(components),
                    "total_modules": len(self._generated_docs),
                    "affected_modules": 0,
                    "total_latency_ms": elapsed,
                    "incremental": True,
                },
            )

        # Step 4: Rebuild ModuleSpec tree from the stored dict
        module_specs = self._dict_to_specs(prev_tree)

        # Step 5: Load existing docs for untouched modules
        self._load_unaffected(prev_tree, affected_modules)

        # Step 6: Re-generate affected modules (concurrent, leaf-first)
        log.info("incremental_step_6_regenerating", count=len(affected_modules))
        regen_tasks = []
        for mod_name in affected_modules:
            spec = module_specs.get(mod_name)
            if spec is None:
                spec = self._find_spec_recursive(module_specs, mod_name)
            if spec is None:
                continue
            regen_tasks.append(
                self._regenerate_module(spec, components, prev_tree)
            )
        await asyncio.gather(*regen_tasks)

        # Step 7: Re-generate overview if any top-level module was touched
        top_level_touched = affected_modules & set(prev_tree.keys())
        overview = self._load_existing_doc("overview") or ""
        if top_level_touched:
            log.info("incremental_step_7_overview")
            overview = await self._generate_repo_overview(repo_name, prev_tree)
            overview_path = os.path.join(self._cfg.docs_dir, "overview.md")
            await asyncio.to_thread(self._save_doc, overview_path, overview)

        # Step 8: Collect results
        elapsed = (time.perf_counter() - t0) * 1000.0
        result = WikiResult(
            repo_name=repo_name,
            overview=overview,
            modules=list(self._generated_docs.values()),
            module_tree=prev_tree,
            stats={
                "total_components": len(components),
                "total_modules": len(self._generated_docs),
                "affected_modules": len(affected_modules),
                "regenerated": list(affected_modules),
                "total_latency_ms": elapsed,
                "incremental": True,
                "provider": self._llm.provider_name,
                "model": self._llm.model_name,
            },
        )

        meta_path = os.path.join(self._cfg.docs_dir, "metadata.json")
        await asyncio.to_thread(self._write_json, meta_path, result.stats)

        log.info(
            "incremental_wiki_done",
            affected=len(affected_modules),
            total_modules=len(self._generated_docs),
            elapsed_ms=round(elapsed),
        )
        return result

    # ── affected-module detection ───────────────────────────────────────────

    def _find_affected_modules(
        self,
        tree: dict[str, Any],
        components: dict[str, ComponentNode],
        changed_files: set[str],
    ) -> set[str]:
        """
        Walk the module tree and find every module that contains at least
        one component whose ``file_path`` is in ``changed_files``.
        Also propagate upward to parent modules.
        """
        # Build component-id → file_path map
        comp_to_file: dict[str, str] = {
            cid: comp.file_path for cid, comp in components.items()
        }

        # Normalise changed file paths (forward slashes, no leading ./)
        normalised = {
            f.replace("\\", "/").lstrip("./") for f in changed_files
        }

        affected: set[str] = set()

        def _walk(subtree: dict[str, Any], parent_chain: list[str]) -> bool:
            any_hit = False
            for mod_name, info in subtree.items():
                mod_hit = False

                # Check direct component membership
                for cid in info.get("components", []):
                    fpath = comp_to_file.get(cid, "")
                    if fpath.replace("\\", "/").lstrip("./") in normalised:
                        mod_hit = True
                        break

                # Recurse into children
                children = info.get("children", {})
                child_hit = False
                if children:
                    child_hit = _walk(children, parent_chain + [mod_name])

                if mod_hit or child_hit:
                    affected.add(mod_name)
                    # Mark all ancestors
                    for ancestor in parent_chain:
                        affected.add(ancestor)
                    any_hit = True

            return any_hit

        _walk(tree, [])
        return affected

    # ── module regeneration ─────────────────────────────────────────────────

    async def _regenerate_module(
        self,
        spec: ModuleSpec,
        components: dict[str, ComponentNode],
        module_tree: dict[str, Any],
    ) -> ModuleDoc:
        """Re-generate a single module doc."""
        module_name = spec.name

        if spec.children:
            # Parent module — just re-generate the overview from child docs
            child_docs = []
            for child_name, child_spec in spec.children.items():
                existing = self._generated_docs.get(child_name)
                if existing:
                    child_docs.append(existing)
            doc = await self._generate_parent_doc(module_name, child_docs, module_tree)
        else:
            doc = await self._generate_leaf_doc(module_name, spec, components, module_tree)

        async with self._lock:
            self._generated_docs[module_name] = doc

        out_path = os.path.join(self._cfg.docs_dir, f"{module_name}.md")
        await asyncio.to_thread(self._save_doc, out_path, doc.content)
        doc.file_path = out_path
        return doc

    # ── leaf / parent doc generation (mirrors AsyncWikiGenerator) ───────────

    async def _generate_leaf_doc(
        self,
        module_name: str,
        spec: ModuleSpec,
        components: dict[str, ComponentNode],
        module_tree: dict[str, Any],
    ) -> ModuleDoc:
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

        async with self._sem:
            resp = await self._llm.achat(messages, max_tokens=self._cfg.max_doc_tokens)

        content = resp.content

        # Iterative refinement
        if self._refiner is not None:
            async with self._sem:
                content, _ = await self._refiner.arefine(
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

        if self._refiner is not None:
            async with self._sem:
                content, _ = await self._refiner.arefine(
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
                "\n\n".join(module_summaries), self._cfg.context_char_budget,
            ),
        )

        async with self._sem:
            resp = await self._llm.acomplete(prompt, max_tokens=self._cfg.max_doc_tokens)
        return resp.content

    # ── repository parsing (sync, called via to_thread) ─────────────────────

    def _parse_repository(
        self,
        repo_path: str,
        project_id: str,
    ) -> dict[str, ComponentNode]:
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

    # ── module tree helpers ──────────────────────────────────────────────────

    def _load_module_tree(self) -> dict[str, Any]:
        """Load the previous module_tree.json from docs_dir."""
        tree_path = os.path.join(self._cfg.docs_dir, "module_tree.json")
        if not os.path.exists(tree_path):
            return {}
        try:
            with open(tree_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("module_tree_load_failed", error=str(exc))
            return {}

    def _dict_to_specs(self, tree: dict[str, Any]) -> dict[str, ModuleSpec]:
        """Reconstruct ModuleSpec objects from the stored dict."""
        specs: dict[str, ModuleSpec] = {}
        for name, info in tree.items():
            children = {}
            if info.get("children"):
                children = self._dict_to_specs(info["children"])
            specs[name] = ModuleSpec(
                name=name,
                path=info.get("path", ""),
                component_ids=info.get("components", []),
                description=info.get("description", ""),
                children=children,
            )
        return specs

    def _find_spec_recursive(
        self,
        specs: dict[str, ModuleSpec],
        target: str,
    ) -> ModuleSpec | None:
        """Depth-first search for a ModuleSpec by name."""
        for name, spec in specs.items():
            if name == target:
                return spec
            if spec.children:
                found = self._find_spec_recursive(spec.children, target)
                if found:
                    return found
        return None

    def _load_existing_doc(self, module_name: str) -> str | None:
        """Load an existing Markdown file from the docs directory."""
        doc_path = os.path.join(self._cfg.docs_dir, f"{module_name}.md")
        if not os.path.exists(doc_path):
            return None
        try:
            return Path(doc_path).read_text(encoding="utf-8")
        except OSError:
            return None

    def _load_all_existing(self, tree: dict[str, Any]) -> None:
        """Load all existing module docs from disk into _generated_docs."""
        for mod_name, info in tree.items():
            if mod_name not in self._generated_docs:
                content = self._load_existing_doc(mod_name)
                if content:
                    doc_path = os.path.join(self._cfg.docs_dir, f"{mod_name}.md")
                    self._generated_docs[mod_name] = ModuleDoc(
                        module_name=mod_name,
                        content=content,
                        file_path=doc_path,
                        component_ids=info.get("components", []),
                    )
            children = info.get("children", {})
            if children:
                self._load_all_existing(children)

    def _load_unaffected(
        self,
        tree: dict[str, Any],
        affected: set[str],
    ) -> None:
        """Load existing docs for all modules NOT in the affected set."""
        for mod_name, info in tree.items():
            if mod_name not in affected:
                content = self._load_existing_doc(mod_name)
                if content:
                    doc_path = os.path.join(self._cfg.docs_dir, f"{mod_name}.md")
                    self._generated_docs[mod_name] = ModuleDoc(
                        module_name=mod_name,
                        content=content,
                        file_path=doc_path,
                        component_ids=info.get("components", []),
                    )
            children = info.get("children", {})
            if children:
                self._load_unaffected(children, affected)

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
