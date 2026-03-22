"""
CodeIntel Platform — Documentation Generator
Generates docstrings, module summaries, and subsystem wikis using Ollama.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import networkx as nx
import structlog

from services.llm import LLMProvider, get_provider
from services.docgen.prompts import (
    FUNCTION_DOC_PROMPT,
    FILE_MODULE_DOC_PROMPT,
    SUBSYSTEM_DOC_PROMPT,
)

log = structlog.get_logger(__name__)


# Prompt templates are now in services.docgen.prompts
# FUNCTION_DOC_PROMPT, FILE_MODULE_DOC_PROMPT, SUBSYSTEM_DOC_PROMPT
# are imported at the top of this file.

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class GeneratedDoc:
    """A single generated documentation artifact."""

    symbol_name: str
    doc_type: str  # "function", "module", "subsystem"
    content: str
    file_path: str
    project_id: str
    language: str = ""
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Doc Generator
# ---------------------------------------------------------------------------


class DocGenerator:
    """
    Generates documentation for code symbols using Ollama LLM.

    Processes functions in topological order (callees before callers) to
    produce richer docs that include context about called functions.

    Documentation is generated at three levels:
    1. Function/class level — docstrings
    2. Module level — file summaries
    3. Subsystem level — directory/service overviews
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        use_batch_model: bool = False,
    ) -> None:
        """
        Initialize the DocGenerator.

        Args:
            llm: LLM provider to use.  Falls back to the global singleton
                 from ``services.llm.get_provider()`` if not supplied.
            use_batch_model: If True and using Ollama, use the batch model.
        """
        if llm is not None:
            self._llm = llm
        else:
            from services.llm.config import LLMConfig
            cfg = LLMConfig()
            if use_batch_model and cfg.llm_provider == "ollama":
                from services.llm.providers.ollama import OllamaProvider
                self._llm = OllamaProvider(cfg, use_batch_model=True)
            else:
                self._llm = get_provider(cfg)
        self._log = structlog.get_logger(__name__)

    def generate_function_doc(
        self,
        symbol: dict[str, Any],
        callee_docs: list[dict[str, Any]] | None = None,
        developer_context: str = "",
    ) -> GeneratedDoc:
        """
        Generate documentation for a single function or class.

        Uses callee documentation as context when available, producing
        richer docs by explaining what called functions do.

        Args:
            symbol: Dict with keys: name, qualified_name, body, language,
                file_path, project_id, params, calls, docstring.
            callee_docs: List of GeneratedDoc for functions called by this one.
                Used as context in the prompt.
            developer_context: Optional additional context (e.g., from app_docs).

        Returns:
            GeneratedDoc with generated docstring content.
        """
        source_code = symbol.get("body", symbol.get("content", ""))
        language = symbol.get("language", "python")
        file_path = symbol.get("file_path", "")
        project_id = symbol.get("project_id", "")

        # Build callee doc summary
        callee_summary = "No callee documentation available."
        if callee_docs:
            callee_lines = []
            for doc in callee_docs[:5]:  # Limit to 5 callees to stay within token budget
                callee_lines.append(
                    f"- `{doc.symbol_name}`: {doc.content[:200].strip()}"
                )
            callee_summary = "\n".join(callee_lines)

        prompt = FUNCTION_DOC_PROMPT.format(
            language=language,
            source_code=self._truncate_to_budget(source_code, budget=3000),
            callee_docs=callee_summary,
            developer_context=developer_context[:500] if developer_context else "None",
        )

        content = self._call_llm(prompt)
        content = self._clean_docstring(content)

        return GeneratedDoc(
            symbol_name=symbol.get("qualified_name", symbol.get("name", "")),
            doc_type="function",
            content=content,
            file_path=file_path,
            project_id=project_id,
            language=language,
        )

    def generate_module_doc(
        self,
        file_path: str,
        symbol_table: list[dict[str, Any]],
        module_summaries: list[GeneratedDoc] | None = None,
    ) -> GeneratedDoc:
        """
        Generate module-level documentation for a file.

        Aggregates documentation from all symbols in the file into a
        cohesive module summary.

        Args:
            file_path: Relative path to the source file.
            symbol_table: List of symbol dicts (functions, classes) in the file.
            module_summaries: Pre-generated docs for symbols in this module.

        Returns:
            GeneratedDoc with module Markdown documentation.
        """
        language = "unknown"
        project_id = ""
        if symbol_table:
            language = symbol_table[0].get("language", "python")
            project_id = symbol_table[0].get("project_id", "")

        # Build symbol summary
        symbol_lines = []
        for sym in symbol_table[:30]:  # Cap at 30 symbols per module
            doc_content = ""
            if module_summaries:
                for doc in module_summaries:
                    if sym.get("name") in doc.symbol_name:
                        doc_content = doc.content[:150].strip()
                        break

            if not doc_content:
                doc_content = sym.get("docstring", "No documentation")[:150]

            symbol_lines.append(
                f"- **{sym.get('symbol_type', 'function')} `{sym.get('name', '')}`**: {doc_content}"
            )

        symbol_summaries = "\n".join(symbol_lines) if symbol_lines else "No symbols found."

        prompt = FILE_MODULE_DOC_PROMPT.format(
            file_path=file_path,
            language=language,
            symbol_summaries=self._truncate_to_budget(symbol_summaries, budget=4000),
        )

        content = self._call_llm(prompt)

        return GeneratedDoc(
            symbol_name=file_path,
            doc_type="module",
            content=content,
            file_path=file_path,
            project_id=project_id,
            language=language,
        )

    def generate_subsystem_doc(
        self,
        subsystem_name: str,
        module_docs: list[GeneratedDoc],
    ) -> GeneratedDoc:
        """
        Generate subsystem-level documentation from module summaries.

        Produces a high-level overview of a directory/service/subsystem
        from its constituent module documents.

        Args:
            subsystem_name: Name of the subsystem (e.g., 'payment-service').
            module_docs: List of module-level GeneratedDoc objects.

        Returns:
            GeneratedDoc with subsystem Markdown documentation.
        """
        module_lines = []
        for doc in module_docs[:15]:  # Cap subsystem size
            module_lines.append(
                f"### {doc.file_path}\n{doc.content[:400].strip()}"
            )

        module_docs_text = "\n\n".join(module_lines)

        prompt = SUBSYSTEM_DOC_PROMPT.format(
            subsystem_name=subsystem_name,
            module_docs=self._truncate_to_budget(module_docs_text, budget=6000),
        )

        content = self._call_llm(prompt)

        return GeneratedDoc(
            symbol_name=subsystem_name,
            doc_type="subsystem",
            content=content,
            file_path=subsystem_name,
            project_id=module_docs[0].project_id if module_docs else "",
        )

    def process_in_topological_order(
        self,
        symbols: list[dict[str, Any]],
    ) -> list[GeneratedDoc]:
        """
        Process all symbols in topological order (callees before callers).

        Builds a call graph using networkx and processes nodes in reverse
        topological order, ensuring callee docs are available when generating
        caller docs.

        Args:
            symbols: List of symbol dicts, each with 'name', 'calls', and other fields.

        Returns:
            List of GeneratedDoc objects in topological processing order.
        """
        # Build directed call graph
        G = nx.DiGraph()
        symbol_map: dict[str, dict[str, Any]] = {}

        for sym in symbols:
            name = sym.get("qualified_name", sym.get("name", ""))
            G.add_node(name)
            symbol_map[name] = sym

        for sym in symbols:
            caller = sym.get("qualified_name", sym.get("name", ""))
            for callee in sym.get("calls", []):
                if callee in symbol_map:
                    G.add_edge(caller, callee)

        # Get topological order (leaves first)
        try:
            order = list(reversed(list(nx.topological_sort(G))))
        except nx.NetworkXUnfeasible:
            # Cycle detected — process in original order
            self._log.warning("call_graph_has_cycles_using_original_order")
            order = [sym.get("qualified_name", sym.get("name", "")) for sym in symbols]

        # Process in topological order
        generated_docs: dict[str, GeneratedDoc] = {}
        results: list[GeneratedDoc] = []

        for sym_name in order:
            sym = symbol_map.get(sym_name)
            if sym is None:
                continue

            # Gather callee docs for context
            callee_docs = [
                generated_docs[callee]
                for callee in sym.get("calls", [])
                if callee in generated_docs
            ]

            doc = self.generate_function_doc(
                symbol=sym,
                callee_docs=callee_docs if callee_docs else None,
            )
            generated_docs[sym_name] = doc
            results.append(doc)

        return results

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt via the provider abstraction.

        Handles errors and returns a fallback string if the call fails.
        """
        resp = self._llm.complete(prompt, temperature=0.2, max_tokens=1024)
        return resp.content

    def _truncate_to_budget(self, text: str, budget: int) -> str:
        """Truncate text to stay within the token budget (rough char estimate)."""
        # Rough estimate: 1 token ~ 4 characters
        char_budget = budget * 4
        if len(text) > char_budget:
            return text[:char_budget] + "\n... [truncated for token budget]"
        return text

    @staticmethod
    def _clean_docstring(content: str) -> str:
        """
        Clean up LLM-generated docstring content.

        Removes code fences and leading/trailing whitespace.
        """
        # Remove ```python ... ``` fences
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content, flags=re.MULTILINE)
        content = re.sub(r"\n?```$", "", content, flags=re.MULTILINE)
        # Remove triple-quote wrappers if LLM included them
        content = content.strip('"""').strip("'''").strip()
        return content.strip()
