"""
CodeIntel Platform -- Iterative Document Refiner (Phase 3)

Self-review loop that critiques and refines LLM-generated documentation.

Flow:
    1. Ask the LLM to critique the doc (produces a structured review)
    2. If the review says "pass", stop early
    3. Otherwise, ask the LLM to refine the doc based on the issues
    4. Repeat up to ``max_iterations``

Both sync (``refine``) and async (``arefine``) entry points are provided
so the refiner can be used from both WikiGenerator and AsyncWikiGenerator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from services.llm.base import ChatMessage, LLMProvider
from services.docgen.prompts import DOC_CRITIQUE_PROMPT, DOC_REFINE_PROMPT

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Review result
# ---------------------------------------------------------------------------

_REVIEW_RE = re.compile(r"<REVIEW>\s*(.*?)\s*</REVIEW>", re.DOTALL)

# Minimum overall score and per-criterion floor to pass
_DEFAULT_PASS_THRESHOLD = 4.0
_DEFAULT_CRITERION_FLOOR = 3


@dataclass
class ReviewResult:
    """Structured output of one critique round."""

    scores: dict[str, int] = field(default_factory=dict)
    overall: float = 0.0
    issues: list[str] = field(default_factory=list)
    passed: bool = False
    raw: str = ""


# ---------------------------------------------------------------------------
# DocRefiner
# ---------------------------------------------------------------------------


class DocRefiner:
    """
    Iterative refinement loop for generated documentation.

    Parameters
    ----------
    llm : LLMProvider
        The LLM backend (same provider used for generation).
    max_iterations : int
        Maximum critique→refine cycles (default 2).
    pass_threshold : float
        Minimum ``overall`` score to accept the doc without further edits.
    max_refine_tokens : int
        Max tokens for the refine call.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        max_iterations: int = 2,
        pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
        max_refine_tokens: int = 2048,
    ) -> None:
        self._llm = llm
        self._max_iter = max_iterations
        self._threshold = pass_threshold
        self._max_tokens = max_refine_tokens

    # ── sync API ────────────────────────────────────────────────────────────

    def refine(
        self,
        module_name: str,
        document: str,
        source_context: str = "",
    ) -> tuple[str, list[ReviewResult]]:
        """
        Run the critique → refine loop synchronously.

        Parameters
        ----------
        module_name : str
            Name of the module being refined (used in prompts).
        document : str
            The current Markdown documentation to refine.
        source_context : str, optional
            Truncated source code context to help the refiner make accurate edits.

        Returns
        -------
        (refined_doc, reviews) : tuple
            The (possibly improved) Markdown and the list of review
            results from each iteration.
        """
        reviews: list[ReviewResult] = []
        current = document

        for iteration in range(self._max_iter):
            log.debug(
                "refine_iteration_start",
                module=module_name,
                iteration=iteration + 1,
            )

            # ── Step 1: Critique ────────────────────────────────────────
            review = self._critique_sync(module_name, current)
            reviews.append(review)

            if review.passed:
                log.info(
                    "refine_passed_early",
                    module=module_name,
                    iteration=iteration + 1,
                    overall=review.overall,
                )
                break

            # ── Step 2: Refine ──────────────────────────────────────────
            current = self._refine_sync(
                module_name, current, review.issues, source_context,
            )
            log.info(
                "refine_iteration_done",
                module=module_name,
                iteration=iteration + 1,
                issues_fixed=len(review.issues),
            )

        return current, reviews

    # ── async API ───────────────────────────────────────────────────────────

    async def arefine(
        self,
        module_name: str,
        document: str,
        source_context: str = "",
    ) -> tuple[str, list[ReviewResult]]:
        """Async variant of :meth:`refine`."""
        reviews: list[ReviewResult] = []
        current = document

        for iteration in range(self._max_iter):
            log.debug(
                "arefine_iteration_start",
                module=module_name,
                iteration=iteration + 1,
            )

            review = await self._critique_async(module_name, current)
            reviews.append(review)

            if review.passed:
                log.info(
                    "arefine_passed_early",
                    module=module_name,
                    iteration=iteration + 1,
                    overall=review.overall,
                )
                break

            current = await self._refine_async(
                module_name, current, review.issues, source_context,
            )
            log.info(
                "arefine_iteration_done",
                module=module_name,
                iteration=iteration + 1,
                issues_fixed=len(review.issues),
            )

        return current, reviews

    # ── internal: sync helpers ──────────────────────────────────────────────

    def _critique_sync(self, module_name: str, document: str) -> ReviewResult:
        prompt = DOC_CRITIQUE_PROMPT.format(
            module_name=module_name,
            document=document,
        )
        resp = self._llm.complete(prompt, max_tokens=1024)
        return self._parse_review(resp.content)

    def _refine_sync(
        self,
        module_name: str,
        document: str,
        issues: list[str],
        source_context: str,
    ) -> str:
        ctx_block = (
            f"## Source Code Context\n{source_context}"
            if source_context
            else ""
        )
        prompt = DOC_REFINE_PROMPT.format(
            module_name=module_name,
            document=document,
            issues="\n".join(f"- {i}" for i in issues),
            source_context=ctx_block,
        )
        messages = [
            ChatMessage(role="system", content="You are an expert documentation writer."),
            ChatMessage(role="user", content=prompt),
        ]
        resp = self._llm.chat(messages, max_tokens=self._max_tokens)
        return resp.content

    # ── internal: async helpers ─────────────────────────────────────────────

    async def _critique_async(self, module_name: str, document: str) -> ReviewResult:
        prompt = DOC_CRITIQUE_PROMPT.format(
            module_name=module_name,
            document=document,
        )
        resp = await self._llm.acomplete(prompt, max_tokens=1024)
        return self._parse_review(resp.content)

    async def _refine_async(
        self,
        module_name: str,
        document: str,
        issues: list[str],
        source_context: str,
    ) -> str:
        ctx_block = (
            f"## Source Code Context\n{source_context}"
            if source_context
            else ""
        )
        prompt = DOC_REFINE_PROMPT.format(
            module_name=module_name,
            document=document,
            issues="\n".join(f"- {i}" for i in issues),
            source_context=ctx_block,
        )
        messages = [
            ChatMessage(role="system", content="You are an expert documentation writer."),
            ChatMessage(role="user", content=prompt),
        ]
        resp = await self._llm.achat(messages, max_tokens=self._max_tokens)
        return resp.content

    # ── review parser ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_review(raw: str) -> ReviewResult:
        """Extract structured review from LLM response."""
        match = _REVIEW_RE.search(raw)
        if not match:
            log.warning("review_parse_failed", raw_length=len(raw))
            # Treat as a fail so we refine at least once
            return ReviewResult(
                overall=0.0,
                issues=["Could not parse review — assuming improvements needed"],
                passed=False,
                raw=raw,
            )

        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            log.warning("review_json_decode_failed")
            return ReviewResult(
                overall=0.0,
                issues=["Review JSON decode error — assuming improvements needed"],
                passed=False,
                raw=raw,
            )

        scores = data.get("scores", {})
        overall = float(data.get("overall", 0.0))
        issues = data.get("issues", [])

        # Determine pass: overall >= threshold AND every criterion >= floor
        criterion_pass = all(
            v >= _DEFAULT_CRITERION_FLOOR for v in scores.values()
        ) if scores else False
        passed = data.get("pass", overall >= _DEFAULT_PASS_THRESHOLD and criterion_pass)

        return ReviewResult(
            scores=scores,
            overall=overall,
            issues=issues,
            passed=bool(passed),
            raw=raw,
        )
