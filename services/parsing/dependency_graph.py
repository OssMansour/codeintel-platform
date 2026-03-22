"""
CodeIntel Platform -- Dependency Graph with Cycle Detection
Enhanced dependency graph using Tarjan's algorithm for cycle detection
and resolution, ported from CodeWiki's approach.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ComponentNode:
    """
    A single code component (function, class, method) in the graph.

    Richer than the base SymbolInfo -- carries dependency edges and
    metadata needed for module clustering and doc generation.
    """

    id: str                              # qualified name  e.g. 'payment.service.process'
    name: str                            # simple name     e.g. 'process'
    component_type: str                  # 'function', 'class', 'method'
    file_path: str                       # relative path to source file
    source_code: str = ""
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""
    params: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)       # outgoing edges (this calls X)
    called_by: list[str] = field(default_factory=list)   # incoming edges (X calls this)
    language: str = "python"
    project_id: str = ""


# ---------------------------------------------------------------------------
# Tarjan's Strongly Connected Components
# ---------------------------------------------------------------------------


def detect_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """
    Detect cycles in a dependency graph using Tarjan's algorithm.

    Args:
        graph: Adjacency list  (node -> set of nodes it depends on).

    Returns:
        List of cycles, where each cycle is a list of node IDs that
        form a strongly connected component of size > 1.
    """
    index_counter = [0]
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    cycles: list[list[str]] = []

    def _strongconnect(node: str) -> None:
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)

        for successor in graph.get(node, set()):
            if successor not in index:
                _strongconnect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in on_stack:
                lowlink[node] = min(lowlink[node], index[successor])

        # Root of an SCC
        if lowlink[node] == index[node]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == node:
                    break
            if len(scc) > 1:
                cycles.append(scc)

    for node in graph:
        if node not in index:
            _strongconnect(node)

    return cycles


def resolve_cycles(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    Break cycles by removing edges in each SCC.

    Strategy: for each cycle, remove the first edge that will break it.
    This is a heuristic -- in production you might prefer to remove edges
    between different modules over edges within the same module.

    Returns:
        A new acyclic graph (original is not modified).
    """
    cycles = detect_cycles(graph)
    if not cycles:
        return graph

    new_graph = {node: deps.copy() for node, deps in graph.items()}

    for cycle in cycles:
        log.debug("breaking_cycle", cycle=cycle)
        for i in range(len(cycle) - 1):
            current = cycle[i]
            next_node = cycle[i + 1]
            if next_node in new_graph.get(current, set()):
                new_graph[current].discard(next_node)
                log.debug("edge_removed", src=current, dst=next_node)
                break

    # Verify -- run again to catch nested cycles
    remaining = detect_cycles(new_graph)
    if remaining:
        log.warning("residual_cycles_after_resolution", count=len(remaining))
        # Second pass
        for cycle in remaining:
            for i in range(len(cycle) - 1):
                current, nxt = cycle[i], cycle[i + 1]
                if nxt in new_graph.get(current, set()):
                    new_graph[current].discard(nxt)
                    break

    return new_graph


# ---------------------------------------------------------------------------
# Topological sort (dependency-first)
# ---------------------------------------------------------------------------


def topological_sort(graph: dict[str, set[str]]) -> list[str]:
    """
    Kahn's algorithm on the acyclic graph.

    Returns nodes in dependency-first order (leaves first, roots last)
    so that a node's dependencies are always processed before the node.
    """
    acyclic = resolve_cycles(graph)

    in_degree: dict[str, int] = {n: 0 for n in acyclic}
    for node, deps in acyclic.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[dep] += 1

    queue = deque(n for n, d in in_degree.items() if d == 0)
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for dependent, deps in acyclic.items():
            if node in deps:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    if len(result) != len(acyclic):
        log.warning("topo_sort_incomplete", sorted=len(result), total=len(acyclic))
        # Append any un-sorted nodes
        remaining = [n for n in acyclic if n not in set(result)]
        result.extend(remaining)

    # Reverse so leaves (no outgoing edges) come first
    return result[::-1]


# ---------------------------------------------------------------------------
# Graph builder from SymbolInfo / ComponentNode
# ---------------------------------------------------------------------------


def build_dependency_graph(
    components: dict[str, ComponentNode],
) -> dict[str, set[str]]:
    """
    Build an adjacency-list graph from a dict of ComponentNodes.

    An edge A -> B means "A calls/depends on B".

    Args:
        components: Mapping of qualified_name -> ComponentNode.

    Returns:
        Adjacency list graph.
    """
    graph: dict[str, set[str]] = {}
    for cid, node in components.items():
        deps: set[str] = set()
        for call in node.calls:
            if call in components:
                deps.add(call)
        graph[cid] = deps

    return graph


def get_leaf_nodes(graph: dict[str, set[str]]) -> list[str]:
    """
    Return nodes that have no outgoing edges (they don't depend on
    anything else in the graph).  These are the "leaves" and should
    be documented first.
    """
    return [n for n, deps in graph.items() if len(deps) == 0]


def get_root_nodes(graph: dict[str, set[str]]) -> list[str]:
    """
    Return nodes that nobody depends on (no incoming edges).
    These are the top-level entry points.
    """
    all_deps: set[str] = set()
    for deps in graph.values():
        all_deps.update(deps)
    return [n for n in graph if n not in all_deps]


def symbols_to_components(
    symbols: list[dict[str, Any]],
) -> dict[str, ComponentNode]:
    """
    Convert a list of raw symbol dicts (from the parser/chunker) into
    ComponentNode instances keyed by qualified name.
    """
    components: dict[str, ComponentNode] = {}
    for sym in symbols:
        qname = sym.get("qualified_name", sym.get("name", ""))
        if not qname:
            continue
        components[qname] = ComponentNode(
            id=qname,
            name=sym.get("name", qname),
            component_type=sym.get("symbol_type", "function"),
            file_path=sym.get("file_path", ""),
            source_code=sym.get("body", sym.get("content", "")),
            start_line=sym.get("start_line", 0),
            end_line=sym.get("end_line", 0),
            docstring=sym.get("docstring", ""),
            params=sym.get("params", []),
            calls=sym.get("calls", []),
            language=sym.get("language", "python"),
            project_id=sym.get("project_id", ""),
        )
    return components
