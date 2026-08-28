from __future__ import annotations

import random
from typing import Any


def is_vertex_cover(graph, cover: set[Any]) -> bool:
    return all(u in cover or v in cover for u, v in graph.edges())


def greedy_degree_vertex_cover(graph, weights: dict[Any, float]) -> set[Any]:
    gc = graph.copy()
    cover: set[Any] = set()
    while gc.number_of_edges() > 0:
        node = max(gc.nodes(), key=lambda x: gc.degree(x) / max(weights[x], 1e-12))
        cover.add(node)
        gc.remove_node(node)
    return cover


def greedy_edge_vertex_cover(graph, weights: dict[Any, float]) -> set[Any]:
    gc = graph.copy()
    cover: set[Any] = set()
    while gc.number_of_edges() > 0:
        u, v = random.choice(list(gc.edges()))
        chosen = u if weights[u] <= weights[v] else v
        cover.add(chosen)
        gc.remove_node(chosen)
    return cover


def mvc_primal_dual_weighted(graph, weights: dict[Any, float]) -> set[Any]:
    remaining_edges = set(graph.edges())
    cover: set[Any] = set()
    while remaining_edges:
        u, v = min(
            remaining_edges,
            key=lambda edge: weights[edge[0]] + weights[edge[1]],
        )
        chosen = u if weights[u] <= weights[v] else v
        cover.add(chosen)
        remaining_edges = {edge for edge in remaining_edges if chosen not in edge}
    return cover


def mvc_exact_cplex(graph, weights: dict[Any, float]) -> set[Any]:
    try:
        from docplex.mp.model import Model
    except ImportError as exc:
        raise ImportError("Method 'exact' requires optional dependency 'docplex'.") from exc

    model = Model("wmvc_exact")
    x = {node: model.binary_var(name=f"x_{node}") for node in graph.nodes()}

    for u, v in graph.edges():
        model.add_constraint(x[u] + x[v] >= 1)

    model.minimize(model.sum(weights[node] * x[node] for node in graph.nodes()))
    model.solve(log_output=False)

    return {node for node in x if x[node].solution_value > 0.5}


def mvc_lp_relaxation(graph, weights: dict[Any, float]) -> set[Any]:
    try:
        from docplex.mp.model import Model
    except ImportError as exc:
        raise ImportError("Method 'lp_relaxation' requires optional dependency 'docplex'.") from exc

    model = Model("wmvc_lp")
    x = {
        node: model.continuous_var(lb=0, ub=1, name=f"x_{node}")
        for node in graph.nodes()
    }

    for u, v in graph.edges():
        model.add_constraint(x[u] + x[v] >= 1)

    model.minimize(model.sum(weights[node] * x[node] for node in graph.nodes()))
    model.solve(log_output=False)

    return {node for node in x if x[node].solution_value >= 0.5}
