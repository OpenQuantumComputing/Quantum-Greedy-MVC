from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx

from ._internal.classical import (
    greedy_degree_vertex_cover,
    is_vertex_cover,
    mvc_exact_cplex,
    mvc_lp_relaxation,
    mvc_primal_dual_weighted,
)
from ._internal.quantum import qeg_ldf_vertex_cover, quantum_greedy_vertex_cover
from .types import QegLdfMetadata, SolveResult


ProblemName = Literal["mvc", "mis"]
MethodName = Literal[
    "quantum_greedy",
    "qeg_ldf",
    "greedy_degree",
    "primal_dual",
    "exact",
    "lp_relaxation",
]


@dataclass
class QuantumGreedySolver:
    method: MethodName = "quantum_greedy"
    shots: int | None = None
    qeg_time: float = 0.35
    qeg_trotter_layers: int = 1

    def __post_init__(self) -> None:
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be a positive integer or None")

        if self.method == "qeg_ldf":
            if self.qeg_time <= 0:
                raise ValueError("qeg_time must be > 0")
            if self.qeg_trotter_layers < 1:
                raise ValueError("qeg_trotter_layers must be >= 1")

    def solve(
        self,
        graph: nx.Graph,
        problem: ProblemName = "mvc",
        weights: dict[Any, float] | None = None,
    ) -> SolveResult:
        if problem == "mvc":
            return self.solve_mvc(graph=graph, weights=weights)
        if problem == "mis":
            return self.solve_mis(graph=graph, weights=weights)
        raise ValueError(f"Unsupported problem '{problem}'. Use 'mvc' or 'mis'.")

    def solve_mvc(self, graph: nx.Graph, weights: dict[Any, float] | None = None) -> SolveResult:
        validated_graph = self._validate_graph(graph)
        normalized_weights = self._normalize_weights(validated_graph, weights)

        cover, method_metadata = self._solve_vertex_cover(validated_graph, normalized_weights)
        return SolveResult(
            problem="mvc",
            method=self.method,
            solution=set(cover),
            objective=float(sum(normalized_weights[node] for node in cover)),
            feasible=is_vertex_cover(validated_graph, cover),
            metadata=self._base_metadata(validated_graph, method_metadata),
        )

    def solve_mis(self, graph: nx.Graph, weights: dict[Any, float] | None = None) -> SolveResult:
        validated_graph = self._validate_graph(graph)
        normalized_weights = self._normalize_weights(validated_graph, weights)

        cover, method_metadata = self._solve_vertex_cover(validated_graph, normalized_weights)
        independent_set = set(validated_graph.nodes()) - set(cover)

        metadata = self._base_metadata(validated_graph, method_metadata)
        metadata["via"] = "complement_of_vertex_cover"

        return SolveResult(
            problem="mis",
            method=self.method,
            solution=independent_set,
            objective=float(sum(normalized_weights[node] for node in independent_set)),
            feasible=self._is_independent_set(validated_graph, independent_set),
            metadata=metadata,
        )

    def _solve_vertex_cover(
        self,
        graph: nx.Graph,
        weights: dict[Any, float],
    ) -> tuple[set[Any], dict[str, Any]]:
        if self.method == "quantum_greedy":
            return quantum_greedy_vertex_cover(graph, weights, shots=self.shots), {}

        if self.method == "qeg_ldf":
            cover, diagnostics = qeg_ldf_vertex_cover(
                graph=graph,
                weights=weights,
                evolution_time=self.qeg_time,
                trotter_layers=self.qeg_trotter_layers,
                shots=self.shots,
            )
            qeg_metadata: QegLdfMetadata = {
                "time": float(self.qeg_time),
                "trotter_layers": int(self.qeg_trotter_layers),
                "shots": self.shots,
                "steps": diagnostics,
            }
            return cover, {"qeg_ldf": qeg_metadata}

        if self.method == "greedy_degree":
            return greedy_degree_vertex_cover(graph, weights), {}
        if self.method == "primal_dual":
            return mvc_primal_dual_weighted(graph, weights), {}
        if self.method == "exact":
            return mvc_exact_cplex(graph, weights), {}
        if self.method == "lp_relaxation":
            return mvc_lp_relaxation(graph, weights), {}

        raise ValueError(f"Unsupported method '{self.method}'.")

    @staticmethod
    def _base_metadata(graph: nx.Graph, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            **extra,
        }

    @staticmethod
    def _validate_graph(graph: nx.Graph) -> nx.Graph:
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be an instance of networkx.Graph")
        if graph.is_directed():
            raise ValueError("Directed graphs are not supported. Provide an undirected graph.")
        return graph

    @staticmethod
    def _normalize_weights(graph: nx.Graph, weights: dict[Any, float] | None) -> dict[Any, float]:
        if weights is None:
            return {node: 1.0 for node in graph.nodes()}

        node_set = set(graph.nodes())
        weight_nodes = set(weights.keys())
        if node_set != weight_nodes:
            missing = node_set - weight_nodes
            extra = weight_nodes - node_set
            raise ValueError(f"weights keys must match graph nodes. missing={missing}, extra={extra}")

        normalized = {node: float(value) for node, value in weights.items()}
        if any(value < 0 for value in normalized.values()):
            raise ValueError("All weights must be non-negative for MVC/MIS solving.")

        return normalized

    @staticmethod
    def _is_independent_set(graph: nx.Graph, nodes: set[Any]) -> bool:
        for u, v in graph.edges():
            if u in nodes and v in nodes:
                return False
        return True
