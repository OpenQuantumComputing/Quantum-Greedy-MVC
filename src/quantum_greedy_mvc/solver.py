from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx

from ._internal.classical import (
    greedy_degree_vertex_cover,
    is_vertex_cover,
    mvc_exact_cplex,
    mvc_lp_relaxation,
    mvc_primal_dual_weighted,
)
from ._internal.quantum import (
    _conditioned_mvc_mixer_circuit,
    _expected_cost_from_circuit,
    _relabel_graph_and_weights,
    _remove_isolated_nodes_inplace,
    qeg_ldf_vertex_cover,
    quantum_greedy_vertex_cover,
)
from .types import SolveResult

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

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
        graph = self._validate_graph(graph)
        weights = self._normalize_weights(graph, weights)
        cover, extra = self._solve_vertex_cover(graph, weights)

        metadata = {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            **extra,
        }

        return SolveResult(
            problem="mvc",
            method=self.method,
            solution=set(cover),
            objective=float(sum(weights[node] for node in cover)),
            feasible=is_vertex_cover(graph, cover),
            metadata=metadata,
        )

    def solve_mis(self, graph: nx.Graph, weights: dict[Any, float] | None = None) -> SolveResult:
        graph = self._validate_graph(graph)
        weights = self._normalize_weights(graph, weights)
        cover, extra = self._solve_vertex_cover(graph, weights)
        independent_set = set(graph.nodes()) - set(cover)

        metadata = {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "via": "complement_of_vertex_cover",
            **extra,
        }

        return SolveResult(
            problem="mis",
            method=self.method,
            solution=independent_set,
            objective=float(sum(weights[node] for node in independent_set)),
            feasible=self._is_independent_set(graph, independent_set),
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
            cover, steps = qeg_ldf_vertex_cover(
                graph=graph,
                weights=weights,
                evolution_time=self.qeg_time,
                trotter_layers=self.qeg_trotter_layers,
                shots=self.shots,
            )
            return cover, {
                "qeg_ldf": {
                    "time": float(self.qeg_time),
                    "trotter_layers": int(self.qeg_trotter_layers),
                    "shots": self.shots,
                    "steps": steps,
                }
            }
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


def _qeg_ldf_first_step_vertex_cover(
    graph: nx.Graph,
    weights: dict[Any, float],
    evolution_time: float = 0.35,
    trotter_layers: int = 1,
    shots: int | None = None,
) -> tuple[Any | None, QuantumCircuit | None]:
    if evolution_time <= 0:
        raise ValueError("evolution_time must be > 0")
    if trotter_layers < 1:
        raise ValueError("trotter_layers must be >= 1")

    working_graph = graph.copy()
    working_weights = {node: float(weights[node]) for node in working_graph.nodes()}
    _remove_isolated_nodes_inplace(working_graph, working_weights)

    if working_graph.number_of_edges() == 0:
        return None, None

    candidates = list(working_graph.nodes())
    ordered_candidates = sorted(candidates, key=lambda node: (str(type(node)), repr(node)))
    deterministic_rank = {
        node: rank for rank, node in enumerate(ordered_candidates)
    }

    graph_int, weights_int, node_to_int, _ = _relabel_graph_and_weights(
        working_graph,
        working_weights,
    )

    best_node: Any | None = None
    best_circuit: QuantumCircuit | None = None
    best_key: tuple[float, int, int] | None = None

    for node in candidates:
        circuit = _conditioned_mvc_mixer_circuit(
            graph_int=graph_int,
            fixed_vertex=node_to_int[node],
            evolution_time=evolution_time,
            trotter_layers=trotter_layers,
        )
        energy = _expected_cost_from_circuit(circuit, weights_int, shots)
        # Selection rule: lowest energy, then highest degree, then deterministic node order.
        key = (energy, -working_graph.degree(node), deterministic_rank[node])
        if best_key is None or key < best_key:
            best_node = node
            best_circuit = circuit
            best_key = key

    return best_node, best_circuit


def first_step_mvc(
    graph: nx.Graph,
    weights: dict[Any, float] | None = None,
) -> tuple[Any | None, QuantumCircuit | None]:
    """Return the first QEG-LDF MVC decision as (selected_vertex, conditioned_circuit)."""
    validated_graph = QuantumGreedySolver._validate_graph(graph)
    normalized_weights = QuantumGreedySolver._normalize_weights(validated_graph, weights)
    return _qeg_ldf_first_step_vertex_cover(validated_graph, normalized_weights)


def first_step_mis(
    graph: nx.Graph,
    weights: dict[Any, float] | None = None,
) -> tuple[Any | None, QuantumCircuit | None]:
    """Return the first cover vertex decision used by solve_mis() complement logic."""
    # solve_mis() is implemented as complement_of_vertex_cover, so this first decision
    # intentionally mirrors the first recursive cover decision.
    return first_step_mvc(graph, weights)
