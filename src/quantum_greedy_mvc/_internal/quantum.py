from __future__ import annotations

import math
import random
from typing import Any

import networkx as nx
import numpy as np


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import RXGate
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit_aer import Aer
    except ImportError as exc:
        raise ImportError(
            "Quantum methods require optional dependencies 'qiskit' and 'qiskit-aer'."
        ) from exc
    return QuantumCircuit, transpile, Parameter, RXGate, SparsePauliOp, Statevector, Aer


def _deterministic_node_order(nodes) -> list[Any]:
    return sorted(nodes, key=lambda node: (str(type(node)), repr(node)))


def node_order_by_cost_degree(graph, weights: dict[Any, float]) -> list[Any]:
    return sorted(graph.nodes(), key=lambda node: (-weights[node], graph.degree(node)))


def _relabel_graph_and_weights(graph: nx.Graph, weights: dict[Any, float]):
    ordered_nodes = _deterministic_node_order(graph.nodes())
    node_to_int = {node: i for i, node in enumerate(ordered_nodes)}
    int_to_node = {i: node for node, i in node_to_int.items()}

    graph_int = nx.relabel_nodes(graph, node_to_int, copy=True)
    weights_int = {node_to_int[node]: float(weights[node]) for node in ordered_nodes}

    return graph_int, weights_int, node_to_int, int_to_node


def mixer_from_graph(graph, weights: dict[Any, float], node_order: list[Any] | None = None):
    QuantumCircuit, _, Parameter, RXGate, _, _, _ = _require_qiskit()

    indexed_graph = nx.convert_node_labels_to_integers(graph)
    n_qubits = indexed_graph.number_of_nodes()

    circuit = QuantumCircuit(n_qubits)
    betas = {node: Parameter(f"β_{node}") for node in indexed_graph.nodes()}

    for i in range(n_qubits):
        circuit.x(i)

    if node_order is None:
        node_order = node_order_by_cost_degree(indexed_graph, weights)

    for target in node_order:
        angle = 2 * betas[target]
        controls = list(indexed_graph.neighbors(target))
        if controls:
            circuit.append(RXGate(angle).control(len(controls)), controls + [target])
        else:
            circuit.rx(angle, target)

    return circuit, betas, indexed_graph


def expectation_value_cost_shifted(
    circuit,
    betas,
    weights: dict[Any, float],
    beta_values: dict[Any, float],
    shots: int | None = None,
):
    _, transpile, _, _, SparsePauliOp, Statevector, Aer = _require_qiskit()

    bind_dict = {betas[i]: beta_values[i] for i in betas}
    circuit_bound = circuit.assign_parameters(bind_dict)

    n_qubits = circuit.num_qubits
    paulis = []
    coeffs = []

    for i, c_i in weights.items():
        p = ["I"] * n_qubits
        p[n_qubits - 1 - i] = "Z"
        paulis.append("".join(p))
        coeffs.append(-0.5 * c_i)

    hamiltonian = SparsePauliOp(paulis, coeffs)
    shift = 0.5 * sum(weights.values())

    if shots is None:
        psi = Statevector.from_instruction(circuit_bound)
        return float(shift + psi.expectation_value(hamiltonian).real)

    measured = circuit_bound.copy()
    measured.measure_all()
    backend = Aer.get_backend("aer_simulator")
    measured = transpile(measured, backend)
    counts = backend.run(measured, shots=shots).result().get_counts()

    exp_val = 0.0
    for bitstring, count in counts.items():
        prob = count / shots
        z_vals = np.array([1 if bit == "0" else -1 for bit in bitstring[::-1]])

        hz_value = 0.0
        for i, c_i in weights.items():
            hz_value += -0.5 * c_i * z_vals[i]

        exp_val += prob * hz_value

    return float(shift + exp_val)


def greedy_optimize(
    circuit,
    betas,
    weights: dict[Any, float],
    beta_values: dict[Any, float],
    shots: int | None = None,
):
    values = beta_values.copy()
    free = list(betas.keys())

    while free:
        i = random.choice(free)
        best_val = values[i]
        best_energy = expectation_value_cost_shifted(circuit, betas, weights, values, shots)

        for candidate in (0.0, math.pi / 2):
            trial = values.copy()
            trial[i] = candidate
            energy = expectation_value_cost_shifted(circuit, betas, weights, trial, shots)
            if energy < best_energy:
                best_energy = energy
                best_val = candidate

        values[i] = best_val
        free.remove(i)

    return values


def quantum_greedy_vertex_cover(graph, weights: dict[Any, float], shots: int | None = None) -> set[Any]:
    indexed_graph = nx.convert_node_labels_to_integers(graph)
    index_to_node = {index: node for index, node in enumerate(graph.nodes())}
    indexed_weights = {index: float(weights[index_to_node[index]]) for index in indexed_graph.nodes()}

    circuit, betas, graph_int = mixer_from_graph(indexed_graph, indexed_weights)
    beta_init = {i: 0.5 * math.pi / 2 for i in graph_int.nodes()}
    solved = greedy_optimize(circuit, betas, indexed_weights, beta_init, shots=shots)

    atol = 1e-9
    cover_idx: set[int] = set()
    for i, beta in solved.items():
        if abs(beta) <= atol:
            cover_idx.add(i)
        elif abs(beta - (math.pi / 2)) <= atol:
            continue
        else:
            cover_idx.add(i)

    for u, v in graph_int.edges():
        if u not in cover_idx and v not in cover_idx:
            chosen = u if indexed_weights[u] <= indexed_weights[v] else v
            cover_idx.add(chosen)

    cover: set[Any] = {index_to_node[i] for i in cover_idx}
    return cover


def _conditioned_mvc_mixer_circuit(
    graph_int: nx.Graph,
    fixed_vertex: int,
    evolution_time: float,
    trotter_layers: int,
):
    QuantumCircuit, _, _, RXGate, _, _, _ = _require_qiskit()

    if trotter_layers < 1:
        raise ValueError("trotter_layers must be >= 1")

    n_qubits = graph_int.number_of_nodes()
    circuit = QuantumCircuit(n_qubits)

    # Initialize all qubits in |1>, the all-ones feasible MVC state.
    for i in range(n_qubits):
        circuit.x(i)

    delta_t = evolution_time / trotter_layers
    # Convention: Qiskit RX(theta) = exp(-i theta X / 2), so exp(+i delta_t X)
    # is implemented as RX(-2 * delta_t).
    theta = -2.0 * delta_t

    for _ in range(trotter_layers):
        for u in range(n_qubits):
            if u == fixed_vertex:
                continue
            controls = sorted(graph_int.neighbors(u))
            if controls:
                circuit.append(RXGate(theta).control(len(controls)), controls + [u])
            else:
                circuit.rx(theta, u)

    return circuit


def _expected_cost_from_state(
    circuit,
    weights_int: dict[int, float],
    shots: int | None = None,
):
    _, transpile, _, _, SparsePauliOp, Statevector, Aer = _require_qiskit()

    n_qubits = circuit.num_qubits
    paulis = []
    coeffs = []

    for i, c_i in weights_int.items():
        p = ["I"] * n_qubits
        p[n_qubits - 1 - i] = "Z"
        paulis.append("".join(p))
        coeffs.append(-0.5 * c_i)

    hamiltonian = SparsePauliOp(paulis, coeffs)
    shift = 0.5 * sum(weights_int.values())

    if shots is None:
        psi = Statevector.from_instruction(circuit)
        return float(shift + psi.expectation_value(hamiltonian).real)

    measured = circuit.copy()
    measured.measure_all()
    backend = Aer.get_backend("aer_simulator")
    measured = transpile(measured, backend)
    counts = backend.run(measured, shots=shots).result().get_counts()

    exp_val = 0.0
    for bitstring, count in counts.items():
        prob = count / shots
        z_vals = np.array([1 if bit == "0" else -1 for bit in bitstring[::-1]])

        hz_value = 0.0
        for i, c_i in weights_int.items():
            hz_value += -0.5 * c_i * z_vals[i]

        exp_val += prob * hz_value

    return float(shift + exp_val)


def qeg_ldf_vertex_cover(
    graph: nx.Graph,
    weights: dict[Any, float],
    evolution_time: float = 0.35,
    trotter_layers: int = 1,
    shots: int | None = None,
):
    if evolution_time < 0:
        raise ValueError("evolution_time must be >= 0")
    if trotter_layers < 1:
        raise ValueError("trotter_layers must be >= 1")

    working_graph = graph.copy()
    cover: set[Any] = set()
    diagnostics: list[dict[str, Any]] = []

    isolated = [node for node, degree in working_graph.degree() if degree == 0]
    if isolated:
        working_graph.remove_nodes_from(isolated)

    step = 0
    while working_graph.number_of_edges() > 0:
        candidate_nodes = [node for node, degree in working_graph.degree() if degree > 0]
        if not candidate_nodes:
            break

        step_weights = {node: float(weights[node]) for node in working_graph.nodes()}
        graph_int, weights_int, node_to_int, int_to_node = _relabel_graph_and_weights(
            working_graph,
            step_weights,
        )

        candidate_energies: dict[Any, float] = {}
        for node in candidate_nodes:
            fixed_vertex = node_to_int[node]
            circuit = _conditioned_mvc_mixer_circuit(
                graph_int,
                fixed_vertex,
                evolution_time=evolution_time,
                trotter_layers=trotter_layers,
            )
            candidate_energies[node] = _expected_cost_from_state(
                circuit,
                weights_int,
                shots=shots,
            )

        chosen = min(
            candidate_nodes,
            key=lambda node: (
                candidate_energies[node],
                -working_graph.degree(node),
                str(type(node)),
                repr(node),
            ),
        )

        diagnostics.append(
            {
                "step": step,
                "chosen": chosen,
                "chosen_energy": float(candidate_energies[chosen]),
                "candidates": [
                    {
                        "node": node,
                        "qubit": node_to_int[node],
                        "energy": float(candidate_energies[node]),
                        "degree": int(working_graph.degree(node)),
                        "weight": float(step_weights[node]),
                    }
                    for node in _deterministic_node_order(candidate_nodes)
                ],
                "mapping": [
                    {"node": int_to_node[q], "qubit": q}
                    for q in sorted(int_to_node.keys())
                ],
                "remaining_edges_before": int(working_graph.number_of_edges()),
            }
        )

        cover.add(chosen)
        working_graph.remove_node(chosen)

        isolated = [node for node, degree in working_graph.degree() if degree == 0]
        if isolated:
            working_graph.remove_nodes_from(isolated)

        diagnostics[-1]["remaining_edges_after"] = int(working_graph.number_of_edges())
        step += 1

    return cover, diagnostics
