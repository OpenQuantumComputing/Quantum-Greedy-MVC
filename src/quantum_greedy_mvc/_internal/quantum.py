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


def _relabel_graph_and_weights(
    graph: nx.Graph,
    weights: dict[Any, float],
) -> tuple[nx.Graph, dict[int, float], dict[Any, int], dict[int, Any]]:
    ordered_nodes = _deterministic_node_order(graph.nodes())
    node_to_int = {node: i for i, node in enumerate(ordered_nodes)}
    int_to_node = {i: node for node, i in node_to_int.items()}

    graph_int = nx.relabel_nodes(graph, node_to_int, copy=True)
    weights_int = {node_to_int[node]: float(weights[node]) for node in ordered_nodes}
    return graph_int, weights_int, node_to_int, int_to_node


def _build_cost_hamiltonian(weights: dict[int, float], n_qubits: int):
    _, _, _, _, SparsePauliOp, _, _ = _require_qiskit()

    paulis = []
    coeffs = []
    for qubit, weight in weights.items():
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - qubit] = "Z"
        paulis.append("".join(pauli))
        coeffs.append(-0.5 * weight)

    return SparsePauliOp(paulis, coeffs), 0.5 * sum(weights.values())


def _expected_cost_from_circuit(circuit, weights: dict[int, float], shots: int | None) -> float:
    _, transpile, _, _, _, Statevector, Aer = _require_qiskit()

    hamiltonian, shift = _build_cost_hamiltonian(weights, circuit.num_qubits)

    if shots is None:
        state = Statevector.from_instruction(circuit)
        return float(shift + state.expectation_value(hamiltonian).real)

    measured = circuit.copy()
    measured.measure_all()
    backend = Aer.get_backend("aer_simulator")
    measured = transpile(measured, backend)
    counts = backend.run(measured, shots=shots).result().get_counts()

    expectation = 0.0
    for bitstring, count in counts.items():
        prob = count / shots
        z_vals = np.array([1 if bit == "0" else -1 for bit in bitstring[::-1]])
        hz = sum(-0.5 * weights[i] * z_vals[i] for i in weights)
        expectation += prob * hz

    return float(shift + expectation)


def _mixer_from_graph(graph: nx.Graph, weights: dict[int, float]):
    QuantumCircuit, _, Parameter, RXGate, _, _, _ = _require_qiskit()

    n_qubits = graph.number_of_nodes()
    circuit = QuantumCircuit(n_qubits)
    betas = {node: Parameter(f"β_{node}") for node in graph.nodes()}

    for qubit in range(n_qubits):
        circuit.x(qubit)

    node_order = sorted(graph.nodes(), key=lambda node: (-weights[node], graph.degree(node)))
    for target in node_order:
        angle = 2 * betas[target]
        controls = list(graph.neighbors(target))
        if controls:
            circuit.append(RXGate(angle).control(len(controls)), controls + [target])
        else:
            circuit.rx(angle, target)

    return circuit, betas


def expectation_value_cost_shifted(circuit, betas, weights, beta_values, shots: int | None = None):
    bound = circuit.assign_parameters({betas[i]: beta_values[i] for i in betas})
    return _expected_cost_from_circuit(bound, weights, shots)


def _greedy_optimize_angles(circuit, betas, weights, beta_values, shots: int | None = None):
    values = beta_values.copy()
    free = list(betas.keys())

    while free:
        idx = random.choice(free)
        best_val = values[idx]
        best_energy = expectation_value_cost_shifted(circuit, betas, weights, values, shots)

        for candidate in (0.0, math.pi / 2):
            trial = values.copy()
            trial[idx] = candidate
            energy = expectation_value_cost_shifted(circuit, betas, weights, trial, shots)
            if energy < best_energy:
                best_energy = energy
                best_val = candidate

        values[idx] = best_val
        free.remove(idx)

    return values


def _is_selected_for_cover(beta: float, atol: float = 1e-9) -> bool:
    # Preserve established behavior:
    # near pi/2 -> not selected; everything else -> selected fallback.
    return not (abs(beta - (math.pi / 2)) <= atol)


def quantum_greedy_vertex_cover(
    graph: nx.Graph,
    weights: dict[Any, float],
    shots: int | None = None,
) -> set[Any]:
    graph_int, weights_int, _, int_to_node = _relabel_graph_and_weights(graph, weights)
    circuit, betas = _mixer_from_graph(graph_int, weights_int)
    beta_init = {i: 0.5 * math.pi / 2 for i in graph_int.nodes()}
    solved = _greedy_optimize_angles(circuit, betas, weights_int, beta_init, shots)

    cover_int = {i for i, beta in solved.items() if _is_selected_for_cover(beta)}

    for u, v in graph_int.edges():
        if u not in cover_int and v not in cover_int:
            cover_int.add(u if weights_int[u] <= weights_int[v] else v)

    return {int_to_node[i] for i in cover_int}


def _conditioned_mvc_mixer_circuit(
    graph_int: nx.Graph,
    fixed_vertex: int,
    evolution_time: float,
    trotter_layers: int,
):
    QuantumCircuit, _, _, RXGate, _, _, _ = _require_qiskit()

    if trotter_layers < 1:
        raise ValueError("trotter_layers must be >= 1")

    circuit = QuantumCircuit(graph_int.number_of_nodes())
    for qubit in range(graph_int.number_of_nodes()):
        circuit.x(qubit)

    delta_t = evolution_time / trotter_layers
    # RX(theta) = exp(-i theta X / 2), so exp(+i delta_t X) => RX(-2*delta_t)
    theta = -2.0 * delta_t

    for _ in range(trotter_layers):
        for qubit in range(graph_int.number_of_nodes()):
            if qubit == fixed_vertex:
                continue
            controls = sorted(graph_int.neighbors(qubit))
            if controls:
                circuit.append(RXGate(theta).control(len(controls)), controls + [qubit])
            else:
                circuit.rx(theta, qubit)

    return circuit


def _remove_isolated_nodes_inplace(graph: nx.Graph, weights: dict[Any, float]) -> None:
    isolated = [node for node, degree in graph.degree() if degree == 0]
    if isolated:
        graph.remove_nodes_from(isolated)
        for node in isolated:
            weights.pop(node, None)


def qeg_ldf_vertex_cover(
    graph: nx.Graph,
    weights: dict[Any, float],
    evolution_time: float = 0.35,
    trotter_layers: int = 1,
    shots: int | None = None,
) -> tuple[set[Any], list[dict[str, Any]]]:
    if evolution_time <= 0:
        raise ValueError("evolution_time must be > 0")
    if trotter_layers < 1:
        raise ValueError("trotter_layers must be >= 1")

    working_graph = graph.copy()
    working_weights = {node: float(weights[node]) for node in working_graph.nodes()}
    cover: set[Any] = set()
    diagnostics: list[dict[str, Any]] = []

    _remove_isolated_nodes_inplace(working_graph, working_weights)

    step = 0
    while working_graph.number_of_edges() > 0:
        candidates = [node for node, degree in working_graph.degree() if degree > 0]
        if not candidates:
            break

        graph_int, weights_int, node_to_int, int_to_node = _relabel_graph_and_weights(
            working_graph,
            working_weights,
        )

        energies: dict[Any, float] = {}
        for node in candidates:
            circuit = _conditioned_mvc_mixer_circuit(
                graph_int=graph_int,
                fixed_vertex=node_to_int[node],
                evolution_time=evolution_time,
                trotter_layers=trotter_layers,
            )
            energies[node] = _expected_cost_from_circuit(circuit, weights_int, shots)

        chosen = min(
            candidates,
            key=lambda node: (energies[node], -working_graph.degree(node), str(type(node)), repr(node)),
        )

        remaining_before = working_graph.number_of_edges()
        cover.add(chosen)
        working_graph.remove_node(chosen)
        working_weights.pop(chosen, None)
        _remove_isolated_nodes_inplace(working_graph, working_weights)

        diagnostics.append(
            {
                "step": step,
                "chosen": chosen,
                "chosen_energy": float(energies[chosen]),
                "remaining_edges_before": int(remaining_before),
                "remaining_edges_after": int(working_graph.number_of_edges()),
                "mapping": [{"node": int_to_node[q], "qubit": q} for q in sorted(int_to_node)],
                "candidates": [
                    {
                        "node": node,
                        "qubit": node_to_int[node],
                        "energy": float(energies[node]),
                        "degree": int(working_graph.degree(node)) if node in working_graph else 0,
                        "weight": float(weights_int[node_to_int[node]]),
                    }
                    for node in _deterministic_node_order(candidates)
                ],
            }
        )
        step += 1

    return cover, diagnostics
