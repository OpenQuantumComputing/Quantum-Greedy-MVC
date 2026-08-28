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
            "Method 'quantum_greedy' requires optional dependencies 'qiskit' and 'qiskit-aer'."
        ) from exc
    return QuantumCircuit, transpile, Parameter, RXGate, SparsePauliOp, Statevector, Aer


def node_order_by_cost_degree(graph, weights: dict[Any, float]) -> list[Any]:
    return sorted(graph.nodes(), key=lambda node: (-weights[node], graph.degree(node)))


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


def expectation_value_cost_shifted(circuit, betas, weights: dict[Any, float], beta_values: dict[Any, float], shots: int | None = None):
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


def greedy_optimize(circuit, betas, weights: dict[Any, float], beta_values: dict[Any, float], shots: int | None = None):
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

    # Map betas to a cover candidate.
    # beta ~= 0 indicates selected (in cover), beta ~= pi/2 indicates not selected.
    # For undecided values, default to selected to avoid silent infeasibility.
    atol = 1e-9
    cover_idx: set[int] = set()
    for i, beta in solved.items():
        if abs(beta) <= atol:
            cover_idx.add(i)
        elif abs(beta - (math.pi / 2)) <= atol:
            continue
        else:
            cover_idx.add(i)

    # Feasibility repair pass: ensure every edge is covered.
    for u, v in graph_int.edges():
        if u not in cover_idx and v not in cover_idx:
            chosen = u if indexed_weights[u] <= indexed_weights[v] else v
            cover_idx.add(chosen)

    cover: set[Any] = {index_to_node[i] for i in cover_idx}
    return cover
