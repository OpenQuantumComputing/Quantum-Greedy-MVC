import networkx as nx
import pytest

from quantum_greedy_mvc import (
    QuantumGreedySolver,
    build_qeg_ldf_circuit,
    first_step_mis,
    first_step_mvc,
)


def test_mvc_unweighted_greedy_degree_returns_vertex_cover():
    graph = nx.cycle_graph(6)
    solver = QuantumGreedySolver(method="greedy_degree")
    result = solver.solve_mvc(graph)

    assert result.problem == "mvc"
    assert result.feasible is True
    assert isinstance(result.solution, set)
    assert result.objective == float(len(result.solution))


def test_mis_unweighted_is_independent_set():
    graph = nx.path_graph(7)
    solver = QuantumGreedySolver(method="primal_dual")
    result = solver.solve_mis(graph)

    assert result.problem == "mis"
    assert result.feasible is True
    for u, v in graph.edges():
        assert not (u in result.solution and v in result.solution)


def test_weighted_mvc_and_mis_objectives_sum_to_total_weight():
    graph = nx.cycle_graph(4)
    weights = {0: 2.0, 1: 1.0, 2: 2.0, 3: 1.0}

    solver = QuantumGreedySolver(method="greedy_degree")
    mvc = solver.solve_mvc(graph, weights=weights)
    mis = solver.solve_mis(graph, weights=weights)

    total_weight = sum(weights.values())
    assert abs((mvc.objective + mis.objective) - total_weight) < 1e-9


def test_invalid_weights_raise():
    graph = nx.path_graph(3)
    solver = QuantumGreedySolver(method="greedy_degree")

    bad_weights = {0: 1.0, 1: 2.0}
    with pytest.raises(ValueError, match="weights keys must match graph nodes"):
        solver.solve_mvc(graph, weights=bad_weights)


def test_qeg_ldf_dispatch_includes_metadata(monkeypatch):
    graph = nx.path_graph(4)

    def fake_qeg(graph, weights, evolution_time, trotter_layers, shots):
        assert evolution_time == 0.7
        assert trotter_layers == 3
        assert shots is None
        return {1, 2}, [{"step": 0, "chosen": 1, "chosen_energy": 1.23}]

    monkeypatch.setattr("quantum_greedy_mvc.solver.qeg_ldf_vertex_cover", fake_qeg)

    solver = QuantumGreedySolver(method="qeg_ldf", qeg_time=0.7, qeg_trotter_layers=3)
    result = solver.solve_mvc(graph)

    assert result.method == "qeg_ldf"
    assert result.solution == {1, 2}
    assert result.metadata["qeg_ldf"]["time"] == 0.7
    assert result.metadata["qeg_ldf"]["trotter_layers"] == 3
    assert result.metadata["qeg_ldf"]["steps"][0]["chosen"] == 1


def test_qeg_parameter_validation():
    with pytest.raises(ValueError, match="qeg_trotter_layers"):
        QuantumGreedySolver(method="qeg_ldf", qeg_trotter_layers=0)

    with pytest.raises(ValueError, match="qeg_time"):
        QuantumGreedySolver(method="qeg_ldf", qeg_time=-0.1)

    with pytest.raises(ValueError, match="qeg_time"):
        QuantumGreedySolver(method="qeg_ldf", qeg_time=0.0)


def test_non_qeg_methods_ignore_qeg_parameters():
    solver = QuantumGreedySolver(method="greedy_degree", qeg_time=-1.0, qeg_trotter_layers=0)
    result = solver.solve_mvc(nx.path_graph(4))
    assert result.feasible is True


def test_first_step_mvc_returns_selected_vertex_and_circuit(monkeypatch):
    graph = nx.path_graph(3)
    weights = {0: 1.0, 1: 1.0, 2: 1.0}

    def fake_circuit(graph_int, fixed_vertex, evolution_time, trotter_layers):
        assert evolution_time == 0.35
        assert trotter_layers == 1
        return f"circ-{fixed_vertex}"

    def fake_energy(circuit, weights_int, shots):
        energies = {"circ-0": 2.0, "circ-1": 1.0, "circ-2": 3.0}
        return energies[circuit]

    monkeypatch.setattr("quantum_greedy_mvc.solver._conditioned_mvc_mixer_circuit", fake_circuit)
    monkeypatch.setattr("quantum_greedy_mvc.solver._expected_cost_from_circuit", fake_energy)

    selected_vertex, circuit = first_step_mvc(graph, weights=weights)
    assert selected_vertex == 1
    assert circuit == "circ-1"


def test_first_step_mis_uses_same_first_decision_as_recursive_cover(monkeypatch):
    graph = nx.path_graph(3)

    def fake_circuit(graph_int, fixed_vertex, evolution_time, trotter_layers):
        return f"circ-{fixed_vertex}"

    def fake_energy(circuit, weights_int, shots):
        energies = {"circ-0": 1.0, "circ-1": 5.0, "circ-2": 2.0}
        return energies[circuit]

    monkeypatch.setattr("quantum_greedy_mvc.solver._conditioned_mvc_mixer_circuit", fake_circuit)
    monkeypatch.setattr("quantum_greedy_mvc.solver._expected_cost_from_circuit", fake_energy)

    selected_vertex, circuit = first_step_mis(graph)
    assert selected_vertex == 0
    assert circuit == "circ-0"


def test_first_step_returns_none_for_edge_free_graph():
    graph = nx.empty_graph(4)
    selected_vertex, circuit = first_step_mvc(graph)
    assert selected_vertex is None
    assert circuit is None


def test_build_qeg_ldf_circuit_builds_without_execution(monkeypatch):
    graph = nx.Graph()
    graph.add_edge("v2", "v10")
    graph.add_node("isolated")

    def fake_circuit(graph_int, fixed_vertex, evolution_time, trotter_layers):
        assert set(graph_int.nodes()) == {0, 1}
        assert evolution_time == 0.4
        assert trotter_layers == 2
        return f"circ-{fixed_vertex}"

    monkeypatch.setattr("quantum_greedy_mvc.solver._conditioned_mvc_mixer_circuit", fake_circuit)

    circuit = build_qeg_ldf_circuit(
        graph=graph,
        fixed_vertex="v2",
        evolution_time=0.4,
        trotter_layers=2,
    )
    assert circuit == "circ-1"


def test_build_qeg_ldf_circuit_rejects_invalid_fixed_vertex():
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match="fixed_vertex must be a node in graph"):
        build_qeg_ldf_circuit(graph=graph, fixed_vertex=9)

    graph_with_isolated = nx.Graph()
    graph_with_isolated.add_edge(0, 1)
    graph_with_isolated.add_node(2)
    with pytest.raises(ValueError, match="fixed_vertex must have degree > 0"):
        build_qeg_ldf_circuit(graph=graph_with_isolated, fixed_vertex=2)
