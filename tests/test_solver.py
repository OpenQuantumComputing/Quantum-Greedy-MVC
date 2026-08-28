import networkx as nx
import pytest

from quantum_greedy_mvc import QuantumGreedySolver


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
