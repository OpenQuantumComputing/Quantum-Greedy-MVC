import networkx as nx

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



def test_weighted_mis_matches_vertex_cover_complement_identity_with_exact_if_available():
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
    try:
        solver.solve_mvc(graph, weights=bad_weights)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "weights keys must match graph nodes" in str(exc)
