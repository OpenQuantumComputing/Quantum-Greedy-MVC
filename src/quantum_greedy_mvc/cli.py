from __future__ import annotations

import argparse
import json

import networkx as nx

from .solver import QuantumGreedySolver


def _parse_graph(spec: str) -> nx.Graph:
    try:
        kind, size_text = spec.split(":", 1)
        n = int(size_text)
    except Exception as exc:
        raise ValueError("--graph must be formatted as '<type>:<n>', e.g. cycle:8") from exc

    if n <= 0:
        raise ValueError("Graph size must be positive.")

    if kind == "cycle":
        return nx.cycle_graph(n)
    if kind == "path":
        return nx.path_graph(n)
    if kind == "complete":
        return nx.complete_graph(n)

    raise ValueError("Supported graph types: cycle, path, complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve MVC/MIS with quantum_greedy_mvc")
    parser.add_argument("--problem", choices=["mvc", "mis"], required=True)
    parser.add_argument(
        "--method",
        choices=["quantum_greedy", "qeg_ldf", "greedy_degree", "primal_dual", "exact", "lp_relaxation"],
        default="greedy_degree",
    )
    parser.add_argument("--graph", required=True, help="Graph spec, e.g. cycle:8")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--qeg-time", type=float, default=0.35, help="QEG-LDF evolution time t")
    parser.add_argument("--qeg-trotter-layers", type=int, default=1, help="QEG-LDF first-order Trotter layers p")

    args = parser.parse_args()

    graph = _parse_graph(args.graph)
    solver = QuantumGreedySolver(
        method=args.method,
        shots=args.shots,
        qeg_time=args.qeg_time,
        qeg_trotter_layers=args.qeg_trotter_layers,
    )
    result = solver.solve(graph=graph, problem=args.problem)

    print(
        json.dumps(
            {
                "problem": result.problem,
                "method": result.method,
                "solution": sorted(result.solution),
                "objective": result.objective,
                "feasible": result.feasible,
                "metadata": result.metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
