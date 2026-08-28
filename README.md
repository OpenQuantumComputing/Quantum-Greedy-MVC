# Quantum-Greedy-MVC

An installable Python package for solving:
- **MVC**: weighted/unweighted Minimum Vertex Cover
- **MIS**: weighted/unweighted Maximum Independent Set

This repository also includes research scripts and Lean formalization artifacts. The runtime package is exposed through `quantum_greedy_mvc`.

## What it solves

Given an undirected graph `G=(V,E)` and optional node weights:
- `solve_mvc(...)` returns a vertex cover `C \subseteq V` minimizing total weight.
- `solve_mis(...)` returns an independent set `I \subseteq V` maximizing total weight.

For weighted MIS, the implementation uses the standard identity:
`weight(MIS) = sum(weights) - weight(minimum weighted vertex cover)`
with `I = V \ C`.

## How it works (high level)

The package wraps the quantum-greedy mixer-style approach from the paper and classical baselines:
- `quantum_greedy` (requires `qiskit` + `qiskit-aer`)
- `greedy_degree`
- `primal_dual`
- `exact` and `lp_relaxation` (require `docplex`)

Reference: https://arxiv.org/pdf/2607.27915

## Install

```bash
pip install .
```

With optional extras:

```bash
pip install .[quantum]
pip install .[cplex]
pip install .[all]
```

## Quickstart

```python
import networkx as nx
from quantum_greedy_mvc import QuantumGreedySolver

G = nx.cycle_graph(6)
solver = QuantumGreedySolver(method="greedy_degree")

mvc = solver.solve_mvc(G)
mis = solver.solve_mis(G)

print(mvc.solution, mvc.objective)
print(mis.solution, mis.objective)
```

## CLI usage

```bash
qgmvc-solve --problem mvc --graph cycle:8 --method greedy_degree
qgmvc-solve --problem mis --graph path:10 --method primal_dual
```

## Public API

- `quantum_greedy_mvc.QuantumGreedySolver`
- `quantum_greedy_mvc.SolveResult`

## Migration notes

Older workflow:
- Run scripts in `MVC/QGMVC.py` and `MVC/QGMVC_parallel.py`
- Notebook/SLURM driven experiments

New workflow:
- `pip install .`
- Import `QuantumGreedySolver`
- Call `solve_mvc(...)` / `solve_mis(...)` directly from Python

The legacy experiment scripts are preserved for research reproduction, while the package API is intended for end users.
