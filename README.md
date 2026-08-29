# Quantum-Greedy-MVC

A small Python package for solving:
- **MVC**: weighted/unweighted Minimum Vertex Cover
- **MIS**: weighted/unweighted Maximum Independent Set

## Install

```bash
pip install .
```

Optional extras:

```bash
pip install .[quantum]
pip install .[cplex]
pip install .[all]
```

## Minimal usage

```python
import networkx as nx
from quantum_greedy_mvc import QuantumGreedySolver

G = nx.cycle_graph(6)
solver = QuantumGreedySolver(method="qeg_ldf", qeg_time=0.35, qeg_trotter_layers=1)

mvc = solver.solve_mvc(G)
mis = solver.solve_mis(G)

print(mvc.solution, mvc.objective)
print(mis.solution, mis.objective)
```

## Notebook example

- `/home/runner/work/Quantum-Greedy-MVC/Quantum-Greedy-MVC/examples/basic_usage.ipynb`

## Methods

- `quantum_greedy` (existing heuristic mixer)
- `qeg_ldf` (recursive Quantum Energy Greedy with LDF-like reduction)
- `greedy_degree`
- `primal_dual`
- `exact` and `lp_relaxation`

Reference: https://arxiv.org/pdf/2607.27915

## Notes

- The main interface is Python API (`QuantumGreedySolver`).
- CLI tools are not required for core usage.
