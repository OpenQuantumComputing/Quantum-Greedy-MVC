# Quantum-Greedy-MVC

An installable Python package for solving:
- **MVC**: weighted/unweighted Minimum Vertex Cover
- **MIS**: weighted/unweighted Maximum Independent Set

This repository includes the installable runtime package (`quantum_greedy_mvc`) and Lean formalization artifacts in `lean/`.

## What it solves

Given an undirected graph `G=(V,E)` and optional node weights:
- `solve_mvc(...)` returns a vertex cover `C \subseteq V` minimizing total weight.
- `solve_mis(...)` returns an independent set `I \subseteq V` maximizing total weight.

For weighted MIS, the implementation uses the standard identity:
`weight(MIS) = sum(weights) - weight(minimum weighted vertex cover)`
with `I = V \ C`.

## How it works (high level)

The package wraps quantum and classical baselines:
- `quantum_greedy` (existing heuristic mixer method, requires `qiskit` + `qiskit-aer`)
- `qeg_ldf` (recursive Quantum Energy Greedy with LDF-like reduction, requires `qiskit` + `qiskit-aer`)
- `greedy_degree`
- `primal_dual`
- `exact` and `lp_relaxation` (require `docplex`)

Reference: https://arxiv.org/pdf/2607.27915


### QEG–LDF method (new)

`qeg_ldf` is implemented as a separate solver method and does **not** change `quantum_greedy` or `greedy_degree`.

At each recursion step on the reduced graph `G_r`:
1. deterministically relabel current nodes to qubits,
2. initialize the all-ones feasible MVC state `|1...1>`,
3. for each non-isolated candidate `v`, build a conditioned mixer that omits the mixer on `v`,
4. approximate `exp(+i t H_MVC^(v))` with a first-order product formula using `p` Trotter layers,
5. evaluate conditioned expected cost and choose the best candidate,
6. include the chosen vertex in the cover and recurse on the reduced graph.

Sign convention used in code:
- Qiskit uses `RX(theta) = exp(-i theta X / 2)`.
- To implement `exp(+i Δt X)`, this package uses `RX(-2 * Δt)`.

The solver records per-step diagnostics in `SolveResult.metadata["qeg_ldf"]["steps"]`.


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
solver = QuantumGreedySolver(method="qeg_ldf", qeg_time=0.35, qeg_trotter_layers=1)

mvc = solver.solve_mvc(G)
mis = solver.solve_mis(G)

print(mvc.solution, mvc.objective)
print(mis.solution, mis.objective)
```

## CLI usage

```bash
qgmvc-solve --problem mvc --graph cycle:8 --method qeg_ldf --qeg-time 0.35 --qeg-trotter-layers 1
qgmvc-solve --problem mis --graph path:10 --method primal_dual
```

## Public API

- `quantum_greedy_mvc.QuantumGreedySolver`
- `quantum_greedy_mvc.SolveResult`

## Migration notes

This repository now focuses on the installable package API.

Current workflow:
- `pip install .`
- Import `QuantumGreedySolver`
- Call `solve_mvc(...)` / `solve_mis(...)` directly from Python

Legacy experiment scripts/notebooks were removed from the repository to keep the layout lean and package-focused.
