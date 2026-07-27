# Lean formalization of the operator light cone

This directory contains a Lean 4 formalization of the support-propagation argument used in the operator-light-cone proposition of the accompanying paper.

The formalization machine-checks:

- the graph-neighbourhood and ball construction;
- the abstract single-step support conditions;
- the induction bounding the full support after n commutators;
- the stronger bound on the X/Y support;
- the binary symplectic representation of finite Pauli strings;
- the implication from local Pauli anticommutation to the support conditions used by the induction.

The formalization does not construct the full complex matrix representation of the many-qubit operators or formalize the coefficients and cancellations in the complete nested-commutator expansion.

## Verification

Install Lean through elan and run:

```bash
lake exe cache get
lake env lean LightCone.lean
