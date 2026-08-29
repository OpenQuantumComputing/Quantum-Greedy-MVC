from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class QegLdfNodeMapping(TypedDict):
    node: Any
    qubit: int


class QegLdfCandidateDiagnostic(TypedDict):
    node: Any
    qubit: int
    energy: float
    degree: int
    weight: float


class QegLdfStepDiagnostic(TypedDict):
    step: int
    chosen: Any
    chosen_energy: float
    candidates: list[QegLdfCandidateDiagnostic]
    mapping: list[QegLdfNodeMapping]
    remaining_edges_before: int
    remaining_edges_after: int


class QegLdfMetadata(TypedDict):
    time: float
    trotter_layers: int
    shots: int | None
    steps: list[QegLdfStepDiagnostic]


@dataclass(frozen=True)
class SolveResult:
    problem: str
    method: str
    solution: set[Any]
    objective: float
    feasible: bool
    metadata: dict[str, Any] = field(default_factory=dict)
