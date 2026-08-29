from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SolveResult:
    problem: str
    method: str
    solution: set[Any]
    objective: float
    feasible: bool
    metadata: dict[str, Any] = field(default_factory=dict)
