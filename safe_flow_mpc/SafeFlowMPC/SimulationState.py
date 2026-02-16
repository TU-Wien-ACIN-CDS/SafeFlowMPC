from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SimulationState:
    """Holds the current state of the simulation."""

    q: np.ndarray
    dq: np.ndarray = field(default_factory=lambda: np.zeros(7))
    ddq: np.ndarray = field(default_factory=lambda: np.zeros(7))
    dddq: np.ndarray = field(default_factory=lambda: np.zeros(7))
    timestep: int = field(default_factory=lambda: 0)

    # History tracking
    q_prev: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if len(self.q_prev) == 0:
            self.q_prev = [self.q.copy()] * 10
