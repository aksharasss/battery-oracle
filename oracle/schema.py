"""Common in-memory format every loader converts to.

Each dataset stores cycling data differently. Loaders turn one discharge of
one cell into a DischargeCurve so the ICA code never needs to know where
the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DischargeCurve:
    voltage: np.ndarray          # V, in time order
    capacity: np.ndarray         # Ah discharged since the start of this discharge
    cell_id: str
    cycle: int
    source: str                  # "severson", "sandia", "upload", "synthetic", ...
    current: np.ndarray | None = None      # A, negative on discharge
    temperature: np.ndarray | None = None  # degC
    c_rate: float | None = None            # nominal discharge C-rate, if known
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.voltage = np.asarray(self.voltage, dtype=float)
        self.capacity = np.asarray(self.capacity, dtype=float)
        if self.voltage.shape != self.capacity.shape:
            raise ValueError("voltage and capacity must have the same length")
        if self.voltage.size < 10:
            raise ValueError("discharge curve has fewer than 10 points")

    @property
    def total_capacity(self) -> float:
        return float(self.capacity.max() - self.capacity.min())
