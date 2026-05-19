
"""
model.py — Linear interpolation (no ML).

Replaces the UNet AE with deterministic, pointwise interpolation between
two planes based on the angular gap.
"""

from dataclasses import dataclass

import numpy as np

from data_utils import select_bracketing_indices


@dataclass
class LinearInterpolationResult:
    plane: np.ndarray
    idx_low: int
    idx_high: int
    weight: float
    clipped: bool


class LinearInterpolator:
    def __init__(self, planes, angles):
        self.planes = np.asarray(planes, dtype=np.float32)
        self.angles = np.asarray(angles, dtype=np.float32)

        if self.planes.ndim != 3:
            raise ValueError("planes must be (N, H, W)")
        if self.planes.shape[0] != self.angles.shape[0]:
            raise ValueError("planes and angles length mismatch")

    def interpolate(self, target_angle):
        idx_low, idx_high, weight, clipped = select_bracketing_indices(
            self.angles, target_angle
        )
        plane = (1.0 - weight) * self.planes[idx_low] + weight * self.planes[idx_high]
        return LinearInterpolationResult(
            plane=plane,
            idx_low=idx_low,
            idx_high=idx_high,
            weight=weight,
            clipped=clipped,
        )

