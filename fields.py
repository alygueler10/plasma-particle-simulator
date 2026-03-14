from __future__ import annotations

import numpy as np


def uniform_b_field(position: np.ndarray, B0: float = 1.0) -> np.ndarray:
    """
    Uniform magnetic field in the z-direction.
    """
    return np.array([0.0, 0.0, B0], dtype=float)


def gradient_b_field(position: np.ndarray, B0: float = 1.0, alpha: float = 0.05) -> np.ndarray:
    """
    Simple magnetic field with a z-gradient:
        B = (0, 0, B0 * (1 + alpha * z))
    """
    z = position[2]
    return np.array([0.0, 0.0, B0 * (1.0 + alpha * z)], dtype=float)


def magnetic_mirror_field(position: np.ndarray, B0: float = 1.0, beta: float = 0.02) -> np.ndarray:
    """
    Toy magnetic mirror-like field:
        Bz = B0 * (1 + beta * z^2)
    """
    z = position[2]
    return np.array([0.0, 0.0, B0 * (1.0 + beta * z**2)], dtype=float)
