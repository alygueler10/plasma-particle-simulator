from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_3d_trajectory(positions: np.ndarray, title: str = "Particle Trajectory") -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], linewidth=1.5)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], s=60, label="Start")
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], s=60, label="End")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_xy_projection(positions: np.ndarray, title: str = "XY Projection") -> None:
    plt.figure(figsize=(7, 7))
    plt.plot(positions[:, 0], positions[:, 1], linewidth=1.5)
    plt.scatter(positions[0, 0], positions[0, 1], label="Start")
    plt.scatter(positions[-1, 0], positions[-1, 1], label="End")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_energy(times: np.ndarray, energy: np.ndarray, title: str = "Kinetic Energy vs Time") -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(times, energy, linewidth=1.5)
    plt.xlabel("Time")
    plt.ylabel("Kinetic Energy")
    plt.title(title)
    plt.tight_layout()
    plt.show()
