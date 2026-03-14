from __future__ import annotations

import numpy as np

from fields import uniform_b_field, gradient_b_field, magnetic_mirror_field
from solver import simulate_particle, kinetic_energy
from plotting import plot_3d_trajectory, plot_xy_projection, plot_energy


def main() -> None:
    # -----------------------------
    # Particle parameters
    # -----------------------------
    charge = 1.0
    mass = 1.0

    # Initial conditions
    initial_position = np.array([0.0, 0.0, 0.0], dtype=float)
    initial_velocity = np.array([1.0, 0.5, 0.8], dtype=float)

    # Simulation parameters
    dt = 0.01
    steps = 5000

    # -----------------------------
    # Choose magnetic field model
    # -----------------------------
    B0 = 1.0

    def chosen_field(position: np.ndarray) -> np.ndarray:
        return uniform_b_field(position, B0=B0)

        # Try these later:
        # return gradient_b_field(position, B0=B0, alpha=0.03)
        # return magnetic_mirror_field(position, B0=B0, beta=0.01)

    # -----------------------------
    # Run simulation
    # -----------------------------
    times, positions, velocities = simulate_particle(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        charge=charge,
        mass=mass,
        magnetic_field=chosen_field,
        dt=dt,
        steps=steps,
    )

    energy = kinetic_energy(mass, velocities)

    # -----------------------------
    # Diagnostics
    # -----------------------------
    print("Simulation complete.")
    print(f"Initial position: {positions[0]}")
    print(f"Final position:   {positions[-1]}")
    print(f"Initial velocity: {velocities[0]}")
    print(f"Final velocity:   {velocities[-1]}")
    print(f"Initial kinetic energy: {energy[0]:.6f}")
    print(f"Final kinetic energy:   {energy[-1]:.6f}")
    print(f"Relative energy drift:  {(energy[-1] - energy[0]) / energy[0]:.6e}")

    # -----------------------------
    # Plots
    # -----------------------------
    plot_3d_trajectory(positions, title="Charged Particle in a Uniform Magnetic Field")
    plot_xy_projection(positions, title="Circular Motion in the XY Plane")
    plot_energy(times, energy, title="Kinetic Energy Conservation Check")


if __name__ == "__main__":
    main()
