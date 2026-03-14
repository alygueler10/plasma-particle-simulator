from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

Vector = np.ndarray
MagneticFieldFunction = Callable[[Vector], Vector]


def lorentz_acceleration(charge: float, mass: float, velocity: Vector, B: Vector) -> Vector:
    return (charge / mass) * np.cross(velocity, B)


def rk4_step(
    position: Vector,
    velocity: Vector,
    dt: float,
    charge: float,
    mass: float,
    magnetic_field: MagneticFieldFunction,
) -> Tuple[Vector, Vector]:

    def drdt(v: Vector) -> Vector:
        return v

    def dvdt(r: Vector, v: Vector) -> Vector:
        B = magnetic_field(r)
        return lorentz_acceleration(charge, mass, v, B)

    k1_r = drdt(velocity)
    k1_v = dvdt(position, velocity)

    k2_r = drdt(velocity + 0.5 * dt * k1_v)
    k2_v = dvdt(position + 0.5 * dt * k1_r, velocity + 0.5 * dt * k1_v)

    k3_r = drdt(velocity + 0.5 * dt * k2_v)
    k3_v = dvdt(position + 0.5 * dt * k2_r, velocity + 0.5 * dt * k2_v)

    k4_r = drdt(velocity + dt * k3_v)
    k4_v = dvdt(position + dt * k3_r, velocity + dt * k3_v)

    new_position = position + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    new_velocity = velocity + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return new_position, new_velocity


def simulate_particle(
    initial_position: Vector,
    initial_velocity: Vector,
    charge: float,
    mass: float,
    magnetic_field: MagneticFieldFunction,
    dt: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.zeros((steps + 1, 3), dtype=float)
    velocities = np.zeros((steps + 1, 3), dtype=float)
    times = np.zeros(steps + 1, dtype=float)

    positions[0] = initial_position
    velocities[0] = initial_velocity

    position = initial_position.astype(float).copy()
    velocity = initial_velocity.astype(float).copy()

    for i in range(steps):
        position, velocity = rk4_step(
            position=position,
            velocity=velocity,
            dt=dt,
            charge=charge,
            mass=mass,
            magnetic_field=magnetic_field,
        )
        positions[i + 1] = position
        velocities[i + 1] = velocity
        times[i + 1] = times[i] + dt

    return times, positions, velocities


def kinetic_energy(mass: float, velocities: np.ndarray) -> np.ndarray:
    speeds_squared = np.sum(velocities**2, axis=1)
    return 0.5 * mass * speeds_squared
