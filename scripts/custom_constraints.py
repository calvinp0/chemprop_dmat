from __future__ import annotations

import numpy as np
from ase.constraints import FixConstraint
from ase.atoms import Atoms


class AdaptiveHookean(FixConstraint):
    """
    A Hookean-like constraint with an adaptive spring constant.

    Applies a harmonic potential V = 0.5 * k(r) * (r - rt)^2 between two atoms.
    The stiffness 'k' can adapt based on the current distance 'r' relative to the
    target 'rt'.

    Default behavior is a constant k if scaling_factor is 0.
    """

    def __init__(self, a1: int, a2: int, k_base: float, rt: float, scaling_factor: float = 0.0):
        """
        Args:
            a1: Index of the first atom.
            a2: Index of the second atom.
            k_base: Base spring constant (eV/Angstrom^2).
            rt: Target distance (Angstrom).
            scaling_factor: Factor to adjust k based on deviation. 
                            Example: k_eff = k_base * (1 + scaling_factor * |r - rt|)
        """
        self.a1 = a1
        self.a2 = a2
        self.indices = [a1, a2]
        self.k_base = k_base
        self.rt = rt
        self.scaling_factor = scaling_factor

    def _get_distance_vector(self, atoms: Atoms):
        p1 = atoms.positions[self.a1]
        p2 = atoms.positions[self.a2]
        diff = p1 - p2
        dist = np.linalg.norm(diff)
        return diff, dist

    def calculate_effective_k(self, dist: float) -> float:
        """
        Calculate the effective spring constant based on current distance.
        Override this method for custom adaptive logic.
        """
        deviation = abs(dist - self.rt)
        # Example: Increase stiffness as we get further away, or vice versa.
        # Here we linearly scale k with deviation.
        k_eff = self.k_base * (1.0 + self.scaling_factor * deviation)
        return k_eff

    def adjust_forces(self, atoms: Atoms, forces: np.ndarray):
        """
        Add spring forces to the atoms.
        
        Potential V(r) = 0.5 * k_base * (1 + alpha * |d|) * d^2
                       = 0.5 * k_base * d^2 + 0.5 * k_base * alpha * |d|^3
        
        Force F(r) = -dV/dr
                   = - (k_base * d + 1.5 * k_base * alpha * d * |d|)
                   = - k_base * d * (1 + 1.5 * alpha * |d|)
                   
        where d = r - rt.
        """
        diff, dist = self._get_distance_vector(atoms)
        if dist < 1e-12:
            return  # Avoid division by zero

        deviation = abs(dist - self.rt)
        
        # Effective stiffness for FORCE calculation (includes the 1.5 factor from derivative)
        k_force = self.k_base * (1.0 + 1.5 * self.scaling_factor * deviation)
        
        f_magnitude = -k_force * (dist - self.rt)
        f_vector = (diff / dist) * f_magnitude

        forces[self.a1] += f_vector
        forces[self.a2] -= f_vector

    def adjust_potential_energy(self, atoms: Atoms) -> float:
        _, dist = self._get_distance_vector(atoms)
        k_eff = self.calculate_effective_k(dist)
        energy = 0.5 * k_eff * (dist - self.rt)**2
        return energy

    def todict(self):
        return {
            "name": "AdaptiveHookean",
            "kwargs": {
                "a1": self.a1,
                "a2": self.a2,
                "k_base": self.k_base,
                "rt": self.rt,
                "scaling_factor": self.scaling_factor,
            },
        }

    def __repr__(self):
        return (
            f"AdaptiveHookean(a1={self.a1}, a2={self.a2}, "
            f"k_base={self.k_base}, rt={self.rt}, scaling={self.scaling_factor})"
        )
