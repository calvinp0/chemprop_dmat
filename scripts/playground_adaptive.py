#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure we can import from local scripts
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from ase import Atoms
from ase.constraints import Hookean
from ase.calculators.emt import EMT
from scripts.custom_constraints import AdaptiveHookean
from ase.optimize import BFGS

def make_water(dist=0.96):
    """Creates a water molecule with one O-H bond at `dist`."""
    # O at origin, H1 on x-axis at `dist`, H2 arbitrary
    atoms = Atoms('H2O', positions=[[dist, 0, 0], [0, 0, 0], [0, 1, 0]])
    calc = EMT()
    atoms.calc = calc
    return atoms

def test_constraints():
    # Parameters
    a1, a2 = 0, 1  # Indices for H1 and O
    rt = 0.96      # Target distance (equilibrium)
    k_base = 10.0  # Base spring constant
    scaling = 15.0  # Adaptive scaling factor
    
    print(f"--- Constraint Playground ---")
    print(f"Bond: O-H (Indices {a1}-{a2})")
    print(f"Target Distance (rt): {rt:.2f} Angstrom")
    print(f"Base k: {k_base}")
    print(f"Adaptive Scaling: {scaling}")
    print(f"Formula: k_eff = k_base * (1 + scaling * |r - rt|)\n")

    print(f"{ 'Dist (A)':<10} | { 'Dev (A)':<10} | { 'Std Force':<10} | { 'Adapt Force':<12} | { 'k_eff':<10} || { 'Final Std':<10} | { 'Final Adapt':<10}")
    print("-" * 95)

    # Test range: slightly compressed to very stretched
    distances = np.linspace(0.90, 1.50, 7)

    for r in distances:
        # Setup Standard Hookean
        atoms_std = make_water(r)
        cons_std = Hookean(a1=a1, a2=a2, k=k_base, rt=rt)
        atoms_std.set_constraint(cons_std)
        
        # Calculate forces (dummy calculation needed to trigger constraints usually, 
        # but adjust_forces is direct in ASE if we call it manually or via get_forces)
        # Note: ASE constraints modify forces *during* get_forces() call.
        # We need a calculator attached to get base forces, but here we just want constraint forces.
        # We can manually call adjust_forces with zero array to isolate constraint contribution.
        
        forces_std = np.zeros((3, 3))
        cons_std.adjust_forces(atoms_std, forces_std)
        f_std_mag = np.linalg.norm(forces_std[a1])

        # Setup Adaptive Hookean
        atoms_adapt = make_water(r)
        cons_adapt = AdaptiveHookean(a1=a1, a2=a2, k_base=k_base, rt=rt, scaling_factor=scaling)
        atoms_adapt.set_constraint(cons_adapt)
        
        forces_adapt = np.zeros((3, 3))
        cons_adapt.adjust_forces(atoms_adapt, forces_adapt)
        f_adapt_mag = np.linalg.norm(forces_adapt[a1])
        
        # Calculate expected k_eff for display
        dev = abs(r - rt)
        k_eff = k_base * (1 + scaling * dev)

        # Optimize Standard
        atoms_std.calc = EMT()
        dyn_std = BFGS(atoms_std, logfile=None)
        dyn_std.run(fmax=0.05)
        final_dist_std = np.linalg.norm(atoms_std.positions[a1] - atoms_std.positions[a2])

        # Optimize Adaptive
        atoms_adapt.calc = EMT()
        dyn_adapt = BFGS(atoms_adapt, logfile=None)
        dyn_adapt.run(fmax=0.05)
        final_dist_adapt = np.linalg.norm(atoms_adapt.positions[a1] - atoms_adapt.positions[a2])

        print(f"{r:<10.2f} | {dev:<10.2f} | {f_std_mag:<10.2f} | {f_adapt_mag:<12.2f} | {k_eff:<10.2f} || {final_dist_std:<10.4f} | {final_dist_adapt:<10.4f}")

if __name__ == "__main__":
    test_constraints()
