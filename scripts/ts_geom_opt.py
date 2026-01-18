from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ase import Atoms
from ase.constraints import FixInternals, Hookean
from ase.optimize import BFGS
from ase.optimize.sciopt import Converged, OptimizerConvergenceError, SciPyFminBFGS, SciPyFminCG
from xtb.ase.calculator import XTB


def convert_list_index_0_to_1(_list: list, direction: int = 1):
    """
    Convert a list from 0-indexed to 1-indexed, or vice versa.
    Ensures positive values in the resulting list.
    """
    new_list = [item + direction for item in _list]
    if any(val < 0 for val in new_list):
        raise ValueError(f"The resulting list from converting {_list} has negative values:\n{new_list}")
    if isinstance(_list, tuple):
        new_list = tuple(new_list)
    return new_list


@dataclass(frozen=True)
class XYZSpec:
    symbols: tuple[str, ...]
    isotopes: tuple[int, ...]
    coords: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ConstraintSpec:
    atom_indices_1based: tuple[int, int]
    distance: float


def build_atoms(xyz: XYZSpec) -> Atoms:
    return Atoms(xyz.symbols, xyz.coords)


def _build_constraints(constraints: Iterable[ConstraintSpec]) -> list:
    hooks = []
    bonds = []
    for c in constraints:
        a1, a2 = c.atom_indices_1based
        hooks.append(Hookean(a1=a1 - 1, a2=a2 - 1, k=15.0, rt=c.distance))
    return [*hooks, FixInternals(bonds=bonds)]


def multiplicity_to_uhf(multiplicity: int) -> int:
    uhf = multiplicity - 1
    if uhf < 0:
        raise ValueError(f"Invalid multiplicity: {multiplicity}")
    return uhf


def run_opt(
    xyz: XYZSpec,
    constraints: Optional[Iterable[ConstraintSpec]] = None,
    fmax: float = 0.001,
    steps: Optional[int] = None,
    engine: str = "SciPyFminBFGS",
    model=None,
    use_xtb: bool = True,
    charge: int = 0,
    multiplicity: int = 1,
    diagnostics: Optional[dict] = None,
) -> Optional[XYZSpec]:
    """
    Run a geometry optimization calculation with optional constraints.
    Converges when all forces are less than ``fmax`` or ``steps`` exceeded.
    """
    steps = steps or 1000
    atoms = build_atoms(xyz)

    if use_xtb:
        uhf = multiplicity_to_uhf(multiplicity)
        atoms.set_calculator(XTB(method="GFN2-xTB", charge=charge, uhf=uhf))
    elif model is not None:
        atoms.set_calculator(model.ase())
    else:
        raise ValueError("Either set use_xtb=True or pass a torchani model.")

    if constraints is not None:
        atoms.set_constraint(_build_constraints(constraints))

    engine_list = ["bfgs", "scipyfminbfgs", "scipyfmincg"]
    engine_set = set([engine] + engine_list)
    engine_dict = {"bfgs": BFGS, "scipyfminbfgs": SciPyFminBFGS, "scipyfmincg": SciPyFminCG}
    for opt_engine_name in engine_set:
        opt_engine = engine_dict[opt_engine_name.lower()]
        opt = opt_engine(atoms, logfile=None)
        try:
            opt.run(fmax=fmax, steps=steps)
        except (Converged, NotImplementedError, OptimizerConvergenceError):
            pass
        else:
            break
    else:
        return None

    if diagnostics is not None:
        forces = atoms.get_forces()
        fmax_actual = ((forces * forces).sum(axis=1) ** 0.5).max()
        diagnostics["fmax_actual"] = float(fmax_actual)
        if constraints is not None:
            coords = atoms.get_positions()
            max_dev = None
            for spec in constraints:
                a1, a2 = spec.atom_indices_1based
                dist = float(((coords[a1 - 1] - coords[a2 - 1]) ** 2).sum() ** 0.5)
                dev = abs(dist - spec.distance)
                max_dev = dev if max_dev is None else max(max_dev, dev)
            diagnostics["max_constraint_deviation"] = max_dev

    opt_xyz = XYZSpec(
        coords=tuple(map(tuple, atoms.get_positions().tolist())),
        isotopes=xyz.isotopes,
        symbols=xyz.symbols,
    )
    return opt_xyz
