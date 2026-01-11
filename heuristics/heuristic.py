from typing import Optional, Tuple, Dict, Any, List, Union

from rdkit import Chem

from .sdf_to_zmat import xyz_to_zmat_rdkit
from .mapping import map_two_mols_rdkit
from .utils import (
    get_parameter_from_atom_indices,
    get_new_map_based_on_zmat_1,
    is_angle_linear,
    is_xyz_linear,
    key_by_val,
    remove_rdkit_atom,
    remove_zmat_atom_0,
    update_new_map_based_on_zmat_2,
    up_param,
    zmat_to_xyz,
)


def _get_single_neighbor_index(mol: Chem.Mol, atom_index: int) -> int:
    neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(atom_index).GetNeighbors()]
    if len(neighbors) != 1:
        raise ValueError(f"Atom {atom_index} must have exactly one neighbor, got {neighbors}")
    return neighbors[0]


def find_distant_neighbor(mol: Chem.Mol, start: int) -> Optional[int]:
    """
    Find a distant neighbor (two steps away) from start; prefer heavy atoms.
    """
    if mol.GetNumAtoms() <= 2:
        return None
    distant_h = None
    start_atom = mol.GetAtomWithIdx(start)
    for neighbor in start_atom.GetNeighbors():
        for distant_neighbor in neighbor.GetNeighbors():
            idx = distant_neighbor.GetIdx()
            if idx == start:
                continue
            if distant_neighbor.GetSymbol() == "H":
                distant_h = idx
            else:
                return idx
    return distant_h


def _validate_combine_coordinates_with_redundant_atoms_args(
    xyz_1: dict,
    xyz_2: dict,
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    h1: int,
    h2: int,
    a2: float,
    d2: Optional[float],
    d3: Optional[float],
    c: Optional[int],
    d: Optional[int],
) -> Tuple[bool, bool, int, int]:
    """
    Validate and normalize all the combine_coordinates… parameters.

    Returns:
      is_a2_linear, is_mol_1_linear, a_idx, b_idx
    """
    if not isinstance(xyz_1, dict) or not isinstance(xyz_2, dict):
        raise TypeError(f"xyz_1 and xyz_2 must be dicts, got {type(xyz_1)} and {type(xyz_2)}")

    is_a2_linear = is_angle_linear(a2)
    is_mol_1_linear = is_xyz_linear(xyz_1)
    num1, num2 = mol_1.GetNumAtoms(), mol_2.GetNumAtoms()

    if num1 < 2 or num2 < 2:
        raise ValueError(f"Each fragment needs ≥2 atoms, got {num1} and {num2}")

    # if angle is linear we ignore d2
    d2 = None if is_a2_linear else d2
    if d2 is None and not is_a2_linear and num1 > 2:
        raise ValueError("d2 must be given when a2 is non-linear and mol_1 has ≥3 atoms")

    if d3 is None and not is_a2_linear and num2 > 2:
        raise ValueError("d3 must be given when a2 is non-linear and mol_2 has ≥3 atoms")

    if c is None and num1 > 2:
        raise ValueError("c must be given when mol_1 has ≥3 atoms")

    if d is None and num2 > 2:
        raise ValueError("d must be given when mol_2 has ≥3 atoms")

    a_idx = _get_single_neighbor_index(mol_1, h1)
    b_idx = _get_single_neighbor_index(mol_2, h2)
    if c == a_idx:
        raise ValueError(f"c ({c}) cannot be the same as A index {a_idx}")
    if d == b_idx:
        raise ValueError(f"d ({d}) cannot be the same as B index {b_idx}")
    return is_a2_linear, is_mol_1_linear, a_idx, b_idx


def generate_the_two_constrained_zmats(
    xyz_1: dict,
    xyz_2: dict,
    rdmol_1: Chem.Mol,
    rdmol_2: Chem.Mol,
    h1: int,
    h2: int,
    a: int,
    b: int,
    c: Optional[int],
    d: Optional[int],
) -> Tuple[dict, dict]:
    # DO NOT add hydrogens – indices must match xyz exactly

    constraints1 = {"A_group": [(h1, a, c)], "R_atom": [(h1, a)]} if c is not None else {"R_atom": [(h1, a)]}
    constraints2 = {"R_group": [(b, h2)], "A_group": [(d, b, h2)]} if d is not None else {"R_group": [(b, h2)]}

    zmat1 = xyz_to_zmat_rdkit(
        xyz=xyz_1,
        rdmol=rdmol_1,
        constraints=constraints1,
        consolidate=False,
        use_rmg_atom_order=True,
    )

    zmat2 = xyz_to_zmat_rdkit(
        xyz=xyz_2,
        rdmol=rdmol_2,
        constraints=constraints2,
        consolidate=False,
        use_rmg_atom_order=True,
    )

    return zmat1, zmat2


def stretch_zmat_bond(zmat: dict, indices: Tuple[int, int], stretch: float) -> None:
    """
    Stretch a bond in a zmat.

    Args:
        zmat: The zmat to process.
        indices (tuple): A length 2 tuple with the 0-indices of the xyz (not zmat) atoms representing the bond to stretch.
        stretch (float): The factor by which to multiply (stretch/shrink) the bond length.
    """
    param = get_parameter_from_atom_indices(zmat=zmat, indices=indices, xyz_indexed=True)
    zmat["vars"][param] *= stretch


def determine_glue_params(
    zmat: dict,
    add_dummy: bool,
    h1: int,
    a: int,
    c: Optional[int],
    d: Optional[int],
) -> Tuple[str, Optional[str], Optional[str]]:
    num_atoms_1 = len(zmat["symbols"])
    z_a = key_by_val(zmat["map"], a)
    z_h1 = key_by_val(zmat["map"], h1)
    z_b = num_atoms_1 + int(add_dummy)
    z_c = key_by_val(zmat["map"], c) if c is not None else None
    z_d = num_atoms_1 + 1 + int(add_dummy) if d is not None else None

    if add_dummy:
        zmat["map"][len(zmat["symbols"])] = f"X{len(zmat['symbols'])}"
        z_x = num_atoms_1
        zmat["symbols"] = tuple(list(zmat["symbols"]) + ["X"])
        r_str = f"RX_{z_x}_{z_h1}"
        a_str = f"AX_{z_x}_{z_h1}_{z_a}"
        d_str = f"DX_{z_x}_{z_h1}_{z_a}_{z_c}" if z_c is not None else None
        zmat["coords"] = tuple(list(zmat["coords"]) + [(r_str, a_str, d_str)])
        zmat["vars"][r_str] = 1.0
        zmat["vars"][a_str] = 90.0
        if d_str is not None:
            zmat["vars"][d_str] = 0.0
        param_a2 = f"A_{z_b}_{z_h1}_{z_a}"
        param_d2 = f"D_{z_b}_{z_h1}_{z_x}_{z_a}"
        param_d3 = f"D_{z_d}_{z_b}_{z_a}_{z_c if c is not None else z_x}" if d is not None else None
    else:
        param_a2 = f"A_{z_b}_{z_h1}_{z_a}"
        param_d2 = f"D_{z_b}_{z_h1}_{z_a}_{z_c}" if z_c is not None else None
        param_d3 = f"D_{z_d}_{z_b}_{z_h1}_{z_a}" if d is not None else None

    return param_a2, param_d2, param_d3


def get_new_zmat_2_map(
    zmat_1: dict,
    zmat_2: dict,
    reactant_2_mol: Chem.Mol,
    zmat_2_mol: Chem.Mol,
    reactants_reversed: bool = False,
) -> Dict[int, Union[int, str]]:
    new_map = get_new_map_based_on_zmat_1(zmat_1=zmat_1, zmat_2=zmat_2, reactants_reversed=reactants_reversed)

    zmat_2_mod = remove_zmat_atom_0(zmat_2)
    redundant_idx = zmat_2["map"][0]
    if not isinstance(redundant_idx, int):
        raise ValueError(f"Expected redundant atom index to be int, got {redundant_idx}")
    zmat_2_mod_mol = remove_rdkit_atom(zmat_2_mol, redundant_idx)

    reactant_2_for_map = reactant_2_mol
    if reactant_2_mol.GetNumAtoms() == zmat_2_mod_mol.GetNumAtoms() + 1:
        reactant_2_for_map = remove_rdkit_atom(reactant_2_mol, redundant_idx)
    atom_map = map_two_mols_rdkit(
        mol1=zmat_2_mod_mol,
        mol2=reactant_2_for_map,
        consider_chirality=False,
        use_geometry_tiebreak=True,
        map_type="list",
    )
    if atom_map is None:
        raise ValueError("Could not map zmat_2_mod_mol to reactant_2_mol.")

    new_map = update_new_map_based_on_zmat_2(
        new_map=new_map,
        zmat_2=zmat_2_mod,
        num_atoms_1=len(zmat_1["symbols"]),
        atom_map=atom_map,
        reactants_reversed=reactants_reversed,
    )

    if len(list(new_map.values())) != len(set(new_map.values())):
        raise ValueError(f"Could not generate a combined zmat map with no repeating values.\n{new_map}")

    return new_map


def get_modified_params_from_zmat_2(
    zmat_1: dict,
    zmat_2: dict,
    reactant_2_mol: Chem.Mol,
    zmat_2_mol: Chem.Mol,
    add_dummy: bool,
    glue_params: Tuple[str, Optional[str], Optional[str]],
    h1: int,
    a: int,
    c: Optional[int],
    a2: float,
    d2: Optional[float],
    d3: Optional[float],
    reactants_reversed: bool = False,
) -> Tuple[tuple, tuple, dict, dict]:
    new_symbols = tuple(zmat_1["symbols"] + zmat_2["symbols"][1:])
    new_coords, new_vars = list(), dict()
    param_a2, param_d2, param_d3 = glue_params
    num_atoms_1 = len(zmat_1["symbols"])

    for i, coords in enumerate(zmat_2["coords"][1:]):
        new_coord = list()
        for j, param in enumerate(coords):
            if param is not None:
                if i == 0:
                    new_param = f"R_{num_atoms_1}_{h1}"
                elif i == 1 and j == 0:
                    new_param = f"R_{num_atoms_1 + 1}_{num_atoms_1}"
                elif i == 1 and j == 1:
                    new_param = f"A_{num_atoms_1 + 1}_{num_atoms_1}_{a}"
                else:
                    new_param = up_param(param=param, increment=num_atoms_1 - 1)
                new_coord.append(new_param)
                new_vars[new_param] = zmat_2["vars"][param]
            else:
                if i == 0 and j == 1:
                    new_coord.append(param_a2)
                    new_vars[param_a2] = a2 + 90 if add_dummy else a2
                elif i == 0 and j == 2 and c is not None:
                    new_coord.append(param_d2)
                    new_vars[param_d2] = 0.0 if d2 is None else d2
                elif i == 1 and j == 2 and param_d3 is not None:
                    new_coord.append(param_d3)
                    new_vars[param_d3] = 0.0 if d3 is None else d3
                else:
                    new_coord.append(None)
        new_coords.append(tuple(new_coord))

    new_map = get_new_zmat_2_map(
        zmat_1=zmat_1,
        zmat_2=zmat_2,
        reactant_2_mol=reactant_2_mol,
        zmat_2_mol=zmat_2_mol,
        reactants_reversed=reactants_reversed,
    )
    new_coords = tuple(list(zmat_1["coords"]) + new_coords)
    new_vars = {**zmat_1["vars"], **new_vars}
    return new_symbols, new_coords, new_vars, new_map


def combine_coordinates_with_redundant_atoms(
    xyz_1: Dict[str, Any],
    xyz_2: Dict[str, Any],
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    h1: int,
    h2: int,
    c: Optional[int] = None,
    d: Optional[int] = None,
    r1_stretch: float = 1.2,
    r2_stretch: float = 1.2,
    a2: float = 180.0,
    d2: Optional[float] = None,
    d3: Optional[float] = None,
    keep_dummy: bool = False,
    reactants_reversed: bool = False,
) -> Dict[str, Any]:
    """
    Combine two coordinates that share an atom.
    For this redundant atom case, only three additional degrees of freedom (here ``a2``, ``d2``, and ``d3``)
    are required.
    """
    if c is None and mol_1.GetNumAtoms() > 2:
        c = find_distant_neighbor(mol_1, h1)
    if d is None and mol_2.GetNumAtoms() > 2:
        d = find_distant_neighbor(mol_2, h2)

    is_a2_linear, is_mol_1_linear, a, b = _validate_combine_coordinates_with_redundant_atoms_args(
        xyz_1, xyz_2, mol_1, mol_2, h1, h2, a2, d2, d3, c, d
    )
    zmat_1, zmat_2 = generate_the_two_constrained_zmats(
        xyz_1, xyz_2, mol_1, mol_2, h1, h2, a, b, c, d
    )
    stretch_zmat_bond(zmat=zmat_1, indices=(h1, a), stretch=r1_stretch)
    stretch_zmat_bond(zmat=zmat_2, indices=(b, h2), stretch=r2_stretch)

    add_dummy = is_a2_linear and len(zmat_1["symbols"]) > 2 and not is_mol_1_linear
    glue_params = determine_glue_params(
        zmat=zmat_1,
        add_dummy=add_dummy,
        a=a,
        h1=h1,
        c=c,
        d=d,
    )
    new_symbols, new_coords, new_vars, new_map = get_modified_params_from_zmat_2(
        zmat_1=zmat_1,
        zmat_2=zmat_2,
        reactant_2_mol=mol_2,
        zmat_2_mol=mol_2,
        add_dummy=add_dummy,
        glue_params=glue_params,
        a=a,
        h1=h1,
        c=c,
        a2=a2,
        d2=d2,
        d3=d3,
        reactants_reversed=reactants_reversed,
    )
    combined_zmat = {"symbols": new_symbols, "coords": new_coords, "vars": new_vars, "map": new_map}
    for i, coords in enumerate(combined_zmat["coords"]):
        if i > 2 and None in coords:
            raise ValueError(
                "Could not combine zmats, got a None parameter beyond the 3rd row:\n"
                f"{coords} in:\n{combined_zmat}"
            )

    xyz = zmat_to_xyz(combined_zmat, keep_dummy=keep_dummy)
    if not keep_dummy:
        return xyz
    return xyz


def combine_coordinates_with_redundant_atoms_scan(
    xyz_1: Dict[str, Any],
    xyz_2: Dict[str, Any],
    mol_1: Chem.Mol,
    mol_2: Chem.Mol,
    h1: int,
    h2: int,
    c: Optional[int] = None,
    d: Optional[int] = None,
    r1_stretch: float = 1.2,
    r2_stretch: float = 1.2,
    a2: float = 180.0,
    dihedral_increment: int = 30,
    keep_dummy: bool = False,
    reactants_reversed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate a set of TS guesses by scanning the d2/d3 dihedral angles (ARC-style).
    """
    d2_values = list()
    d3_values = list()

    if len(mol_1.GetAtoms()) > 2 and not is_angle_linear(a2):
        d2_values = list(range(0, 360, dihedral_increment))
    if len(mol_2.GetAtoms()) > 2:
        d3_values = list(range(0, 360, dihedral_increment))

    if d2_values and d3_values:
        d2_d3_pairs = [(d2, d3) for d2 in d2_values for d3 in d3_values]
    elif d2_values:
        d2_d3_pairs = [(d2, None) for d2 in d2_values]
    elif d3_values:
        d2_d3_pairs = [(None, d3) for d3 in d3_values]
    else:
        d2_d3_pairs = [(None, None)]

    guesses: List[Dict[str, Any]] = []
    for d2, d3 in d2_d3_pairs:
        guesses.append(
            combine_coordinates_with_redundant_atoms(
                xyz_1=xyz_1,
                xyz_2=xyz_2,
                mol_1=mol_1,
                mol_2=mol_2,
                h1=h1,
                h2=h2,
                c=c,
                d=d,
                r1_stretch=r1_stretch,
                r2_stretch=r2_stretch,
                a2=a2,
                d2=d2,
                d3=d3,
                keep_dummy=keep_dummy,
                reactants_reversed=reactants_reversed,
            )
        )

    return guesses
