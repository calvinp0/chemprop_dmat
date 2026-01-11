from typing import Union, List, Dict, Optional

from rdkit import Chem

from .utils import MolWithXYZ, _check_mols_before_mapping, _identity_map, _choose_best_by_rmsd

def map_two_mols_rdkit(
    mol1: Union[Chem.Mol, MolWithXYZ],
    mol2: Union[Chem.Mol, MolWithXYZ],
    map_type: str = "list",
    consider_chirality: bool = True,
    use_geometry_tiebreak: bool = True,
    inc_vals: Optional[int] = None,
) -> Optional[Union[List[int], Dict[int, int]]]:
    """
    RDKit replacement for map_two_species():
    returns mapping from mol1 atom indices -> mol2 atom indices.

    Strategy:
    1) validate atom counts, formula counts, bond signatures
    2) special-cases for 1 atom, homonuclear diatomic, all-unique-elements
    3) otherwise: RDKit graph isomorphism via GetSubstructMatches (mol1 as query on mol2)
       - optionally chirality sensitive
       - if multiple matches exist, optionally pick best by RMSD

    Requirements:
    - If use_geometry_tiebreak=True, both mols should have 3D conformer 0.
    """
    if isinstance(mol1, MolWithXYZ):
        m1 = mol1.mol
    else:
        m1 = mol1
    if isinstance(mol2, MolWithXYZ):
        m2 = mol2.mol
    else:
        m2 = mol2

    if map_type not in ("list", "dict"):
        raise ValueError(f"map_type must be 'list' or 'dict', got {map_type}")

    if not _check_mols_before_mapping(m1, m2):
        return None

    n = m1.GetNumAtoms()

    # mono-atomic
    if n == 1:
        out = _identity_map(1, map_type)
        return out

    # homonuclear diatomic (e.g., H2, O2, N2, Cl2...)
    if n == 2 and len({a.GetSymbol() for a in m1.GetAtoms()}) == 1:
        out = _identity_map(2, map_type)
        return out

    # all different elements -> deterministic by symbol match
    if len({a.GetSymbol() for a in m1.GetAtoms()}) == n:
        sym_to_idx2 = {a.GetSymbol(): a.GetIdx() for a in m2.GetAtoms()}
        m12 = [sym_to_idx2[a.GetSymbol()] for a in m1.GetAtoms()]
        if inc_vals is not None:
            m12 = [j + inc_vals for j in m12]
        if map_type == "dict":
            return {i: m12[i] for i in range(n)}
        return m12

    # general case: graph isomorphism
    # Use mol1 as the query; for true isomorphism, this works because sizes are equal.
    matches = m2.GetSubstructMatches(m1, useChirality=consider_chirality, uniquify=True)
    if not matches:
        # try flipping chirality flag like ARC does
        matches = m2.GetSubstructMatches(m1, useChirality=not consider_chirality, uniquify=True)
    if not matches:
        return None

    if len(matches) == 1 or not use_geometry_tiebreak:
        best = matches[0]
    else:
        best = _choose_best_by_rmsd(m1, m2, matches)

    m12 = [int(best[i]) for i in range(n)]  # mol1 idx -> mol2 idx

    if inc_vals is not None:
        m12 = [j + inc_vals for j in m12]

    if map_type == "dict":
        return {i: m12[i] for i in range(n)}
    return m12
