from pathlib import Path

import pytest
from rdkit import Chem

from heuristics.heuristic import combine_coordinates_with_redundant_atoms
import json

from heuristics.sdf_to_zmat import (
    read_rdkit_mols_from_sdf,
    rdkit_mol_to_xyz_dict,
    select_mol_from_sdf,
)
from heuristics.utils import ZMatError


def _pick_sdf_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "DATA" / "SDF" / "deduped_sdf" / "kfir_rxn_10218.sdf",
        repo_root / "DATA" / "SDF" / "rmg_rxn_276.sdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    pytest.skip("No SDF available for heuristic tests")


def _pick_terminal_h(mol: Chem.Mol) -> int:
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H" and atom.GetDegree() == 1:
            return atom.GetIdx()
    return -1


def _pick_c_or_d(mol: Chem.Mol, h_idx: int) -> int:
    h_atom = mol.GetAtomWithIdx(h_idx)
    neighbors = list(h_atom.GetNeighbors())
    if len(neighbors) != 1:
        return -1
    a_idx = neighbors[0].GetIdx()
    a_atom = mol.GetAtomWithIdx(a_idx)
    for neighbor in a_atom.GetNeighbors():
        idx = neighbor.GetIdx()
        if idx not in (a_idx, h_idx):
            return idx
    return -1


def _pick_indices_from_mol_properties(
    mol: Chem.Mol,
    a_label: str,
    h_label: str,
) -> tuple[int, int, int]:
    if not mol.HasProp("mol_properties"):
        return -1, -1, -1
    props = json.loads(mol.GetProp("mol_properties"))
    a_idx = next((int(idx) for idx, v in props.items() if v.get("label") == a_label), -1)
    h_idx = next((int(idx) for idx, v in props.items() if v.get("label") == h_label), -1)
    if a_idx == -1 or h_idx == -1:
        return a_idx, h_idx, -1
    c_idx = -1
    for neighbor in mol.GetAtomWithIdx(a_idx).GetNeighbors():
        n_idx = neighbor.GetIdx()
        if n_idx not in (a_idx, h_idx):
            c_idx = n_idx
            break
    return a_idx, h_idx, c_idx


def test_combine_coordinates_with_redundant_atoms_smoke():
    sdf_path = _pick_sdf_path()
    prop_key = "type"
    r1h_value = "r1h"
    r2h_value = "r2h"

    mols = read_rdkit_mols_from_sdf(str(sdf_path))
    if len(mols) < 2:
        pytest.skip("SDF must contain at least two molecules (R1H and R2H)")

    if prop_key and r1h_value and r2h_value:
        try:
            mol_1 = select_mol_from_sdf(str(sdf_path), prop_key=prop_key, prop_value=r1h_value)
            mol_2 = select_mol_from_sdf(str(sdf_path), prop_key=prop_key, prop_value=r2h_value)
        except ZMatError:
            mol_1, mol_2 = mols[0], mols[1]
    else:
        mol_1, mol_2 = mols[0], mols[1]

    xyz_1 = rdkit_mol_to_xyz_dict(mol_1)
    xyz_2 = rdkit_mol_to_xyz_dict(mol_2)

    a1, h1, c = _pick_indices_from_mol_properties(mol_1, "donator", "d_hydrogen")
    a2, h2, d = _pick_indices_from_mol_properties(mol_2, "acceptor", "a_hydrogen")

    if h1 == -1:
        h1 = _pick_terminal_h(mol_1)
    if h2 == -1:
        h2 = _pick_terminal_h(mol_2)
    if h1 == -1 or h2 == -1:
        pytest.skip("No terminal hydrogens found in SDF")

    if c == -1:
        c = _pick_c_or_d(mol_1, h1)
    if d == -1:
        d = _pick_c_or_d(mol_2, h2)
    if mol_1.GetNumAtoms() > 2 and c == -1:
        pytest.skip("Unable to select C for redundant-atom combine")
    if mol_2.GetNumAtoms() > 2 and d == -1:
        pytest.skip("Unable to select C/D neighbors for redundant-atom combine")

    coords = combine_coordinates_with_redundant_atoms(
        xyz_1=xyz_1,
        xyz_2=xyz_2,
        mol_1=mol_1,
        mol_2=mol_2,
        h1=h1,
        h2=h2,
        c=None if c == -1 else c,
        d=None if d == -1 else d,
        a2=180.0,
    )

    assert len(coords["symbols"]) == len(coords["coords"])
    assert all(sym != "X" for sym in coords["symbols"])
