#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Set

from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from heuristics.sdf_to_zmat import get_connectivity_from_rdkit  # noqa: E402

sys.path.insert(0, "/home/calvin/code/RMG-Py")
from rmgpy.molecule.molecule import Molecule  # noqa: E402
from rmgpy.molecule import converter  # noqa: E402


def _adj_from_rmg(mol: Molecule) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {}
    for i, atom in enumerate(mol.atoms):
        adj[i] = {mol.atoms.index(nbr) for nbr in atom.edges}
    return adj


def _compare_adjs(label: str, rd_adj: Dict[int, Set[int]], rmg_adj: Dict[int, Set[int]]) -> None:
    rd_atoms = len(rd_adj)
    rmg_atoms = len(rmg_adj)
    print(f"{label}: RDKit atoms={rd_atoms}, RMG atoms={rmg_atoms}")
    if rd_atoms != rmg_atoms:
        print("  atom count mismatch (likely implicit/explicit H handling)")

    max_atoms = max(rd_atoms, rmg_atoms)
    diffs = 0
    for i in range(max_atoms):
        rd_nbrs = rd_adj.get(i, set())
        rmg_nbrs = rmg_adj.get(i, set())
        if rd_nbrs != rmg_nbrs:
            diffs += 1
            missing = sorted(rmg_nbrs - rd_nbrs)
            extra = sorted(rd_nbrs - rmg_nbrs)
            print(f"  atom {i}: missing={missing} extra={extra}")
    if diffs == 0:
        print("  connectivity matches")


def _remap_rmg_adj(rmg_adj: Dict[int, Set[int]], rmg_to_orig: Dict[int, int]) -> Dict[int, Set[int]]:
    remapped: Dict[int, Set[int]] = {}
    for rmg_idx, nbrs in rmg_adj.items():
        if rmg_idx not in rmg_to_orig:
            continue
        orig_idx = rmg_to_orig[rmg_idx]
        remapped[orig_idx] = {rmg_to_orig[n] for n in nbrs if n in rmg_to_orig}
    return remapped


def _select_mols(mols, prop_key: Optional[str], prop_value: Optional[str], index: Optional[int]):
    if prop_key and prop_value:
        filtered = [m for m in mols if m is not None and m.HasProp(prop_key) and m.GetProp(prop_key) == prop_value]
        if not filtered:
            raise ValueError(f"No mol with {prop_key}={prop_value}")
        return filtered
    if index is not None:
        if index < 0 or index >= len(mols):
            raise ValueError(f"Index {index} out of range (0..{len(mols)-1})")
        return [mols[index]]
    return [m for m in mols if m is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RDKit vs RMG connectivity on an SDF.")
    parser.add_argument("sdf_path")
    parser.add_argument("--prop-key", default=None)
    parser.add_argument("--prop-value", default=None)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    supplier = Chem.SDMolSupplier(args.sdf_path, removeHs=False, sanitize=True)
    mols = _select_mols(list(supplier), args.prop_key, args.prop_value, args.index)

    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        print(f"\nMolecule #{idx}")
        rd_adj = get_connectivity_from_rdkit(mol)

        mol_with_h = Chem.AddHs(mol, addCoords=True)
        rd_adj_with_h = get_connectivity_from_rdkit(mol_with_h)

        rmg_mol = Molecule()
        converter.from_rdkit_mol(rmg_mol, mol)
        rmg_adj = _adj_from_rmg(rmg_mol)

        _compare_adjs("RDKit (as-is) vs RMG (raw index)", rd_adj, rmg_adj)
        _compare_adjs("RDKit (+Hs) vs RMG (raw index)", rd_adj_with_h, rmg_adj)

        rmg_rdkit, rmg_to_rdk = converter.to_rdkit_mol(
            rmg_mol, remove_h=False, return_mapping=True, save_order=True
        )
        rmg_idx_to_rdk = {
            i: rmg_to_rdk[atom]
            for i, atom in enumerate(rmg_mol.vertices)
            if atom in rmg_to_rdk
        }
        match = mol.GetSubstructMatch(rmg_rdkit)
        if not match:
            inverse = rmg_rdkit.GetSubstructMatch(mol)
            if inverse:
                match = [None] * len(rmg_rdkit.GetAtoms())
                for orig_idx, rmg_idx in enumerate(inverse):
                    if rmg_idx < len(match):
                        match[rmg_idx] = orig_idx
                match = tuple(i for i in match if i is not None)
        if match:
            rmg_to_orig = {
                rmg_idx: match[rmg_idx_to_rdk[rmg_idx]]
                for rmg_idx in rmg_idx_to_rdk
                if rmg_idx_to_rdk[rmg_idx] < len(match)
            }
            rmg_adj_remap = _remap_rmg_adj(rmg_adj, rmg_to_orig)
            _compare_adjs("RDKit (as-is) vs RMG (remapped)", rd_adj, rmg_adj_remap)
        else:
            print("RDKit substructure match failed; cannot remap RMG indices")


if __name__ == "__main__":
    main()
