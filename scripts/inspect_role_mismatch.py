#!/usr/bin/env python
"""List TS entries whose donor/acceptor/moving-H atom type disagrees with mol_properties."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from rdkit import Chem

from scripts.ts_hpo import (
    BASE_ROLE_COLS,
    get_role_cols,
    load_ts_mol_from_sdf,
    sanitize_partial,
)


Role = Literal["donor", "moving_h", "acceptor", "donor_adjacent", "acceptor_adjacent"]


def inspect(ts_path: Path, sdf_dir: Path, add_adj_roles: bool = False) -> None:
    role_keys = list(get_role_cols(add_adj_roles).keys())
    mismatches: list[tuple[str, Role, str]] = []
    ts_mols = [json.loads(line) for line in ts_path.read_text().splitlines() if line.strip()]
    for entry in ts_mols:
        name = entry["sdf_file"]
        sdf_path = sdf_dir / name
        try:
            mol = load_ts_mol_from_sdf(sdf_path, sanitize=False)
        except FileNotFoundError:
            continue
        mol = sanitize_partial(mol)
        atom_types = entry.get("role_atom_types") or entry.get("mol_properties")
        role_order = entry["role_order"]
        role_indices = entry["role_indices_ordered"]
        role_map = dict(zip(role_order, role_indices))
        for role in BASE_ROLE_COLS:
            idx = role_map.get(f"*{ROLE_ORDER_MAP[role]}")
            if idx is None:
                continue
            if idx >= mol.GetNumAtoms():
                continue
            atom = mol.GetAtomWithIdx(idx)
            sym = atom.GetSymbol()
            expected = _expected_symbol(idx, atom_types)
            if expected and sym != expected:
                mismatches.append((name, role, f"{sym} != {expected}"))
    if not mismatches:
        print("No type mismatches found.")
        return
    print("Found type mismatches:")
    for name, role, message in mismatches:
        print(f"{name}: {role} -> {message}")


def _expected_symbol(idx: int, atom_types: dict | None) -> str | None:
    if not atom_types:
        return None
    if isinstance(next(iter(atom_types.values())), dict):
        atom_type = atom_types.get(str(idx), {}).get("atom_type")
    else:
        atom_type = atom_types.get(str(idx))
    if not atom_type:
        return None
    if atom_type.startswith(("Cl", "Br")):
        return atom_type[:2].capitalize()
    return atom_type[0].upper()


ROLE_ORDER_MAP = {"donor": 1, "moving_h": 2, "acceptor": 3, "donor_adjacent": 0, "acceptor_adjacent": 4}


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Find TS entries with role label/symbol mismatches.")
    p.add_argument("--ts-path", type=Path, default="DATA/ts_molecules.ndjson")
    p.add_argument("--sdf-dir", type=Path, default="DATA/SDF")
    p.add_argument("--add-adj-roles", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect(args.ts_path, args.sdf_dir, add_adj_roles=args.add_adj_roles)
