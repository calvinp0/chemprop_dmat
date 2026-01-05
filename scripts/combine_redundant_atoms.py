#!/usr/bin/env python
"""
CLI wrapper around ARC's combine_coordinates_with_redundant_atoms.

Supports XYZ files/strings and SDF blocks (selected by type or index).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from rdkit import Chem


def _ensure_arc_on_path() -> None:
    arc_root = os.environ.get("ARC_ROOT", "/home/calvin/code/ARC")
    arc_path = Path(arc_root)
    if arc_path.exists():
        sys.path.insert(0, str(arc_path.resolve()))
        return
    raise FileNotFoundError(
        f"ARC_ROOT not found: {arc_root}. Set ARC_ROOT to your ARC repo path."
    )


def _parse_int_prop(mol: Chem.Mol, keys: Tuple[str, ...]) -> Optional[int]:
    for key in keys:
        if mol.HasProp(key):
            try:
                return int(float(mol.GetProp(key)))
            except ValueError:
                return None
    return None


def _get_str_prop(mol: Chem.Mol, keys: Tuple[str, ...]) -> Optional[str]:
    for key in keys:
        if mol.HasProp(key):
            value = mol.GetProp(key).strip()
            return value if value else None
    return None


def _parse_mol_properties(mol: Chem.Mol) -> Dict[str, Dict[str, str]]:
    if not mol.HasProp("mol_properties"):
        return {}
    raw = mol.GetProp("mol_properties").strip()
    if not raw:
        return {}
    try:
        import json

        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _get_label_index(mol_props: Dict[str, Dict[str, str]], labels: Tuple[str, ...]) -> Optional[int]:
    if not mol_props:
        return None
    labels = tuple(lab.lower() for lab in labels)
    for idx_str, meta in mol_props.items():
        label = str(meta.get("label", "")).lower()
        if any(lab in label for lab in labels):
            try:
                return int(idx_str)
            except ValueError:
                return None
    return None


def _load_sdf_mol(path: Path, sdf_type: Optional[str], sdf_index: Optional[int]) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(
        str(path), removeHs=False, sanitize=False, strictParsing=False
    )
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"No molecules found in SDF: {path}")
    all_types = sorted(
        {m.GetProp("type").strip().lower() for m in mols if m.HasProp("type")}
    )

    if sdf_type is not None:
        sdf_type = sdf_type.strip().lower()
        filtered = []
        for mol in mols:
            if mol.HasProp("type") and mol.GetProp("type").strip().lower() == sdf_type:
                filtered.append(mol)
        mols = filtered
        if not mols:
            raise ValueError(
                f"No SDF blocks with type={sdf_type} in {path}. Available: {all_types}"
            )

    if sdf_index is not None:
        if sdf_index < 0 or sdf_index >= len(mols):
            raise IndexError(f"sdf_index {sdf_index} out of range for {path} ({len(mols)} mols)")
        return mols[sdf_index]

    if len(mols) > 1:
        types = [m.GetProp("type") if m.HasProp("type") else "" for m in mols]
        raise ValueError(
            f"Multiple molecules found in {path}. Provide --sdf-type or --sdf-index. Types: {types}"
        )
    return mols[0]


def _rdkit_to_xyz(mol: Chem.Mol) -> dict:
    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit mol has no conformers; cannot extract coordinates.")
    conf = mol.GetConformer()
    coords = []
    symbols = []
    for idx, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(idx)
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
        symbols.append(atom.GetSymbol())
    from arc.species.converter import xyz_from_data

    return xyz_from_data(coords=coords, symbols=symbols)


def _load_sdf_block_info(
    path: Path, sdf_type: Optional[str], sdf_index: Optional[int]
) -> Tuple[dict, Optional[int], Optional[int], Dict[str, Dict[str, str]], Optional[str]]:
    mol = _load_sdf_mol(path, sdf_type=sdf_type, sdf_index=sdf_index)
    xyz = _rdkit_to_xyz(mol)
    mult = _parse_int_prop(mol, ("multiplicity", "spin_multiplicity"))
    charge = _parse_int_prop(mol, ("charge", "formal_charge"))
    mol_props = _parse_mol_properties(mol)
    adjlist = _get_str_prop(mol, ("rmg_adjacency_list", "adjacency_list"))
    return xyz, mult, charge, mol_props, adjlist


def _load_xyz_source(
    path_str: str,
    sdf_type: Optional[str],
    sdf_index: Optional[int],
) -> Tuple[dict, Optional[int], Optional[int], Dict[str, Dict[str, str]]]:
    path = Path(path_str)
    if path.exists() and path.suffix.lower() == ".sdf":
        xyz, mult, charge, mol_props, _adjlist = _load_sdf_block_info(
            path, sdf_type=sdf_type, sdf_index=sdf_index
        )
        return xyz, mult, charge, mol_props

    from arc.species.converter import str_to_xyz

    xyz = str_to_xyz(path_str)
    return xyz, None, None, {}


def _build_mol_from_xyz(xyz: dict, multiplicity: int, charge: int):
    from arc.species.perceive import perceive_molecule_from_xyz

    mol = perceive_molecule_from_xyz(
        xyz=xyz, charge=charge, multiplicity=multiplicity, n_radicals=None
    )
    if mol is None:
        raise ValueError("Failed to perceive molecule from xyz.")
    return mol


def _remove_atom_from_xyz(xyz: dict, idx: int) -> dict:
    from arc.species.converter import xyz_from_data

    coords = [coord for i, coord in enumerate(xyz["coords"]) if i != idx]
    symbols = [sym for i, sym in enumerate(xyz["symbols"]) if i != idx]
    isotopes = None
    if "isotopes" in xyz:
        isotopes = [iso for i, iso in enumerate(xyz["isotopes"]) if i != idx]
    return xyz_from_data(coords=coords, symbols=symbols, isotopes=isotopes)


def main() -> None:
    _ensure_arc_on_path()

    from arc.job.adapters.ts.heuristics import (
        combine_coordinates_with_redundant_atoms,
        find_distant_neighbor,
    )
    from arc.species.species import ARCSpecies
    from arc.species.converter import xyz_to_xyz_file_format

    ap = argparse.ArgumentParser(
        description="Combine coordinates with redundant atoms using ARC heuristics."
    )
    ap.add_argument("--xyz1", required=True, help="XYZ or SDF path for reactant R1H.")
    ap.add_argument("--xyz2", required=True, help="XYZ or SDF path for product R2H.")
    ap.add_argument(
        "--reactant2",
        required=True,
        help="XYZ or SDF path for reactant R2 (radical), i.e., xyz2 without the redundant H.",
    )
    ap.add_argument("--xyz1-type", help="SDF block type for xyz1 (e.g., r1h).")
    ap.add_argument("--xyz2-type", help="SDF block type for xyz2 (e.g., r2h).")
    ap.add_argument("--reactant2-type", help="SDF block type for reactant2 (e.g., r2).")
    ap.add_argument("--xyz1-index", type=int, help="SDF mol index for xyz1 (0-based).")
    ap.add_argument("--xyz2-index", type=int, help="SDF mol index for xyz2 (0-based).")
    ap.add_argument("--reactant2-index", type=int, help="SDF mol index for reactant2 (0-based).")
    ap.add_argument("--h1", type=int, default=None, help="Index of redundant H in xyz1 (0-based).")
    ap.add_argument("--h2", type=int, default=None, help="Index of redundant H in xyz2 (0-based).")
    ap.add_argument("--c", type=int, default=None, help="Index of atom C in xyz1 (0-based).")
    ap.add_argument("--d", type=int, default=None, help="Index of atom D in xyz2 (0-based).")
    ap.add_argument("--a2", type=float, default=180.0, help="Angle B-H-A in degrees.")
    ap.add_argument("--d2", type=float, default=None, help="Dihedral B-H-A-C in degrees.")
    ap.add_argument("--d3", type=float, default=None, help="Dihedral D-B-H-A in degrees.")
    ap.add_argument("--r1-stretch", type=float, default=1.2, help="Stretch factor for A-H1.")
    ap.add_argument("--r2-stretch", type=float, default=1.2, help="Stretch factor for B-H2.")
    ap.add_argument("--keep-dummy", action="store_true", help="Keep dummy atom if added.")
    ap.add_argument(
        "--reactants-reversed",
        action="store_true",
        help="Set if reactants are reversed relative to the RMG template.",
    )
    ap.add_argument(
        "--allow-collisions",
        action="store_true",
        help="Bypass ARC colliding-atoms checks when building intermediate species.",
    )
    ap.add_argument(
        "--allow-map-fallback",
        action="store_true",
        help="Fallback to identity atom-map when ARC mapping fails (same atom count).",
    )
    ap.add_argument(
        "--auto-reactants-reversed",
        action="store_true",
        help="Infer reactants_reversed using ARCReaction built from r1h/r2/r2h/r1 SDF blocks.",
    )
    ap.add_argument(
        "--rxn-sdf",
        type=Path,
        default=None,
        help="SDF path to load r1h/r2/r2h/r1 blocks for auto reactant reversal.",
    )
    ap.add_argument("--mult1", type=int, default=None, help="Multiplicity for xyz1.")
    ap.add_argument("--mult2", type=int, default=None, help="Multiplicity for xyz2.")
    ap.add_argument("--mult3", type=int, default=None, help="Multiplicity for reactant2.")
    ap.add_argument("--charge1", type=int, default=None, help="Charge for xyz1.")
    ap.add_argument("--charge2", type=int, default=None, help="Charge for xyz2.")
    ap.add_argument("--charge3", type=int, default=None, help="Charge for reactant2.")
    ap.add_argument(
        "--radicals3",
        type=int,
        default=None,
        help="Number of radicals for reactant2 (optional).",
    )
    ap.add_argument(
        "--output", type=Path, default=Path("combined_ts.xyz"), help="Output XYZ path."
    )
    args = ap.parse_args()

    xyz1, mult1_sdf, charge1_sdf, props1 = _load_xyz_source(
        args.xyz1, args.xyz1_type, args.xyz1_index
    )
    xyz2, mult2_sdf, charge2_sdf, props2 = _load_xyz_source(
        args.xyz2, args.xyz2_type, args.xyz2_index
    )
    xyz3, mult3_sdf, charge3_sdf, _props3 = _load_xyz_source(
        args.reactant2, args.reactant2_type, args.reactant2_index
    )

    mult1 = args.mult1 if args.mult1 is not None else (mult1_sdf or 1)
    mult2 = args.mult2 if args.mult2 is not None else (mult2_sdf or 1)
    mult3 = args.mult3 if args.mult3 is not None else (mult3_sdf or 1)
    charge1 = args.charge1 if args.charge1 is not None else (charge1_sdf or 0)
    charge2 = args.charge2 if args.charge2 is not None else (charge2_sdf or 0)
    charge3 = args.charge3 if args.charge3 is not None else (charge3_sdf or 0)

    mol1 = _build_mol_from_xyz(xyz1, multiplicity=mult1, charge=charge1)
    mol2 = _build_mol_from_xyz(xyz2, multiplicity=mult2, charge=charge2)
    reactant_2 = ARCSpecies(
        label="reactant_2",
        xyz=xyz3,
        multiplicity=mult3,
        charge=charge3,
        number_of_radicals=args.radicals3,
    )
    reactant_2.mol_from_xyz()

    h1 = args.h1 if args.h1 is not None else _get_label_index(props1, ("d_hydrogen",))
    h2 = args.h2 if args.h2 is not None else _get_label_index(props2, ("a_hydrogen",))
    if h1 is None or h2 is None:
        raise ValueError(
            "Missing h1/h2 indices. Provide --h1/--h2 or ensure SDF mol_properties "
            "contains labels 'd_hydrogen' (r1h) and 'a_hydrogen' (r2h)."
        )

    c_idx = args.c if args.c is not None else find_distant_neighbor(mol1, h1)
    d_idx = args.d if args.d is not None else find_distant_neighbor(mol2, h2)

    from arc.species import species as arc_species
    from arc.mapping import engine as arc_mapping_engine
    from arc.job.adapters.ts import heuristics as arc_heuristics
    from arc.reaction.reaction import ARCReaction

    orig_colliding_atoms = arc_species.colliding_atoms
    orig_map_two_species = arc_mapping_engine.map_two_species
    orig_heuristics_map = arc_heuristics.map_two_species
    if args.allow_collisions:
        arc_species.colliding_atoms = lambda _xyz: False
    if args.allow_map_fallback:
        def _map_two_species_fallback(spc_1, spc_2, *args, **kwargs):
            result = orig_map_two_species(spc_1, spc_2, *args, **kwargs)
            if result is not None:
                return result
            try:
                spc_1 = arc_mapping_engine.get_arc_species(spc_1)
                spc_2 = arc_mapping_engine.get_arc_species(spc_2)
                if spc_1.number_of_atoms != spc_2.number_of_atoms:
                    return None
                map_type = kwargs.get("map_type", "list")
                if map_type == "dict":
                    return {i: i for i in range(spc_1.number_of_atoms)}
                return list(range(spc_1.number_of_atoms))
            except Exception:
                return None
        arc_mapping_engine.map_two_species = _map_two_species_fallback
        arc_heuristics.map_two_species = _map_two_species_fallback
    def _combine(reversed_flag: bool, reactant_2_override):
        return combine_coordinates_with_redundant_atoms(
            xyz_1=xyz1,
            xyz_2=xyz2,
            mol_1=mol1,
            mol_2=mol2,
            reactant_2=reactant_2_override,
            h1=h1,
            h2=h2,
            c=c_idx,
            d=d_idx,
            r1_stretch=args.r1_stretch,
            r2_stretch=args.r2_stretch,
            a2=args.a2,
            d2=args.d2,
            d3=args.d3,
            keep_dummy=args.keep_dummy,
            reactants_reversed=reversed_flag,
        )

    def _maybe_auto_reversed() -> Optional[bool]:
        if not args.auto_reactants_reversed:
            return None
        sdf_path = args.rxn_sdf
        if sdf_path is None:
            if Path(args.xyz1).suffix.lower() == ".sdf":
                sdf_path = Path(args.xyz1)
            else:
                return None
        try:
            r1h_xyz, r1h_mult, r1h_charge, _p1, _r1h_adj = _load_sdf_block_info(
                sdf_path, sdf_type="r1h", sdf_index=None
            )
            r2_xyz, r2_mult, r2_charge, _p2, _r2_adj = _load_sdf_block_info(
                sdf_path, sdf_type="r2", sdf_index=None
            )
            r2h_xyz, r2h_mult, r2h_charge, _p3, _r2h_adj = _load_sdf_block_info(
                sdf_path, sdf_type="r2h", sdf_index=None
            )
            r1_xyz, r1_mult, r1_charge, _p4, _r1_adj = _load_sdf_block_info(
                sdf_path, sdf_type="r1", sdf_index=None
            )
            r1h_spc = ARCSpecies(label="r1h", xyz=r1h_xyz, multiplicity=r1h_mult or 1, charge=r1h_charge or 0)
            r2_spc = ARCSpecies(label="r2", xyz=r2_xyz, multiplicity=r2_mult or 1, charge=r2_charge or 0)
            r2h_spc = ARCSpecies(label="r2h", xyz=r2h_xyz, multiplicity=r2h_mult or 1, charge=r2h_charge or 0)
            r1_spc = ARCSpecies(label="r1", xyz=r1_xyz, multiplicity=r1_mult or 1, charge=r1_charge or 0)
            rxn = ARCReaction(r_species=[r1h_spc, r2_spc], p_species=[r1_spc, r2h_spc])
            if not rxn.product_dicts:
                print("auto-reactants-reversed: family not recognized (no product_dicts)")
                return None
            reactants_reversed, _products_reversed = arc_heuristics.are_h_abs_wells_reversed(
                rxn, rxn.product_dicts[0]
            )
            family = rxn.product_dicts[0].get("family", "unknown")
            print(
                f"auto-reactants-reversed: family={family}, reactants_reversed={reactants_reversed}"
            )
            return reactants_reversed
        except Exception as exc:
            print(f"auto-reactants-reversed: failed to build ARCReaction ({exc})")
            return None

    def _attempt_combine(reactant_2_override):
        auto_reversed = _maybe_auto_reversed()
        if auto_reversed is True or args.reactants_reversed:
            return _combine(True, reactant_2_override)
        if auto_reversed is False:
            return _combine(False, reactant_2_override)
        last_exc: Exception | None = None
        for reversed_flag in (False, True):
            try:
                return _combine(reversed_flag, reactant_2_override)
            except ValueError as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise ValueError("Failed to combine coordinates.")

    try:
        try:
            combined = _attempt_combine(reactant_2)
        except ValueError as exc:
            if "atom_map" not in str(exc):
                raise
            xyz2_no_h = _remove_atom_from_xyz(xyz2, h2)
            reactant_2_fallback = ARCSpecies(
                label="reactant_2_fallback",
                xyz=xyz2_no_h,
                multiplicity=mult3,
                charge=charge3,
                number_of_radicals=args.radicals3,
            )
            reactant_2_fallback.mol_from_xyz()
            combined = _attempt_combine(reactant_2_fallback)
    finally:
        arc_species.colliding_atoms = orig_colliding_atoms
        arc_mapping_engine.map_two_species = orig_map_two_species
        arc_heuristics.map_two_species = orig_heuristics_map

    args.output.write_text(xyz_to_xyz_file_format(combined))
    print(f"Wrote combined TS guess to {args.output}")


if __name__ == "__main__":
    main()
