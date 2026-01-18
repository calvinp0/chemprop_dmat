#!/usr/bin/env python
"""
Batch TS guess generator for R1H/R2H pairs using ``simple_ts_merge``.

This reads each ``*_updated.sdf`` in the updated SDF directory, pulls out the
R1H (donor) and R2H (acceptor) blocks (by ``type`` property), finds the labeled
hydrogens/anchors from ``mol_properties``, builds a TS guess, and writes an XYZ
to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from scripts.simple_ts_merge import _as_xyz, make_ts_guess, scan_variants


def _parse_properties(lines: List[str], start: int) -> Dict[str, str]:
    props: Dict[str, str] = {}
    i = start
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith(">") and "<" in line and ">" in line:
            key = line.split("<", 1)[1].split(">", 1)[0].strip().lower()
            i += 1
            values: List[str] = []
            while i < n and lines[i].strip() != "":
                values.append(lines[i].rstrip())
                i += 1
            props[key] = "\n".join(values).strip()
        else:
            i += 1
        while i < n and lines[i].strip() == "":
            i += 1
    return props


def _parse_int_prop(props: Dict[str, str], keys: Iterable[str]) -> Optional[int]:
    for key in keys:
        value = props.get(key, "")
        if not value:
            continue
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _select_charge_multiplicity(ts: Optional[Dict], r1h: Dict, r2h: Dict) -> Tuple[Optional[int], Optional[int]]:
    blocks = [ts, r1h, r2h]
    charge = None
    multiplicity = None
    for block in blocks:
        if not block:
            continue
        props = block.get("props", {})
        if charge is None:
            charge = _parse_int_prop(props, ("charge", "formal_charge"))
        if multiplicity is None:
            multiplicity = _parse_int_prop(props, ("multiplicity", "spin_multiplicity"))
        if charge is not None and multiplicity is not None:
            break
    return charge, multiplicity


def _parse_sdf_block(block: str) -> Optional[Dict]:
    lines = block.strip("\n").splitlines()
    if len(lines) < 3:
        return None

    counts_idx = None
    num_atoms = num_bonds = 0
    for i, line in enumerate(lines[:6]):  # counts should be near the top
        parts = line.split()
        if len(parts) >= 2:
            try:
                num_atoms, num_bonds = int(parts[0]), int(parts[1])
                counts_idx = i
                break
            except Exception:
                continue
    if counts_idx is None:
        return None

    atom_start = counts_idx + 1
    atom_lines = lines[atom_start : atom_start + num_atoms]
    bond_lines = lines[atom_start + num_atoms : atom_start + num_atoms + num_bonds]

    symbols: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            return None
        x, y, z = map(float, parts[:3])
        symbols.append(parts[3])
        coords.append((x, y, z))

    adjacency: List[List[int]] = [[] for _ in range(num_atoms)]
    for line in bond_lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            a, b = int(parts[0]) - 1, int(parts[1]) - 1
        except Exception:
            continue
        if 0 <= a < num_atoms and 0 <= b < num_atoms:
            adjacency[a].append(b)
            adjacency[b].append(a)

    prop_start = atom_start + num_atoms + num_bonds
    for idx in range(prop_start, len(lines)):
        if lines[idx].startswith("M  END"):
            prop_start = idx + 1
            break
    props = _parse_properties(lines, prop_start)

    return {
        "symbols": symbols,
        "coords": np.array(coords, dtype=float),
        "adjacency": adjacency,
        "props": props,
        "type": props.get("type", "").strip().lower(),
    }


def load_r1h_r2h(path: Path) -> Tuple[Dict, Dict, Optional[Dict]]:
    text = path.read_text(errors="ignore")
    blocks = [blk for blk in text.split("$$$$") if blk.strip()]
    parsed = [_parse_sdf_block(blk) for blk in blocks]
    parsed = [p for p in parsed if p is not None and p.get("type") in {"r1h", "r2h", "ts"}]
    r1h = next((p for p in parsed if p["type"] == "r1h"), None)
    r2h = next((p for p in parsed if p["type"] == "r2h"), None)
    ts = next((p for p in parsed if p["type"] == "ts"), None)
    if r1h is None or r2h is None:
        raise ValueError("Missing r1h or r2h block")
    return r1h, r2h, ts


def _get_labeled_index(mol_props: str, labels: Iterable[str]) -> Optional[int]:
    if not mol_props:
        return None
    try:
        props = json.loads(mol_props)
    except Exception:
        return None
    labels = [lab.lower() for lab in labels]
    for idx_str, meta in props.items():
        label = str(meta.get("label", "")).lower()
        if any(lab in label for lab in labels):
            try:
                return int(idx_str)
            except Exception:
                continue
    return None


def _heavy_neighbor(adjacency: List[List[int]], symbols: List[str], idx: int) -> Optional[int]:
    for nbr in adjacency[idx]:
        if symbols[nbr].upper() != "H":
            return nbr
    return adjacency[idx][0] if adjacency[idx] else None


def _farthest_heavy(
    adjacency: List[List[int]], symbols: List[str], start: int, exclude: Iterable[int]
) -> Optional[int]:
    exclude_set = set(exclude)
    n = len(adjacency)
    dist = [-1] * n
    dist[start] = 0
    queue = [start]
    for node in queue:
        for nbr in adjacency[node]:
            if dist[nbr] == -1:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    candidates = [(d, i) for i, d in enumerate(dist) if i not in exclude_set and d >= 0 and symbols[i].upper() != "H"]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _second_neighbor(
    adjacency: List[List[int]], symbols: List[str], anchor: int, exclude: int
) -> Optional[int]:
    neighbors = [nbr for nbr in adjacency[anchor] if nbr != exclude]
    heavy = [nbr for nbr in neighbors if symbols[nbr].upper() != "H"]
    if heavy:
        return heavy[0]
    if neighbors:
        return neighbors[0]
    far_heavy = _farthest_heavy(adjacency, symbols, anchor, exclude=(exclude, anchor))
    if far_heavy is not None:
        return far_heavy
    for idx, sym in enumerate(symbols):
        if idx not in {exclude, anchor}:
            return idx
    return None


def find_indices(block: Dict, role: str) -> Tuple[int, int]:
    mol_props = block["props"].get("mol_properties", "")
    symbols = block["symbols"]
    adjacency = block["adjacency"]
    if role == "r1h":
        h_idx = _get_labeled_index(mol_props, ["d_hydrogen", "donor_h"])
        anchor_idx = _get_labeled_index(mol_props, ["donator", "donor"])
    else:
        h_idx = _get_labeled_index(mol_props, ["a_hydrogen", "acceptor_h"])
        anchor_idx = _get_labeled_index(mol_props, ["acceptor"])
    if h_idx is None:
        h_idx = next((i for i, sym in enumerate(symbols) if sym.upper() == "H"), None)
    if h_idx is None:
        raise ValueError(f"No hydrogen found for {role}")
    if anchor_idx is None:
        anchor_idx = _heavy_neighbor(adjacency, symbols, h_idx)
    if anchor_idx is None:
        raise ValueError(f"No anchor neighbor found for {role}")
    return h_idx, anchor_idx


def find_second_neighbor(block: Dict, h_idx: int, anchor_idx: int) -> Optional[int]:
    adjacency = block["adjacency"]
    symbols = block["symbols"]
    return _second_neighbor(adjacency, symbols, anchor_idx, h_idx)


def _ts_star2_fragment_sizes(ts_block: Dict) -> Optional[Tuple[int, int]]:
    props = ts_block.get("props", {})
    ts_smiles = props.get("ordered_mapped_smiles", "")
    role_mapnums = props.get("role_mapnums", "")
    if not ts_smiles or not role_mapnums:
        return None
    try:
        mapnums = json.loads(role_mapnums)
        star2 = int(mapnums.get("*2"))
    except Exception:
        return None
    for frag in ts_smiles.split("."):
        mol = Chem.MolFromSmiles(frag, sanitize=False)
        if mol is None:
            continue
        if any(atom.GetAtomMapNum() == star2 for atom in mol.GetAtoms()):
            return mol.GetNumAtoms(), mol.GetNumHeavyAtoms()
    return None


def _fragment_sizes(block: Dict) -> Tuple[int, int]:
    symbols = block["symbols"]
    heavy = sum(1 for s in symbols if s.upper() != "H")
    return len(symbols), heavy


def should_swap_by_ts(r1h: Dict, r2h: Dict, ts: Optional[Dict]) -> Optional[bool]:
    """Return True if TS *2 fragment matches r2h (swap), False if matches r1h, None if ambiguous."""
    if ts is None:
        return None
    ts_sizes = _ts_star2_fragment_sizes(ts)
    if ts_sizes is None:
        return None
    r1_sizes = _fragment_sizes(r1h)
    r2_sizes = _fragment_sizes(r2h)
    if ts_sizes == r1_sizes and ts_sizes != r2_sizes:
        return False
    if ts_sizes == r2_sizes and ts_sizes != r1_sizes:
        return True
    # tie or mismatch: ambiguous
    return None


def _ordered_mapnums(block: Dict) -> Optional[List[int]]:
    props = block.get("props", {})
    smiles = props.get("ordered_mapped_smiles", "")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def _ts_ordered_mapnums(ts_block: Dict) -> Optional[List[int]]:
    props = ts_block.get("props", {})
    smiles = props.get("ordered_mapped_smiles", "")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def write_xyz(path: Path, symbols: List[str], coords: np.ndarray, comment: str | None = None) -> None:
    lines = [str(len(symbols)), comment or "generated by simple_ts_merge"]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:2s} {x: .6f} {y: .6f} {z: .6f}")
    path.write_text("\n".join(lines))


def write_props(path: Path, props: Dict[str, object]) -> None:
    path.write_text(json.dumps(props, indent=2))


def build_ts_guess(
    r1h: Dict,
    r2h: Dict,
    ts: Optional[Dict],
    swap_by_ts: bool = False,
    remap_to_ts: bool = False,
) -> Tuple[Dict, Optional[Dict[str, object]], Optional[bool]]:
    swapped = False
    if swap_by_ts:
        swap = should_swap_by_ts(r1h, r2h, ts)
        if swap is True:
            r1h, r2h = r2h, r1h
            swapped = True
    role_1 = r1h.get("type", "r1h") if isinstance(r1h, dict) else "r1h"
    role_2 = r2h.get("type", "r2h") if isinstance(r2h, dict) else "r2h"
    h1, a = find_indices(r1h, role_1)
    h2, b = find_indices(r2h, role_2)
    c = find_second_neighbor(r1h, h1, a)
    d = find_second_neighbor(r2h, h2, b)
    xyz1 = {"symbols": r1h["symbols"], "coords": r1h["coords"]}
    xyz2 = {"symbols": r2h["symbols"], "coords": r2h["coords"]}

    angles = [160.0, 170.0, 175.0, 180.0]
    dihedrals = [None, 60.0, -60.0, 120.0, -120.0]
    variants = scan_variants(
        xyz1,
        xyz2,
        h1=h1,
        h2=h2,
        a=a,
        b=b,
        c=c,
        d=d,
        a2_list=angles,
        d3_list=dihedrals,
        clash_buffer=0.35,
    )
    if variants:
        ts_guess = variants[0][2]
    else:
        ts_guess = make_ts_guess(xyz1, xyz2, h1=h1, h2=h2, a=a, b=b, c=c, d=d, a2=175.0)

    mapping_info: Optional[Dict[str, object]] = None
    mapnums_1 = _ordered_mapnums(r1h)
    mapnums_2 = _ordered_mapnums(r2h)
    if mapnums_1 and mapnums_2 and len(mapnums_1) == len(r1h["symbols"]) and len(mapnums_2) == len(r2h["symbols"]):
        ts_props = ts.get("props", {}) if ts is not None else {}
        try:
            ts_role_mapnums = json.loads(ts_props.get("role_mapnums", "")) if ts_props.get("role_mapnums") else {}
        except Exception:
            ts_role_mapnums = {}

        roles: Dict[str, Tuple[int, int]] = {}
        mapping: List[Dict[str, object]] = []
        for idx in range(len(r1h["symbols"])):
            mapnum = mapnums_1[idx]
            mapping.append(
                {
                    "mapnum": mapnum,
                    "source": 1,
                    "source_role": r1h.get("type", "r1h"),
                    "old_index": idx,
                    "symbol": r1h["symbols"][idx],
                }
            )
        for idx in range(len(r2h["symbols"])):
            if idx == h2:
                continue
            mapnum = mapnums_2[idx]
            mapping.append(
                {
                    "mapnum": mapnum,
                    "source": 2,
                    "source_role": r2h.get("type", "r2h"),
                    "old_index": idx,
                    "symbol": r2h["symbols"][idx],
                }
            )
        # Assign TS roles based on known reaction atoms.
        roles["*1"] = (1, a)
        roles["*2"] = (1, h1)
        roles["*3"] = (2, b)
        if c is not None:
            roles["*0"] = (1, c)
        if d is not None:
            roles["*4"] = (2, d)

        # Add output indices and tag special roles on the mapping entries.
        out_idx = 0
        for entry in mapping:
            entry["output_index"] = out_idx
            entry["ts_role"] = None
            out_idx += 1
        for ts_role, (src, src_idx) in roles.items():
            for entry in mapping:
                if entry["source"] == src and entry["old_index"] == src_idx:
                    entry["ts_role"] = ts_role

        mapping_info = {"roles": roles, "atoms": mapping}

        if remap_to_ts and ts is not None:
            ts_order = _ts_ordered_mapnums(ts)
            combined_mapnums = [m["mapnum"] for m in mapping if m.get("mapnum", 0) > 0]
            if ts_order and all(m.get("mapnum", 0) > 0 for m in mapping) and len(combined_mapnums) == len(set(combined_mapnums)):
                index_by_map = {m["mapnum"]: i for i, m in enumerate(mapping)}
                new_order = [index_by_map[m] for m in ts_order if m in index_by_map]
                if new_order and len(new_order) == len(mapping):
                    coords = ts_guess["coords"][new_order]
                    symbols = [ts_guess["symbols"][i] for i in new_order]
                    ts_guess = {"symbols": symbols, "coords": coords}
                    remapped = [mapping[i] for i in new_order]
                    for out_idx, entry in enumerate(remapped):
                        entry["output_index"] = out_idx
                    mapping_info = {"roles": roles, "atoms": remapped}

    if mapping_info is not None:
        charge, multiplicity = _select_charge_multiplicity(ts, r1h, r2h)
        if charge is not None:
            mapping_info["charge"] = charge
        if multiplicity is not None:
            mapping_info["multiplicity"] = multiplicity

    return ts_guess, mapping_info, swapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TS guess builder using simple_ts_merge.")
    parser.add_argument("--input_dir", type=Path, default=Path("DATA/SDF/updated"), help="Directory of .sdf files.")
    parser.add_argument("--out_dir", type=Path, default=Path("ts_guesses"), help="Directory for XYZ outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing XYZ files.")
    parser.add_argument("--swap_by_ts", action="store_true", help="Swap r1h/r2h if TS *2 fragment matches r2h.")
    parser.add_argument("--remap_to_ts", action="store_true", help="Reorder output atoms to match TS ordered_mapped_smiles.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    errors: List[Tuple[str, str]] = []
    swap_records: List[Dict[str, object]] = []

    sdf_paths = sorted(args.input_dir.glob("*.sdf"))
    for idx, sdf_path in enumerate(sdf_paths, start=1):
        stem = sdf_path.stem
        out_xyz = args.out_dir / f"{stem}_ts_guess.xyz"
        if out_xyz.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            r1h, r2h, ts = load_r1h_r2h(sdf_path)
            ts_xyz, mapping, swapped = build_ts_guess(
                r1h, r2h, ts, swap_by_ts=args.swap_by_ts, remap_to_ts=args.remap_to_ts
            )
            comment = None
            if mapping is not None:
                mapping["swap_by_ts"] = swapped
                props_path = out_xyz.with_suffix(".ts_props.json")
                write_props(props_path, mapping)
                roles_out = {}
                for entry in mapping.get("atoms", []):
                    role = entry.get("ts_role")
                    if role:
                        roles_out[role] = entry.get("output_index")
                comment = json.dumps(
                    {"ts_roles": roles_out, "props_path": props_path.name},
                    separators=(",", ":"),
                )
            write_xyz(out_xyz, ts_xyz["symbols"], ts_xyz["coords"], comment=comment)
            if args.swap_by_ts:
                swap_records.append({"rxn": sdf_path.stem, "swapped": bool(swapped)})
            generated += 1
        except Exception as exc:  # noqa: BLE001
            errors.append((sdf_path.name, str(exc)))
        if idx % 200 == 0:
            print(f"Processed {idx}/{len(sdf_paths)} SDFs...", flush=True)

    print(f"Generated {generated} XYZ files; skipped {skipped} (existing).")
    if errors:
        print(f"Encountered {len(errors)} errors; showing first 5:")
        for name, msg in errors[:5]:
            print(f"  {name}: {msg}")
    if swap_records:
        swap_log = args.out_dir / "ts_swap_log.csv"
        with swap_log.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rxn", "swapped"])
            writer.writeheader()
            writer.writerows(swap_records)


if __name__ == "__main__":
    main()
