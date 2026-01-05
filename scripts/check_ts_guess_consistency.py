#!/usr/bin/env python
"""
Consistency checker for TS guesses vs SDF components and TS labels.

For each `<stem>_updated.sdf` and matching `ts_guesses/<stem>_ts_guess.xyz`, this script:
- loads R1H and R2H components (type=='r1h'/'r2h') and their element symbols
- loads the TS block (type=='ts') and its mol_properties labels
- loads the TS guess XYZ symbols
- reports:
    * symbol/count mismatches between TS guess and combined R1H+R2H (and TS block)
    * TS mol_properties atom_type element letter vs actual element symbol
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from rdkit import Chem


def load_block_by_type(sdf_path: Path, block_type: str) -> Chem.Mol | None:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    for m in suppl:
        if m is None or not m.HasProp("type"):
            continue
        if m.GetProp("type").lower() == block_type.lower():
            return m
    return None


def load_symbols(mol: Chem.Mol) -> List[str]:
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def drop_labeled_h(mol: Chem.Mol, label_substrings: List[str]) -> Chem.Mol:
    """Remove the first H whose mol_properties label contains any of label_substrings."""
    if not mol.HasProp("mol_properties"):
        return mol
    import json

    try:
        props = json.loads(mol.GetProp("mol_properties"))
    except Exception:
        return mol
    drop_idx = None
    for k, v in props.items():
        lbl = str(v.get("label", "")).lower()
        if any(sub in lbl for sub in label_substrings):
            try:
                idx = int(k) - 1
                drop_idx = idx
                break
            except Exception:
                continue
    if drop_idx is None or drop_idx < 0 or drop_idx >= mol.GetNumAtoms():
        return mol
    if mol.GetAtomWithIdx(drop_idx).GetAtomicNum() != 1:
        return mol
    em = Chem.EditableMol(mol)
    em.RemoveAtom(drop_idx)
    new = em.GetMol()
    # copy conformer positions for remaining atoms
    if mol.GetNumConformers():
        conf = mol.GetConformer()
        new_conf = Chem.Conformer(new.GetNumAtoms())
        j = 0
        for i in range(mol.GetNumAtoms()):
            if i == drop_idx:
                continue
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(j, pos)
            j += 1
        new.AddConformer(new_conf, assignId=True)
    return new


def get_label_indices(mol: Chem.Mol, targets: List[str]) -> List[int]:
    """Return zero-based indices whose mol_properties label contains any target (case-insensitive)."""
    if not mol.HasProp("mol_properties"):
        return []
    import json

    try:
        props = json.loads(mol.GetProp("mol_properties"))
    except Exception:
        return []
    keys_int: List[int] = []
    for k in props:
        try:
            keys_int.append(int(k))
        except Exception:
            continue
    base = 0 if 0 in keys_int else 1
    out = []
    for k, v in props.items():
        lbl = str(v.get("label", "")).lower()
        if any(t in lbl for t in targets):
            try:
                idx_raw = int(k)
                out.append(idx_raw - base)
            except Exception:
                continue
    return out


def load_ts_guess_symbols(xyz_path: Path) -> List[str]:
    lines = xyz_path.read_text().strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{xyz_path} is too short to be an XYZ file.")
    try:
        n = int(lines[0].strip())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse atom count in {xyz_path}") from exc
    sym = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ line in {xyz_path}: {line}")
        sym.append(parts[0])
    if len(sym) != n:
        raise ValueError(f"{xyz_path} reported {n} atoms but parsed {len(sym)}")
    return sym


def atom_type_to_element(atom_type: str) -> str:
    """
    Derive element symbol from atom_type string like "Ct", "Cs", "H0", "N3d".
    Rule: take first character (uppercased); strip any leading digits if present.
    """
    atom_type = atom_type.strip()
    if not atom_type:
        return ""
    first = atom_type[0]
    if first.isdigit() and len(atom_type) > 1:
        first = atom_type[1]
    return first.upper()


def check_props(ts_mol: Chem.Mol) -> List[str]:
    issues: List[str] = []
    if not ts_mol.HasProp("mol_properties"):
        return issues
    import json

    try:
        props: Dict[str, Dict[str, str]] = json.loads(ts_mol.GetProp("mol_properties"))
    except Exception:
        issues.append("TS mol_properties not JSON-decodable.")
        return issues

    # TS blocks use 0-based keys (per provided examples)
    base = 0

    for k, v in props.items():
        try:
            idx_raw = int(k)
        except Exception:
            continue
        idx = idx_raw - base
        if idx < 0 or idx >= ts_mol.GetNumAtoms():
            issues.append(f"Index {idx+1} out of range for TS with {ts_mol.GetNumAtoms()} atoms.")
            continue
        atom_type = str(v.get("atom_type", ""))
        expected = atom_type_to_element(atom_type)
        actual = ts_mol.GetAtomWithIdx(idx).GetSymbol().upper()
        if expected and expected != actual:
            issues.append(f"Atom {idx+1}: atom_type '{atom_type}' -> '{expected}' but element is '{actual}'")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate TS guesses against SDF components and TS labels.")
    ap.add_argument("--sdf_dir", type=Path, default=Path("DATA/SDF/updated"))
    ap.add_argument("--ts_guess_dir", type=Path, default=Path("ts_guesses"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report_path", type=Path, default=None, help="Write a CSV report of mismatches to this path.")
    args = ap.parse_args()

    sdf_paths = sorted(args.sdf_dir.glob("*_updated.sdf"))
    if args.limit:
        sdf_paths = sdf_paths[: args.limit]

    total = 0
    count_symbol_mismatch = 0
    count_ts_symbol_mismatch = 0
    count_prop_issues = 0
    order_counts = Counter()

    report_rows = []

    for sdf_path in sdf_paths:
        stem = sdf_path.stem.replace("_updated", "")
        # support either *_ts_guess.xyz or *.xyz naming
        xyz_path = args.ts_guess_dir / f"{stem}_ts_guess.xyz"
        if not xyz_path.exists():
            alt = args.ts_guess_dir / f"{stem}.xyz"
            if alt.exists():
                xyz_path = alt
            else:
                continue
        total += 1

        try:
            r1h_raw = load_block_by_type(sdf_path, "r1h")
            r2h_raw = load_block_by_type(sdf_path, "r2h")
            ts = load_block_by_type(sdf_path, "ts")
            if r1h_raw is None or r2h_raw is None or ts is None:
                print(f"{stem}: missing r1h/r2h/ts block; skipping.")
                continue
            r1h = drop_labeled_h(r1h_raw, ["d_hydrogen", "donor_h"])
            r2h = drop_labeled_h(r2h_raw, ["a_hydrogen", "acceptor_h"])
            order1 = load_symbols(r1h) + load_symbols(r2h)
            order2 = load_symbols(r2h) + load_symbols(r1h)
            ts_symbols = load_symbols(ts)
            guess_symbols = load_ts_guess_symbols(xyz_path)
        except Exception as exc:  # noqa: BLE001
            print(f"{stem}: failed to load ({exc}); skipping.")
            continue

        # choose best ordering by segment-wise mismatches (first vs r1h, second vs r2h)
        sym_r1_raw = load_symbols(r1h_raw)
        sym_r2_raw = load_symbols(r2h_raw)
        sym_r1 = load_symbols(r1h)
        sym_r2 = load_symbols(r2h)
        donor_idxs_raw = get_label_indices(r1h_raw, ["d_hydrogen", "donor"])
        acceptor_idxs_raw = get_label_indices(r2h_raw, ["a_hydrogen"])
        last_flags = {
            "r1h_donor_is_last": bool(donor_idxs_raw) and max(donor_idxs_raw) == len(sym_r1_raw) - 1,
            "r2h_acceptor_is_last": bool(acceptor_idxs_raw) and max(acceptor_idxs_raw) == len(sym_r2_raw) - 1,
        }

        def seg_mismatches(guess: List[str], first: List[str], second: List[str]) -> Tuple[int, int, int, List[int]]:
            if len(guess) != len(first) + len(second):
                return 10**9, 10**9, 10**9, []  # impossible length
            mismatches_idx: List[int] = []
            m1 = 0
            for i, (a, b) in enumerate(zip(guess[: len(first)], first)):
                if a != b:
                    m1 += 1
                    mismatches_idx.append(i)
            m2 = 0
            offset = len(first)
            for i, (a, b) in enumerate(zip(guess[len(first) :], second)):
                if a != b:
                    m2 += 1
                    mismatches_idx.append(offset + i)
            return m1 + m2, m1, m2, mismatches_idx

        # Only consider physically sensible combinations: one fragment dropped, one raw
        candidates = [
            ("r1_raw+r2_drop", sym_r1_raw, sym_r2),
            ("r2_raw+r1_drop", sym_r2_raw, sym_r1),
        ]

        results = []
        for name, first, second in candidates:
            tot, m_first, m_second, idxs = seg_mismatches(guess_symbols, first, second)
            results.append((tot, m_first, m_second, idxs, name, first, second))

        # choose minimal total; tie-break by preference order in candidates list
        results_sorted = sorted(results, key=lambda x: (x[0], candidates.index((x[4], x[5], x[6]))))
        best_tot, best_f, best_s, best_idx, best_name, best_first, best_second = results_sorted[0]
        reactant_symbols = best_first + best_second
        order_counts[best_name] += 1
        order_choice = best_name
        # donor/acceptor hint: if tie among top totals, adjust preference
        top_tot = results_sorted[0][0]
        tied = [r for r in results_sorted if r[0] == top_tot]
        if len(tied) > 1:
            if last_flags["r1h_donor_is_last"]:
                for r in tied:
                    if r[4].startswith("r1"):
                        best_tot, best_f, best_s, best_idx, best_name, best_first, best_second = r
                        order_choice = best_name + " (hint)"
                        reactant_symbols = best_first + best_second
                        order_counts[best_name] += 1
                        break
            elif last_flags["r2h_acceptor_is_last"]:
                for r in tied:
                    if r[4].startswith("r2"):
                        best_tot, best_f, best_s, best_idx, best_name, best_first, best_second = r
                        order_choice = best_name + " (hint)"
                        reactant_symbols = best_first + best_second
                        order_counts[best_name] += 1
                        break

        print(f"{stem}: chosen order {order_choice}")

        reactant_counts = Counter(reactant_symbols)
        guess_counts = Counter(guess_symbols)
        ts_counts = Counter(ts_symbols)

        if reactant_counts != guess_counts:
            count_symbol_mismatch += 1
            print(f"{stem}: TS guess symbols mismatch vs R1H+R2H counts.")
            print(f"  reactants counts: {reactant_counts}")
            print(f"  guess counts    : {guess_counts}")
            # show first mismatch positions
            min_len = min(len(reactant_symbols), len(guess_symbols))
            pos_mismatches = [i for i in range(min_len) if reactant_symbols[i] != guess_symbols[i]]
            print(f"  first mismatching positions: {pos_mismatches[:10]}")
            print(f"  guess ({len(guess_symbols)}): {guess_symbols}")
            print(f"  r1h_raw ({len(sym_r1_raw)}): {sym_r1_raw}")
            print(f"  r2h_raw ({len(sym_r2_raw)}): {sym_r2_raw}")
            print(f"  r1h_drop({len(sym_r1)}): {sym_r1}")
            print(f"  r2h_drop({len(sym_r2)}): {sym_r2}")
            report_rows.append(
                {
                    "reaction": stem,
                    "issue": "symbol_mismatch",
                    "reactant_counts": dict(reactant_counts),
                    "guess_counts": dict(guess_counts),
                    "order": order_choice,
                }
            )
        if ts_counts != guess_counts:
            count_ts_symbol_mismatch += 1
            print(f"{stem}: TS guess symbols mismatch vs TS block counts. ts={ts_counts}, guess={guess_counts}")
            report_rows.append(
                {
                    "reaction": stem,
                    "issue": "ts_symbol_mismatch",
                    "ts_counts": dict(ts_counts),
                    "guess_counts": dict(guess_counts),
                    "order": order_choice,
                }
            )

        prop_issues = check_props(ts)
        if prop_issues:
            count_prop_issues += 1
            print(f"{stem}: TS mol_properties issues:")
            for msg in prop_issues:
                print(f"  - {msg}")
                report_rows.append({"reaction": stem, "issue": "mol_properties", "detail": msg, "order": order_choice})

    print(f"Checked {total} TS guesses with matching SDFs.")
    print(f"Symbol mismatches vs R1H+R2H: {count_symbol_mismatch}")
    print(f"Symbol mismatches vs TS block: {count_ts_symbol_mismatch}")
    print(f"TS mol_properties element issues: {count_prop_issues}")
    if order_counts:
        print(f"Chosen ordering counts: {dict(order_counts)}")

    # write report if requested
    if args.report_path and report_rows:
        import csv

        keys = sorted({k for row in report_rows for k in row.keys()})
        with args.report_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in report_rows:
                writer.writerow(row)
        print(f"Wrote report to {args.report_path}")


if __name__ == "__main__":
    main()
