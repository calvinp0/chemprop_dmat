"""
Heuristic TS guess builder for R1H + R2H pairs (H-abstraction style).

Goal: make a viewable combined geometry (TS-like) without ARC dependencies.

Method (pure RDKit + NumPy):
1) Load R1H (donor) and R2H (acceptor) from `<stem>_updated.sdf` (type props).
2) Find donor H (`d_hydrogen`) and acceptor H (`a_hydrogen`) indices from
   mol_properties. Find their heavy-atom neighbors A (donor) and B (acceptor).
3) Remove acceptor H from R2H.
4) Rigidly rotate + translate the remaining R2H so that:
     - the old B→H direction aligns to the donor A→H direction
     - the new B position sits at H_donor + r_target * dir(A→H)
       with r_target = |B-H| * r2_stretch (default 1.2).
   Rotation: Rodrigues formula to map old_dir to new_dir about origin B.
5) Fuse R1H and transformed R2H, add a bridge bond H_donor–B_new.
6) Write combined SDF/XYZ to `--out_dir` for inspection.

This is only for visualization / atom-order inspection; it is not an energy-
screened TS. Defaults mimic ARC's stretching (r2_stretch=1.2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem


def load_components(sdf_path: Path) -> Tuple[Chem.Mol, Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    comps = {m.GetProp("type").lower(): m for m in mols if m.HasProp("type")}
    if "r1h" not in comps or "r2h" not in comps:
        raise ValueError(f"Expected r1h/r2h in {sdf_path.name}, found {list(comps.keys())}")
    return comps["r1h"], comps["r2h"]


def find_label_idx(mol: Chem.Mol, label_substring: str) -> Optional[int]:
    if not mol.HasProp("mol_properties"):
        return None
    props = json.loads(mol.GetProp("mol_properties"))
    for k, v in props.items():
        if label_substring in v.get("label", ""):
            try:
                idx = int(k) - 1
                if 0 <= idx < mol.GetNumAtoms():
                    return idx
            except Exception:
                continue
    return None


def fallback_first_h(mol: Chem.Mol) -> Optional[int]:
    """Fallback: first hydrogen atom index."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            return atom.GetIdx()
    return None


def neighbor_heavy(mol: Chem.Mol, idx: int) -> int:
    atom = mol.GetAtomWithIdx(idx)
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() > 1:
            return nbr.GetIdx()
    return atom.GetNeighbors()[0].GetIdx()


def rodrigues(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        # parallel or antiparallel
        if c > 0:
            return np.eye(3)
        # 180 deg: pick arbitrary orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
    vx = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=float,
    )
    s = np.linalg.norm(v)
    r = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-12))
    return r


def transform_r2h(
    r2h: Chem.Mol,
    acc_H_idx: int,
    donor_H_pos: np.ndarray,
    donor_A_pos: np.ndarray,
    r2_stretch: float = 1.2,
    r2_shift: float = 0.3,
    r2_ortho: float = 0.5,
) -> Tuple[Chem.Mol, int]:
    """
    Remove acceptor H, rotate/translate R2H so B->H aligns with A->H (donor),
    and place B at donor_H + r_target * dir(A->H).
    Returns transformed mol and new index of B.
    """
    conf = r2h.GetConformer()
    B_idx = neighbor_heavy(r2h, acc_H_idx)
    B_pos = np.array(conf.GetAtomPosition(B_idx), dtype=float)
    H_pos = np.array(conf.GetAtomPosition(acc_H_idx), dtype=float)

    old_dir = H_pos - B_pos  # B->H
    new_dir = donor_H_pos - donor_A_pos  # A->H
    if np.linalg.norm(new_dir) < 1e-8:
        new_dir = np.array([1.0, 0.0, 0.0])
    r_target = np.linalg.norm(old_dir) * r2_stretch + r2_shift
    new_dir_unit = new_dir / (np.linalg.norm(new_dir) + 1e-12)

    R = rodrigues(old_dir, new_dir_unit)
    # small orthogonal bump to reduce collisions
    bump_axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(bump_axis, new_dir_unit)) > 0.9:
        bump_axis = np.array([0.0, 1.0, 0.0])
    bump = np.cross(new_dir_unit, bump_axis)
    bump = bump / (np.linalg.norm(bump) + 1e-12) * r2_ortho

    # build new mol without acc_H
    em = Chem.EditableMol(r2h)
    em.RemoveAtom(acc_H_idx)
    r2h_new = em.GetMol()

    # remap B index if needed
    B_new = B_idx if acc_H_idx > B_idx else B_idx - 1

    # shift coords: center on B, rotate, place B at target
    conf_new = Chem.Conformer(r2h_new.GetNumAtoms())
    for old_idx in range(r2h.GetNumAtoms()):
        if old_idx == acc_H_idx:
            continue
        new_idx = old_idx if old_idx < acc_H_idx else old_idx - 1
        pos = np.array(conf.GetAtomPosition(old_idx), dtype=float)
        pos_rel = pos - B_pos
        pos_rot = R @ pos_rel
        pos_final = donor_H_pos + r_target * new_dir_unit + bump + pos_rot  # place B at target + shifts
        conf_new.SetAtomPosition(new_idx, pos_final.tolist())

    r2h_new.RemoveAllConformers()
    r2h_new.AddConformer(conf_new, assignId=True)
    return r2h_new, B_new


def fuse(r1h: Chem.Mol, r2h_t: Chem.Mol, donor_H_idx: int, B_idx_new: int) -> Chem.Mol:
    combo = Chem.CombineMols(r1h, r2h_t)
    em = Chem.EditableMol(combo)
    offset = r1h.GetNumAtoms()
    em.AddBond(donor_H_idx, offset + B_idx_new, order=Chem.BondType.SINGLE)
    combo_b = em.GetMol()

    # conformer
    conf1 = r1h.GetConformer()
    conf2 = r2h_t.GetConformer()
    conf_c = Chem.Conformer(combo_b.GetNumAtoms())
    for i in range(r1h.GetNumAtoms()):
        conf_c.SetAtomPosition(i, conf1.GetAtomPosition(i))
    for j in range(r2h_t.GetNumAtoms()):
        conf_c.SetAtomPosition(offset + j, conf2.GetAtomPosition(j))
    combo_b.RemoveAllConformers()
    combo_b.AddConformer(conf_c, assignId=True)
    return combo_b


def write_outputs(mol: Chem.Mol, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = out_dir / f"{stem}_ts_guess.sdf"
    w = Chem.SDWriter(str(sdf_path))
    w.write(mol)
    w.close()

    xyz_path = out_dir / f"{stem}_ts_guess.xyz"
    conf = mol.GetConformer()
    with xyz_path.open("w") as f:
        f.write(f"{mol.GetNumAtoms()}\n{stem}_ts_guess\n")
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol():<2} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")


def process(stem: str, sdf_dir: Path, out_dir: Path, r2_stretch: float, r2_shift: float, r2_ortho: float) -> bool:
    sdf_path = sdf_dir / f"{stem}_updated.sdf"
    if not sdf_path.exists():
        print(f"[skip] {stem}: missing {sdf_path}")
        return False
    try:
        r1h, r2h = load_components(sdf_path)
    except ValueError as e:
        print(f"[skip] {stem}: {e}")
        return False

    dH = find_label_idx(r1h, "d_hydrogen")
    aH = find_label_idx(r2h, "a_hydrogen")
    if dH is None:
        dH = fallback_first_h(r1h)
    if aH is None:
        aH = fallback_first_h(r2h)
    if dH is None or aH is None:
        print(f"[skip] {stem}: missing donor/acceptor H (labels or fallback)")
        return False

    A = neighbor_heavy(r1h, dH)
    conf1 = r1h.GetConformer()
    donor_H_pos = np.array(conf1.GetAtomPosition(dH), dtype=float)
    donor_A_pos = np.array(conf1.GetAtomPosition(A), dtype=float)

    r2h_t, B_new = transform_r2h(
        r2h,
        aH,
        donor_H_pos,
        donor_A_pos,
        r2_stretch=r2_stretch,
        r2_shift=r2_shift,
        r2_ortho=r2_ortho,
    )
    combo = fuse(r1h, r2h_t, dH, B_new)
    write_outputs(combo, stem, out_dir)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Build heuristic TS guesses (geometry) from R1H/R2H pairs.")
    p.add_argument("--ndjson", type=str, default="examples/ts_molecules.ndjson")
    p.add_argument("--sdf_dir", type=str, default="DATA/SDF/updated")
    p.add_argument("--out_dir", type=str, default="ts_guesses")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--r2_stretch", type=float, default=1.2, help="Scale for B–H bond length when placing acceptor.")
    p.add_argument("--r2_shift", type=float, default=0.3, help="Extra shift (Å) along donor A→H direction to reduce collisions.")
    p.add_argument("--r2_ortho", type=float, default=0.5, help="Orthogonal bump (Å) to separate fragments laterally.")
    args = p.parse_args()

    stems = []
    with Path(args.ndjson).open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            stems.append(Path(entry["sdf_file"]).stem)
            if args.limit is not None and len(stems) >= args.limit:
                break

    ok = 0
    for stem in stems:
        ok += int(
            process(
                stem,
                Path(args.sdf_dir),
                Path(args.out_dir),
                args.r2_stretch,
                args.r2_shift,
                args.r2_ortho,
            )
        )
    print(f"Done. Wrote {ok} guesses to {args.out_dir}.")


if __name__ == "__main__":
    main()
