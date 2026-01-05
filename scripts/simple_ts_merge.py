"""
Lightweight H‑abstraction TS guess builder that mirrors the ARC heuristics merge
without depending on ARCSpecies or RMG Molecule.

Inputs are plain xyz dictionaries: ``{"symbols": [...], "coords": np.ndarray shape (n, 3)}``.
Provide the indices of the abstracted hydrogens (``h1``, ``h2``) and their
neighbors (``a``, ``b``), plus optional second neighbors ``c`` and ``d`` for
dihedral control. The algorithm keeps fragment 1 fixed (after stretching A–H)
and places fragment 2 around the shared H to satisfy the desired B–H–A angle.

Example:

    ts_xyz = make_ts_guess(
        xyz1, xyz2, h1=3, h2=0, a=2, b=1,
        c=None, d=None, a2=175.0, d2=None, d3=60.0,
    )

This is intentionally simple: it moves only the two H atoms to stretched bond
lengths, aligns fragment 2 to the target B–H–A angle, and optionally rotates
around the A–H or B–H axes to apply dihedrals. Collision checking is left to
the caller.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    """Return a unit vector; if the vector is near‐zero, return the input."""
    norm = np.linalg.norm(vec)
    return vec if norm < 1e-12 else vec / norm


def orthogonal_vector(vec: np.ndarray) -> np.ndarray:
    """Return any unit vector orthogonal to ``vec``."""
    if abs(vec[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    return normalize(np.cross(vec, ref))


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for rotating about ``axis`` by ``angle_rad``."""
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def rotate_points(points: np.ndarray, axis: np.ndarray, angle_rad: float, origin: np.ndarray) -> np.ndarray:
    """Rotate a set of points around ``axis`` passing through ``origin``."""
    rot = rotation_matrix(axis, angle_rad)
    shifted = points - origin
    return shifted @ rot.T + origin


def stretch_terminal_h(coords: np.ndarray, h_idx: int, anchor_idx: int, stretch: float) -> np.ndarray:
    """Move the H atom along the anchor→H direction to stretch/shrink the bond."""
    new = coords.copy()
    vec = new[h_idx] - new[anchor_idx]
    dist = np.linalg.norm(vec)
    if dist < 1e-8:
        return new
    target = new[anchor_idx] + normalize(vec) * (dist * stretch)
    new[h_idx] = target
    return new


def align_vector(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps ``v_from`` to ``v_to``."""
    v_from_u = normalize(v_from)
    v_to_u = normalize(v_to)
    dot = float(np.clip(np.dot(v_from_u, v_to_u), -1.0, 1.0))
    if abs(dot - 1.0) < 1e-10:
        return np.eye(3)
    if abs(dot + 1.0) < 1e-10:
        axis = orthogonal_vector(v_from_u)
        return rotation_matrix(axis, math.pi)
    axis = np.cross(v_from_u, v_to_u)
    angle = math.acos(dot)
    return rotation_matrix(axis, angle)


def _signed_angle_about_axis(v_from: np.ndarray, v_to: np.ndarray, axis: np.ndarray) -> float | None:
    """Return signed angle to rotate v_from to v_to about axis; None if undefined."""
    axis_u = normalize(axis)
    v_from_p = v_from - axis_u * float(np.dot(v_from, axis_u))
    v_to_p = v_to - axis_u * float(np.dot(v_to, axis_u))
    if np.linalg.norm(v_from_p) < 1e-8 or np.linalg.norm(v_to_p) < 1e-8:
        return None
    v_from_u = normalize(v_from_p)
    v_to_u = normalize(v_to_p)
    return math.atan2(
        float(np.dot(axis_u, np.cross(v_from_u, v_to_u))),
        float(np.dot(v_from_u, v_to_u)),
    )


def make_ts_guess(
    xyz1: dict,
    xyz2: dict,
    h1: int,
    h2: int,
    a: int,
    b: int,
    c: int | None = None,
    d: int | None = None,
    r1_stretch: float = 1.2,
    r2_stretch: float = 1.2,
    a2: float = 180.0,
    d2: float | None = None,
    d3: float | None = None,
) -> dict:
    """
    Build a TS guess by merging two fragments that share the transferring H.

    ``xyz1`` supplies H1 (kept) and its neighbor A. ``xyz2`` supplies H2 (dropped)
    and its neighbor B. Coordinates are modified minimally: both H atoms are
    stretched, fragment 2 is rotated to realize ``B–H–A = a2``, and optional
    dihedrals are applied as whole-fragment rotations about the A–H or B–H axes.
    """

    coords1 = np.array(xyz1["coords"], dtype=float)
    coords2 = np.array(xyz2["coords"], dtype=float)

    # Stretch A–H1 and B–H2 bonds by moving the H atoms.
    coords1 = stretch_terminal_h(coords1, h1, a, r1_stretch)
    coords2 = stretch_terminal_h(coords2, h2, b, r2_stretch)

    # Translate so H1 sits at the origin.
    h1_pos = coords1[h1].copy()
    coords1 -= h1_pos
    a_vec = coords1[a].copy()

    # Prepare fragment 2: translate so H2 is at the origin.
    h2_pos = coords2[h2].copy()
    coords2 -= h2_pos
    b_vec = coords2[b].copy()

    # Define a deterministic target B direction that makes angle a2 with H->A.
    a_dir = normalize(a_vec)
    theta = math.radians(a2)
    u = orthogonal_vector(a_dir)
    target_b_dir = normalize(math.cos(theta) * a_dir + math.sin(theta) * u)

    # Rotate fragment 2 so its H->B aligns with the target direction.
    b_dir = normalize(b_vec)
    rot_align = align_vector(b_dir, target_b_dir)
    coords2 = coords2 @ rot_align.T
    b_vec = coords2[b].copy()

    # Optional d3: rotate fragment 2 about the B–H axis.
    axis_bh = normalize(b_vec)
    if d3 is not None:
        coords2 = rotate_points(coords2, axis_bh, math.radians(d3), origin=np.zeros(3))
        b_vec = coords2[b].copy()
    elif c is not None and d is not None:
        c_vec = coords1[c].copy()
        d_vec = coords2[d].copy()
        n1 = np.cross(a_vec, c_vec)
        n2 = np.cross(b_vec, d_vec)
        angle = _signed_angle_about_axis(n2, n1, axis_bh)
        if angle is not None:
            coords2 = rotate_points(coords2, axis_bh, angle, origin=np.zeros(3))
            b_vec = coords2[b].copy()

    # Optional d2: rotate fragment 1 about the A–H axis.
    if d2 is not None:
        axis_ah = normalize(a_vec)
        coords1 = rotate_points(coords1, axis_ah, math.radians(d2), origin=np.zeros(3))
        a_vec = coords1[a].copy()

    # Merge: keep H from fragment 1, drop H from fragment 2.
    symbols2_wo = [sym for i, sym in enumerate(xyz2["symbols"]) if i != h2]
    coords2_wo = np.delete(coords2, h2, axis=0)

    symbols = list(xyz1["symbols"]) + symbols2_wo
    coords = np.vstack([coords1, coords2_wo])

    return {"symbols": symbols, "coords": coords}


def _as_xyz(symbols: Iterable[str], coords: np.ndarray) -> str:
    """Helper: format an xyz dict as an xyz string."""
    lines = [str(len(symbols)), "generated by simple_ts_merge"]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:2s} {x: .6f} {y: .6f} {z: .6f}")
    return "\n".join(lines)


VDW_RADII = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8}


def has_clash(symbols: list[str], coords: np.ndarray, buffer: float = 0.3) -> bool:
    """Simple van der Waals clash check."""
    n = len(symbols)
    for i in range(n):
        ri = VDW_RADII.get(symbols[i], 1.7)
        for j in range(i + 1, n):
            rj = VDW_RADII.get(symbols[j], 1.7)
            cutoff = ri + rj - buffer
            if np.linalg.norm(coords[i] - coords[j]) < cutoff:
                return True
    return False


def scan_variants(
    xyz1: dict,
    xyz2: dict,
    h1: int,
    h2: int,
    a: int,
    b: int,
    a2_list: Iterable[float],
    d3_list: Iterable[float] | None = None,
    c: int | None = None,
    d: int | None = None,
    r1_stretch: float = 1.2,
    r2_stretch: float = 1.2,
    clash_buffer: float = 0.3,
):
    """
    Generate multiple TS guesses over angle/dihedral grids and filter clashes.
    Returns a list of (a2, d3, xyz_dict).
    """
    d3_list = d3_list or [None]
    results = []
    for a2 in a2_list:
        for d3 in d3_list:
            ts = make_ts_guess(
                xyz1,
                xyz2,
                h1=h1,
                h2=h2,
                a=a,
                b=b,
                c=c,
                d=d,
                a2=a2,
                d3=d3,
                r1_stretch=r1_stretch,
                r2_stretch=r2_stretch,
            )
            if not has_clash(ts["symbols"], ts["coords"], buffer=clash_buffer):
                results.append((a2, d3, ts))
    return results


if __name__ == "__main__":
    # Minimal demonstration with dummy data (A–H1 along +x, B–H2 roughly opposite).
    xyz1_demo = {
        "symbols": ["C", "H", "H", "H"],
        "coords": np.array([
            [0.0, 0.0, 0.0],  # A
            [1.09, 0.0, 0.0],  # H1 (abstracted)
            [-0.5, 0.9, 0.0],
            [-0.5, -0.9, 0.0],
        ]),
    }
    xyz2_demo = {
        "symbols": ["H", "O", "H"],
        "coords": np.array([
            [0.0, 0.0, 0.0],   # H2 (abstracted, will be dropped)
            [-1.0, 0.0, 0.0],  # B
            [-1.0, 0.95, 0.0],
        ]),
    }

    ts = make_ts_guess(
        xyz1_demo,
        xyz2_demo,
        h1=1,
        h2=0,
        a=0,
        b=1,
        a2=175.0,
        d3=60.0,
    )

    print(_as_xyz(ts["symbols"], ts["coords"]))
