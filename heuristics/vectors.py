import numpy as np
from typing import Sequence, Union

ArrayLikeCoords = Union[Sequence[Sequence[float]], np.ndarray, dict]

class VectorsError(Exception):
    pass


def _as_coords(coords: ArrayLikeCoords) -> np.ndarray:
    if isinstance(coords, dict) and "coords" in coords:
        coords = coords["coords"]
    return np.asarray(coords, dtype=np.float64)


def _as_indices(atoms, index: int) -> np.ndarray:
    if index not in (0, 1):
        raise VectorsError(f"index must be 0 or 1, got {index}")

    out = []
    for a in atoms:
        if isinstance(a, str) and a.startswith("X"):
            out.append(int(a[1:]))
        else:
            out.append(a)

    if not all(isinstance(a, int) for a in out):
        raise VectorsError(f"all atom indices must be ints, got {out}")

    out = np.asarray(out, dtype=int) - index  # convert 1-indexed to 0-indexed if needed
    if len(np.unique(out)) != len(out):
        raise VectorsError(f"repeated atom indices in {atoms}")
    return out


def distance(coords: ArrayLikeCoords, atoms, index: int = 0) -> float:
    c = _as_coords(coords)
    a = _as_indices(atoms, index)
    if len(a) != 2:
        raise VectorsError(f"distance needs 2 atoms, got {len(a)}")
    v = c[a[1]] - c[a[0]]
    return float(np.linalg.norm(v))


def angle(coords: ArrayLikeCoords, atoms, index: int = 0, degrees: bool = True) -> float:
    c = _as_coords(coords)
    a = _as_indices(atoms, index)
    if len(a) != 3:
        raise VectorsError(f"angle needs 3 atoms, got {len(a)}")

    # vectors with vertex at a[1]
    v1 = c[a[0]] - c[a[1]]
    v2 = c[a[2]] - c[a[1]]

    # normalize
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        raise VectorsError("zero-length vector in angle")

    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    ang = np.arccos(cos)
    return float(np.degrees(ang) if degrees else ang)


def dihedral(coords: ArrayLikeCoords, atoms, index: int = 0, degrees: bool = True, wrap_360: bool = True) -> float:
    c = _as_coords(coords)
    a = _as_indices(atoms, index)
    if len(a) != 4:
        raise VectorsError(f"dihedral needs 4 atoms, got {len(a)}")

    p0, p1, p2, p3 = c[a[0]], c[a[1]], c[a[2]], c[a[3]]
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 for stable projection
    b1n = np.linalg.norm(b1)
    if b1n == 0.0:
        raise VectorsError("zero-length central bond in dihedral")
    b1u = b1 / b1n

    # normals to the planes
    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u

    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv == 0.0 or nw == 0.0:
        return float("nan")  # colinear / ill-defined
    v /= nv
    w /= nw

    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)
    phi = np.arctan2(y, x)  # (-pi, pi]

    if wrap_360 and phi < 0:
        phi += 2 * np.pi

    return float(np.degrees(phi) if degrees else phi)


def calculate_param(coords: ArrayLikeCoords, atoms, index: int = 0) -> float:
    n = len(atoms)
    if n == 2:
        return distance(coords, atoms, index=index)
    if n == 3:
        return angle(coords, atoms, index=index, degrees=True)
    if n == 4:
        return dihedral(coords, atoms, index=index, degrees=True, wrap_360=True)
    raise ValueError(f"Expected 2, 3, or 4 atoms, got {n}: {atoms}")
