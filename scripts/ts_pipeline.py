#!/usr/bin/env python
"""
End-to-end TS pipeline:
1) (optional) train a baseline model and write predictions to SQLite
2) generate TS guess XYZs from SDFs (simple merge)
3) (optional) run a geometry optimization on the guesses
4) record outputs in a single SQLite DB
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
from typing import Iterable, Optional, Tuple

import numpy as np

from scripts import batch_simple_ts_merge as merge
from scripts.ts_geom_opt import ConstraintSpec, XYZSpec, run_opt

_LOG = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ts_guesses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merge_run_id TEXT,
            model_run_id TEXT,
            created_at TEXT,
            rxn_name TEXT,
            sdf_path TEXT,
            guess_xyz_path TEXT,
            props_json_path TEXT,
            swap_by_ts INTEGER,
            remap_to_ts INTEGER,
            error TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ts_optimizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merge_run_id TEXT,
            created_at TEXT,
            rxn_name TEXT,
            guess_xyz_path TEXT,
            opt_xyz_path TEXT,
            engine TEXT,
            fmax REAL,
            steps INTEGER,
            charge INTEGER,
            multiplicity INTEGER,
            fmax_actual REAL,
            max_constraint_deviation REAL,
            ah_dist REAL,
            hb_dist REAL,
            ahb_angle REAL,
            status TEXT,
            error TEXT,
            constraints_json_path TEXT
        )
        """
    )
    cur.execute("PRAGMA table_info(ts_optimizations)")
    columns = {row[1] for row in cur.fetchall()}
    if "constraints_json_path" not in columns:
        cur.execute("ALTER TABLE ts_optimizations ADD COLUMN constraints_json_path TEXT")
    if "charge" not in columns:
        cur.execute("ALTER TABLE ts_optimizations ADD COLUMN charge INTEGER")
    if "multiplicity" not in columns:
        cur.execute("ALTER TABLE ts_optimizations ADD COLUMN multiplicity INTEGER")
    if "fmax_actual" not in columns:
        cur.execute("ALTER TABLE ts_optimizations ADD COLUMN fmax_actual REAL")
    if "max_constraint_deviation" not in columns:
        cur.execute("ALTER TABLE ts_optimizations ADD COLUMN max_constraint_deviation REAL")
    if "ah_dist" not in columns:
        cur.execute("ALTER TABLE ts_optimizations ADD COLUMN ah_dist REAL")
    if "hb_dist" not in columns:
        cur.execute("ALTER TABLE ts_optimizations ADD COLUMN hb_dist REAL")
    if "ahb_angle" not in columns:
        cur.execute("ALTER TABLE ts_optimizations ADD COLUMN ahb_angle REAL")
    conn.commit()
    return conn


def _latest_run_id(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    if cur.fetchone() is None:
        return None
    cur.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None


def run_training(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_xyz(path: Path) -> XYZSpec:
    lines = path.read_text().strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Not enough lines for XYZ: {path}")
    try:
        n_atoms = int(lines[0].strip())
    except Exception as exc:
        raise ValueError(f"Invalid atom count in {path}") from exc
    coord_lines = lines[2 : 2 + n_atoms]
    symbols = []
    coords = []
    for line in coord_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad XYZ line in {path}: {line}")
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return XYZSpec(symbols=tuple(symbols), isotopes=tuple([0] * n_atoms), coords=tuple(coords))


def _write_xyz(path: Path, xyz: XYZSpec) -> None:
    lines = [str(len(xyz.symbols)), "optimized by ts_geom_opt"]
    for sym, (x, y, z) in zip(xyz.symbols, xyz.coords):
        lines.append(f"{sym:2s} {x: .6f} {y: .6f} {z: .6f}")
    path.write_text("\n".join(lines))


TS_CONSTRAINT_ROLE_PAIRS: tuple[Tuple[str, str], ...] = (
    ("*1", "*2"),
    ("*2", "*3"),
)


def _load_ts_props(props_path: Path) -> Optional[dict]:
    if not props_path.exists():
        return None
    try:
        return json.loads(props_path.read_text())
    except Exception:
        return None


def _parse_int_value(value: str, allow_negative: bool) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(float(str(value).strip()))
    except ValueError:
        return None
    if parsed < 0 and not allow_negative:
        return None
    return parsed


def _parse_props_int(props: dict, keys: tuple[str, ...], allow_negative: bool) -> Optional[int]:
    for key in keys:
        if key in props:
            parsed = _parse_int_value(props.get(key), allow_negative=allow_negative)
            if parsed is not None:
                return parsed
    return None


def _parse_sdf_props_block(block: str) -> dict:
    props: dict[str, str] = {}
    lines = block.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith(">") and "<" in line and ">" in line:
            key = line.split("<", 1)[1].split(">", 1)[0].strip().lower()
            i += 1
            values = []
            while i < n and lines[i].strip() != "":
                values.append(lines[i].rstrip())
                i += 1
            props[key] = "\n".join(values).strip()
        else:
            i += 1
    return props


def _charge_multiplicity_from_sdf(sdf_path: Path) -> Tuple[Optional[int], Optional[int]]:
    text = sdf_path.read_text(errors="ignore")
    blocks = [blk for blk in text.split("$$$$") if blk.strip()]
    by_type: dict[str, dict] = {}
    for block in blocks:
        props = _parse_sdf_props_block(block)
        block_type = props.get("type", "").strip().lower()
        if block_type:
            by_type[block_type] = props
    ordered = [by_type.get("ts"), by_type.get("r1h"), by_type.get("r2h")]
    charge = None
    multiplicity = None
    for props in ordered:
        if not props:
            continue
        if charge is None:
            charge = _parse_props_int(props, ("charge", "formal_charge"), allow_negative=True)
        if multiplicity is None:
            multiplicity = _parse_props_int(props, ("multiplicity", "spin_multiplicity"), allow_negative=False)
        if charge is not None and multiplicity is not None:
            break
    return charge, multiplicity


def _charge_multiplicity_for_guess(
    guess_xyz: Path,
    sdf_dir: Optional[Path],
    default_charge: int,
    default_multiplicity: int,
) -> Tuple[int, int]:
    charge = None
    multiplicity = None
    props = _load_ts_props(guess_xyz.with_suffix(".ts_props.json"))
    if props:
        charge = _parse_props_int(props, ("charge",), allow_negative=True)
        multiplicity = _parse_props_int(props, ("multiplicity",), allow_negative=False)
    if (charge is None or multiplicity is None) and sdf_dir is not None:
        rxn_name = guess_xyz.stem.replace("_ts_guess", "")
        sdf_path = sdf_dir / f"{rxn_name}.sdf"
        if sdf_path.exists():
            sdf_charge, sdf_multiplicity = _charge_multiplicity_from_sdf(sdf_path)
            if charge is None:
                charge = sdf_charge
            if multiplicity is None:
                multiplicity = sdf_multiplicity
    if charge is None:
        charge = default_charge
    if multiplicity is None:
        multiplicity = default_multiplicity
    return charge, multiplicity


def _build_constraints_from_props(xyz: XYZSpec, props: dict) -> list[ConstraintSpec]:
    atoms = props.get("atoms", [])
    role_to_output: dict[str, int] = {}
    for atom in atoms:
        role = atom.get("ts_role")
        out_idx = atom.get("output_index")
        if role and isinstance(out_idx, int):
            role_to_output[role] = out_idx

    coords = np.array(xyz.coords, dtype=float)
    constraints: list[ConstraintSpec] = []
    for role_a, role_b in TS_CONSTRAINT_ROLE_PAIRS:
        idx_a = role_to_output.get(role_a)
        idx_b = role_to_output.get(role_b)
        if idx_a is None or idx_b is None:
            continue
        distance = float(np.linalg.norm(coords[idx_a] - coords[idx_b]))
        constraints.append(
            ConstraintSpec(atom_indices_1based=(idx_a + 1, idx_b + 1), distance=distance)
        )
    return constraints


def _write_constraints_json(path: Path, constraints: list[ConstraintSpec], props_name: Optional[str]) -> None:
    payload = {
        "ts_props": props_name,
        "constraints": [
            {
                "atom_indices_1based": list(spec.atom_indices_1based),
                "distance": spec.distance,
            }
            for spec in constraints
        ],
        "recorded_at": _now(),
    }
    path.write_text(json.dumps(payload, indent=2))


def _constraints_for_guess(guess_xyz: Path, xyz: XYZSpec) -> Tuple[Optional[list[ConstraintSpec]], Optional[Path]]:
    props_path = guess_xyz.with_suffix(".ts_props.json")
    props = _load_ts_props(props_path)
    if props is None:
        return None, None

    constraint_specs = _build_constraints_from_props(xyz, props)
    if not constraint_specs:
        return None, None

    constraints_path = guess_xyz.with_suffix(".hook_constraints.json")
    _write_constraints_json(constraints_path, constraint_specs, props_path.name)
    return constraint_specs, constraints_path


def _angle_metrics_from_props(xyz: XYZSpec, props: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    atoms = props.get("atoms", [])
    role_to_output: dict[str, int] = {}
    for atom in atoms:
        role = atom.get("ts_role")
        out_idx = atom.get("output_index")
        if role and isinstance(out_idx, int):
            role_to_output[role] = out_idx

    idx_a = role_to_output.get("*1")
    idx_h = role_to_output.get("*2")
    idx_b = role_to_output.get("*3")
    if idx_a is None or idx_h is None or idx_b is None:
        return None, None, None

    coords = np.array(xyz.coords, dtype=float)
    a = coords[idx_a]
    h = coords[idx_h]
    b = coords[idx_b]
    ah_vec = a - h
    hb_vec = b - h
    ah_dist = float(np.linalg.norm(ah_vec))
    hb_dist = float(np.linalg.norm(hb_vec))
    denom = np.linalg.norm(ah_vec) * np.linalg.norm(hb_vec)
    if denom == 0.0:
        return ah_dist, hb_dist, None
    cosang = np.clip(np.dot(ah_vec, hb_vec) / denom, -1.0, 1.0)
    ahb_angle = float(np.degrees(np.arccos(cosang)))
    return ah_dist, hb_dist, ahb_angle


def _append_opt_failure(log_path: Path, rxn_name: str, guess_xyz: Path, error: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{_now()}\t{rxn_name}\t{guess_xyz}\t{error}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


@dataclass
class MergeResult:
    rxn_name: str
    sdf_path: str
    guess_xyz_path: str
    props_json_path: Optional[str]
    swap_by_ts: bool
    remap_to_ts: bool
    error: Optional[str] = None


def merge_one(
    sdf_path: Path,
    out_dir: Path,
    swap_by_ts: bool,
    remap_to_ts: bool,
) -> MergeResult:
    rxn_name = sdf_path.stem
    out_xyz = out_dir / f"{rxn_name}_ts_guess.xyz"
    try:
        r1h, r2h, ts = merge.load_r1h_r2h(sdf_path)
        ts_xyz, mapping, swapped = merge.build_ts_guess(
            r1h, r2h, ts, swap_by_ts=swap_by_ts, remap_to_ts=remap_to_ts
        )
        props_path = None
        comment = None
        if mapping is not None:
            mapping["swap_by_ts"] = bool(swapped)
            props_path = out_xyz.with_suffix(".ts_props.json")
            merge.write_props(props_path, mapping)
            roles_out = {}
            for entry in mapping.get("atoms", []):
                role = entry.get("ts_role")
                if role:
                    roles_out[role] = entry.get("output_index")
            comment = json.dumps(
                {"ts_roles": roles_out, "props_path": props_path.name},
                separators=(",", ":"),
            )
        merge.write_xyz(out_xyz, ts_xyz["symbols"], ts_xyz["coords"], comment=comment)
        return MergeResult(
            rxn_name=rxn_name,
            sdf_path=str(sdf_path.resolve()),
            guess_xyz_path=str(out_xyz.resolve()),
            props_json_path=str(props_path.resolve()) if props_path else None,
            swap_by_ts=bool(swapped),
            remap_to_ts=bool(remap_to_ts),
        )
    except Exception as exc:  # noqa: BLE001
        return MergeResult(
            rxn_name=rxn_name,
            sdf_path=str(sdf_path.resolve()),
            guess_xyz_path=str(out_xyz.resolve()),
            props_json_path=None,
            swap_by_ts=False,
            remap_to_ts=bool(remap_to_ts),
            error=str(exc),
        )


def record_merge(conn: sqlite3.Connection, merge_run_id: str, model_run_id: Optional[str], result: MergeResult) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ts_guesses (
            merge_run_id, model_run_id, created_at, rxn_name, sdf_path, guess_xyz_path,
            props_json_path, swap_by_ts, remap_to_ts, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            merge_run_id,
            model_run_id,
            _now(),
            result.rxn_name,
            result.sdf_path,
            result.guess_xyz_path,
            result.props_json_path,
            int(result.swap_by_ts),
            int(result.remap_to_ts),
            result.error,
        ),
    )
    conn.commit()


def optimize_one(
    guess_xyz: Path,
    out_dir: Path,
    fmax: float,
    steps: int,
    engine: str,
    use_xtb: bool,
    charge: int,
    multiplicity: int,
    sdf_dir: Optional[Path],
    constraints: Optional[Iterable[ConstraintSpec]] = None,
) -> Tuple[
    Optional[Path],
    Optional[str],
    Optional[Path],
    int,
    int,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    xyz = _read_xyz(guess_xyz)
    auto_constraints, constraints_path = _constraints_for_guess(guess_xyz, xyz)
    resolved_charge, resolved_multiplicity = _charge_multiplicity_for_guess(
        guess_xyz,
        sdf_dir=sdf_dir,
        default_charge=charge,
        default_multiplicity=multiplicity,
    )
    diagnostics: dict = {}
    props = _load_ts_props(guess_xyz.with_suffix(".ts_props.json"))
    try:
        opt = run_opt(
            xyz=xyz,
            constraints=auto_constraints if auto_constraints is not None else constraints,
            fmax=fmax,
            steps=steps,
            engine=engine,
            use_xtb=use_xtb,
            charge=resolved_charge,
            multiplicity=resolved_multiplicity,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        return (
            None,
            str(exc),
            constraints_path,
            resolved_charge,
            resolved_multiplicity,
            None,
            None,
            None,
            None,
            None,
        )
    if opt is None:
        return (
            None,
            "optimizer_failed",
            constraints_path,
            resolved_charge,
            resolved_multiplicity,
            diagnostics.get("fmax_actual"),
            diagnostics.get("max_constraint_deviation"),
            None,
            None,
            None,
        )
    ah_dist, hb_dist, ahb_angle = (None, None, None)
    if props:
        ah_dist, hb_dist, ahb_angle = _angle_metrics_from_props(opt, props)
    out_path = out_dir / guess_xyz.name.replace("_ts_guess", "_ts_opt")
    _write_xyz(out_path, opt)
    return (
        out_path,
        None,
        constraints_path,
        resolved_charge,
        resolved_multiplicity,
        diagnostics.get("fmax_actual"),
        diagnostics.get("max_constraint_deviation"),
        ah_dist,
        hb_dist,
        ahb_angle,
    )


def record_opt(
    conn: sqlite3.Connection,
    merge_run_id: str,
    rxn_name: str,
    guess_xyz_path: str,
    opt_xyz_path: Optional[str],
    engine: str,
    fmax: float,
    steps: int,
    charge: int,
    multiplicity: int,
    fmax_actual: Optional[float],
    max_constraint_deviation: Optional[float],
    ah_dist: Optional[float],
    hb_dist: Optional[float],
    ahb_angle: Optional[float],
    status: str,
    error: Optional[str],
    constraints_json_path: Optional[str],
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ts_optimizations (
            merge_run_id, created_at, rxn_name, guess_xyz_path, opt_xyz_path, engine, fmax, steps,
            charge, multiplicity, fmax_actual, max_constraint_deviation, ah_dist, hb_dist, ahb_angle,
            status, error, constraints_json_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            merge_run_id,
            _now(),
            rxn_name,
            guess_xyz_path,
            opt_xyz_path,
            engine,
            fmax,
            steps,
            charge,
            multiplicity,
            fmax_actual,
            max_constraint_deviation,
            ah_dist,
            hb_dist,
            ahb_angle,
            status,
            error,
            constraints_json_path,
        ),
    )
    conn.commit()


def main() -> None:
    p = argparse.ArgumentParser(description="Run TS pipeline end-to-end.")
    p.add_argument("--train", action="store_true", help="Train baseline model before merging.")
    p.add_argument("--ts-path", type=str, default="DATA/ts_molecules.ndjson")
    p.add_argument("--sdf-dir", type=str, default="DATA/SDF")
    p.add_argument("--db", type=str, default="ts_predictions.sqlite")
    p.add_argument("--guess-dir", type=str, default="ts_guesses")
    p.add_argument("--opt-dir", type=str, default="ts_guesses_opt")
    p.add_argument("--swap-by-ts", action="store_true")
    p.add_argument("--remap-to-ts", action="store_true")
    p.add_argument("--optimize", action="store_true", help="Run geometry optimization on guesses.")
    p.add_argument("--opt-fmax", type=float, default=0.05)
    p.add_argument("--opt-steps", type=int, default=400)
    p.add_argument("--opt-engine", type=str, default="SciPyFminBFGS")
    p.add_argument("--opt-use-xtb", action="store_true", help="Use xTB (GFN2-xTB) for optimization.")
    p.add_argument("--opt-charge", type=int, default=0, help="Total charge for xTB calculations.")
    p.add_argument(
        "--opt-multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity (M=2S+1) for xTB; converted to UHF via M-1.",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of SDFs to process.")
    args = p.parse_args()

    db_path = Path(args.db)
    conn = _init_db(db_path)

    if args.train:
        train_cmd = [
            "python",
            "scripts/ts_basic.py",
            "--ts-path",
            args.ts_path,
            "--sdf-dir",
            args.sdf_dir,
            "--db",
            args.db,
        ]
        run_training(train_cmd)

    model_run_id = _latest_run_id(conn)
    merge_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    sdf_paths = sorted(Path(args.sdf_dir).glob("*.sdf"))
    if args.limit:
        sdf_paths = sdf_paths[: args.limit]
    guess_dir = Path(args.guess_dir)
    guess_dir.mkdir(parents=True, exist_ok=True)

    for sdf_path in sdf_paths:
        result = merge_one(
            sdf_path=sdf_path,
            out_dir=guess_dir,
            swap_by_ts=args.swap_by_ts,
            remap_to_ts=args.remap_to_ts,
        )
        record_merge(conn, merge_run_id, model_run_id, result)

    if args.optimize:
        opt_dir = Path(args.opt_dir)
        opt_dir.mkdir(parents=True, exist_ok=True)
        failure_log = opt_dir / "opt_failures.log"
        guess_paths = sorted(guess_dir.glob("*_ts_guess.xyz"))
        for guess_path in guess_paths:
            _LOG.warning("Optimizing guess %s", guess_path.name)
            rxn_name = guess_path.stem.replace("_ts_guess", "")
            (
                opt_xyz_path,
                error,
                constraints_path,
                resolved_charge,
                resolved_multiplicity,
                fmax_actual,
                max_constraint_deviation,
                ah_dist,
                hb_dist,
                ahb_angle,
            ) = optimize_one(
                guess_xyz=guess_path,
                out_dir=opt_dir,
                fmax=args.opt_fmax,
                steps=args.opt_steps,
                engine=args.opt_engine,
                use_xtb=args.opt_use_xtb,
                charge=args.opt_charge,
                multiplicity=args.opt_multiplicity,
                sdf_dir=Path(args.sdf_dir),
                constraints=None,
            )
            status = "ok" if opt_xyz_path else "failed"
            if status == "failed" and error:
                _append_opt_failure(failure_log, rxn_name, guess_path, error)
            record_opt(
                conn,
                merge_run_id,
                rxn_name,
                str(guess_path.resolve()),
                str(opt_xyz_path.resolve()) if opt_xyz_path else None,
                args.opt_engine,
                args.opt_fmax,
                args.opt_steps,
                resolved_charge,
                resolved_multiplicity,
                fmax_actual,
                max_constraint_deviation,
                ah_dist,
                hb_dist,
                ahb_angle,
                status,
                error,
                str(constraints_path.resolve()) if constraints_path else None,
            )

    conn.close()


if __name__ == "__main__":
    main()
