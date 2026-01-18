#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class BaselineStats:
    runtime_seconds: Optional[float]
    job_count: int
    job_types: str
    charge: Optional[int]
    multiplicity: Optional[int]
    pbs_ncpus_max: Optional[int]
    pbs_mem_bytes_max: Optional[int]
    gjf_nprocshared_max: Optional[int]
    gjf_mem_bytes_max: Optional[int]


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _agg(values: List[float], mode: str) -> Optional[float]:
    if not values:
        return None
    if mode == "sum":
        return sum(values)
    if mode == "max":
        return max(values)
    if mode == "mean":
        return sum(values) / len(values)
    raise ValueError(f"Unknown agg mode: {mode}")


def load_baseline(
    csv_path: Path, agg_mode: str, job_type_filter: Optional[set[str]]
) -> Dict[str, BaselineStats]:
    grouped: Dict[str, Dict[str, List]] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("reaction_name")
            if not name:
                continue
            job_type = row.get("job_type")
            if job_type_filter and job_type not in job_type_filter:
                continue
            entry = grouped.setdefault(
                name,
                {
                    "runtime_seconds": [],
                    "job_types": set(),
                    "charge": [],
                    "multiplicity": [],
                    "pbs_ncpus": [],
                    "pbs_mem_bytes": [],
                    "gjf_nprocshared": [],
                    "gjf_mem_bytes": [],
                },
            )
            runtime = _parse_float(row.get("runtime_seconds", ""))
            if runtime is not None:
                entry["runtime_seconds"].append(runtime)
            if job_type:
                entry["job_types"].add(job_type)
            charge = _parse_int(row.get("charge", ""))
            if charge is not None:
                entry["charge"].append(charge)
            multiplicity = _parse_int(row.get("multiplicity", ""))
            if multiplicity is not None:
                entry["multiplicity"].append(multiplicity)
            pbs_ncpus = _parse_int(row.get("pbs_ncpus", ""))
            if pbs_ncpus is not None:
                entry["pbs_ncpus"].append(pbs_ncpus)
            pbs_mem_bytes = _parse_int(row.get("pbs_mem_bytes", ""))
            if pbs_mem_bytes is not None:
                entry["pbs_mem_bytes"].append(pbs_mem_bytes)
            gjf_nproc = _parse_int(row.get("gjf_nprocshared", ""))
            if gjf_nproc is not None:
                entry["gjf_nprocshared"].append(gjf_nproc)
            mem_bytes = _parse_int(row.get("gjf_mem_bytes", ""))
            if mem_bytes is not None:
                entry["gjf_mem_bytes"].append(mem_bytes)

    stats: Dict[str, BaselineStats] = {}
    for name, entry in grouped.items():
        runtime = _agg(entry["runtime_seconds"], agg_mode)
        stats[name] = BaselineStats(
            runtime_seconds=runtime,
            job_count=len(entry["runtime_seconds"]),
            job_types=";".join(sorted(entry["job_types"])),
            charge=entry["charge"][0] if entry["charge"] else None,
            multiplicity=entry["multiplicity"][0] if entry["multiplicity"] else None,
            pbs_ncpus_max=max(entry["pbs_ncpus"]) if entry["pbs_ncpus"] else None,
            pbs_mem_bytes_max=max(entry["pbs_mem_bytes"]) if entry["pbs_mem_bytes"] else None,
            gjf_nprocshared_max=max(entry["gjf_nprocshared"]) if entry["gjf_nprocshared"] else None,
            gjf_mem_bytes_max=max(entry["gjf_mem_bytes"]) if entry["gjf_mem_bytes"] else None,
        )
    return stats


def reaction_name_from_xyz(xyz_path: Path) -> str:
    stem = xyz_path.stem
    for suffix in ("_ts_opt", "_ts_guess"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def select_reactions(
    xyz_paths: Iterable[Path],
    baseline: Dict[str, BaselineStats],
    select_all: bool,
    top_n: Optional[int],
    ids: Optional[List[str]],
    exclude_existing: Optional[set[str]],
) -> List[Tuple[str, Path]]:
    mapping = {reaction_name_from_xyz(p): p for p in xyz_paths}
    if ids:
        return [(name, mapping[name]) for name in ids if name in mapping]
    if exclude_existing:
        mapping = {name: path for name, path in mapping.items() if name not in exclude_existing}
    if select_all or top_n is None:
        return sorted(mapping.items())
    ranked = []
    for name, path in mapping.items():
        base = baseline.get(name)
        if base and base.runtime_seconds is not None:
            ranked.append((base.runtime_seconds, name, path))
    ranked.sort(reverse=True)
    return [(name, path) for _, name, path in ranked[:top_n]]


def write_gjf(
    xyz_path: Path,
    gjf_path: Path,
    title: str,
    charge: int,
    multiplicity: int,
    route: str,
    mem: Optional[str],
    nprocshared: Optional[int],
    link0: List[str],
) -> None:
    lines = xyz_path.read_text().splitlines()
    coords = lines[2:]
    link0_lines = []
    if nprocshared:
        link0_lines.append(f"%nprocshared={nprocshared}")
    if mem:
        link0_lines.append(f"%mem={mem}")
    link0_lines.extend(link0)
    content = "\n".join(
        link0_lines
        + [
            route,
            "",
            title,
            "",
            f"{charge} {multiplicity}",
            *coords,
            "",
            "",
        ]
    )
    gjf_path.write_text(content)


def write_submit_sh(
    path: Path,
    job_name: str,
    pbs_queue: str,
    ncpus: int,
    mem_bytes: int,
) -> None:
    text = "\n".join(
        [
            "#!/bin/bash",
            f"#PBS -q {pbs_queue}",
            f"#PBS -N {job_name}",
            f"#PBS -l select=1:ncpus={ncpus}:mem={mem_bytes}:mpiprocs={ncpus}",
            "#PBS -o out.txt",
            "#PBS -e err.txt",
            "",
            ". ~/.bashrc",
            "",
            'cd "$PBS_O_WORKDIR"',
            "",
            "source /usr/local/g09/setup.sh",
            "",
            'GAUSS_SCRDIR="/gtmp/calvin.p/scratch/g09/$PBS_JOBID"',
            "",
            'mkdir -p "$GAUSS_SCRDIR"',
            "",
            'export GAUSS_SCRDIR="$GAUSS_SCRDIR"',
            "",
            "touch initial_time",
            "",
            'cd "$GAUSS_SCRDIR"',
            "",
            'cp "$PBS_O_WORKDIR/input.gjf" "$GAUSS_SCRDIR"',
            "",
            'if [ -f "$PBS_O_WORKDIR/check.chk" ]; then',
            '    cp "$PBS_O_WORKDIR/check.chk" "$GAUSS_SCRDIR/"',
            "fi",
            "",
            "g09 < input.gjf > input.log",
            "",
            'cp input.* "$PBS_O_WORKDIR/"',
            "",
            "if [ -f check.chk ]; then",
            '    cp check.chk "$PBS_O_WORKDIR/"',
            "fi",
            "",
            'rm -vrf "$GAUSS_SCRDIR"',
            "",
            'cd "$PBS_O_WORKDIR"',
            "",
            "touch final_time",
            "",
        ]
    )
    path.write_text(text)


def run_gaussian(g09_cmd: str, gjf_path: Path, log_path: Path) -> Tuple[str, float]:
    cmd = shlex.split(g09_cmd)
    start = time.monotonic()
    with gjf_path.open("rb") as inp, log_path.open("wb") as out:
        proc = subprocess.run(cmd, stdin=inp, stdout=out, stderr=subprocess.STDOUT)
    runtime = time.monotonic() - start
    status = "ok" if proc.returncode == 0 else "failed"
    return status, runtime


def _mem_bytes_to_mb(mem_bytes: Optional[int]) -> Optional[int]:
    if mem_bytes is None:
        return None
    return max(1, int(mem_bytes / (1024 * 1024)))


def _record_path(path: Path, seen: set[Path], paths: List[Path]) -> None:
    if path in seen:
        return
    seen.add(path)
    paths.append(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gaussian09 inputs for ts_guesses_opt and optionally benchmark runtimes."
    )
    parser.add_argument("--xyz-dir", type=str, default="ts_guesses_opt")
    parser.add_argument("--baseline-csv", type=str, default="ts0_opt_resources.csv")
    parser.add_argument("--baseline-agg", type=str, choices=["sum", "max", "mean"], default="sum")
    parser.add_argument(
        "--baseline-job-type",
        type=str,
        default="opt",
        help='Comma-separated job_type filter for baseline (default: "opt"; use "all" for no filter).',
    )
    parser.add_argument("--all", action="store_true", help="Select all reactions from xyz-dir.")
    parser.add_argument("--top", type=int, help="Select top N by baseline runtime.")
    parser.add_argument(
        "--ids",
        type=str,
        help="Comma-separated reaction_name list to select explicitly.",
    )
    parser.add_argument("--gjf-dir", type=str, default="ts_guesses_gjf")
    parser.add_argument("--job-dir", type=str, default="ts_guesses_g09_jobs")
    parser.add_argument("--log-dir", type=str, default="ts_guesses_g09_logs")
    parser.add_argument("--results-csv", type=str, default="ts_guesses_g09_results.csv")
    parser.add_argument(
        "--route",
        type=str,
        default="#p opt=(ts,calcfc,noeigentest,maxcycles=200) IOp(2/9=2000) wb97xd/def2tzvp",
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--mem", type=str, default="8GB")
    parser.add_argument("--nprocshared", type=int, default=8)
    parser.add_argument(
        "--no-baseline-resources",
        action="store_true",
        help="Ignore per-reaction resources/charge/multiplicity from baseline CSV.",
    )
    parser.add_argument(
        "--link0",
        action="append",
        default=[],
        help="Extra %%link0 lines (repeatable), e.g. --link0 %chk=foo.chk",
    )
    parser.add_argument("--make-pbs", action="store_true", help="Write submit.sh in job dir.")
    parser.add_argument("--pbs-queue", type=str, default="zeus_combined_q")
    parser.add_argument("--pbs-ncpus", type=int, default=8)
    parser.add_argument("--pbs-mem-bytes", type=int, default=36045000000)
    parser.add_argument("--pbs-job-prefix", type=str, default="")
    parser.add_argument("--run", action="store_true", help="Run Gaussian09 and time each job.")
    parser.add_argument("--g09-cmd", type=str, default="g09", help="Gaussian09 command.")
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Overwrite results CSV instead of append/update mode.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Allow reactions already present in the results CSV to be selected again.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a backup copy of the results CSV before updating.",
    )
    parser.add_argument(
        "--print-created",
        action="store_true",
        help="Print paths for files written during this run.",
    )

    args = parser.parse_args()

    xyz_dir = Path(args.xyz_dir)
    gjf_dir = Path(args.gjf_dir)
    job_dir = Path(args.job_dir)
    log_dir = Path(args.log_dir)
    results_csv = Path(args.results_csv)
    baseline_csv = Path(args.baseline_csv)

    xyz_paths = sorted(xyz_dir.glob("*.xyz"))
    if not xyz_paths:
        raise SystemExit(f"No .xyz files found in {xyz_dir}")

    job_type_filter = None
    if args.baseline_job_type.lower() != "all":
        job_type_filter = {item.strip() for item in args.baseline_job_type.split(",") if item.strip()}
    baseline = (
        load_baseline(baseline_csv, args.baseline_agg, job_type_filter)
        if baseline_csv.exists()
        else {}
    )
    ids = [item.strip() for item in args.ids.split(",")] if args.ids else None
    existing_rows: Dict[str, Dict[str, str]] = {}
    existing_fieldnames: List[str] = []
    if not args.overwrite_results and results_csv.exists():
        with results_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            for row in reader:
                name = row.get("reaction_name")
                if name:
                    existing_rows[name] = row
    exclude_existing = None if args.include_existing else set(existing_rows)

    selected = select_reactions(
        xyz_paths, baseline, args.all, args.top, ids, exclude_existing
    )
    if not selected:
        raise SystemExit(
            "No reactions selected; all candidates may already exist in results CSV."
        )

    gjf_dir.mkdir(parents=True, exist_ok=True)
    if args.make_pbs:
        job_dir.mkdir(parents=True, exist_ok=True)
    if args.run:
        log_dir.mkdir(parents=True, exist_ok=True)

    default_fieldnames = [
        "reaction_name",
        "guess_xyz_path",
        "gjf_path",
        "job_dir",
        "submit_sh_path",
        "log_path",
        "status",
        "runtime_seconds",
        "baseline_runtime_seconds",
        "baseline_job_count",
        "baseline_job_types",
        "charge",
        "multiplicity",
        "baseline_pbs_ncpus_max",
        "baseline_pbs_mem_bytes_max",
        "baseline_gjf_nprocshared_max",
        "baseline_gjf_mem_bytes_max",
    ]
    if existing_fieldnames:
        fieldnames = list(existing_fieldnames)
        for field in default_fieldnames:
            if field not in fieldnames:
                fieldnames.append(field)
    else:
        fieldnames = default_fieldnames

    written_paths: List[Path] = []
    written_seen: set[Path] = set()

    if results_csv.exists() and not args.no_backup:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = results_csv.with_suffix(results_csv.suffix + f".bak_{timestamp}")
        shutil.copy2(results_csv, backup_path)
        _record_path(backup_path, written_seen, written_paths)

    new_rows: Dict[str, Dict[str, str]] = {}
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        _record_path(results_csv, written_seen, written_paths)

        for reaction_name, xyz_path in selected:
            base = baseline.get(reaction_name)
            nproc = args.nprocshared
            mem = args.mem
            charge = args.charge
            multiplicity = args.multiplicity
            pbs_ncpus = args.pbs_ncpus
            pbs_mem_bytes = args.pbs_mem_bytes
            use_baseline = not args.no_baseline_resources
            if use_baseline and base:
                if base.charge is not None:
                    charge = base.charge
                if base.multiplicity is not None:
                    multiplicity = base.multiplicity
                if base.gjf_nprocshared_max:
                    nproc = base.gjf_nprocshared_max
                mem_mb = _mem_bytes_to_mb(base.gjf_mem_bytes_max)
                if mem_mb:
                    mem = f"{mem_mb}MB"
                if base.pbs_ncpus_max:
                    pbs_ncpus = base.pbs_ncpus_max
                if base.pbs_mem_bytes_max:
                    pbs_mem_bytes = base.pbs_mem_bytes_max

            job_path = job_dir / reaction_name
            if args.make_pbs:
                job_path.mkdir(parents=True, exist_ok=True)
                gjf_path = job_path / "input.gjf"
            else:
                gjf_path = gjf_dir / f"{reaction_name}.gjf"
            log_path = log_dir / f"{reaction_name}.log" if args.run else Path("")
            submit_path = job_path / "submit.sh" if args.make_pbs else Path("")

            link0 = list(args.link0)
            if not any(line.startswith("%chk=") for line in link0):
                link0.append("%chk=check.chk")
            write_gjf(
                xyz_path=xyz_path,
                gjf_path=gjf_path,
                title=reaction_name,
                charge=charge,
                multiplicity=multiplicity,
                route=args.route,
                mem=mem,
                nprocshared=nproc,
                link0=link0,
            )
            _record_path(gjf_path, written_seen, written_paths)
            if args.make_pbs:
                job_name = f"{args.pbs_job_prefix}{reaction_name}"
                write_submit_sh(
                    path=submit_path,
                    job_name=job_name,
                    pbs_queue=args.pbs_queue,
                    ncpus=pbs_ncpus,
                    mem_bytes=pbs_mem_bytes,
                )
                _record_path(submit_path, written_seen, written_paths)

            status = "not_run"
            runtime = ""
            if args.run:
                _record_path(log_path, written_seen, written_paths)
                status, runtime_val = run_gaussian(args.g09_cmd, gjf_path, log_path)
                runtime = f"{runtime_val:.3f}"

            new_rows[reaction_name] = {
                "reaction_name": reaction_name,
                "guess_xyz_path": str(xyz_path),
                "gjf_path": str(gjf_path),
                "job_dir": str(job_path) if args.make_pbs else "",
                "submit_sh_path": str(submit_path) if args.make_pbs else "",
                "log_path": str(log_path) if args.run else "",
                "status": status,
                "runtime_seconds": runtime,
                "baseline_runtime_seconds": base.runtime_seconds if base else "",
                "baseline_job_count": base.job_count if base else "",
                "baseline_job_types": base.job_types if base else "",
                "charge": charge,
                "multiplicity": multiplicity,
                "baseline_pbs_ncpus_max": base.pbs_ncpus_max if base else "",
                "baseline_pbs_mem_bytes_max": base.pbs_mem_bytes_max if base else "",
                "baseline_gjf_nprocshared_max": base.gjf_nprocshared_max if base else "",
                "baseline_gjf_mem_bytes_max": base.gjf_mem_bytes_max if base else "",
            }

        if not args.overwrite_results and existing_rows:
            for name, row in existing_rows.items():
                if name not in new_rows:
                    new_rows[name] = row

        for name in sorted(new_rows):
            writer.writerow(new_rows[name])

    if args.print_created and written_paths:
        print("Wrote files:")
        for path in written_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
