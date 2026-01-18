#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re

from gaussian_parser import GaussianParser


def _job_dirs(root: Path) -> List[Path]:
    dirs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if (path / "submit.sh").exists() and (path / "input.gjf").exists():
            dirs.append(path)
    return sorted(dirs)


def _rsync(src: str, dst: str, verbose: bool) -> None:
    cmd = ["rsync", "-az", "-e", "ssh -q", src, dst]
    if verbose:
        cmd.insert(2, "-v")
    subprocess.run(cmd, check=True)


def _ssh(remote: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-q", "-o", "LogLevel=ERROR", remote, command],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _remote_job_path(remote_base: str, job_name: str) -> str:
    return f"{remote_base.rstrip('/')}/{job_name}"


def _remote_dir_exists(remote: str, path: str) -> bool:
    try:
        _ssh(remote, f"test -d {path} && echo OK")
        return True
    except subprocess.CalledProcessError:
        return False


def _remote_file_exists(remote: str, path: str) -> bool:
    try:
        _ssh(remote, f"test -f {path} && echo OK")
        return True
    except subprocess.CalledProcessError:
        return False


def _remote_running_jobs(remote: str) -> Tuple[set[str], set[str]]:
    try:
        out = _ssh(remote, "/opt/pbs/bin/qstat -f -u $USER")
    except subprocess.CalledProcessError:
        return set(), set()
    workdirs = set()
    names = set()
    current_vars: List[str] = []
    in_var_list = False
    for raw in out.splitlines():
        line = raw.rstrip()
        if line.startswith("Job Id:"):
            in_var_list = False
            current_vars = []
            continue
        if line.strip().startswith("Job_Name ="):
            parts = line.split("=", 1)
            if len(parts) == 2:
                names.add(parts[1].strip())
            continue
        if line.strip().startswith("Variable_List ="):
            in_var_list = True
            current_vars = [line.split("=", 1)[1].strip()]
            continue
        if in_var_list and line.startswith((" ", "\t")):
            current_vars.append(line.strip())
            continue
        if in_var_list and not line.startswith((" ", "\t")):
            in_var_list = False
            vars_joined = ",".join(current_vars)
            for item in vars_joined.split(","):
                if item.startswith("PBS_O_WORKDIR="):
                    workdirs.add(item.split("=", 1)[1])
            current_vars = []
        if "PBS_O_WORKDIR=" in line:
            for chunk in line.split(","):
                if "PBS_O_WORKDIR=" in chunk:
                    workdirs.add(chunk.split("PBS_O_WORKDIR=", 1)[1].strip())
    if current_vars:
        vars_joined = ",".join(current_vars)
        for item in vars_joined.split(","):
            if item.startswith("PBS_O_WORKDIR="):
                workdirs.add(item.split("=", 1)[1])
    return workdirs, names


def sync_to_remote(job_dirs: Iterable[Path], remote: str, remote_base: str, verbose: bool) -> None:
    for job_dir in job_dirs:
        remote_path = _remote_job_path(remote_base, job_dir.name)
        if verbose:
            print(f"[sync-to] {job_dir} -> {remote_path}")
        _ssh(remote, f"mkdir -p {remote_path}")
        _rsync(f"{job_dir}/", f"{remote}:{remote_path}/", verbose)


def _parse_job_id(qsub_output: str) -> str:
    match = re.search(r"\b(\d+(?:\.\w+)?)\b", qsub_output)
    return match.group(1) if match else ""


def submit_remote(
    job_dirs: Iterable[Path],
    remote: str,
    remote_base: str,
    submit_log: Optional[Path],
    resubmit: bool,
    verbose: bool,
) -> Dict[str, Dict[str, str]]:
    existing = _load_submit_log(submit_log) if submit_log else {}
    running_workdirs, running_names = _remote_running_jobs(remote) if not resubmit else (set(), set())
    job_meta: Dict[str, Dict[str, str]] = {}
    for job_dir in job_dirs:
        remote_path = _remote_job_path(remote_base, job_dir.name)
        if not resubmit:
            prior = existing.get(job_dir.name, {})
            if prior.get("job_id"):
                continue
            if remote_path in running_workdirs or job_dir.name in running_names:
                if verbose:
                    print(f"[submit] skip running job on server: {job_dir.name}")
                continue
            if _remote_file_exists(remote, f"{remote_path}/initial_time") or _remote_file_exists(
                remote, f"{remote_path}/final_time"
            ):
                if verbose:
                    print(f"[submit] skip job with existing time markers: {job_dir.name}")
                continue
        if verbose:
            print(f"[submit] {job_dir.name} -> {remote_path}")
        out = _ssh(remote, f"cd {remote_path} && /usr/local/bin/qsub submit.sh && date +%s")
        lines = [
            line
            for line in out.splitlines()
            if line.strip() and not line.lstrip().startswith(("**", "WARNING:"))
        ]
        qsub_output = lines[0] if lines else ""
        submit_epoch = lines[1] if len(lines) > 1 else ""
        job_meta[job_dir.name] = {
            "qsub_output": qsub_output,
            "job_id": _parse_job_id(qsub_output),
            "submit_epoch": submit_epoch,
            "remote_path": remote_path,
        }
    return job_meta


def sync_from_remote(job_dirs: Iterable[Path], remote: str, remote_base: str, verbose: bool) -> None:
    for job_dir in job_dirs:
        remote_path = _remote_job_path(remote_base, job_dir.name)
        if verbose:
            print(f"[sync-from] {remote_path} -> {job_dir}")
        if not _remote_dir_exists(remote, remote_path):
            if verbose:
                print(f"[sync-from] skip missing remote dir: {remote_path}")
            continue
        try:
            _rsync(f"{remote}:{remote_path}/", f"{job_dir}/", verbose)
        except subprocess.CalledProcessError:
            if verbose:
                print(f"[sync-from] rsync failed for {remote_path}")
            continue


def _read_results(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames or [], rows


def _write_results(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_baseline(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        baseline = {}
        fine_opt = {}
        for row in reader:
            name = row.get("reaction_name")
            job_type = row.get("job_type")
            if not name:
                continue
            if job_type == "opt":
                baseline[name] = row
            elif job_type == "fine_opt" and name not in fine_opt:
                fine_opt[name] = row
        for name, row in fine_opt.items():
            if name not in baseline:
                row = dict(row)
                row["baseline_note"] = "fallback_fine_opt"
                baseline[name] = row
        return baseline


def _backfill_rows(
    results_csv: Path,
    job_root: Path,
    baseline_csv: Path,
) -> None:
    fieldnames, rows = _read_results(results_csv) if results_csv.exists() else ([], [])
    existing = {row.get("reaction_name", ""): row for row in rows if row.get("reaction_name")}
    baseline = _read_baseline(baseline_csv)

    required_fields = [
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
        "baseline_note",
        "baseline_opt_steps",
        "baseline_opt_steps_source",
        "new_opt_steps",
        "new_opt_steps_source",
        "real_time_seconds",
        "job_id",
        "submit_epoch",
    ]
    if not fieldnames:
        fieldnames = required_fields
    else:
        for field in required_fields:
            if field not in fieldnames:
                fieldnames.append(field)

    for job_dir in _job_dirs(job_root):
        name = job_dir.name
        if name in existing:
            continue
        base = baseline.get(name, {})
        existing[name] = {
            "reaction_name": name,
            "guess_xyz_path": str(Path("ts_guesses_opt") / f"{name}_ts_opt.xyz"),
            "gjf_path": str(job_dir / "input.gjf"),
            "job_dir": str(job_dir),
            "submit_sh_path": str(job_dir / "submit.sh"),
            "log_path": "",
            "status": "not_run",
            "runtime_seconds": "",
            "baseline_runtime_seconds": base.get("runtime_seconds", ""),
            "baseline_job_count": "1" if base else "",
            "baseline_job_types": base.get("job_type", ""),
            "charge": base.get("charge", ""),
            "multiplicity": base.get("multiplicity", ""),
            "baseline_pbs_ncpus_max": base.get("pbs_ncpus", ""),
            "baseline_pbs_mem_bytes_max": base.get("pbs_mem_bytes", ""),
            "baseline_gjf_nprocshared_max": base.get("gjf_nprocshared", ""),
            "baseline_gjf_mem_bytes_max": base.get("gjf_mem_bytes", ""),
            "baseline_note": base.get("baseline_note", ""),
            "baseline_opt_steps": base.get("opt_steps", ""),
            "baseline_opt_steps_source": base.get("opt_steps_source", ""),
            "new_opt_steps": "",
            "new_opt_steps_source": "",
            "real_time_seconds": "",
            "job_id": "",
            "submit_epoch": "",
        }

    _write_results(results_csv, fieldnames, [existing[k] for k in sorted(existing)])


def _load_submit_log(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        cleaned = {}
        for row in reader:
            name = row.get("reaction_name", "")
            if not name:
                continue
            job_id = row.get("job_id", "")
            row["job_id"] = _parse_job_id(job_id) if job_id else ""
            submit_epoch = row.get("submit_epoch", "")
            row["submit_epoch"] = submit_epoch if submit_epoch.isdigit() else ""
            cleaned[name] = row
        return cleaned


def update_real_time(
    results_csv: Path,
    job_root: Path,
    remote: Optional[str],
    remote_base: Optional[str],
    use_remote_times: bool,
    submit_log: Optional[Path],
    baseline_csv: Optional[Path],
    verbose: bool,
) -> None:
    fieldnames, rows = _read_results(results_csv)
    extra_fields = []
    if "real_time_seconds" not in fieldnames:
        extra_fields.append("real_time_seconds")
    if "job_id" not in fieldnames:
        extra_fields.append("job_id")
    if "submit_epoch" not in fieldnames:
        extra_fields.append("submit_epoch")
    if "baseline_opt_steps" not in fieldnames:
        extra_fields.append("baseline_opt_steps")
    if "baseline_opt_steps_source" not in fieldnames:
        extra_fields.append("baseline_opt_steps_source")
    if "new_opt_steps" not in fieldnames:
        extra_fields.append("new_opt_steps")
    if "new_opt_steps_source" not in fieldnames:
        extra_fields.append("new_opt_steps_source")
    if extra_fields:
        fieldnames = fieldnames + extra_fields

    job_index = {p.name: p for p in _job_dirs(job_root)}
    submit_index = _load_submit_log(submit_log) if submit_log else {}
    baseline = _read_baseline(baseline_csv) if baseline_csv else {}
    for row in rows:
        job_dir = row.get("job_dir") or ""
        reaction_name = row.get("reaction_name") or ""
        path = Path(job_dir) if job_dir else job_index.get(reaction_name)
        if not path:
            continue
        initial = path / "initial_time"
        final = path / "final_time"
        log_path = path / "input.log"
        normal_termination = False
        max_steps_exceeded = False
        parser = None
        if log_path.exists():
            log_text = log_path.read_text(errors="ignore")
            normal_termination = "Normal termination of Gaussian" in log_text
            max_steps_exceeded = (
                "Maximum number of optimization cycles exceeded" in log_text
                or "Max steps exceeded" in log_text
                or "Maximum number of optimization steps exceeded" in log_text
            )
            try:
                parser = GaussianParser(str(log_path))
                opt_steps = parser.opt_cycles
                if opt_steps is not None:
                    row["new_opt_steps"] = str(opt_steps)
                    row["new_opt_steps_source"] = "gaussian_parser"
            except Exception:
                pass
        skip_time = False
        if use_remote_times:
            if not (remote and remote_base):
                continue
            remote_path = _remote_job_path(remote_base, reaction_name)
            if row.get("job_dir"):
                remote_path = _remote_job_path(remote_base, Path(row["job_dir"]).name)
            try:
                if verbose:
                    print(f"[time] {reaction_name} (remote)")
                init_ts = _ssh(remote, f"stat -c %Y {remote_path}/initial_time")
                fin_ts = _ssh(remote, f"stat -c %Y {remote_path}/final_time")
                if log_path.exists() and not normal_termination:
                    if verbose:
                        print(f"[time] {reaction_name} no normal termination in input.log")
                    skip_time = True
                if not skip_time:
                    elapsed = float(fin_ts.strip()) - float(init_ts.strip())
                    row["real_time_seconds"] = f"{elapsed:.3f}"
                    if verbose:
                        print(f"[time] {reaction_name} elapsed={elapsed:.3f}s")
            except subprocess.CalledProcessError:
                if verbose:
                    print(f"[time] {reaction_name} missing initial_time/final_time on remote")
                if initial.exists() and final.exists():
                    if verbose:
                        print(f"[time] {reaction_name} falling back to local times")
                    if log_path.exists() and not normal_termination:
                        if verbose:
                            print(f"[time] {reaction_name} no normal termination in input.log")
                        skip_time = True
                    if not skip_time:
                        elapsed = final.stat().st_mtime - initial.stat().st_mtime
                        row["real_time_seconds"] = f"{elapsed:.3f}"
                        if verbose:
                            print(f"[time] {reaction_name} elapsed={elapsed:.3f}s")
                elif normal_termination and parser is not None:
                    fallback = parser.real_time_seconds
                    if fallback is not None:
                        row["real_time_seconds"] = f"{fallback:.3f}"
                        if verbose:
                            print(f"[time] {reaction_name} elapsed={fallback:.3f}s (log-derived)")
        else:
            if not (initial.exists() and final.exists()):
                if verbose:
                    print(f"[time] {reaction_name} missing initial_time/final_time locally")
                if normal_termination and parser is not None:
                    fallback = parser.real_time_seconds
                    if fallback is not None:
                        row["real_time_seconds"] = f"{fallback:.3f}"
                        if verbose:
                            print(f"[time] {reaction_name} elapsed={fallback:.3f}s (log-derived)")
                skip_time = True
            if verbose:
                print(f"[time] {reaction_name} (local)")
            if log_path.exists() and not normal_termination:
                if verbose:
                    print(f"[time] {reaction_name} no normal termination in input.log")
                skip_time = True
            if not skip_time:
                elapsed = final.stat().st_mtime - initial.stat().st_mtime
                row["real_time_seconds"] = f"{elapsed:.3f}"
                if verbose:
                    print(f"[time] {reaction_name} elapsed={elapsed:.3f}s")

        if normal_termination:
            row["status"] = "ok"
        elif max_steps_exceeded:
            row["status"] = "failed_max_steps"
        elif final.exists():
            row["status"] = "failed"
        submit_row = submit_index.get(reaction_name)
        if submit_row:
            job_id = submit_row.get("job_id", "")
            submit_epoch = submit_row.get("submit_epoch", "")
            if job_id:
                row["job_id"] = job_id
            if submit_epoch:
                row["submit_epoch"] = submit_epoch

        base = baseline.get(reaction_name)
        if base:
            if base.get("opt_steps"):
                row["baseline_opt_steps"] = base.get("opt_steps", "")
            if base.get("opt_steps_source"):
                row["baseline_opt_steps_source"] = base.get("opt_steps_source", "")

    _write_results(results_csv, fieldnames, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Gaussian09 job folders to/from Zeus and update real_time_seconds."
    )
    parser.add_argument("--job-dir", type=str, default="ts_guesses_g09_jobs")
    parser.add_argument("--remote", type=str, required=True)
    parser.add_argument("--remote-base", type=str, required=True)
    parser.add_argument("--sync-to", action="store_true", help="Rsync local job dirs to remote.")
    parser.add_argument("--submit", action="store_true", help="Submit jobs via /usr/local/bin/qsub.")
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="Allow re-submission even if a job_id exists in the submit log.",
    )
    parser.add_argument("--sync-from", action="store_true", help="Rsync remote job dirs back locally.")
    parser.add_argument(
        "--results-csv",
        type=str,
        default="ts_guesses_g09_results.csv",
        help="Results CSV to update with real_time_seconds.",
    )
    parser.add_argument("--update-results", action="store_true")
    parser.add_argument(
        "--use-remote-times",
        action="store_true",
        help="Use remote initial_time/final_time mtimes via ssh for real_time_seconds.",
    )
    parser.add_argument(
        "--submit-log",
        type=str,
        default="ts_guesses_g09_submit_log.csv",
        help="CSV log of qsub submissions.",
    )
    parser.add_argument(
        "--backfill-results",
        action="store_true",
        help="Add missing rows for any job dirs not in results CSV.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=str,
        default="ts0_opt_resources.csv",
        help="Baseline CSV for backfill fields.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-job progress.")

    args = parser.parse_args()

    job_root = Path(args.job_dir)
    job_dirs = _job_dirs(job_root)
    if not job_dirs:
        raise SystemExit(f"No job dirs found in {job_root}")

    verbose = not args.quiet
    if args.backfill_results:
        _backfill_rows(Path(args.results_csv), job_root, Path(args.baseline_csv))
    if args.sync_to:
        sync_to_remote(job_dirs, args.remote, args.remote_base, verbose)
    if args.submit:
        submit_log = Path(args.submit_log)
        existing = _load_submit_log(submit_log)
        job_meta = submit_remote(
            job_dirs,
            args.remote,
            args.remote_base,
            submit_log,
            args.resubmit,
            verbose,
        )
        combined = {**existing}
        for reaction_name, meta in job_meta.items():
            combined[reaction_name] = {
                "reaction_name": reaction_name,
                "job_id": meta.get("job_id", ""),
                "submit_epoch": meta.get("submit_epoch", ""),
                "qsub_output": meta.get("qsub_output", ""),
                "remote_path": meta.get("remote_path", ""),
            }
        with submit_log.open("w", newline="") as handle:
            fieldnames = ["reaction_name", "job_id", "submit_epoch", "qsub_output", "remote_path"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for reaction_name in sorted(combined):
                row = combined[reaction_name]
                row.setdefault("reaction_name", reaction_name)
                writer.writerow(row)
    if args.sync_from:
        sync_from_remote(job_dirs, args.remote, args.remote_base, verbose)
    if args.update_results:
        update_real_time(
            Path(args.results_csv),
            job_root,
            args.remote,
            args.remote_base,
            args.use_remote_times,
            Path(args.submit_log),
            Path(args.baseline_csv),
            verbose,
        )


if __name__ == "__main__":
    main()
