#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re


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
    job_meta: Dict[str, Dict[str, str]] = {}
    for job_dir in job_dirs:
        if not resubmit:
            prior = existing.get(job_dir.name, {})
            if prior.get("job_id"):
                continue
        remote_path = _remote_job_path(remote_base, job_dir.name)
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
        _rsync(f"{remote}:{remote_path}/", f"{job_dir}/", verbose)


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


def _load_submit_log(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row.get("reaction_name", ""): row for row in reader if row.get("reaction_name")}


def update_real_time(
    results_csv: Path,
    job_root: Path,
    remote: Optional[str],
    remote_base: Optional[str],
    use_remote_times: bool,
    submit_log: Optional[Path],
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
    if extra_fields:
        fieldnames = fieldnames + extra_fields

    job_index = {p.name: p for p in _job_dirs(job_root)}
    submit_index = _load_submit_log(submit_log) if submit_log else {}
    for row in rows:
        job_dir = row.get("job_dir") or ""
        reaction_name = row.get("reaction_name") or ""
        path = Path(job_dir) if job_dir else job_index.get(reaction_name)
        if not path:
            continue
        initial = path / "initial_time"
        final = path / "final_time"
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
                elapsed = float(fin_ts.strip()) - float(init_ts.strip())
                row["real_time_seconds"] = f"{elapsed:.3f}"
            except subprocess.CalledProcessError:
                if verbose:
                    print(f"[time] {reaction_name} missing initial_time/final_time on remote")
        else:
            if not (initial.exists() and final.exists()):
                if verbose:
                    print(f"[time] {reaction_name} missing initial_time/final_time locally")
                continue
            if verbose:
                print(f"[time] {reaction_name} (local)")
            elapsed = final.stat().st_mtime - initial.stat().st_mtime
            row["real_time_seconds"] = f"{elapsed:.3f}"
        submit_row = submit_index.get(reaction_name)
        if submit_row:
            row["job_id"] = submit_row.get("job_id", row.get("job_id", ""))
            row["submit_epoch"] = submit_row.get("submit_epoch", row.get("submit_epoch", ""))

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
    parser.add_argument("--quiet", action="store_true", help="Suppress per-job progress.")

    args = parser.parse_args()

    job_root = Path(args.job_dir)
    job_dirs = _job_dirs(job_root)
    if not job_dirs:
        raise SystemExit(f"No job dirs found in {job_root}")

    verbose = not args.quiet
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
            verbose,
        )


if __name__ == "__main__":
    main()
