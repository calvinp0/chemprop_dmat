#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


OUTPUT_ARG_KEYS = {
    "out",
    "out_dir",
    "output",
    "output_dir",
    "report_path",
    "preds",
    "save_preds",
    "ckpt_dir",
    "motif_out_dir",
    "log_dir",
}


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "index.md").exists():
            return parent
    return start


def load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required. Install it with: pip install pyyaml"
        ) from exc
    return yaml.safe_load(path.read_text())


def load_yaml_from_index(index_path: Path) -> dict:
    text = index_path.read_text(encoding="utf-8")
    m = re.search(r"^##\s*MCP\s*\n.*?```yaml\s*(.*?)\s*```", text, flags=re.S | re.M)
    if m is None:
        m = re.search(r"```yaml\s*(.*?)\s*```", text, flags=re.S)
    if m is None:
        raise SystemExit("No ```yaml``` block found in index.md.")
    yaml_text = m.group(1)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required. Install it with: pip install pyyaml"
        ) from exc
    return yaml.safe_load(yaml_text)


def dump_yaml(path: Path, payload: dict) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required. Install it with: pip install pyyaml"
        ) from exc
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def to_flag(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def iter_list(value: Iterable[Any]) -> Iterable[str]:
    for item in value:
        yield str(item)


def is_output_key(key: str) -> bool:
    return key in OUTPUT_ARG_KEYS


def rewrite_output_arg(value: Any, run_dir: Path) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    return str(run_dir / path.name)


def build_args(args: dict[str, Any], run_dir: Path) -> list[str]:
    cli: list[str] = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cli.append(to_flag(key))
            continue
        if is_output_key(key):
            value = rewrite_output_arg(value, run_dir)
        if isinstance(value, (list, tuple)):
            for item in iter_list(value):
                cli.append(to_flag(key))
                cli.append(item)
            continue
        cli.append(to_flag(key))
        cli.append(str(value))
    return cli


def unique_run_dir(base_dir: Path, label: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    stem = f"{ts}_{label}"
    candidate = base_dir / stem
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        retry = base_dir / f"{stem}_{idx}"
        if not retry.exists():
            return retry
        idx += 1


def update_index_results(project_root: Path, line: str) -> None:
    index_path = project_root / "index.md"
    if not index_path.exists():
        return
    text = index_path.read_text(encoding="utf-8")
    header = "## Results (AUTO)"
    if header not in text:
        text = text.rstrip() + f"\n\n{header}\n\n{line}\n"
    else:
        before, after = text.split(header, 1)
        before = before.rstrip()
        after = after.lstrip()
        existing_lines = after.splitlines()
        existing_lines = existing_lines[:60]
        new_after = "\n\n" + line + "\n" + "\n".join(existing_lines).rstrip() + "\n"
        text = before + "\n\n" + header + new_after
    index_path.write_text(text, encoding="utf-8")


def run_action(
    project_root: Path,
    outputs_dir: Path,
    name: str,
    action: dict[str, Any],
    conda_env: str,
    dry_run: bool,
    fail_fast: bool,
) -> None:
    script = action.get("script")
    if not script:
        raise SystemExit(f"Action '{name}' missing script.")

    args = action.get("args", {})
    if not isinstance(args, dict):
        raise SystemExit(f"Action '{name}' args must be a mapping.")

    run_dir = unique_run_dir(outputs_dir, name)
    run_dir.mkdir(parents=True, exist_ok=False)

    rewritten_args = build_args(args, run_dir)

    script_path = (project_root / script).resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found for action '{name}': {script_path}")
    cmd = ["conda", "run", "-n", conda_env, "python", str(script_path), *rewritten_args]

    meta = {
        "action": name,
        "script": script,
        "args": args,
        "rewritten_args": rewritten_args,
        "command": cmd,
        "run_dir": str(run_dir),
        "project_root": str(project_root),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    dump_yaml(run_dir / "meta.yaml", meta)

    if dry_run:
        print(" ".join(cmd))
        return

    with (run_dir / "stdout.txt").open("w") as out_fh, (run_dir / "stderr.txt").open(
        "w"
    ) as err_fh:
        res = subprocess.run(cmd, cwd=project_root, stdout=out_fh, stderr=err_fh)
    meta["return_code"] = res.returncode
    dump_yaml(run_dir / "meta.yaml", meta)
    status = "OK" if res.returncode == 0 else "FAIL"
    stamp = meta["timestamp"]
    rel = run_dir.relative_to(project_root)
    update_index_results(project_root, f"- **{status}** `{name}` ({stamp}) -> `{rel}/`")
    if res.returncode != 0:
        print(f"[FAIL] {name}: rc={res.returncode} (see {run_dir})", file=sys.stderr)
        if fail_fast:
            raise SystemExit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MCP-style actions with per-run outputs.")
    ap.add_argument("--spec", default=None, type=Path, help="Path to MCP YAML spec.")
    ap.add_argument(
        "--project-root",
        required=True,
        type=Path,
        help="Path to the code repository root.",
    )
    ap.add_argument("--action", default=None, help="Run a single action by name.")
    ap.add_argument("--conda-env", default="chemprop_cmpnn", help="Conda env name.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument("--fail-fast", action="store_true", help="Stop after first failure.")
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    if args.spec is not None:
        if args.spec.suffix.lower() in {".md", ".markdown"}:
            payload = load_yaml_from_index(args.spec)
        else:
            payload = load_yaml(args.spec)
    else:
        index_path = project_root / "index.md"
        if not index_path.exists():
            raise SystemExit("index.md not found; provide --spec.")
        payload = load_yaml_from_index(index_path)
    mcp = payload.get("mcp")
    if not isinstance(mcp, dict):
        raise SystemExit("Top-level 'mcp' mapping is required.")

    outputs_dir = mcp.get("outputs_dir")
    if not outputs_dir:
        raise SystemExit("'mcp.outputs_dir' is required.")
    outputs_dir = (project_root / outputs_dir).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    actions = mcp.get("actions", {})
    if not isinstance(actions, dict) or not actions:
        raise SystemExit("'mcp.actions' must be a non-empty mapping.")

    if args.action:
        action = actions.get(args.action)
        if action is None:
            raise SystemExit(f"Action '{args.action}' not found.")
        run_action(
            project_root,
            outputs_dir,
            args.action,
            action,
            args.conda_env,
            args.dry_run,
            args.fail_fast,
        )
        return

    for name, action in actions.items():
        run_action(
            project_root,
            outputs_dir,
            name,
            action,
            args.conda_env,
            args.dry_run,
            args.fail_fast,
        )


if __name__ == "__main__":
    main()
