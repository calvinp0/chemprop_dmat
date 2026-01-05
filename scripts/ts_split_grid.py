#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_base_cmd(args) -> list[str]:
    cmd = [
        "python",
        "scripts/ts_hpo.py",
        "--ts-path",
        args.ts_path,
        "--sdf-dir",
        args.sdf_dir,
        "--n-trials",
        str(args.n_trials),
        "--max-epochs",
        str(args.max_epochs),
        "--final-epochs",
        str(args.final_epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--ensemble-size",
        str(args.ensemble_size),
        "--ensemble-seed",
        str(args.ensemble_seed),
        "--hpo-splits",
        str(args.hpo_splits),
    ]
    if args.add_adj_roles:
        cmd.append("--add-adj-roles")
    if args.molecule_featurizers:
        cmd.append("--molecule-featurizers")
        cmd.extend(args.molecule_featurizers)
    if args.no_descriptor_scaling:
        cmd.append("--no-descriptor-scaling")
    if args.split_no_chirality:
        cmd.append("--split-no-chirality")
    if args.save_preds:
        cmd.extend(["--save-preds", "__PRED_PLACEHOLDER__"])
    if args.run_motif_analysis:
        cmd.append("--run-motif-analysis")
        cmd.extend(["--motif-out-dir", "__MOTIF_PLACEHOLDER__"])
    return cmd


def run_one(
    cmd: list[str],
    out_path: Path,
    ckpt_dir: Path,
    pred_path: Path | None,
    motif_out_dir: Path | None,
    dry_run: bool,
):
    cmd = cmd + ["--out", str(out_path), "--ckpt-dir", str(ckpt_dir)]
    if pred_path is not None:
        cmd = [str(pred_path) if x == "__PRED_PLACEHOLDER__" else x for x in cmd]
    if motif_out_dir is not None:
        cmd = [str(motif_out_dir) if x == "__MOTIF_PLACEHOLDER__" else x for x in cmd]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Run TS HPO across a split grid.")
    p.add_argument("--ts-path", type=str, default="examples/ts_molecules.ndjson")
    p.add_argument("--sdf-dir", type=str, default="DATA/SDF")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--final-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--ensemble-seed", type=int, default=1000)
    p.add_argument("--hpo-splits", type=int, default=3)
    p.add_argument("--add-adj-roles", action="store_true")
    p.add_argument("--molecule-featurizers", nargs="+", default=None)
    p.add_argument("--no-descriptor-scaling", action="store_true")
    p.add_argument("--split-no-chirality", action="store_true")
    p.add_argument(
        "--save-preds",
        action="store_true",
        help="Save test preds/indices for motif analysis (writes one .npz per split).",
    )
    p.add_argument(
        "--run-motif-analysis",
        action="store_true",
        help="Run motif analysis after each HPO run (writes one output dir per split).",
    )
    p.add_argument("--out-dir", type=str, default="hpo_grid")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--donor-holdouts",
        type=str,
        default="",
        help="Comma-separated donor elements to hold out (e.g. O,N,S).",
    )
    p.add_argument(
        "--acceptor-holdouts",
        type=str,
        default="",
        help="Comma-separated acceptor elements to hold out (e.g. O,N,S).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = build_base_cmd(args)

    jobs: list[tuple[str, list[str]]] = [
        ("random", ["--splitter", "random"]),
        ("rc_r2_core", ["--splitter", "reaction_center", "--split-radius", "2"]),
        ("rc_r2_adj", ["--splitter", "reaction_center", "--split-radius", "2", "--add-adj-roles"]),
        ("rc_r3_adj", ["--splitter", "reaction_center", "--split-radius", "3", "--add-adj-roles"]),
        ("donor_acceptor_pair", ["--splitter", "donor_acceptor_pair", "--add-adj-roles"]),
    ]

    for donor in [d.strip() for d in args.donor_holdouts.split(",") if d.strip()]:
        jobs.append(
            (
                f"donor_holdout_{donor}",
                ["--splitter", "donor_element", "--holdout-donor-element", donor, "--add-adj-roles"],
            )
        )

    for acceptor in [a.strip() for a in args.acceptor_holdouts.split(",") if a.strip()]:
        jobs.append(
            (
                f"acceptor_holdout_{acceptor}",
                ["--splitter", "acceptor_element", "--holdout-acceptor-element", acceptor, "--add-adj-roles"],
            )
        )

    for name, split_args in jobs:
        out_path = out_dir / f"{name}.json"
        ckpt_dir = out_dir / f"ckpt_{name}"
        pred_path = out_dir / f"{name}_preds.npz" if args.save_preds else None
        motif_out_dir = out_dir / f"motif_{name}" if args.run_motif_analysis else None
        run_one(
            base_cmd + split_args,
            out_path,
            ckpt_dir,
            pred_path,
            motif_out_dir,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
