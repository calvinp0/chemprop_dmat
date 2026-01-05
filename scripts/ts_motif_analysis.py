#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem

from chemprop import data

from ts_splits import SplitConfig, make_split_indices, reaction_center_signature


ROLE_ORDER = ["*0", "*1", "*2", "*3", "*4"]


def mol_from_smiles_keep_h(smiles: str) -> Chem.Mol | None:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            )
        except Exception:
            pass
    return mol


def load_ts_mol_from_sdf(sdf_path: Path, sanitize: bool = False) -> Chem.Mol:
    if not sdf_path.exists():
        candidates = []
        if not sdf_path.name.startswith("rmg_"):
            alt = sdf_path.with_name(f"rmg_{sdf_path.name}")
            if alt.exists():
                candidates.append(alt)
        if sdf_path.name.startswith("rmg_"):
            alt = sdf_path.with_name(sdf_path.name.replace("rmg_", "", 1))
            if alt.exists():
                candidates.append(alt)
        glob_hits = list(sdf_path.parent.glob(f"*{sdf_path.name}"))
        candidates.extend([p for p in glob_hits if p.exists()])
        if candidates:
            sdf_path = candidates[0]
        else:
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    suppl = Chem.SDMolSupplier(
        str(sdf_path), removeHs=False, sanitize=sanitize, strictParsing=False
    )
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"Could not load TS molecule from {sdf_path}")
    ts = None
    for m in mols:
        if m.HasProp("type") and m.GetProp("type").strip().lower() == "ts":
            ts = m
            break
    m = ts or mols[0]
    if sanitize:
        try:
            Chem.SanitizeMol(m)
        except Exception:
            try:
                Chem.SanitizeMol(
                    m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                )
            except Exception:
                pass
    return m


def sanitize_partial(mol: Chem.Mol) -> Chem.Mol:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            )
        except Exception:
            pass
    return mol


def load_ts_entries(ts_path: Path, sdf_root: Path):
    ts_molecules = [json.loads(line) for line in ts_path.read_text().splitlines() if line.strip()]
    data_rows = []
    skipped_missing = 0
    for mol in ts_molecules:
        name = mol["sdf_file"].replace(".sdf", "")
        sdf_path = sdf_root / mol["sdf_file"]
        try:
            rdkit_mol = load_ts_mol_from_sdf(sdf_path, sanitize=False)
        except FileNotFoundError:
            rdkit_mol = mol_from_smiles_keep_h(mol["smiles"])
            if rdkit_mol is None:
                skipped_missing += 1
                continue
        rdkit_mol = sanitize_partial(rdkit_mol)

        dmat = np.array(mol["flat_reaction_dmat"], dtype=float)
        mask = np.array(mol["flat_reaction_dmat_mask"], dtype=float)
        dmat[~mask.astype(bool)] = np.nan

        if not rdkit_mol.HasProp("_Name"):
            rdkit_mol.SetProp("_Name", name)
        if not rdkit_mol.HasProp("role_order"):
            rdkit_mol.SetProp("role_order", json.dumps(mol["role_order"]))
        if not rdkit_mol.HasProp("role_indices_ordered"):
            rdkit_mol.SetProp("role_indices_ordered", json.dumps(mol["role_indices_ordered"]))

        data_rows.append(
            {
                "name": name,
                "mol": rdkit_mol,
                "y": dmat,
            }
        )
    if skipped_missing:
        print(f"Skipped {skipped_missing} entries due to missing SDF/SMILES.")
    return data_rows


def motif_ids(
    mols: Iterable[Chem.Mol],
    add_adj_roles: bool,
    radius: int,
    nbits: int,
    use_chirality: bool,
) -> list[str]:
    role_keys = ROLE_ORDER if add_adj_roles else ["*1", "*2", "*3"]
    return [
        reaction_center_signature(
            mol, role_keys, radius=radius, nbits=nbits, use_chirality=use_chirality
        )
        for mol in mols
    ]


def train_val_test_indices_random(mols: list[Chem.Mol], split_sizes, seed: int):
    train_idx, val_idx, test_idx = data.make_split_indices(
        mols, "random", split_sizes, seed=seed
    )
    return train_idx[0], val_idx[0], test_idx[0]


def _flat_indices(idxs):
    if len(idxs) == 0:
        return []
    first = idxs[0]
    if isinstance(first, (int, np.integer)):
        return list(map(int, idxs))
    return list(map(int, idxs[0]))


def train_val_test_indices_reaction_center(
    mols: list[Chem.Mol], split_sizes, seed: int, args
):
    split_cfg = SplitConfig(
        splitter="reaction_center",
        split_sizes=split_sizes,
        seed=seed,
        group_kfolds=args.group_kfolds,
        group_test_fold=args.group_test_fold,
        split_radius=args.split_radius,
        split_nbits=args.split_nbits,
        split_use_chirality=not args.split_no_chirality,
        holdout_donor_element=None,
        holdout_acceptor_element=None,
    )
    return make_split_indices(mols, split_cfg, add_adj_roles=args.add_adj_roles)


def overlap_stats(motif_ids_list: list[str], train_idx, test_idx):
    train_motifs = {motif_ids_list[i] for i in train_idx}
    test_motifs = {motif_ids_list[i] for i in test_idx}
    overlap = train_motifs & test_motifs
    pct = 0.0 if not test_motifs else (len(overlap) / len(test_motifs)) * 100.0
    return pct, len(train_motifs), len(test_motifs), len(overlap)


def overlap_stats_samples(motif_ids_list: list[str], train_idx, test_idx):
    train_motifs = {motif_ids_list[i] for i in train_idx}
    test_motifs = [motif_ids_list[i] for i in test_idx]
    seen = sum(m in train_motifs for m in test_motifs)
    pct_samples = 0.0 if len(test_motifs) == 0 else 100.0 * seen / len(test_motifs)
    return pct_samples, seen, len(test_motifs)


def perf_by_motif_frequency(y_true: np.ndarray, y_pred: np.ndarray, train_counts, test_idx, bins):
    def mae_row(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0:
            return np.nan
        return np.abs(a[m] - b[m]).mean()

    test_mae = np.array([mae_row(y_true[i], y_pred[i]) for i in test_idx])
    test_counts = np.array([train_counts[i] for i in test_idx], dtype=int)

    bin_edges = sorted(bins)
    labels = ["unseen", "seen-rare", "seen-medium", "seen-common", "seen-very-common"]
    edges = [0] + bin_edges + [np.inf]

    bin_ids = []
    for c in test_counts:
        idx = 0
        while idx + 1 < len(edges) and c >= edges[idx + 1]:
            idx += 1
        bin_ids.append(idx)

    rows = []
    for i, label in enumerate(labels):
        mask = np.array(bin_ids) == i
        vals = test_mae[mask]
        rows.append(
            {
                "bin": label,
                "n": int(mask.sum()),
                "mean_mae": float(np.nanmean(vals)) if mask.any() else np.nan,
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser(description="Motif analysis for TS dataset.")
    p.add_argument("--ts-path", type=str, default="DATA/ts_molecules.ndjson")
    p.add_argument("--sdf-dir", type=str, default="DATA/SDF")
    p.add_argument("--out-dir", type=str, default="ts_motif_analysis")
    p.add_argument("--add-adj-roles", action="store_true")
    p.add_argument("--split-sizes", type=str, default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--group-kfolds", type=int, default=5)
    p.add_argument("--group-test-fold", type=int, default=0)
    p.add_argument("--split-radius", type=int, default=2)
    p.add_argument("--split-nbits", type=int, default=2048)
    p.add_argument("--split-no-chirality", action="store_true")
    p.add_argument("--min-motif-size", type=int, default=5)
    p.add_argument(
        "--preds",
        type=str,
        default=None,
        help="Optional path to predictions aligned to NDJSON order (.npy or .npz).",
    )
    p.add_argument(
        "--perf-split",
        choices=["random", "reaction_center"],
        default="reaction_center",
        help="Split to use for performance vs motif frequency.",
    )
    p.add_argument(
        "--splitter",
        choices=["random", "reaction_center"],
        default=None,
        help="Alias for --perf-split (kept for parity with ts_hpo).",
    )
    p.add_argument(
        "--freq-bins",
        type=str,
        default="1,3,11,51",
        help="Comma-separated bin edges for motif frequency.",
    )
    args = p.parse_args()

    args.split_sizes = tuple(float(x.strip()) for x in args.split_sizes.split(","))
    if len(args.split_sizes) != 3:
        raise ValueError("--split-sizes must have three comma-separated values")
    bins = [int(x.strip()) for x in args.freq_bins.split(",") if x.strip()]

    ts_path = Path(args.ts_path)
    sdf_root = Path(args.sdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_rows = load_ts_entries(ts_path, sdf_root)
    mols = [d["mol"] for d in data_rows]
    ys = np.vstack([d["y"] for d in data_rows]).astype(np.float64)

    motif_list = motif_ids(
        mols, args.add_adj_roles, args.split_radius, args.split_nbits, not args.split_no_chirality
    )
    motif_counts = pd.Series(motif_list).value_counts()
    motif_counts.to_csv(out_dir / "motif_counts.csv", header=["count"])

    total = len(motif_list)
    top_frac = 0.1
    top_k = max(int(np.ceil(len(motif_counts) * top_frac)), 1)
    top_coverage = motif_counts.iloc[:top_k].sum() / total if total else 0.0
    print(f"Motifs: {len(motif_counts)} total, {total} samples")
    print(f"Top {top_frac:.0%} motifs cover {top_coverage:.2%} of samples")

    try:
        import matplotlib.pyplot as plt

        counts = motif_counts.values
        plt.figure(figsize=(8, 4))
        plt.hist(counts, bins=30, color="#4C78A8", alpha=0.85)
        plt.xlabel("Motif size (count)")
        plt.ylabel("Number of motifs")
        plt.title("Motif size distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "motif_size_hist.png", dpi=200)
        plt.close()

        sorted_counts = np.sort(counts)[::-1]
        cum = np.cumsum(sorted_counts) / sorted_counts.sum()
        x = np.arange(1, len(sorted_counts) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(x, cum, color="#F58518")
        plt.xlabel("Top motifs (rank)")
        plt.ylabel("Cumulative coverage")
        plt.title("Motif coverage curve")
        plt.tight_layout()
        plt.savefig(out_dir / "motif_coverage_curve.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting skipped (matplotlib error): {e}")

    train_idx_r, _, test_idx_r = train_val_test_indices_random(mols, args.split_sizes, args.seed)
    train_idx_rc, _, test_idx_rc = train_val_test_indices_reaction_center(
        mols, args.split_sizes, args.seed, args
    )
    train_idx_rc = _flat_indices(train_idx_rc)
    test_idx_rc = _flat_indices(test_idx_rc)

    overlap_random = overlap_stats(motif_list, train_idx_r, test_idx_r)
    overlap_rc = overlap_stats(motif_list, train_idx_rc, test_idx_rc)
    overlap_samples_random = overlap_stats_samples(motif_list, train_idx_r, test_idx_r)
    overlap_samples_rc = overlap_stats_samples(motif_list, train_idx_rc, test_idx_rc)
    print(
        f"Overlap random: {overlap_random[0]:.2f}% "
        f"(train motifs={overlap_random[1]}, test motifs={overlap_random[2]}, overlap={overlap_random[3]})"
    )
    print(
        f"Overlap reaction_center: {overlap_rc[0]:.2f}% "
        f"(train motifs={overlap_rc[1]}, test motifs={overlap_rc[2]}, overlap={overlap_rc[3]})"
    )
    print(
        f"Sample overlap random: {overlap_samples_random[0]:.2f}% "
        f"(seen={overlap_samples_random[1]}, test_samples={overlap_samples_random[2]})"
    )
    print(
        f"Sample overlap reaction_center: {overlap_samples_rc[0]:.2f}% "
        f"(seen={overlap_samples_rc[1]}, test_samples={overlap_samples_rc[2]})"
    )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        labels = ["random", "reaction_center"]
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(
            x - width / 2,
            [overlap_random[0], overlap_rc[0]],
            width=width,
            label="unique motifs",
            color="#54A24B",
        )
        plt.bar(
            x + width / 2,
            [overlap_samples_random[0], overlap_samples_rc[0]],
            width=width,
            label="test samples",
            color="#E45756",
        )
        plt.xticks(x, labels)
        plt.ylabel("% test seen in train")
        plt.ylim(0, 100)
        plt.title("Train-test motif overlap")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "motif_overlap.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"Plotting skipped (matplotlib error): {e}")

    motif_to_indices = {}
    for idx, motif_id in enumerate(motif_list):
        motif_to_indices.setdefault(motif_id, []).append(idx)

    rows = []
    for motif_id, idxs in motif_to_indices.items():
        if len(idxs) < args.min_motif_size:
            continue
        y_block = ys[idxs]
        stds = np.nanstd(y_block, axis=0)
        mean_std = float(np.nanmean(stds))
        median_std = float(np.nanmedian(stds))
        mean_abs = np.nanmean(np.abs(y_block), axis=0)
        cv = stds / (mean_abs + 1e-6)
        mean_cv = float(np.nanmean(cv))
        mean_vec = np.nanmean(y_block, axis=0)
        cos_sims = []
        for row in y_block:
            mask = np.isfinite(row) & np.isfinite(mean_vec)
            if mask.sum() < 2:
                continue
            a = row[mask]
            b = mean_vec[mask]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                continue
            cos_sims.append(float(np.dot(a, b) / denom))
        rows.append(
            {
                "motif_id": motif_id,
                "count": len(idxs),
                "mean_std": mean_std,
                "median_std": median_std,
                "mean_cv": mean_cv,
                "mean_cos_sim": float(np.mean(cos_sims)) if cos_sims else np.nan,
            }
        )
    df_var = pd.DataFrame(rows)
    if df_var.empty:
        print(
            "No motifs with size >= "
            f"{args.min_motif_size}; skipping variance summary."
        )
    else:
        df_var = df_var.sort_values("count", ascending=False)
        df_var.to_csv(out_dir / "motif_variance.csv", index=False)
        print(
            "Intra-motif variance (motifs >= "
            f"{args.min_motif_size}): mean_std={df_var.mean_std.mean():.4f} "
            f"median_std={df_var.median_std.mean():.4f} "
            f"mean_cos_sim={df_var.mean_cos_sim.mean():.4f} "
            f"mean_cv={df_var.mean_cv.mean():.4f}"
        )

    if args.splitter is not None:
        args.perf_split = args.splitter

    summary_rows = []
    for split_name, train_idx, test_idx, overlap_motif, overlap_samples in [
        ("random", train_idx_r, test_idx_r, overlap_random, overlap_samples_random),
        ("reaction_center", train_idx_rc, test_idx_rc, overlap_rc, overlap_samples_rc),
    ]:
        train_motif_counts = pd.Series([motif_list[i] for i in train_idx]).value_counts()
        test_counts = [int(train_motif_counts.get(motif_list[i], 0)) for i in test_idx]
        median_cnt = float(np.median(test_counts)) if test_counts else 0.0
        summary_rows.append(
            {
                "split": split_name,
                "test_samples": len(test_idx),
                "pct_test_samples_seen_motif": overlap_samples[0],
                "pct_unique_test_motifs_seen": overlap_motif[0],
                "median_train_count_for_test": median_cnt,
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "motif_overlap_summary.csv", index=False)

    if args.preds:
        pred_path = Path(args.preds)
        if pred_path.suffix == ".npy":
            y_pred = np.load(pred_path)
            pred_indices = None
            pred_train_indices = None
        elif pred_path.suffix == ".npz":
            loaded = np.load(pred_path)
            y_pred = loaded["preds"] if "preds" in loaded else loaded["arr_0"]
            if "test_indices" in loaded:
                pred_indices = loaded["test_indices"].tolist()
            elif "indices" in loaded:
                pred_indices = loaded["indices"].tolist()
            else:
                pred_indices = None
            pred_train_indices = (
                loaded["train_indices"].tolist() if "train_indices" in loaded else None
            )
        else:
            raise ValueError("Preds must be .npy or .npz")

        if pred_indices is None and y_pred.shape != ys.shape:
            raise ValueError(
                f"Preds shape {y_pred.shape} does not match targets shape {ys.shape}"
            )

        if pred_indices is not None:
            test_idx = list(map(int, pred_indices))
            y_pred_full = np.full_like(ys, np.nan, dtype=float)
            for k, idx in enumerate(test_idx):
                y_pred_full[idx] = y_pred[k]
            y_pred = y_pred_full
            if pred_train_indices is not None:
                train_idx = list(map(int, pred_train_indices))
            else:
                train_idx = train_idx_rc if args.perf_split == "reaction_center" else train_idx_r
        else:
            if args.perf_split == "random":
                train_idx, _, test_idx = train_idx_r, None, test_idx_r
            else:
                train_idx, _, test_idx = train_idx_rc, None, test_idx_rc

        train_counts = {i: 0 for i in range(len(motif_list))}
        train_motif_counts = pd.Series([motif_list[i] for i in train_idx]).value_counts()
        for i, motif_id in enumerate(motif_list):
            train_counts[i] = int(train_motif_counts.get(motif_id, 0))

        rows_perf = perf_by_motif_frequency(ys, y_pred, train_counts, test_idx, bins)
        df_perf = pd.DataFrame(rows_perf)
        df_perf.to_csv(out_dir / "perf_by_motif_freq.csv", index=False)

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 4))
            plt.bar(df_perf["bin"], df_perf["mean_mae"], color="#B279A2")
            plt.ylabel("Mean MAE")
            plt.xlabel("Motif frequency in training")
            plt.title(f"Performance vs motif frequency ({args.perf_split})")
            plt.tight_layout()
            plt.savefig(out_dir / "perf_vs_motif_freq.png", dpi=200)
            plt.close()
        except Exception as e:
            print(f"Plotting skipped (matplotlib error): {e}")
    else:
        print("No --preds provided; skipping performance vs motif frequency.")


if __name__ == "__main__":
    main()
