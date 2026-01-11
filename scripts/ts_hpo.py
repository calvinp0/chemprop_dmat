#!/usr/bin/env python
"""
Hyperparameter search for the TS distance-matrix model.

This script mirrors the preprocessing done in examples/quick_dmat_predict.ipynb:
* loads ts_molecules.ndjson
* reads SDFs to preserve explicit hydrogens and atom order (does not add new Hs)
* builds role-specific atom flags (donor, moving H, acceptor) as V_f
* scales targets ignoring NaNs
* trains an MPNN with a Huber loss and runs Optuna search over a small set of knobs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import optuna
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem
import torch

from chemprop import data, featurizers, models, nn
from chemprop.featurizers import MoleculeFeaturizerRegistry
from chemprop.nn.transforms import ScaleTransform

sys.path.append(str(Path(__file__).resolve().parent))
from ts_splits import SplitConfig, make_split_indices  # noqa: E402


ROLE_ORDER = ["*0", "*1", "*2", "*3", "*4"]
BASE_ROLE_COLS = {"donor": 0, "moving_h": 1, "acceptor": 2}


def get_role_cols(add_adj_roles: bool) -> dict[str, int]:
    role_cols = dict(BASE_ROLE_COLS)
    if add_adj_roles:
        role_cols.update(
            {
                "donor_adjacent": len(role_cols),
                "acceptor_adjacent": len(role_cols) + 1,
            }
        )
    return role_cols


def role_atom_features_from_mol(
    mol: Chem.Mol,
    role_order: list[str],
    role_indices: list[int | None],
    atom_types: dict[str, str] | dict[str, dict[str, str]] | None,
    counts: dict[str, dict[str, int]],
    errors: list[str],
    add_adj_roles: bool = False,
) -> np.ndarray:
    """One-hot flags per atom for donor / moving H / acceptor (expects explicit Hs already present).

    `counts` is mutated to track missing/out-of-range/type-mismatch issues per role.
    """
    n_atoms = mol.GetNumAtoms()
    role_cols = get_role_cols(add_adj_roles)
    feats = np.zeros((n_atoms, len(role_cols)), dtype=np.float32)

    role_map = dict(zip(role_order, role_indices))
    targets = {
        "donor": role_map.get("*1"),
        "moving_h": role_map.get("*2"),
        "acceptor": role_map.get("*3"),
    }
    if add_adj_roles:
        targets.update(
            {
                "donor_adjacent": role_map.get("*0"),
                "acceptor_adjacent": role_map.get("*4"),
            }
        )

    def atom_type_to_symbol(atom_type: str | None) -> str | None:
        if not atom_type:
            return None
        # strip trailing digits/valence markers, keep leading element symbol (handle Cl/Br explicitly)
        s = atom_type
        if s.startswith(("Cl", "Br")):
            return s[:2].capitalize()
        return s[0].upper()

    def expected_symbol(idx: int) -> str | None:
        if not atom_types:
            return None
        # role_atom_types is a mapping idx -> atom_type str; mol_properties stores atom_type in nested dict
        if isinstance(next(iter(atom_types.values())) if atom_types else "", dict):
            atom_type = atom_types.get(str(idx), {}).get("atom_type")
        else:
            atom_type = atom_types.get(str(idx))
        return atom_type_to_symbol(atom_type)

    for role, idx in targets.items():
        if idx is None:
            counts["missing"][role] += 1
            continue
        if idx >= n_atoms:
            counts["out_of_range"][role] += 1
            errors.append(
                f"out_of_range: role={role}, idx={idx}, n_atoms={n_atoms}, name={mol.GetProp('_Name') if mol.HasProp('_Name') else 'unknown'}"
            )
            continue
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        expected_sym = expected_symbol(idx)
        if expected_sym is not None and sym != expected_sym:
            counts["type_mismatch"][role] += 1  # soft flag only; do not skip/raise
            errors.append(
                f"symbol_mismatch: role={role}, idx={idx}, rdkit_sym={sym}, expected={expected_sym}"
            )
        feats[idx, role_cols[role]] = 1.0

    return feats


def mol_from_smiles_keep_h(smiles: str) -> Chem.Mol | None:
    """Parse SMILES keeping explicit Hs, with relaxed sanitization to avoid valence complaints."""
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
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass
    return mol


def load_ts_mol_from_sdf(sdf_path: Path, sanitize: bool = False) -> Chem.Mol:
    """Load a TS mol from SDF, preferring unsanitized read to preserve order/explicit Hs."""
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

    # Prefer unsanitized read to preserve atom order/explicit Hs; sanitize only if requested.
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=sanitize, strictParsing=False)
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
                Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                pass
    return m


def sanitize_partial(mol: Chem.Mol) -> Chem.Mol:
    """Run a relaxed sanitize to set aromaticity/conjugation while tolerating TS valence quirks."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass
    return mol


def fit_scaler_ignore_nan(y: np.ndarray) -> Any:
    """Fit mean/std per task ignoring NaNs; returns a sklearn-like scaler stub."""
    y = np.asarray(y, float)
    n_tasks = y.shape[1]

    class _Scaler:
        pass

    scaler = _Scaler()
    scaler.mean_ = np.zeros(n_tasks, float)
    scaler.scale_ = np.ones(n_tasks, float)
    for t in range(n_tasks):
        finite = np.isfinite(y[:, t])
        vals = y[finite, t]
        if vals.size:
            scaler.mean_[t] = vals.mean()
            std = vals.std(ddof=0)
            scaler.scale_[t] = std if std != 0 else 1.0
    return scaler


def load_dataset(
    ts_path: Path,
    sdf_root: Path,
    strict_roles: bool = False,
    max_logged_errors: int = 5,
    add_adj_roles: bool = False,
    molecule_featurizers: list[str] | None = None,
) -> tuple[list[data.MoleculeDatapoint], np.ndarray, int]:
    ts_molecules = [json.loads(line) for line in ts_path.read_text().splitlines() if line.strip()]

    mol_featurizers = None
    if molecule_featurizers:
        mol_featurizers = [MoleculeFeaturizerRegistry[mf]() for mf in molecule_featurizers]

    role_keys = list(get_role_cols(add_adj_roles).keys())
    counts = {
        "missing": {r: 0 for r in role_keys},
        "out_of_range": {r: 0 for r in role_keys},
        "type_mismatch": {r: 0 for r in role_keys},
    }
    errors: list[str] = []
    skipped_missing_sdf = 0
    skipped_role_errors = 0
    skipped_out_of_range = 0
    logged_errors = 0
    data_ts = []
    for mol in ts_molecules:
        name = mol["sdf_file"].replace(".sdf", "")
        sdf_path = sdf_root / mol["sdf_file"]
        # Prefer SDF to preserve atom order; fallback to SMILES if SDF missing.
        try:
            rdkit_mol = load_ts_mol_from_sdf(sdf_path, sanitize=False)
        except FileNotFoundError as e:
            try:
                rdkit_mol = mol_from_smiles_keep_h(mol["smiles"])
            except Exception:
                rdkit_mol = None
            if rdkit_mol is None:
                print(f"Skipping {name}: {e}")
                skipped_missing_sdf += 1
                continue

        rdkit_mol = sanitize_partial(rdkit_mol)
        n_atoms = rdkit_mol.GetNumAtoms()
        max_idx = max((idx for idx in mol["role_indices_ordered"] if idx is not None), default=-1)
        if max_idx >= n_atoms:
            skipped_out_of_range += 1
            if logged_errors < max_logged_errors:
                logged_errors += 1
                print(
                    f"Skipping {name}: role index {max_idx} out of range for mol with {n_atoms} atoms "
                    f"(indices={mol['role_indices_ordered']})"
                )
            continue
        dmat = np.array(mol["flat_reaction_dmat"], dtype=float)
        mask = np.array(mol["flat_reaction_dmat_mask"], dtype=float)
        dmat[~mask.astype(bool)] = np.nan

        if not rdkit_mol.HasProp("_Name"):
            rdkit_mol.SetProp("_Name", name)
        if not rdkit_mol.HasProp("sdf_file"):
            rdkit_mol.SetProp("sdf_file", mol["sdf_file"])
        if not rdkit_mol.HasProp("role_order"):
            rdkit_mol.SetProp("role_order", json.dumps(mol["role_order"]))
        if not rdkit_mol.HasProp("role_indices_ordered"):
            rdkit_mol.SetProp("role_indices_ordered", json.dumps(mol["role_indices_ordered"]))
        err_before = len(errors)
        atom_types = mol.get("role_atom_types") or mol.get("mol_properties")
        role_feat = role_atom_features_from_mol(
            rdkit_mol,
            mol["role_order"],
            mol["role_indices_ordered"],
            atom_types,
            counts,
            errors,
            add_adj_roles=add_adj_roles,
        )
        if len(errors) > err_before and strict_roles:
            skipped_role_errors += 1
            if logged_errors < max_logged_errors:
                logged_errors += 1
                print(f"Skipping {name} due to role label issues: {errors[err_before]}")
            continue

        x_d = None
        if mol_featurizers is not None:
            feats = [mf(rdkit_mol) for mf in mol_featurizers]
            x_d = np.hstack(feats).astype(np.float32, copy=False)

        data_ts.append(
            {
                "rxn_name": name,
                "mol": rdkit_mol,
                "reaction_dmat": dmat,
                "mask": mask,
                "role_feat": role_feat,
                "x_d": x_d,
            }
        )

    rdkit_mols = [d["mol"] for d in data_ts]
    names = [d["rxn_name"] for d in data_ts]
    ys = np.vstack([d["reaction_dmat"] for d in data_ts]).astype(np.float64)
    role_feats = [d["role_feat"] for d in data_ts]
    x_ds = [d["x_d"] for d in data_ts]

    all_data = [
        data.MoleculeDatapoint(mol, name=nme, y=y, V_f=vf, x_d=xd)
        for mol, nme, y, vf, xd in zip(rdkit_mols, names, ys, role_feats, x_ds)
    ]

    # Simple summary of role-label health to catch silent corruption.
    total = len(ts_molecules)
    used = len(data_ts)
    print(
        f"Role label checks over {used} usable entries (from {total}, skipped_missing_sdf={skipped_missing_sdf}, "
        f"skipped_out_of_range={skipped_out_of_range}, skipped_role_errors={skipped_role_errors}):"
    )
    base_roles = set(BASE_ROLE_COLS.keys())
    for category, roles in counts.items():
        for role, cnt in roles.items():
            if cnt == 0:
                continue
            if category == "missing" and role not in base_roles:
                continue
            if category != "type_mismatch" and category != "missing":
                continue
            denom = used if used else total
            print(f"  {category} {role}: {cnt} ({cnt/denom:.3%})")
    if errors and strict_roles:
        first = errors[0]
        raise ValueError(f"Role label issues detected ({len(errors)} total). First: {first}")
    x_d_dim = 0
    for xd in x_ds:
        if xd is not None:
            x_d_dim = xd.shape[0]
            break
    return all_data, ys, x_d_dim


def load_best_weights_into_model(model: pl.LightningModule, ckpt_path: str) -> pl.LightningModule:
    """Load best checkpoint weights into an existing model instance (weights_only=False, trusted)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    model_keys = set(model.state_dict().keys())
    if not model_keys.issuperset(state.keys()):
        for prefix in ("model.", "mpnn."):
            if all(k.startswith(prefix) for k in state.keys()):
                state = {k[len(prefix) :]: v for k, v in state.items()}
                break

    model.load_state_dict(state, strict=False)
    return model


def build_loaders(
    all_data: list[data.MoleculeDatapoint],
    featurizer,
    batch_size: int,
    seed: int,
    shuffle_train_targets: bool = False,
    shuffle_seed: int = 0,
    split_indices: tuple[list[int], list[int], list[int]] | None = None,
):
    mols = [d.mol for d in all_data]
    pl.seed_everything(seed, workers=True)
    if split_indices is None:
        train_idx, val_idx, test_idx = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
    else:
        train_idx, val_idx, test_idx = split_indices

    def _flat(idxs):
        if len(idxs) == 0:
            return []
        first = idxs[0]
        if isinstance(first, (int, np.integer)):
            return list(map(int, idxs))
        return list(map(int, idxs[0]))

    train_idx = _flat(train_idx)
    val_idx = _flat(val_idx)
    test_idx = _flat(test_idx)

    if shuffle_train_targets:
        rng = np.random.default_rng(shuffle_seed + seed)
        y_values = [all_data[i].y for i in train_idx]
        if any(y is not None for y in y_values):
            perm = rng.permutation(train_idx)
            y_train_src = [all_data[i].y for i in perm]
            for k, idx in enumerate(train_idx):
                src_y = y_train_src[k]
                all_data[idx].y = None if src_y is None else np.array(src_y, copy=True)
            print(f"Shuffled training targets (global) with seed={shuffle_seed + seed}.")

            before = np.stack([np.array(y, copy=False) for y in y_values])
            after = np.stack([np.array(all_data[i].y, copy=False) for i in train_idx])

            same_rows = np.isclose(before, after, equal_nan=True).all(axis=1).mean()

            def row_corr(a, b):
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 2:
                    return np.nan
                aa, bb = a[m], b[m]
                if aa.std() == 0 or bb.std() == 0:
                    return np.nan
                return np.corrcoef(aa, bb)[0, 1]

            corrs = [row_corr(before[i], after[i]) for i in range(len(before))]
            corrs = np.array([c for c in corrs if np.isfinite(c)])

            print(f"[shuffle-debug] identical rows fraction: {same_rows:.3f}")
            if corrs.size:
                print(
                    f"[shuffle-debug] mean row corr: {corrs.mean():.3f}  median: {np.median(corrs):.3f}"
                )
            else:
                print("[shuffle-debug] no finite corrs (too many NaNs per row)")

            print("[shuffle-debug] first 3 names / first 5 y entries (after shuffle):")
            for dp in [all_data[i] for i in train_idx[:3]]:
                print(dp.name, dp.y[:5])
        else:
            print("Requested training target shuffling, but targets are missing; skipping shuffle.")

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, [train_idx], [val_idx], [test_idx]
    )

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    test_dset = data.MoleculeDataset(test_data[0], featurizer)
    return (
        train_dset,
        val_dset,
        test_dset,
        data.build_dataloader(train_dset, shuffle=True, batch_size=batch_size),
        data.build_dataloader(val_dset, shuffle=False, batch_size=batch_size),
        data.build_dataloader(test_dset, shuffle=False, batch_size=batch_size),
        train_idx,
        val_idx,
        test_idx,
    )


@dataclass
class SearchSpace:
    hidden_min: int = 256
    hidden_max: int = 768
    depth_min: int = 2
    depth_max: int = 6
    dropout_min: float = 0.0
    dropout_max: float = 0.4
    lr_min: float = 1e-4
    lr_max: float = 5e-3
    beta_min: float = 0.05
    beta_max: float = 1.5
    batch_norm: bool = True
    wd_min: float = 1e-6
    wd_max: float = 1e-2


def build_model(
    featurizer,
    scaler,
    y_dim: int,
    trial: optuna.trial.Trial,
    space: SearchSpace,
    x_d_dim: int = 0,
    task_weights=None,
    x_d_transform=None,
):
    hidden = trial.suggest_int("hidden_dim", space.hidden_min, space.hidden_max, step=64)
    depth = trial.suggest_int("mp_depth", space.depth_min, space.depth_max)
    dropout = trial.suggest_float("dropout", space.dropout_min, space.dropout_max)
    lr = trial.suggest_float("lr", space.lr_min, space.lr_max, log=True)
    beta = trial.suggest_float("huber_beta", space.beta_min, space.beta_max)
    n_layers = trial.suggest_int("ffn_layers", 1, 4)
    weight_decay = trial.suggest_float("weight_decay", space.wd_min, space.wd_max, log=True)

    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_h=hidden, depth=depth, dropout=dropout)
    agg_choice = trial.suggest_categorical("agg", ["mean", "sum"])
    agg = nn.MeanAggregation() if agg_choice == "mean" else nn.SumAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    huber = nn.metrics.Huber(beta=beta, task_weights=task_weights if task_weights is not None else 1.0)
    ffn_input = hidden + x_d_dim
    ffn = nn.RegressionFFN(
        n_tasks=y_dim,
        input_dim=ffn_input,  # match message-passing output (+ global feats if present)
        hidden_dim=hidden,
        n_layers=n_layers,
        dropout=dropout,
        criterion=huber,
        output_transform=output_transform,
    )
    metrics = [nn.metrics.Huber(beta=beta), nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]

    model = models.MPNN(
        mp,
        agg,
        ffn,
        batch_norm=space.batch_norm,
        metrics=metrics,
        init_lr=lr,
        max_lr=lr,
        final_lr=lr,
        X_d_transform=x_d_transform,
        weight_decay=weight_decay,
    )
    return model


def run_search(args):
    ts_path = Path(args.ts_path)
    sdf_root = Path(args.sdf_dir)
    all_data, ys, x_d_dim = load_dataset(
        ts_path,
        sdf_root,
        strict_roles=args.strict_roles,
        add_adj_roles=args.add_adj_roles,
        molecule_featurizers=args.molecule_featurizers,
    )
    from collections import Counter
    names = [dp.name for dp in all_data]
    name_counts = Counter(names)
    dups = sum(v > 1 for v in name_counts.values())
    print(
        f"[dup-debug] duplicated names: {dups} / {len(name_counts)} unique / {len(names)} total"
    )

    def y_hash(y):
        yy = np.array(y, dtype=np.float64)
        zz = np.where(np.isfinite(yy), yy, 1e20)
        return hashlib.md5(zz.tobytes()).hexdigest()

    yh = [y_hash(dp.y) for dp in all_data]
    y_counts = Counter(yh)
    print("[dup-debug] top y-hash counts:", y_counts.most_common(5))
    extra_atom_fdim = 5 if args.add_adj_roles else 3
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra_atom_fdim)
    n_tasks = ys.shape[1]
    manual_task_weights = None
    if args.task_weights:
        tw = [float(x.strip()) for x in args.task_weights.split(",") if x.strip()]
        if len(tw) != n_tasks:
            raise ValueError(f"--task-weights length {len(tw)} != n_tasks {n_tasks}")
        manual_task_weights = np.asarray(tw, dtype=np.float32)
        manual_task_weights = manual_task_weights / manual_task_weights.mean()

    scheme_names = ["uniform"]
    if args.weighted_huber:
        scheme_names.extend(["inv_freq", "inv_std", "inv_freq_inv_std"])
    if manual_task_weights is not None:
        scheme_names.append("manual")
    default_scheme = scheme_names[0]

    space = SearchSpace()

    split_indices_fixed = None
    if args.splitter != "random":
        split_cfg = SplitConfig(
            splitter=args.splitter,
            split_sizes=args.split_sizes,
            seed=args.seed,
            group_kfolds=args.group_kfolds,
            group_test_fold=args.group_test_fold,
            split_radius=args.split_radius,
            split_nbits=args.split_nbits,
            split_use_chirality=not args.split_no_chirality,
            holdout_donor_element=args.holdout_donor_element,
            holdout_acceptor_element=args.holdout_acceptor_element,
        )
        split_indices_fixed = make_split_indices(
            [d.mol for d in all_data], split_cfg, add_adj_roles=args.add_adj_roles
        )
    else:
        split_indices_fixed = data.make_split_indices(
            [d.mol for d in all_data], "random", args.split_sizes, seed=args.seed
        )

    def prepare_split(seed: int):
        (
            train_dset,
            val_dset,
            test_dset,
            train_loader,
            val_loader,
            test_loader,
            train_idx,
            val_idx,
            test_idx,
        ) = build_loaders(
            all_data,
            featurizer,
            args.batch_size,
            seed,
            shuffle_train_targets=args.shuffle_train_targets,
            shuffle_seed=args.shuffle_seed,
            split_indices=split_indices_fixed,
        )

        x_d_transform = None
        if train_dset.d_xd > 0 and not args.no_descriptor_scaling:
            scaler_xd = train_dset.normalize_inputs("X_d")
            val_dset.normalize_inputs("X_d", scaler_xd)
            scaler_xd = scaler_xd if not isinstance(scaler_xd, list) else scaler_xd[0]
            if scaler_xd is not None:
                x_d_transform = ScaleTransform.from_standard_scaler(scaler_xd)

        y_train_raw = train_dset._Y
        y_val_raw = val_dset._Y
        scaler = fit_scaler_ignore_nan(y_train_raw)

        counts = np.isfinite(y_train_raw).sum(axis=0).astype(float)
        counts[counts == 0] = 1.0
        inv = counts.max() / counts
        invfreq_task_weights = (inv / inv.mean()).astype(np.float32)

        stds = np.zeros(n_tasks, dtype=np.float32)
        eps = 1e-6
        for t in range(n_tasks):
            vals = y_train_raw[np.isfinite(y_train_raw[:, t]), t]
            std = vals.std(ddof=0) if vals.size else 1.0
            if std == 0:
                std = 1.0
            stds[t] = std
        invstd_task_weights = (1.0 / (stds + eps)).astype(np.float32)
        invstd_task_weights = invstd_task_weights / invstd_task_weights.mean()

        invfreq_invstd = invfreq_task_weights * invstd_task_weights
        invfreq_invstd = invfreq_invstd / invfreq_invstd.mean()

        weight_schemes: dict[str, np.ndarray | None] = {"uniform": None}
        if args.weighted_huber:
            weight_schemes["inv_freq"] = invfreq_task_weights
            weight_schemes["inv_std"] = invstd_task_weights
            weight_schemes["inv_freq_inv_std"] = invfreq_invstd
        if manual_task_weights is not None:
            weight_schemes["manual"] = manual_task_weights

        train_dset.Y = (y_train_raw - scaler.mean_) / scaler.scale_
        val_dset.Y = (y_val_raw - scaler.mean_) / scaler.scale_
        test_dset.Y = test_dset._Y
        return train_loader, val_loader, test_loader, scaler, weight_schemes, x_d_transform

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_root = Path(args.ckpt_dir) if args.ckpt_dir else Path(".tmp_ckpts") / f"run_{run_id}"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial):
        scheme = (
            trial.suggest_categorical("task_weight_scheme", scheme_names)
            if len(scheme_names) > 1
            else default_scheme
        )
        split_losses = []
        for split_idx in range(args.hpo_splits):
            split_seed = args.seed + split_idx
            pl.seed_everything(split_seed, workers=True)
            train_loader, val_loader, _, scaler, weight_schemes, x_d_transform = prepare_split(split_seed)
            model = build_model(
                featurizer,
                scaler,
                ys.shape[1],
                trial,
                space,
                x_d_dim,
                task_weights=weight_schemes[scheme],
                x_d_transform=x_d_transform,
            )
            ckpt_cb = pl.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_root / f"optuna_trial_{trial.number}_seed_{split_seed}"),
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            )
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=True,
                max_epochs=args.max_epochs,
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience), ckpt_cb],
            )
            trainer.fit(model, train_loader, val_loader)
            if ckpt_cb.best_model_path:
                model = load_best_weights_into_model(model, ckpt_cb.best_model_path)
            split_losses.append(trainer.callback_metrics["val_loss"].item())
        return float(np.mean(split_losses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best_params = study.best_params
    best_scheme = best_params.get("task_weight_scheme", default_scheme)
    print(f"Best trial: val_loss={study.best_value:.4f}")
    print("Best params:", {**best_params, "task_weight_scheme": best_scheme})

    # Retrain best model and evaluate on test set
    class DummyTrial:
        def __init__(self, params):
            self.params = params
            self._user_attrs = {}

        def suggest_int(self, name, low, high, step=1):
            return self.params[name]

        def suggest_float(self, name, low, high, log=False):
            return self.params[name]

        def suggest_categorical(self, name, choices):
            return self.params[name]

        def set_user_attr(self, key, value):
            self._user_attrs[key] = value

        @property
        def user_attrs(self):
            return self._user_attrs

    dummy = DummyTrial(best_params)
    (
        train_dset,
        val_dset,
        test_dset,
        train_loader,
        val_loader,
        test_loader,
        train_idx,
        val_idx,
        test_idx,
    ) = build_loaders(
        all_data,
        featurizer,
        args.batch_size,
        args.seed,
        shuffle_train_targets=args.shuffle_train_targets,
        shuffle_seed=args.shuffle_seed,
        split_indices=split_indices_fixed,
    )

    x_d_transform = None
    if train_dset.d_xd > 0 and not args.no_descriptor_scaling:
        scaler_xd = train_dset.normalize_inputs("X_d")
        val_dset.normalize_inputs("X_d", scaler_xd)
        scaler_xd = scaler_xd if not isinstance(scaler_xd, list) else scaler_xd[0]
        if scaler_xd is not None:
            x_d_transform = ScaleTransform.from_standard_scaler(scaler_xd)

    y_train_raw = train_dset._Y
    y_val_raw = val_dset._Y
    scaler = fit_scaler_ignore_nan(y_train_raw)

    counts = np.isfinite(y_train_raw).sum(axis=0).astype(float)
    counts[counts == 0] = 1.0
    inv = counts.max() / counts
    invfreq_task_weights = (inv / inv.mean()).astype(np.float32)

    stds = np.zeros(n_tasks, dtype=np.float32)
    eps = 1e-6
    for t in range(n_tasks):
        vals = y_train_raw[np.isfinite(y_train_raw[:, t]), t]
        std = vals.std(ddof=0) if vals.size else 1.0
        if std == 0:
            std = 1.0
        stds[t] = std
    invstd_task_weights = (1.0 / (stds + eps)).astype(np.float32)
    invstd_task_weights = invstd_task_weights / invstd_task_weights.mean()

    invfreq_invstd = invfreq_task_weights * invstd_task_weights
    invfreq_invstd = invfreq_invstd / invfreq_invstd.mean()

    weight_schemes: dict[str, np.ndarray | None] = {"uniform": None}
    if args.weighted_huber:
        weight_schemes["inv_freq"] = invfreq_task_weights
        weight_schemes["inv_std"] = invstd_task_weights
        weight_schemes["inv_freq_inv_std"] = invfreq_invstd
    if manual_task_weights is not None:
        weight_schemes["manual"] = manual_task_weights
    train_dset.Y = (y_train_raw - scaler.mean_) / scaler.scale_
    val_dset.Y = (y_val_raw - scaler.mean_) / scaler.scale_
    test_dset.Y = test_dset._Y

    best_model = build_model(
        featurizer,
        scaler,
        ys.shape[1],
        dummy,
        space,
        x_d_dim,
        task_weights=weight_schemes[best_scheme],
        x_d_transform=x_d_transform,
    )
    best_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_root / "final"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    final_trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        max_epochs=args.final_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience), best_ckpt],
    )
    final_trainer.fit(best_model, train_loader, val_loader)
    if best_ckpt.best_model_path:
        best_model = load_best_weights_into_model(best_model, best_ckpt.best_model_path)
    test_metrics = final_trainer.test(best_model, test_loader, verbose=False)[0]
    print("Test metrics:", {k: float(v) for k, v in test_metrics.items()})

    # Ensemble training with fixed split, varying seeds
    def train_one(model, seed: int):
        pl.seed_everything(seed, workers=True)
        ckpt = pl.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_root / f"ensemble_seed_{seed}"),
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,
            max_epochs=args.final_epochs,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=args.patience), ckpt],
        )
        trainer.fit(model, train_loader, val_loader)
        if ckpt.best_model_path:
            model = load_best_weights_into_model(model, ckpt.best_model_path)
        return model

    @torch.no_grad()
    def predict_model(model, loader):
        model.eval()
        preds_all, y_all, mask_all = [], [], []
        device = next(model.parameters()).device
        for b in loader:
            # support both tuple batches and TrainingBatch-like objects
            if isinstance(b, (tuple, list)):
                bmg, V_d, X_d, y, *_ = b
            else:
                bmg, V_d, X_d, y = b.bmg, getattr(b, "V_d", None), getattr(b, "X_d", None), b.Y

            tmp = bmg.to(device)
            bmg = tmp if tmp is not None else bmg  # BatchMolGraph.to may be in-place and return None
            V_d = V_d.to(device) if V_d is not None else None
            X_d = X_d.to(device) if X_d is not None else None
            y = y.to(device)
            pred = model(bmg, V_d, X_d)
            mask = torch.isfinite(y)
            preds_all.append(pred.cpu())
            y_all.append(y.cpu())
            mask_all.append(mask.cpu())
        return torch.cat(preds_all, 0), torch.cat(y_all, 0), torch.cat(mask_all, 0)

    @torch.no_grad()
    def ensemble_predict(models, loader):
        pred_list = []
        for m in models:
            p, y, mask = predict_model(m, loader)
            pred_list.append(p)
        preds = torch.stack(pred_list, 0)  # [K, N, T]
        mean = preds.mean(0)
        std = preds.std(0)
        return mean, std, y, mask

    if args.save_preds:
        preds, y_true, y_mask = predict_model(best_model, test_loader)
        out_path = Path(args.save_preds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            preds=preds.cpu().numpy(),
            train_indices=np.array(train_idx, dtype=int),
            val_indices=np.array(val_idx, dtype=int),
            test_indices=np.array(test_idx, dtype=int),
        )
        print(f"Wrote test preds to {out_path}")

    if args.run_motif_analysis:
        cmd = [
            "python",
            "scripts/ts_motif_analysis.py",
            "--ts-path",
            str(args.ts_path),
            "--sdf-dir",
            str(args.sdf_dir),
            "--seed",
            str(args.seed),
            "--group-kfolds",
            str(args.group_kfolds),
            "--group-test-fold",
            str(args.group_test_fold),
            "--split-radius",
            str(args.split_radius),
            "--split-nbits",
            str(args.split_nbits),
        ]
        if args.add_adj_roles:
            cmd.append("--add-adj-roles")
        if args.split_no_chirality:
            cmd.append("--split-no-chirality")
        if args.motif_out_dir:
            cmd.extend(["--out-dir", str(args.motif_out_dir)])
        if args.save_preds:
            cmd.extend(["--preds", str(args.save_preds)])
        if args.splitter in ("random", "reaction_center"):
            cmd.extend(["--perf-split", args.splitter])
        subprocess.run(cmd, check=True)

    ensemble_models = []
    for k in range(args.ensemble_size):
        member_seed = args.ensemble_seed + k
        dt = DummyTrial(best_params)
        m = build_model(
            featurizer,
            scaler,
            ys.shape[1],
            dt,
            space,
            x_d_dim,
            task_weights=weight_schemes[best_scheme],
            x_d_transform=x_d_transform,
        )
        m_trained = train_one(m, member_seed)
        ensemble_models.append(m_trained)

    ensemble_mean, ensemble_std, y_true, y_mask = ensemble_predict(ensemble_models, test_loader)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # per-sample mean MAE
        per_sample_mae = (
            (ensemble_mean - torch.nan_to_num(y_true, nan=0.0))
            .abs()
            .masked_fill(~y_mask, 0.0)
            .sum(1)
            .div(y_mask.sum(1).clamp_min(1))
        )
        # per-sample risk: mean std over valid dimensions
        std_masked = ensemble_std.masked_fill(~y_mask, 0.0)
        per_sample_risk = std_masked.sum(1).div(y_mask.sum(1).clamp_min(1))
        payload = {
            "best_params": {**best_params, "task_weight_scheme": best_scheme},
            "best_val_loss": study.best_value,
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "ensemble": {
                "n": args.ensemble_size,
                "mean_per_sample_mae": float(per_sample_mae.mean().item()),
                "mean_uncertainty": float(ensemble_std[y_mask].mean().item()),
                "mean_risk": float(per_sample_risk.mean().item()),
            },
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote results to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Optuna HPO for TS distance-matrix model.")
    p.add_argument("--ts-path", type=str, default="DATA/ts_molecules.ndjson", help="Path to ts_molecules NDJSON.")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None, help="Global Optuna timeout (seconds).")
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--final-epochs", type=int, default=80, help="Epochs for final training of best params.")
    p.add_argument(
        "--hpo-splits",
        type=int,
        default=3,
        help="Number of random splits (via seed offsets) to average per Optuna trial.",
    )
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default=None, help="Optional JSON output path for best params/metrics.")
    p.add_argument("--ensemble-size", type=int, default=5, help="Number of ensemble members.")
    p.add_argument("--ensemble-seed", type=int, default=1000, help="Base seed for ensemble members.")
    p.add_argument("--sdf-dir", type=str, default="DATA/SDF", help="Directory containing TS SDF files (for atom order).")
    p.add_argument(
        "--shuffle-train-targets",
        action="store_true",
        help="Randomly permute training targets after split indices are formed.",
    )
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Random seed used when shuffling training targets (only used with --shuffle-train-targets).",
    )
    p.add_argument(
        "--strict-roles",
        action="store_true",
        help="If set, abort on any role label mismatch; otherwise skip bad entries after logging.",
    )
    p.add_argument(
        "--add-adj-roles",
        action="store_true",
        help="Include *0/*4 roles as donor/acceptor adjacent atom flags.",
    )
    p.add_argument(
        "--splitter",
        type=str,
        default="random",
        choices=["random", "reaction_center", "donor_element", "acceptor_element", "donor_acceptor_pair"],
        help="Data split strategy.",
    )
    p.add_argument(
        "--split-sizes",
        type=str,
        default="0.8,0.1,0.1",
        help="Comma-separated train,val,test split sizes (must sum to 1).",
    )
    p.add_argument(
        "--group-kfolds",
        type=int,
        default=5,
        help="Number of folds for GroupKFold-based splits.",
    )
    p.add_argument(
        "--group-test-fold",
        type=int,
        default=0,
        help="Which fold to use as the test fold for GroupKFold-based splits.",
    )
    p.add_argument(
        "--split-radius",
        type=int,
        default=2,
        help="Morgan radius for reaction-center split signatures.",
    )
    p.add_argument(
        "--split-nbits",
        type=int,
        default=2048,
        help="Morgan fingerprint size for reaction-center split signatures.",
    )
    p.add_argument(
        "--split-no-chirality",
        action="store_true",
        help="Disable chirality in reaction-center split signatures.",
    )
    p.add_argument(
        "--holdout-donor-element",
        type=str,
        default=None,
        help="If set with --splitter donor_element, hold out this donor element in test.",
    )
    p.add_argument(
        "--holdout-acceptor-element",
        type=str,
        default=None,
        help="If set with --splitter acceptor_element, hold out this acceptor element in test.",
    )
    p.add_argument(
        "--molecule-featurizers",
        nargs="+",
        choices=sorted(MoleculeFeaturizerRegistry.keys()),
        default=None,
        help="Molecule-level featurizers to append as global descriptors (e.g. morgan_binary rdkit_2d).",
    )
    p.add_argument(
        "--no-descriptor-scaling",
        action="store_true",
        help="Disable scaling for extra molecule descriptors (X_d).",
    )
    p.add_argument(
        "--weighted-huber",
        action="store_true",
        help="Add inv_freq / inv_std / inv_freq_inv_std task-weight schemes to the sweep.",
    )
    p.add_argument(
        "--task-weights",
        type=str,
        default=None,
        help="Comma-separated list of 10 weights for the 10 distance targets (added as a sweep option).",
    )
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Optional checkpoint root; defaults to a unique run directory under .tmp_ckpts.",
    )
    p.add_argument(
        "--save-preds",
        type=str,
        default=None,
        help="Optional .npz path to save test preds and indices for motif analysis.",
    )
    p.add_argument(
        "--run-motif-analysis",
        action="store_true",
        help="Run motif analysis script after training (uses saved preds if provided).",
    )
    p.add_argument(
        "--motif-out-dir",
        type=str,
        default=None,
        help="Optional output directory for motif analysis results.",
    )
    args = p.parse_args()
    args.split_sizes = tuple(float(x.strip()) for x in args.split_sizes.split(","))
    if len(args.split_sizes) != 3:
        raise ValueError("--split-sizes must have three comma-separated values")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_search(args)
