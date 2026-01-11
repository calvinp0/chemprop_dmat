#!/usr/bin/env python
"""
Train a baseline TS distance-matrix model and record test-set predictions.

Uses the same preprocessing/model as scripts/ts_hpo.py, but with fixed defaults and a single run.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch

from chemprop import data, featurizers
from chemprop.featurizers import MoleculeFeaturizerRegistry
from chemprop.nn.transforms import ScaleTransform

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))
from ts_hpo import (  # noqa: E402
    SearchSpace,
    build_loaders,
    build_model,
    fit_scaler_ignore_nan,
    load_dataset,
    load_best_weights_into_model,
)
from ts_splits import SplitConfig, make_split_indices  # noqa: E402


@dataclass
class ModelConfig:
    hidden_dim: int = 512
    mp_depth: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    huber_beta: float = 0.2
    ffn_layers: int = 2
    weight_decay: float = 1e-4
    agg: str = "mean"
    batch_norm: bool = True


class DummyTrial:
    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, low, high, step=1):
        return self.params[name]

    def suggest_float(self, name, low, high, log=False):
        return self.params[name]

    def suggest_categorical(self, name, choices):
        return self.params[name]


def _array_to_jsonable(row: np.ndarray) -> list[float | None]:
    out = []
    for val in row.tolist():
        if val is None:
            out.append(None)
        elif isinstance(val, float) and np.isnan(val):
            out.append(None)
        else:
            out.append(float(val))
    return out


def _init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE,
            created_at TEXT,
            args_json TEXT,
            model_json TEXT,
            split_json TEXT,
            metrics_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS test_predictions (
            run_id TEXT,
            dataset_index INTEGER,
            rxn_name TEXT,
            sdf_file TEXT,
            sdf_path TEXT,
            y_true_json TEXT,
            y_mask_json TEXT,
            y_pred_json TEXT,
            PRIMARY KEY (run_id, dataset_index)
        )
        """
    )
    conn.commit()
    return conn


@torch.no_grad()
def predict_model(model, loader):
    model.eval()
    preds_all, y_all, mask_all = [], [], []
    device = next(model.parameters()).device
    for b in loader:
        if isinstance(b, (tuple, list)):
            bmg, V_d, X_d, y, *_ = b
        else:
            bmg, V_d, X_d, y = b.bmg, getattr(b, "V_d", None), getattr(b, "X_d", None), b.Y

        tmp = bmg.to(device)
        bmg = tmp if tmp is not None else bmg
        V_d = V_d.to(device) if V_d is not None else None
        X_d = X_d.to(device) if X_d is not None else None
        y = y.to(device)
        pred = model(bmg, V_d, X_d)
        mask = torch.isfinite(y)
        preds_all.append(pred.cpu())
        y_all.append(y.cpu())
        mask_all.append(mask.cpu())
    return torch.cat(preds_all, 0), torch.cat(y_all, 0), torch.cat(mask_all, 0)


def run_basic(args):
    ts_path = Path(args.ts_path)
    sdf_root = Path(args.sdf_dir)
    all_data, ys, x_d_dim = load_dataset(
        ts_path,
        sdf_root,
        strict_roles=args.strict_roles,
        add_adj_roles=args.add_adj_roles,
        molecule_featurizers=args.molecule_featurizers,
    )
    extra_atom_fdim = 5 if args.add_adj_roles else 3
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra_atom_fdim)

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
        split_indices = make_split_indices(
            [d.mol for d in all_data], split_cfg, add_adj_roles=args.add_adj_roles
        )
    else:
        split_indices = data.make_split_indices(
            [d.mol for d in all_data], "random", args.split_sizes, seed=args.seed
        )

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
        split_indices=split_indices,
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
    train_dset.Y = (y_train_raw - scaler.mean_) / scaler.scale_
    val_dset.Y = (y_val_raw - scaler.mean_) / scaler.scale_
    test_dset.Y = test_dset._Y

    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        mp_depth=args.mp_depth,
        dropout=args.dropout,
        lr=args.lr,
        huber_beta=args.huber_beta,
        ffn_layers=args.ffn_layers,
        weight_decay=args.weight_decay,
        agg=args.agg,
        batch_norm=not args.no_batch_norm,
    )
    trial = DummyTrial(
        {
            "hidden_dim": model_cfg.hidden_dim,
            "mp_depth": model_cfg.mp_depth,
            "dropout": model_cfg.dropout,
            "lr": model_cfg.lr,
            "huber_beta": model_cfg.huber_beta,
            "ffn_layers": model_cfg.ffn_layers,
            "weight_decay": model_cfg.weight_decay,
            "agg": model_cfg.agg,
        }
    )
    space = SearchSpace(batch_norm=model_cfg.batch_norm)
    model = build_model(
        featurizer,
        scaler,
        ys.shape[1],
        trial,
        space,
        x_d_dim=x_d_dim,
        task_weights=None,
        x_d_transform=x_d_transform,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_root = Path(args.ckpt_dir) if args.ckpt_dir else Path(".tmp_ckpts") / f"basic_{run_id}"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_root),
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

    test_metrics = trainer.test(model, test_loader, verbose=False)[0]
    print("Test metrics:", {k: float(v) for k, v in test_metrics.items()})

    preds, y_true, y_mask = predict_model(model, test_loader)
    preds_np = preds.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    y_mask_np = y_mask.cpu().numpy()

    db_path = Path(args.db)
    conn = _init_db(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO runs (run_id, created_at, args_json, model_json, split_json, metrics_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            run_id,
            datetime.now().isoformat(timespec="seconds"),
            json.dumps(vars(args), sort_keys=True),
            json.dumps(asdict(model_cfg), sort_keys=True),
            json.dumps(
                {
                    "splitter": args.splitter,
                    "split_sizes": args.split_sizes,
                    "seed": args.seed,
                    "group_kfolds": args.group_kfolds,
                    "group_test_fold": args.group_test_fold,
                    "split_radius": args.split_radius,
                    "split_nbits": args.split_nbits,
                    "split_no_chirality": args.split_no_chirality,
                    "holdout_donor_element": args.holdout_donor_element,
                    "holdout_acceptor_element": args.holdout_acceptor_element,
                },
                sort_keys=True,
            ),
            json.dumps({k: float(v) for k, v in test_metrics.items()}, sort_keys=True),
        ),
    )

    for row, dataset_idx in enumerate(test_idx):
        dp = all_data[dataset_idx]
        rxn_name = dp.name
        sdf_file = dp.mol.GetProp("sdf_file") if dp.mol.HasProp("sdf_file") else None
        sdf_path = str((sdf_root / sdf_file).resolve()) if sdf_file else None
        cur.execute(
            "INSERT OR REPLACE INTO test_predictions "
            "(run_id, dataset_index, rxn_name, sdf_file, sdf_path, y_true_json, y_mask_json, y_pred_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                int(dataset_idx),
                rxn_name,
                sdf_file,
                sdf_path,
                json.dumps(_array_to_jsonable(y_true_np[row])),
                json.dumps(_array_to_jsonable(y_mask_np[row].astype(float))),
                json.dumps(_array_to_jsonable(preds_np[row])),
            ),
        )

    conn.commit()
    conn.close()
    print(f"Wrote test predictions for {len(test_idx)} samples to {db_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train a baseline TS model and save test predictions.")
    p.add_argument("--ts-path", type=str, default="DATA/ts_molecules.ndjson", help="Path to ts_molecules NDJSON.")
    p.add_argument("--sdf-dir", type=str, default="DATA/SDF", help="Directory containing TS SDF files.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ckpt-dir", type=str, default=None, help="Optional checkpoint root.")
    p.add_argument(
        "--db",
        type=str,
        default="ts_predictions.sqlite",
        help="SQLite DB path to store test-set predictions.",
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
        "--molecule-featurizers",
        nargs="+",
        choices=sorted(MoleculeFeaturizerRegistry.keys()),
        default=None,
        help="Molecule-level featurizers to append as global descriptors.",
    )
    p.add_argument(
        "--no-descriptor-scaling",
        action="store_true",
        help="Disable scaling for extra molecule descriptors (X_d).",
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
    defaults = ModelConfig()
    p.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    p.add_argument("--mp-depth", type=int, default=defaults.mp_depth)
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--huber-beta", type=float, default=defaults.huber_beta)
    p.add_argument("--ffn-layers", type=int, default=defaults.ffn_layers)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--agg", type=str, choices=["mean", "sum"], default=defaults.agg)
    p.add_argument("--no-batch-norm", action="store_true", help="Disable batch norm in the MPNN.")
    args = p.parse_args()
    args.split_sizes = tuple(float(x.strip()) for x in args.split_sizes.split(","))
    if len(args.split_sizes) != 3:
        raise ValueError("--split-sizes must have three comma-separated values")
    return args


if __name__ == "__main__":
    run_basic(parse_args())
