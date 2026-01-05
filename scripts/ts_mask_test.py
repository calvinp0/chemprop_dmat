#!/usr/bin/env python
"""
Quick sanity check that loss/metrics are mask-aware for TS distance targets.
Forces an entire task column to NaN and checks loss/gradients remain finite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lightning import pytorch as pl
import torch

from chemprop import data, featurizers, models, nn
from scripts.ts_hpo import fit_scaler_ignore_nan, load_dataset, build_loaders, SearchSpace


def build_simple_model(featurizer, scaler, y_dim: int, x_d_dim: int):
    # Minimal fixed model for the masking test.
    space = SearchSpace()
    hidden = 256
    depth = 3
    dropout = 0.0
    lr = 1e-3
    beta = 0.5
    n_layers = 2

    mp = nn.BondMessagePassing(d_v=featurizer.atom_fdim, d_h=hidden, depth=depth, dropout=dropout)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    huber = nn.metrics.Huber(beta=beta, task_weights=1.0)
    ffn_input = hidden + x_d_dim
    ffn = nn.RegressionFFN(
        n_tasks=y_dim,
        input_dim=ffn_input,
        hidden_dim=hidden,
        n_layers=n_layers,
        dropout=dropout,
        criterion=huber,
        output_transform=output_transform,
    )
    metrics = [nn.metrics.Huber(beta=beta), nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]
    return models.MPNN(
        mp,
        agg,
        ffn,
        batch_norm=space.batch_norm,
        metrics=metrics,
        init_lr=lr,
        max_lr=lr,
        final_lr=lr,
        weight_decay=0.0,
    )


def grads_finite(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def main():
    p = argparse.ArgumentParser(description="Check mask-aware loss/metrics for TS targets.")
    p.add_argument("--ts-path", type=str, default="examples/ts_molecules.ndjson")
    p.add_argument("--sdf-dir", type=str, default="DATA/SDF")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--use-global-feats", action="store_true")
    p.add_argument("--nan-task", type=int, default=0, help="Task column to force to NaN.")
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)
    all_data, ys, x_d_dim = load_dataset(
        Path(args.ts_path), Path(args.sdf_dir), strict_roles=False, use_global_feats=args.use_global_feats
    )
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=3)
    train_dset, _, _, train_loader, _, _ = build_loaders(all_data, featurizer, args.batch_size, args.seed)

    y_train_raw = train_dset._Y
    scaler = fit_scaler_ignore_nan(y_train_raw)
    train_dset.Y = (y_train_raw - scaler.mean_) / scaler.scale_

    model = build_simple_model(featurizer, scaler, ys.shape[1], x_d_dim)
    model.train()

    batch = next(iter(train_loader))
    bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

    # Baseline loss/grad
    model.zero_grad(set_to_none=True)
    loss_base = model.training_step(batch, 0)
    loss_base.backward()
    base_ok = torch.isfinite(loss_base).item() and grads_finite(model)

    # Force one task column to NaN.
    targets_nan = targets.clone()
    col = args.nan_task
    if col < 0 or col >= targets_nan.shape[1]:
        raise ValueError(f"--nan-task {col} out of range for n_tasks={targets_nan.shape[1]}")
    targets_nan[:, col] = float("nan")
    batch_nan = (bmg, V_d, X_d, targets_nan, weights, lt_mask, gt_mask)

    model.zero_grad(set_to_none=True)
    loss_nan = model.training_step(batch_nan, 0)
    loss_nan.backward()
    nan_ok = torch.isfinite(loss_nan).item() and grads_finite(model)

    print(f"base_loss={float(loss_base):.6f}, finite_grads={base_ok}")
    print(f"nan_loss={float(loss_nan):.6f}, finite_grads={nan_ok}")
    if not nan_ok:
        raise SystemExit("Masking test failed: loss or gradients are not finite after NaN injection.")


if __name__ == "__main__":
    main()
