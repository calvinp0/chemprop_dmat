"""
Minimal Lightning training script for transition-state geometry + mode prediction.

Single-component TS setup (one TS molecule per SDF) with TransitionStateEncoder +
TSLightningModel. Builds a dataset from `examples/ts_molecules.ndjson` and runs a
Lightning Trainer end-to-end using default Chemprop components.

Defaults drop conformers (no coord leakage). Pass `--keep_coords` to append xyz as
extra atom features instead.

Example:
    conda run -n cmpnn_rocm python examples/run_ts_training.py --epochs 3 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from rdkit import Chem
from torch import Tensor

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import MeanAggregation
from chemprop.nn.message_passing import CommunicativeMessagePassing, MessagePassing
from chemprop.models import TransitionStateEncoder
from chemprop.models.ts_lightning_model import TSLightningModel


def load_entries(ndjson_path: Path, limit: int | None) -> list[dict]:
    entries: list[dict] = []
    with ndjson_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))
            if limit is not None and limit > 0 and len(entries) >= limit:
                break
    return entries


def load_ts_mode(ts_dir: Path, sdf_name: str) -> np.ndarray | None:
    stem = Path(sdf_name).stem
    npy = ts_dir / f"{stem}.npy"
    txt = ts_dir / f"{stem}.txt"
    if npy.exists():
        arr = np.load(npy)
    elif txt.exists():
        arr = np.loadtxt(txt)
    else:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"TS mode for {sdf_name} has shape {arr.shape}, expected (N, 3)")
    return arr.astype(np.float32)


def load_ts_molecule(sdf_path: Path, drop_conformers: bool) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"No molecules found in {sdf_path}")
    ts_mols = [m for m in mols if m.HasProp("type") and m.GetProp("type").lower() == "ts"]
    if not ts_mols:
        raise ValueError(f"No TS molecule (type=='ts') found in {sdf_path}")
    mol = ts_mols[0]
    if drop_conformers:
        mol = Chem.Mol(mol)
        for cid in [conf.GetId() for conf in mol.GetConformers()]:
            mol.RemoveConformer(cid)
    return mol


def pad_modes(modes: List[Tensor]) -> tuple[Tensor, Tensor]:
    counts = [m.shape[0] for m in modes]
    max_atoms = max(counts)
    B = len(modes)
    padded = torch.zeros((B, max_atoms, 3), dtype=torch.float32)
    mask = torch.zeros((B, max_atoms), dtype=torch.bool)
    for i, m in enumerate(modes):
        n = m.shape[0]
        padded[i, :n] = m
        mask[i, :n] = True
    return padded, mask


def make_batch(graphs: list, coords_all: list[np.ndarray], geom_targets: list[Tensor], mode_targets: list[Tensor], idxs: list[int]):
    batch_graphs = [graphs[i] for i in idxs]
    bmg = BatchMolGraph(batch_graphs)
    geom_target = torch.stack([geom_targets[i] for i in idxs], dim=0)
    mode_target, mode_mask = pad_modes([mode_targets[i] for i in idxs])
    max_atoms = mode_target.shape[1]
    coords_list = []
    edge_indices = []
    edge_masks = []
    for i in idxs:
        n = graphs[i].V.shape[0]
        coords_src = coords_all[i]
        coords = torch.zeros((max_atoms, 3), dtype=torch.float32)
        src_n = min(n, coords_src.shape[0])
        coords[:src_n] = torch.tensor(coords_src[:src_n], dtype=torch.float32)
        coords_list.append(coords)
        ei = torch.tensor(graphs[i].edge_index, dtype=torch.long)
        edge_indices.append(ei)
        edge_mask = torch.zeros(ei.shape[1], dtype=torch.bool)
        edge_mask[:] = True
        edge_masks.append(edge_mask)
    coords = torch.stack(coords_list, dim=0)
    # pad edge indices to max_edges
    max_edges = max(ei.shape[1] for ei in edge_indices) if edge_indices else 0
    edge_index_b = torch.zeros((len(idxs), max_edges, 2), dtype=torch.long)
    edge_mask_b = torch.zeros((len(idxs), max_edges), dtype=torch.bool)
    for b, ei in enumerate(edge_indices):
        e = ei.shape[1]
        edge_index_b[b, :e, :] = ei.t()
        edge_mask_b[b, :e] = True
    return {
        "bmg": bmg,
        "V_d": None,
        "geom_target": geom_target,
        "mode_target": mode_target,
        "mode_mask": mode_mask,
        "coords": coords,
        "edge_index": edge_index_b,
        "edge_mask": edge_mask_b,
    }


def random_modes_like(mode_target: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Generate random unit vectors per atom respecting the mask."""
    rand = torch.randn_like(mode_target)
    rand = rand / rand.norm(dim=-1, keepdim=True).clamp_min(eps)
    rand = rand * mask.unsqueeze(-1)
    return rand


def cosine_alignment(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Returns 1D tensor of |cos θ| for valid atoms."""
    pred_unit = pred / pred.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_unit = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
    cosine = (pred_unit * target_unit).sum(dim=-1).abs()
    return cosine[mask.bool()]


def evaluate_random_baseline(dataloader, mode_loss_fn):
    """Evaluates a random direction baseline for TS modes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_loss_mode = []
    all_cos_vals = []

    for batch in dataloader:
        mode_target = batch["mode_target"].to(device)
        mode_mask = batch["mode_mask"].to(device)

        rand_pred = random_modes_like(mode_target, mode_mask)
        loss_mode = mode_loss_fn(rand_pred, mode_target, mode_mask)
        all_loss_mode.append(loss_mode.item())

        cos_vals = cosine_alignment(rand_pred, mode_target, mode_mask)
        all_cos_vals.append(cos_vals.cpu())

    loss_mode_mean = float(np.mean(all_loss_mode))
    cos_all = torch.cat(all_cos_vals, dim=0)
    mean_abs_cos = cos_all.mean().item()

    print(f"Random baseline mode_loss: {loss_mode_mean:.4f}")
    print(f"Random baseline mean |cos θ|: {mean_abs_cos:.4f}")


def main(args: argparse.Namespace) -> None:
    ndjson_path = Path(args.ndjson)
    sdf_dir = Path(args.sdf_dir)
    ts_dir = Path(args.ts_mode_dir)

    entries = load_entries(ndjson_path, args.limit)

    featurizer = SimpleMoleculeMolGraphFeaturizer(
        extra_atom_fdim=3 if (args.use_ts_coords and not args.drop_coords) else 0
    )

    graphs: list = []
    coords_all: list[np.ndarray] = []
    geom_targets_raw: list[Tensor] = []
    mode_targets: list[Tensor] = []

    skipped = 0
    for entry in entries:
        sdf_file = entry["sdf_file"]
        mol = load_ts_molecule(sdf_dir / sdf_file, drop_conformers=args.drop_coords)

        atom_extra = None
        coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
        if args.use_ts_coords and not args.drop_coords and mol.GetNumConformers() > 0:
            coords = np.array(
                [
                    [
                        mol.GetConformer().GetAtomPosition(i).x,
                        mol.GetConformer().GetAtomPosition(i).y,
                        mol.GetConformer().GetAtomPosition(i).z,
                    ]
                    for i in range(mol.GetNumAtoms())
                ],
                dtype=np.float32,
            )
            atom_extra = coords
        mg = featurizer(mol, atom_features_extra=atom_extra)
        graphs.append(mg)
        coords_all.append(coords)

        geom_targets_raw.append(torch.tensor(entry["flat_reaction_dmat"], dtype=torch.float32))

        ts_mode = load_ts_mode(ts_dir, sdf_file)
        if ts_mode is None:
            skipped += 1
            graphs.pop()
            geom_targets_raw.pop()
            continue
        if ts_mode.shape[0] != mol.GetNumAtoms():
            raise ValueError(f"TS mode atoms ({ts_mode.shape[0]}) != total atoms ({mol.GetNumAtoms()}) for {sdf_file}")
        mode_targets.append(torch.from_numpy(ts_mode))

    if len(mode_targets) == 0:
        print("No datapoints found with available TS modes; exiting.")
        return
    print(f"Loaded {len(mode_targets)} datapoints; skipped {skipped} missing TS modes.")

    # Standardize geometry targets after split; keep raw copies.
    geom_targets_raw_tensor = torch.stack(geom_targets_raw)
    geom_mean = geom_targets_raw_tensor.mean(dim=0)
    geom_std = geom_targets_raw_tensor.std(dim=0).clamp_min(1e-8)
    geom_targets = [(g - geom_mean) / geom_std for g in geom_targets_raw]

    hidden_dim = args.hidden_dim
    if args.encoder.lower() == "cmpnn":
        mp = CommunicativeMessagePassing(d_v=featurizer.atom_fdim, d_h=hidden_dim)
    elif args.encoder.lower() == "dmpnn":
        from chemprop.nn.message_passing.base import BondMessagePassing as DMPNN

        mp = DMPNN(d_v=featurizer.atom_fdim, d_h=hidden_dim)
    else:
        raise ValueError(f"Unknown encoder '{args.encoder}'. Expected one of: cmpnn, dmpnn.")
    agg = MeanAggregation()
    encoder = TransitionStateEncoder(mp, agg, batch_norm=args.batch_norm)

    geom_dim = geom_targets_raw[0].shape[0]
    model_hidden = mp.output_dim
    model = TSLightningModel(
        encoder=encoder,
        hidden_dim=model_hidden,
        n_geom_features=geom_dim,
        geom_hidden_dim=args.geom_hidden_dim or hidden_dim,
        geom_n_layers=args.geom_n_layers,
        mode_hidden_dim=args.mode_hidden_dim,
        mode_n_layers=args.mode_n_layers,
        equivariant_mode_head=args.equivariant_mode_head,
        lambda_geom=args.lambda_geom,
        lambda_mode=args.lambda_mode,
        use_scheduler=args.use_scheduler,
        warmup_epochs=args.warmup_epochs,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
        lr=args.lr,
    )

    class TSDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(graphs)

        def __getitem__(self, idx: int):
            return graphs[idx], coords_all[idx], geom_targets[idx], mode_targets[idx]

    def collate(samples):
        g, c, gt, mt = zip(*samples)
        batch = make_batch(list(g), list(c), list(gt), list(mt), list(range(len(samples))))
        return batch

    # Build dataset/splits/loaders
    dataset = TSDataset()
    n_total = len(dataset)
    n_val = max(1, int(args.val_frac * n_total)) if args.val_frac > 0 else 0
    n_test = max(1, int(args.test_frac * n_total)) if args.test_frac > 0 else 0
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough data after splitting; reduce val/test fractions.")

    splits = [n_train]
    if n_val > 0:
        splits.append(n_val)
    if n_test > 0:
        splits.append(n_test)

    subsets = torch.utils.data.random_split(
        dataset, splits, generator=torch.Generator().manual_seed(args.split_seed)
    )
    if n_val > 0 and n_test > 0:
        train_set, val_set, test_set = subsets
    elif n_val > 0:
        train_set, val_set = subsets
        test_set = None
    elif n_test > 0:
        train_set, test_set = subsets
        val_set = None
    else:
        train_set = subsets[0]
        val_set = test_set = None

    def make_loader(split, shuffle):
        return torch.utils.data.DataLoader(
            split,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=args.num_workers,
        )

    train_loader = make_loader(train_set, shuffle=True)
    val_loader = make_loader(val_set, shuffle=False) if val_set is not None else None
    test_loader = make_loader(test_set, shuffle=False) if test_set is not None else None

    accelerator = "gpu" if args.use_cuda and torch.cuda.is_available() else "cpu"
    logger = CSVLogger(args.log_dir, name="ts_training", flush_logs_every_n_steps=10)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        logger=logger,
    )
    trainer.fit(model, train_loader, val_loader)
    if test_loader is not None:
        trainer.test(model, test_loader)

    # Optional random baseline on test if available
    if test_loader is not None and args.eval_random_baseline:
        print("Evaluating random baseline on test split...")
        evaluate_random_baseline(
            test_loader,
            lambda pred, target, mask: model._mode_loss(pred, target, mask),
        )

    # Optional plotting
    if args.plot_metrics:
        metrics_path = Path(logger.log_dir) / "metrics.csv"
        if not metrics_path.exists():
            print(f"No metrics.csv found at {metrics_path}")
        else:
            try:
                import csv
                import math
                import matplotlib.pyplot as plt

                def add_point(rows, key):
                    by_epoch = {}
                    for row in rows:
                        val = row.get(key, "")
                        if val in ("", None):
                            continue
                        try:
                            v = float(val)
                        except ValueError:
                            continue
                        if math.isfinite(v):
                            e = int(row["epoch"])
                            by_epoch[e] = v  # keep last value per epoch
                    if not by_epoch:
                        return [], []
                    xs = sorted(by_epoch.keys())
                    ys = [by_epoch[e] for e in xs]
                    return xs, ys

                with metrics_path.open() as f:
                    rows = list(csv.DictReader(f))

                series = {
                    "train_loss_epoch": add_point(rows, "train_loss_epoch"),
                    "train_loss_mode_epoch": add_point(rows, "train_loss_mode_epoch"),
                    "train_loss_geom_epoch": add_point(rows, "train_loss_geom_epoch"),
                    "val_loss": add_point(rows, "val_loss"),
                    "val_loss_mode": add_point(rows, "val_loss_mode"),
                    "val_loss_geom": add_point(rows, "val_loss_geom"),
                }

                plt.figure()
                plotted = False
                for label, (xs, ys) in series.items():
                    if ys:
                        plt.plot(xs, ys, label=label)
                        plotted = True
                if not plotted:
                    print("No finite loss values found in metrics.csv to plot.")
                else:
                    plt.xlabel("epoch")
                    plt.ylabel("loss")
                    plt.legend()
                    plt.title("TS training losses")
                    out_path = Path(logger.log_dir) / "loss_curve.png"
                    plt.savefig(out_path)
                    print(f"Saved loss plot to {out_path}")
            except ImportError:
                print("matplotlib not installed; skipping plot.")

    # end collate

    # (dataset/splits/trainers defined below in main scope)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndjson", type=str, default="examples/ts_molecules.ndjson")
    parser.add_argument("--sdf_dir", type=str, default="DATA/SDF")
    parser.add_argument("--ts_mode_dir", type=str, default="DATA/TSModes")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of datapoints to load (omit or <=0 to load all)",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Message passing hidden size")
    parser.add_argument(
        "--encoder",
        type=str,
        default="cmpnn",
        choices=["cmpnn", "dmpnn"],
        help="Message passing encoder to use.",
    )
    parser.add_argument("--geom_hidden_dim", type=int, default=None, help="Geom head hidden dim (defaults to hidden_dim)")
    parser.add_argument("--geom_n_layers", type=int, default=1, help="Geom head hidden layers")
    parser.add_argument("--mode_hidden_dim", type=int, default=64, help="TS mode head hidden dim")
    parser.add_argument("--mode_n_layers", type=int, default=2, help="TS mode head hidden layers")
    parser.add_argument("--lambda_geom", type=float, default=1.0, help="Weight for geometry loss")
    parser.add_argument("--lambda_mode", type=float, default=0.5, help="Weight for mode loss")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=13)
    parser.add_argument(
        "--drop_coords",
        action="store_true",
        help="Drop 3D coordinates from SDF; default keeps coords off unless --use_ts_coords is set.",
    )
    parser.add_argument(
        "--use_ts_coords",
        action="store_true",
        help="Append TS SDF coordinates as extra atom features and provide coords to the equivariant head.",
    )
    parser.add_argument(
        "--batch_norm",
        action="store_true",
        help="Enable batch norm in encoder (can fail with very small batches).",
    )
    parser.add_argument("--use_scheduler", action="store_true", help="Use Noam-like LR scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--final_lr", type=float, default=1e-4)
    parser.add_argument(
        "--equivariant_mode_head",
        action="store_true",
        help="Use EGNN-based TS mode head (needs coords/edge_index in batch).",
    )
    parser.add_argument(
        "--eval_random_baseline",
        action="store_true",
        help="Also evaluate a random direction baseline on the test split.",
    )
    parser.add_argument(
        "--plot_metrics",
        action="store_true",
        help="Plot train/val loss curves (requires matplotlib).",
    )
    args = parser.parse_args()

    main(args)
