"""
Tiny overfit test for TS mode head. Trains on a small subset to check if the model
can drive TS mode loss down (high |cos θ|) when focusing on the mode task.

Usage example:
    python examples/ts_mode_overfit_test.py --limit 32 --epochs 200 --lambda_mode 1.0 --lambda_geom 0.0 --hidden_dim 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from lightning import pytorch as pl
from rdkit import Chem
from torch import Tensor

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import MeanAggregation
from chemprop.nn.message_passing import CommunicativeMessagePassing
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


def make_batch(graphs: list, geom_targets: list[Tensor], mode_targets: list[Tensor], idxs: list[int]):
    batch_graphs = [graphs[i] for i in idxs]
    bmg = BatchMolGraph(batch_graphs)
    geom_target = torch.stack([geom_targets[i] for i in idxs], dim=0)
    mode_target, mode_mask = pad_modes([mode_targets[i] for i in idxs])
    return {
        "bmg": bmg,
        "V_d": None,
        "geom_target": geom_target,
        "mode_target": mode_target,
        "mode_mask": mode_mask,
    }


def cosine_alignment(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    pred_unit = pred / pred.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_unit = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
    cosine = (pred_unit * target_unit).sum(dim=-1).abs()
    return cosine[mask.bool()]


def main(args: argparse.Namespace) -> None:
    ndjson_path = Path(args.ndjson)
    sdf_dir = Path(args.sdf_dir)
    ts_dir = Path(args.ts_mode_dir)

    entries = load_entries(ndjson_path, args.limit)
    featurizer = SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=0 if args.drop_coords else 3)

    graphs: list = []
    geom_targets: list[Tensor] = []
    mode_targets: list[Tensor] = []
    skipped = 0

    for entry in entries:
        sdf_file = entry["sdf_file"]
        mol = load_ts_molecule(sdf_dir / sdf_file, drop_conformers=args.drop_coords)

        atom_extra = None
        if not args.drop_coords and mol.GetNumConformers() > 0:
            coords = np.array([list(mol.GetConformer().GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
            atom_extra = coords
        graphs.append(featurizer(mol, atom_features_extra=atom_extra))

        geom_targets.append(torch.tensor(entry["flat_reaction_dmat"], dtype=torch.float32))

        ts_mode = load_ts_mode(ts_dir, sdf_file)
        if ts_mode is None:
            skipped += 1
            graphs.pop()
            geom_targets.pop()
            continue
        if ts_mode.shape[0] != mol.GetNumAtoms():
            raise ValueError(f"TS mode atoms ({ts_mode.shape[0]}) != total atoms ({mol.GetNumAtoms()}) for {sdf_file}")
        mode_targets.append(torch.from_numpy(ts_mode))

    if len(mode_targets) == 0:
        print("No datapoints found with available TS modes; exiting.")
        return
    print(f"Loaded {len(mode_targets)} datapoints; skipped {skipped} missing TS modes.")

    # Standardize geometry targets (not critical when lambda_geom=0)
    geom_targets_tensor = torch.stack(geom_targets)
    geom_mean = geom_targets_tensor.mean(dim=0)
    geom_std = geom_targets_tensor.std(dim=0).clamp_min(1e-8)
    geom_targets = [(g - geom_mean) / geom_std for g in geom_targets]

    hidden_dim = args.hidden_dim
    mp = CommunicativeMessagePassing(d_v=featurizer.atom_fdim, d_h=hidden_dim)
    agg = MeanAggregation()
    encoder = TransitionStateEncoder(mp, agg, batch_norm=args.batch_norm)

    geom_dim = geom_targets[0].shape[0]
    model_hidden = mp.output_dim
    model = TSLightningModel(
        encoder=encoder,
        hidden_dim=model_hidden,
        n_geom_features=geom_dim,
        geom_hidden_dim=args.geom_hidden_dim or hidden_dim,
        geom_n_layers=args.geom_n_layers,
        mode_hidden_dim=args.mode_hidden_dim,
        mode_n_layers=args.mode_n_layers,
        lambda_geom=args.lambda_geom,
        lambda_mode=args.lambda_mode,
        use_scheduler=False,
        lr=args.lr,
    )

    class TSDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(graphs)

        def __getitem__(self, idx: int):
            return graphs[idx], geom_targets[idx], mode_targets[idx]

    def collate(samples):
        g, gt, mt = zip(*samples)
        batch = make_batch(list(g), list(gt), list(mt), list(range(len(samples))))
        return batch

    dataset = TSDataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )

    accelerator = "gpu" if args.use_cuda and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(model, loader)

    # Evaluate |cos θ| on the training set to confirm overfitting
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model.eval().to(device)
    cos_vals = []
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch["bmg"], "to"):
                maybe = batch["bmg"].to(device)
                batch["bmg"] = maybe if maybe is not None else batch["bmg"]
            batch["geom_target"] = batch["geom_target"].to(device)
            batch["mode_target"] = batch["mode_target"].to(device)
            batch["mode_mask"] = batch["mode_mask"].to(device)
            _, mode_pred, atom_mask = model(batch)
            cos_vals.append(cosine_alignment(mode_pred, batch["mode_target"], atom_mask).cpu())
    if cos_vals:
        mean_abs_cos = torch.cat(cos_vals).mean().item()
        print(f"Train overfit mean |cos θ|: {mean_abs_cos:.4f}")


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
    parser.add_argument("--geom_hidden_dim", type=int, default=None, help="Geom head hidden dim (defaults to hidden_dim)")
    parser.add_argument("--geom_n_layers", type=int, default=1, help="Geom head hidden layers")
    parser.add_argument("--mode_hidden_dim", type=int, default=64, help="TS mode head hidden dim")
    parser.add_argument("--mode_n_layers", type=int, default=2, help="TS mode head hidden layers")
    parser.add_argument("--lambda_geom", type=float, default=0.0, help="Weight for geometry loss")
    parser.add_argument("--lambda_mode", type=float, default=1.0, help="Weight for mode loss")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument(
        "--drop_coords",
        action="store_true",
        help="Drop 3D coordinates from SDF; default keeps coords off (no leakage).",
    )
    parser.add_argument(
        "--batch_norm",
        action="store_true",
        help="Enable batch norm in encoder (can fail with very small batches).",
    )
    args = parser.parse_args()

    main(args)
