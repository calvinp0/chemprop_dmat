"""
Quick smoke test for the transition-state setup (single-component TS, geom + TS mode).

What it does:
- loads a handful of entries from examples/ts_molecules.ndjson
- reads the TS conformer from each SDF (type=='ts') in DATA/SDF
- featurizes the TS molecule into a MolGraph
- loads TS mode targets from DATA/TSModes (npy or txt)
- builds a single-component batch, pads TS mode targets, and runs a forward pass through
  TransitionStateEncoder + TSLightningModel
- prints shapes and loss values to confirm wiring end-to-end

Requirements:
- rdkit must be installed
- paths: examples/ts_molecules.ndjson, DATA/SDF/*.sdf, DATA/TSModes/*.npy|*.txt

Run:
    python examples/ts_mode_smoke_test.py --limit 2 --shared-enc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import MeanAggregation
from chemprop.nn.message_passing import CommunicativeMessagePassing
from chemprop.models import TransitionStateEncoder
from chemprop.models.ts_lightning_model import TSLightningModel
from chemprop.schedulers import build_NoamLike_LRSched


def load_entries(ndjson_path: Path, limit: int) -> list[dict]:
    entries: list[dict] = []
    with ndjson_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))
            if len(entries) >= limit:
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


def load_ts_molecule(sdf_path: Path, drop_conformers: bool = True) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"No molecules found in {sdf_path}")
    ts_mols = [m for m in mols if m.HasProp("type") and m.GetProp("type").lower() == "ts"]
    if not ts_mols:
        raise ValueError(f"No TS molecule (type=='ts') found in {sdf_path}")
    mol = ts_mols[0]
    if drop_conformers:
        # keep atom ordering but strip 3D conformers to avoid leaking geometry
        mol = Chem.Mol(mol)  # copy
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        for cid in conf_ids:
            mol.RemoveConformer(cid)
    return mol


def pad_mode_targets(mode_targets: list[Tensor]) -> tuple[Tensor, Tensor]:
    counts = [mt.shape[0] for mt in mode_targets]
    max_atoms = max(counts)
    B = len(mode_targets)
    padded = torch.zeros((B, max_atoms, 3), dtype=torch.float32)
    mask = torch.zeros((B, max_atoms), dtype=torch.bool)
    for i, mt in enumerate(mode_targets):
        n = mt.shape[0]
        padded[i, :n] = mt
        mask[i, :n] = True
    return padded, mask


def build_batch(graphs: list, geom_targets: list[Tensor], mode_targets: list[Tensor]) -> dict:
    bmg = BatchMolGraph(graphs)
    geom_target = torch.stack(geom_targets, dim=0)
    mode_target, target_mask = pad_mode_targets(mode_targets)
    return {
        "bmg": bmg,
        "V_d": None,
        "geom_target": geom_target,
        "mode_target": mode_target,
        "target_mask": target_mask,
    }


def main(args: argparse.Namespace) -> None:
    ndjson_path = Path(args.ndjson)
    sdf_dir = Path(args.sdf_dir)
    ts_dir = Path(args.ts_mode_dir)

    entries = load_entries(ndjson_path, args.limit)
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    graphs: list = []
    geom_targets: list[Tensor] = []
    mode_targets: list[Tensor] = []
    skipped = 0

    for entry in entries:
        sdf_file = entry["sdf_file"]
        mol = load_ts_molecule(sdf_dir / sdf_file, drop_conformers=not args.keep_coords)
        graphs.append(featurizer(mol))

        geom_targets.append(torch.tensor(entry["flat_reaction_dmat"], dtype=torch.float32))

        ts_mode = load_ts_mode(ts_dir, sdf_file)
        if ts_mode is None:
            skipped += 1
            graphs.pop()
            geom_targets.pop()
            continue
        total_atoms = mol.GetNumAtoms()
        if ts_mode.shape[0] != total_atoms:
            raise ValueError(f"TS mode atoms ({ts_mode.shape[0]}) != total atoms ({total_atoms}) for {sdf_file}")
        mode_targets.append(torch.from_numpy(ts_mode))

    if len(mode_targets) == 0:
        print("No datapoints found with available TS modes; exiting.")
        return
    batch = build_batch(graphs, geom_targets, mode_targets)
    print(f"Loaded {len(mode_targets)} datapoints; skipped {skipped} missing TS modes.")

    # Model setup
    hidden_dim = args.hidden_dim
    mp_block = CommunicativeMessagePassing(d_h=hidden_dim)
    mp = mp_block
    agg = MeanAggregation()
    encoder = TransitionStateEncoder(mp, agg, batch_norm=True)

    geom_dim = geom_targets[0].shape[0]
    model_hidden = mp_block.output_dim
    model = TSLightningModel(
        encoder=encoder,
        hidden_dim=model_hidden,
        n_geom_features=geom_dim,
        mode_hidden_dim=args.mode_hidden_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model.to(device)
    batch = {k: v for k, v in batch.items()}
    if hasattr(batch["bmg"], "to"):
        maybe = batch["bmg"].to(device)
        batch["bmg"] = maybe if maybe is not None else batch["bmg"]
    batch["geom_target"] = batch["geom_target"].to(device)
    batch["mode_target"] = batch["mode_target"].to(device)

    model.train()
    geom_pred, mode_pred, atom_mask = model(batch)

    # basic checks
    if not torch.equal(atom_mask.bool(), batch["target_mask"].to(device)):
        print("Warning: encoder mask does not match padded target mask.")

    loss_geom = torch.nn.functional.mse_loss(geom_pred, batch["geom_target"])
    loss_mode = model._mode_loss(mode_pred, batch["mode_target"], atom_mask)
    total_loss = loss_geom + loss_mode

    if args.backprop:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = build_NoamLike_LRSched(
            opt, warmup_steps=1, cooldown_steps=9, init_lr=args.lr, max_lr=args.lr, final_lr=args.lr
        )
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        sched.step()
    print(f"geom_pred shape: {tuple(geom_pred.shape)}")
    print(f"mode_pred shape: {tuple(mode_pred.shape)}, atom_mask shape: {tuple(atom_mask.shape)}")
    print(f"Loss_geom: {loss_geom.item():.4f}, Loss_mode: {loss_mode.item():.4f}")
    if args.backprop:
        print("Backprop step completed (Adam, lr={:.1e}).".format(args.lr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndjson", type=str, default="examples/ts_molecules.ndjson")
    parser.add_argument("--sdf_dir", type=str, default="DATA/SDF")
    parser.add_argument("--ts_mode_dir", type=str, default="DATA/TSModes")
    parser.add_argument("--limit", type=int, default=2, help="Number of datapoints to load")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Message passing hidden size")
    parser.add_argument(
        "--mode_hidden_dim", type=int, default=64, help="Hidden size for the TS mode head"
    )
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--keep_coords",
        action="store_true",
        help="Keep 3D coordinates from SDF (by default conformers are dropped to avoid leakage).",
    )
    parser.add_argument(
        "--backprop",
        action="store_true",
        help="Perform a single optimizer step to verify gradients.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for backprop smoke.")
    args = parser.parse_args()

    main(args)
