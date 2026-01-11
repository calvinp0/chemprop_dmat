"""
Smoke test for multicomponent TS prediction using reactants (R1H/R2H) as input only.

What it does:
- reads entries from examples/ts_molecules.ndjson
- loads R1H and R2H components from each SDF (type=='r1h'/'r2h'), keeping coords by default
- featurizes reactants into MolGraphs, appending xyz coords as extra atom features (no TS geometry is used as input)
- loads TS mode targets from DATA/TSModes (npy or txt) and pads them with a mask
- builds a multicomponent batch, runs TransitionStateEncoder + decoders, and prints shapes/losses

Run (example):
    python examples/ts_mode_multicomponent_smoke_test.py --limit 2 --hidden_dim 256 --mode_hidden_dim 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor, nn

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import MeanAggregation
from chemprop.nn.ffn import MLP
from chemprop.nn.message_passing import CommunicativeMessagePassing, MulticomponentMessagePassing
from chemprop.models import TransitionStateEncoder


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


def load_mode(ts_dir: Path, sdf_name: str) -> np.ndarray | None:
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


def load_reactants(sdf_path: Path, drop_conformers: bool) -> list[Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"No molecules found in {sdf_path}")
    reactants = [m for m in mols if m.HasProp("type") and m.GetProp("type").lower() in {"r1h", "r2h"}]
    if len(reactants) != 2:
        raise ValueError(f"Expected 2 reactant components (r1h, r2h) in {sdf_path}, found {len(reactants)}")
    cleaned = []
    for m in reactants:
        if drop_conformers:
            m = Chem.Mol(m)
            for cid in [conf.GetId() for conf in m.GetConformers()]:
                m.RemoveConformer(cid)
        cleaned.append(m)
    return cleaned


def featurize_with_coords(featurizer: SimpleMoleculeMolGraphFeaturizer, mol: Chem.Mol):
    """Featurize a molecule, appending xyz coordinates as extra atom features if present."""
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    atom_extra = None
    if conf is not None:
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
        atom_extra = coords
    return featurizer(mol, atom_features_extra=atom_extra)


def pad_modes(modes: list[Tensor]) -> tuple[Tensor, Tensor]:
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


class TSModeDecoder(nn.Module):
    """Decode a global reaction latent into a fixed-length padded TS mode tensor."""

    def __init__(self, enc_dim: int, max_atoms: int, hidden_dim: int = 128):
        super().__init__()
        self.max_atoms = max_atoms
        self.net = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(),
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * max_atoms),
        )

    def forward(self, mol_repr: Tensor) -> Tensor:
        B, _ = mol_repr.shape
        out = self.net(mol_repr)
        return out.view(B, self.max_atoms, 3)


def mode_loss(pred: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Sign-invariant cosine loss over padded TS atoms, masked."""
    pred_unit = pred / pred.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_unit = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
    cosine = (pred_unit * target_unit).sum(dim=-1).abs()
    loss = (1 - cosine) * mask.float()
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def main(args: argparse.Namespace) -> None:
    ndjson_path = Path(args.ndjson)
    sdf_dir = Path(args.sdf_dir)
    ts_dir = Path(args.ts_mode_dir)

    entries = load_entries(ndjson_path, args.limit)
    featurizer = SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=3)

    component_graphs: list[list] = [[], []]
    geom_targets: list[Tensor] = []
    mode_targets: list[Tensor] = []
    skipped = 0

    for entry in entries:
        sdf_file = entry["sdf_file"]
        reactants = load_reactants(sdf_dir / sdf_file, drop_conformers=args.drop_coords)
        for i, mol in enumerate(reactants):
            component_graphs[i].append(featurize_with_coords(featurizer, mol))

        geom_targets.append(torch.tensor(entry["flat_reaction_dmat"], dtype=torch.float32))

        ts_mode = load_mode(ts_dir, sdf_file)
        if ts_mode is None:
            skipped += 1
            for comp_list in component_graphs:
                comp_list.pop()
            geom_targets.pop()
            continue
        mode_targets.append(torch.from_numpy(ts_mode))

    if len(mode_targets) == 0:
        print("No datapoints found with available TS modes; exiting.")
        return

    mode_target, mode_mask = pad_modes(mode_targets)
    geom_target = torch.stack(geom_targets, dim=0)
    bmgs = [BatchMolGraph(graphs) for graphs in component_graphs]
    print(f"Loaded {len(mode_targets)} datapoints; skipped {skipped} missing TS modes.")

    hidden_dim = args.hidden_dim
    mp_block = CommunicativeMessagePassing(d_v=featurizer.atom_fdim, d_h=hidden_dim)
    mp = MulticomponentMessagePassing(
        blocks=[mp_block], n_components=2, shared=args.shared_enc
    )
    agg = MeanAggregation()
    encoder = TransitionStateEncoder(mp, agg, batch_norm=True)

    geom_dim = geom_target.shape[1]
    enc_dim = mp_block.output_dim * 2
    geom_head = MLP.build(
        input_dim=enc_dim,
        output_dim=geom_dim,
        hidden_dim=enc_dim,
        n_layers=1,
        dropout=0.0,
        activation="relu",
    )
    mode_head = TSModeDecoder(enc_dim=enc_dim, max_atoms=mode_target.shape[1], hidden_dim=args.mode_hidden_dim)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    encoder.to(device)
    geom_head.to(device)
    mode_head.to(device)

    bmgs_device = []
    for bmg in bmgs:
        if hasattr(bmg, "to"):
            maybe = bmg.to(device)
            bmg = maybe if maybe is not None else bmg
        bmgs_device.append(bmg)

    batch = {
        "bmgs": bmgs_device,
        "V_ds": [None, None],
    }
    geom_target = geom_target.to(device)
    mode_target = mode_target.to(device)
    mode_mask = mode_mask.to(device)

    encoder.train()
    geom_head.train()
    mode_head.train()

    atom_repr, mol_repr, atom_mask = encoder(batch)
    geom_pred = geom_head(mol_repr)
    mode_pred = mode_head(mol_repr)

    loss_geom = torch.nn.functional.mse_loss(geom_pred, geom_target)
    loss_mode = mode_loss(mode_pred, mode_target, mode_mask)
    total_loss = loss_geom + loss_mode

    if args.backprop:
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(geom_head.parameters()) + list(mode_head.parameters()),
            lr=args.lr,
        )
        opt.zero_grad()
        total_loss.backward()
        opt.step()

    print(f"geom_pred shape: {tuple(geom_pred.shape)}")
    print(f"mode_pred shape: {tuple(mode_pred.shape)}, mode_mask shape: {tuple(mode_mask.shape)}")
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
        "--mode_hidden_dim", type=int, default=128, help="Hidden size for the TS mode decoder"
    )
    parser.add_argument(
        "--shared_enc",
        action="store_true",
        help="Share the same encoder across reactant components",
    )
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--drop_coords",
        action="store_true",
        help="Drop 3D coordinates from SDF; default keeps coords and appends xyz as atom features.",
    )
    parser.add_argument(
        "--backprop",
        action="store_true",
        help="Perform a single optimizer step to verify gradients.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for backprop smoke.")
    args = parser.parse_args()

    main(args)
