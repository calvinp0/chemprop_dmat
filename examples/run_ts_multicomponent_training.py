"""
Lightning training script for multicomponent TS prediction using reactants (R1H/R2H) as input.

What it does
------------
- reads entries from examples/ts_molecules.ndjson
- loads R1H and R2H components from each SDF (type=='r1h'/'r2h'), keeping coords as atom features
- optionally swaps EGNN input coords to the TS geometry using precomputed TS guesses (ts_guess_dir)
- builds multicomponent BatchMolGraphs (reactants only; no TS geometry as input to CMPNN/DMPNN)
- loads TS mode targets from DATA/TSModes (npy/txt), pads with mask
- standardizes geometry targets using the train split only
- trains a LightningModule with:
    geom head on mol_repr
    TS mode decoder on mol_repr (global latent -> padded TS atoms)
- reports mean |cosθ| for TS modes on val/test

Example
-------
    conda run -n cmpnn_rocm python examples/run_ts_multicomponent_training.py --epochs 3 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from chemprop.data.collate import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.nn import MeanAggregation
from chemprop.nn.predictors import EquivariantTSModeHead
from chemprop.nn.ffn import MLP
from chemprop.nn.message_passing import CommunicativeMessagePassing, MulticomponentMessagePassing
from chemprop.nn.predictors import EquivariantTSModeHead
from chemprop.models import TransitionStateEncoder
from chemprop.schedulers import build_NoamLike_LRSched


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


def load_reactants(sdf_path: Path) -> list[Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"No molecules found in {sdf_path}")
    reactants = [m for m in mols if m.HasProp("type") and m.GetProp("type").lower() in {"r1h", "r2h"}]
    if len(reactants) != 2:
        raise ValueError(f"Expected 2 reactant components (r1h, r2h) in {sdf_path}, found {len(reactants)}")
    return reactants


def load_ts_coords(sdf_path: Path, drop_migrating: bool) -> np.ndarray:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    ts_mols = [m for m in mols if m.HasProp("type") and m.GetProp("type").lower() == "ts"]
    if not ts_mols:
        raise ValueError(f"No TS geometry (type=='ts') found in {sdf_path}")
    ts = ts_mols[0]
    if drop_migrating:
        ts = drop_migrating_hydrogens(ts)
    if ts.GetNumConformers() == 0:
        raise ValueError(f"TS geometry missing conformer in {sdf_path}")
    conf = ts.GetConformer()
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(ts.GetNumAtoms())],
        dtype=np.float32,
    )
    return coords


def load_ts_guess_xyz(ts_guess_dir: Path, stem: str) -> np.ndarray:
    """Load pre-optimized TS guess coords from ts_guesses/<stem>_ts_guess.xyz."""
    path = ts_guess_dir / f"{stem}_ts_guess.xyz"
    if not path.exists():
        raise FileNotFoundError(f"TS guess XYZ not found at {path}")
    lines = path.read_text().strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"TS guess XYZ at {path} is too short.")
    try:
        n_atoms = int(lines[0].strip())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse atom count in {path}") from exc
    coord_lines = lines[2:]
    if len(coord_lines) < n_atoms:
        raise ValueError(f"TS guess XYZ at {path} has {len(coord_lines)} atoms, expected {n_atoms}")
    coords = []
    for line in coord_lines[:n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ line in {path}: {line}")
        x, y, z = map(float, parts[1:4])
        coords.append((x, y, z))
    return np.array(coords, dtype=np.float32)


def load_ts_props(sdf_path: Path) -> dict:
    """Load the TS block's mol_properties as a dict keyed by atom index (int)."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    ts_mols = [m for m in mols if m.HasProp("type") and m.GetProp("type").lower() == "ts"]
    if not ts_mols:
        return {}
    ts = ts_mols[0]
    if not ts.HasProp("mol_properties"):
        return {}
    import json

    try:
        props = json.loads(ts.GetProp("mol_properties"))
    except Exception:
        return {}
    out = {}
    for k, v in props.items():
        try:
            idx = int(k)
            out[idx] = v
        except Exception:
            continue
    return out


def _label_indices_from_props(prop_str: str, targets: list[str]) -> list[int]:
    import json

    if not prop_str:
        return []
    try:
        props = json.loads(prop_str)
    except Exception:
        return []
    out = []
    for k, v in props.items():
        label = str(v.get("label", "")).lower()
        if any(t in label for t in targets):
            try:
                out.append(int(k) - 1)  # mol_properties keys are 1-based
            except Exception:
                continue
    return out


def remap_indices_after_drop(idxs: list[int], drop_idxs: list[int]) -> list[int]:
    """Map raw indices to indices after dropping drop_idxs; drop any that were removed."""
    if not drop_idxs:
        return idxs
    drop_set = set(drop_idxs)
    drop_sorted = sorted(drop_idxs)
    mapped = []
    for idx in idxs:
        if idx in drop_set:
            continue
        shift = sum(1 for d in drop_sorted if d < idx)
        mapped.append(idx - shift)
    return mapped


def get_drop_indices(mol: Chem.Mol) -> list[int]:
    """Return zero-based indices of migrating H atoms marked as d_hydrogen."""
    if not mol.HasProp("mol_properties"):
        return []
    import json

    props = json.loads(mol.GetProp("mol_properties"))
    drop_idxs: list[int] = []
    for k, v in props.items():
        label = v.get("label", "")
        if "d_hydrogen" in label:
            try:
                idx = int(k) - 1  # mol_properties keys are 1-based
                if idx >= 0:
                    drop_idxs.append(idx)
            except Exception:
                continue
    return sorted(set(drop_idxs))


def drop_migrating_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Remove donor hydrogen (label contains 'd_hydrogen') in mol_properties."""
    drop_idxs = get_drop_indices(mol)
    if not drop_idxs:
        return mol
    em = Chem.EditableMol(mol)
    for idx in sorted(set(drop_idxs), reverse=True):
        if 0 <= idx < mol.GetNumAtoms():
            em.RemoveAtom(idx)
    cleaned = em.GetMol()
    # keep conformers aligned
    for conf_id in range(mol.GetNumConformers()):
        conf = mol.GetConformer(conf_id)
        new_conf = Chem.Conformer(cleaned.GetNumAtoms())
        keep_idx = 0
        for i in range(mol.GetNumAtoms()):
            if i in drop_idxs:
                continue
            pos = conf.GetAtomPosition(i)
            new_conf.SetAtomPosition(keep_idx, pos)
            keep_idx += 1
        new_conf.SetId(conf_id)
        cleaned.AddConformer(new_conf, assignId=True)
    return cleaned


class GeometryMolGraphFeaturizer:
    """
    Geometry-only featurizer: uses 3D positions to compute per-edge features:
      - RBF(distance)
      - sin/cos of angle A–B–C (+ availability mask)
      - sin/cos of dihedral A–B–C–D (+ availability mask)
    Missing geometry is encoded with zeros plus mask=0 to keep it distinct from a true 0 radian value.
    """

    def __init__(
        self,
        rbf_D_min: float = 0.0,
        rbf_D_max: float = 5.0,
        rbf_D_count: int = 10,
        rbf_gamma: float = 10.0,
    ):
        self.mu = np.linspace(rbf_D_min, rbf_D_max, rbf_D_count)
        self.gamma = rbf_gamma
        self.rbf_dim = rbf_D_count
        self.atom_featurizer = MultiHotAtomFeaturizer.v2()
        self.atom_fdim = len(self.atom_featurizer)
        # distance RBF + angle (sin, cos, mask) + dihedral (sin, cos, mask)
        self.bond_fdim = self.rbf_dim + 2 + 1 + 2 + 1

    def _rbf(self, d: float) -> np.ndarray:
        return np.exp(-self.gamma * (d - self.mu) ** 2)

    @staticmethod
    def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        ba /= (np.linalg.norm(ba) + 1e-8)
        bc /= (np.linalg.norm(bc) + 1e-8)
        cos_a = np.dot(ba, bc)
        return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    @staticmethod
    def compute_dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        # rdMolTransforms matches RDKit conventions
        return float(rdMolTransforms.GetDihedralRadFromVect(p0, p1, p2, p3))

    def __call__(self, mol, atom_features_extra=None, bond_features_extra=None):
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if n_atoms == 0:
            raise ValueError("GeometryMolGraphFeaturizer requires at least one atom.")
        if mol.GetNumConformers() == 0:
            raise ValueError("GeometryMolGraphFeaturizer requires a conformer with 3D coordinates.")

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(f"Expected {n_atoms} atom extras, got {len(atom_features_extra)}")
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(f"Expected {n_bonds} bond extras, got {len(bond_features_extra)}")

        V = np.stack([self.atom_featurizer(a) for a in mol.GetAtoms()], dtype=np.float32)
        if atom_features_extra is not None:
            V = np.hstack((V, atom_features_extra))

        bond_features_extra = (
            None
            if bond_features_extra is None
            else [np.asarray(f, dtype=np.float32).ravel() for f in bond_features_extra]
        )
        bond_extra_dim = 0
        if bond_features_extra is not None and n_bonds > 0:
            bond_extra_dim = len(bond_features_extra[0])
            if any(len(f) != bond_extra_dim for f in bond_features_extra):
                raise ValueError("All bond_features_extra entries must have the same length.")

        conf = mol.GetConformer()
        coords = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n_atoms)],
            dtype=np.float32,
        )
        bond_fdim = self.bond_fdim + bond_extra_dim

        torsions_by_bond: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
        for bond in mol.GetBonds():
            j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            j_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
            k_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
            for i in j_nbrs:
                for l in k_nbrs:
                    torsions_by_bond.setdefault((j, k), []).append((i, j, k, l))
                    torsions_by_bond.setdefault((k, j), []).append((l, k, j, i))

        def angle_feat(src: int, dst: int) -> Tuple[np.ndarray, float]:
            vals = []
            for k in [n.GetIdx() for n in mol.GetAtomWithIdx(dst).GetNeighbors() if n.GetIdx() != src]:
                ang = self.compute_angle(coords[src], coords[dst], coords[k])
                vals.append([np.sin(ang), np.cos(ang)])
            if not vals:
                return np.array([0.0, 0.0], dtype=np.float32), 0.0
            return np.mean(vals, axis=0).astype(np.float32), 1.0

        def torsion_feat(src: int, dst: int) -> Tuple[np.ndarray, float]:
            vals = []
            for (i, j, k, l) in torsions_by_bond.get((src, dst), []):
                dih = rdMolTransforms.GetDihedralRad(conf, i, j, k, l)
                vals.append([np.sin(dih), np.cos(dih)])
            if not vals:
                return np.array([0.0, 0.0], dtype=np.float32), 0.0
            return np.mean(vals, axis=0).astype(np.float32), 1.0

        E_list, src_list, dst_list = [], [], []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            dist = np.linalg.norm(coords[u] - coords[v])
            rbf_feat = self._rbf(dist).astype(np.float32)

            for src, dst in ((u, v), (v, u)):
                ang_sc, ang_has = angle_feat(src, dst)
                dih_sc, dih_has = torsion_feat(src, dst)
                feat_parts = [
                    rbf_feat,
                    ang_sc,
                    np.array([ang_has], dtype=np.float32),
                    dih_sc,
                    np.array([dih_has], dtype=np.float32),
                ]
                if bond_features_extra is not None:
                    feat_parts.append(bond_features_extra[bond.GetIdx()])
                feat = np.concatenate(feat_parts, axis=0).astype(np.float32)
                if feat.shape[0] != bond_fdim:
                    raise ValueError(f"Expected bond feature dim {bond_fdim}, got {feat.shape[0]}")
                E_list.append(feat)
                src_list.append(src)
                dst_list.append(dst)

        if E_list:
            E = np.stack(E_list, axis=0)
            edge_index = np.vstack((src_list, dst_list))
            rev_edge_index = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
        else:
            E = np.zeros((0, bond_fdim), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=int)
            rev_edge_index = np.zeros((0,), dtype=int)

        if np.isnan(V).any() or np.isinf(V).any() or np.isnan(E).any() or np.isinf(E).any():
            raise ValueError("Non-finite features encountered in geometry featurization.")

        return MolGraph(V=V.astype(np.float32, copy=False), E=E.astype(np.float32, copy=False), edge_index=edge_index, rev_edge_index=rev_edge_index)


def pad_modes(modes: list[Tensor], max_atoms: int | None = None) -> tuple[Tensor, Tensor]:
    counts = [m.shape[0] for m in modes]
    max_atoms_local = max(counts)
    max_atoms = max_atoms if max_atoms is not None else max_atoms_local
    B = len(modes)
    padded = torch.zeros((B, max_atoms, 3), dtype=torch.float32)
    mask = torch.zeros((B, max_atoms), dtype=torch.bool)
    for i, m in enumerate(modes):
        n = m.shape[0]
        padded[i, :n] = m
        mask[i, :n] = True
    return padded, mask


class TSModeDecoder(nn.Module):
    """Decode a global latent mol_repr into a padded TS mode tensor."""

    def __init__(self, enc_dim: int, max_atoms: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        layers: list[nn.Module] = []
        in_dim = enc_dim
        for _ in range(max(n_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3 * max_atoms))
        self.net = nn.Sequential(*layers)
        self.max_atoms = max_atoms

    def forward(self, mol_repr: Tensor) -> Tensor:
        out = self.net(mol_repr)
        B = mol_repr.size(0)
        return out.view(B, self.max_atoms, 3)


class MultiComponentTSLightning(pl.LightningModule):
    """LightningModule for multicomponent TS prediction with mol-level mode decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        mol_hidden_dim: int,
        atom_hidden_dim: int,
        n_geom_features: int,
        geom_hidden_dim: int,
        geom_n_layers: int,
        mode_hidden_dim: int,
        mode_n_layers: int,
        max_mode_atoms: int,
        lambda_geom: float,
        lambda_mode: float,
        lr: float,
        use_scheduler: bool,
        warmup_epochs: int,
        init_lr: float,
        max_lr: float,
        final_lr: float,
        equivariant_mode_head: bool = False,
        freeze_encoder_epochs: int = 0,
        optimizer_cls: type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.save_hyperparameters(
            ignore=["encoder", "optimizer_cls", "optimizer_kwargs"]
        )
        self.lambda_geom = lambda_geom
        self.lambda_mode = lambda_mode
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.eps = eps
        self.equivariant_mode_head = equivariant_mode_head
        self.freeze_encoder_epochs = freeze_encoder_epochs

        self.geom_head = MLP.build(
            input_dim=mol_hidden_dim,
            output_dim=n_geom_features,
            hidden_dim=geom_hidden_dim,
            n_layers=geom_n_layers,
            dropout=0.0,
            activation="relu",
        )
        if equivariant_mode_head:
            self.mode_head = EquivariantTSModeHead(
                d_h=atom_hidden_dim, n_layers=mode_n_layers, d_edge=0, d_msg=mode_hidden_dim
            )
        else:
            self.mode_head = TSModeDecoder(
                enc_dim=mol_hidden_dim, max_atoms=max_mode_atoms, hidden_dim=mode_hidden_dim, n_layers=mode_n_layers
            )

        if self.freeze_encoder_epochs > 0:
            self._set_encoder_trainable(False)

    def _set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def on_train_epoch_start(self) -> None:
        # Toggle encoder freeze schedule; keep encoder frozen for entire mode-only runs.
        if self.freeze_encoder_epochs > 0:
            trainable = self.current_epoch >= self.freeze_encoder_epochs
            self._set_encoder_trainable(trainable)
        if self.lambda_geom == 0.0:
            self._set_encoder_trainable(False)
        # Keep geom head frozen during mode-only stage (lambda_geom == 0)
        geom_trainable = self.lambda_geom != 0.0
        for p in self.geom_head.parameters():
            p.requires_grad = geom_trainable
        return super().on_train_epoch_start()

    def _move_bmgs_to_device(self, batch: Dict[str, Any]) -> None:
        for bmg in batch.get("bmgs", []):
            if hasattr(bmg, "to"):
                bmg.to(self.device)

    def forward(self, batch: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """Runs the encoder and heads."""
        self._move_bmgs_to_device(batch)
        atom_repr, mol_repr, atom_mask = self.encoder(batch)
        geom_pred = self.geom_head(mol_repr)
        if self.equivariant_mode_head:
            coords: Tensor = batch["coords"].to(self.device)
            edge_index: Tensor = batch["edge_index"].to(self.device)
            edge_mask: Tensor | None = batch.get("edge_mask")
            # pad atom_repr to coords length if needed
            if atom_repr.size(1) < coords.size(1):
                pad = coords.size(1) - atom_repr.size(1)
                atom_repr = F.pad(atom_repr, (0, 0, 0, pad))
            elif atom_repr.size(1) > coords.size(1):
                atom_repr = atom_repr[:, : coords.size(1), :]
            mode_pred = self.mode_head(
                coords=coords,
                h=atom_repr,
                edge_index=edge_index,
                mask=batch["mode_mask"].to(self.device),
                edge_attr=None,
                edge_mask=edge_mask,
            )
        else:
            mode_pred = self.mode_head(mol_repr)
        return geom_pred, mode_pred, atom_mask

    def _mode_loss(self, mode_pred: Tensor, mode_target: Tensor, mode_mask: Tensor, mode_weight: Tensor | None = None) -> Tensor:
        """Masked, sign-invariant cosine loss on per-atom mode directions."""
        pred_unit = mode_pred / mode_pred.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        target_unit = mode_target / mode_target.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        cosine = (pred_unit * target_unit).sum(dim=-1).abs()
        weights = mode_mask.float() if mode_weight is None else mode_weight * mode_mask.float()
        masked_loss = (1 - cosine) * weights
        normalizer = weights.sum().clamp_min(self.eps)
        return masked_loss.sum() / normalizer

    def _shared_step(self, batch: Dict[str, Any], stage: str):
        geom_pred, mode_pred, _ = self(batch)
        geom_target: Tensor = batch["geom_target"].to(self.device)
        mode_target: Tensor = batch["mode_target"].to(self.device)
        mode_mask: Tensor = batch["mode_mask"].to(self.device)
        mode_weight: Tensor | None = batch.get("mode_weight")
        if mode_weight is not None:
            mode_weight = mode_weight.to(self.device)
        reactive_mask: Tensor | None = batch.get("reactive_mask")
        if reactive_mask is not None:
            reactive_mask = reactive_mask.to(self.device)

        # Align shapes in case padding differs
        if mode_pred.size(1) > mode_target.size(1):
            pad = mode_pred.size(1) - mode_target.size(1)
            mode_target = F.pad(mode_target, (0, 0, 0, pad))
            mode_mask = F.pad(mode_mask, (0, pad))
        elif mode_pred.size(1) < mode_target.size(1):
            mode_target = mode_target[:, : mode_pred.size(1)]
            mode_mask = mode_mask[:, : mode_pred.size(1)]

        if not torch.isfinite(mode_pred).all():
            raise ValueError("Non-finite values in mode_pred (check coords/edges).")

        loss_geom = F.mse_loss(geom_pred, geom_target)
        if self.lambda_geom == 0.0:
            # prevent geom_head updates during mode-only warmup
            loss_geom = loss_geom.detach()
        loss_mode = self._mode_loss(mode_pred, mode_target, mode_mask, mode_weight)
        loss_total = self.lambda_geom * loss_geom + self.lambda_mode * loss_mode

        batch_size = geom_target.size(0)
        self.log(f"{stage}_loss", loss_total, on_step=(stage == "train"), on_epoch=True, prog_bar=(stage != "test"), batch_size=batch_size)
        self.log(f"{stage}_loss_geom", loss_geom, on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_loss_mode", loss_mode, on_step=(stage == "train"), on_epoch=True, batch_size=batch_size)

        pred_unit = mode_pred / mode_pred.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        target_unit = mode_target / mode_target.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        cosine = (pred_unit * target_unit).sum(dim=-1).abs()
        if mode_mask.any():
            mean_abs_cos = cosine[mode_mask.bool()].mean()
            self.log(f"{stage}_mean_abs_cos_mode", mean_abs_cos, batch_size=batch_size, prog_bar=False)
            # top-3 magnitude atoms
            norms = mode_target.norm(dim=-1)
            top_cos_vals = []
            for b in range(mode_target.size(0)):
                mask_b = mode_mask[b]
                if not mask_b.any():
                    continue
                norms_b = norms[b] * mask_b
                k = min(3, int(mask_b.sum().item()))
                if k > 0:
                    topk = torch.topk(norms_b, k=k).indices
                    top_cos_vals.append(cosine[b, topk])
            if top_cos_vals:
                top_cos = torch.cat(top_cos_vals).mean()
                self.log(f"{stage}_mean_abs_cos_top3", top_cos, batch_size=batch_size, prog_bar=False)
            # donor/acceptor mask
            if reactive_mask is not None and reactive_mask.any():
                reactive_cos = cosine[reactive_mask.bool()].mean()
                self.log(f"{stage}_mean_abs_cos_reactive", reactive_cos, batch_size=batch_size, prog_bar=False)

        return loss_total

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = self.optimizer_cls(
            self.parameters(),
            lr=self.init_lr if self.use_scheduler else self.lr,
            **self.optimizer_kwargs,
        )
        if not self.use_scheduler:
            return opt

        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs if self.trainer.max_epochs != -1 else 100
        cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}


class TSMultiDataset(Dataset):
    def __init__(
        self,
        comp_graphs: Sequence[Tuple[Any, Any]],
        coords_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
        geom_targets: Sequence[Tensor],
        mode_targets: Sequence[Tensor],
        reactive_indices: Sequence[list[int]],
        reactive_cross_edges: Sequence[list[Tuple[int, int]]],
    ):
        self.comp_graphs = comp_graphs
        self.coords_pairs = coords_pairs
        self.geom_targets = geom_targets
        self.mode_targets = mode_targets
        self.reactive_indices = reactive_indices
        self.reactive_cross_edges = reactive_cross_edges

    def __len__(self) -> int:
        return len(self.comp_graphs)

    def __getitem__(self, idx: int):
        g1, g2 = self.comp_graphs[idx]
        return (
            g1,
            g2,
            self.coords_pairs[idx],
            self.geom_targets[idx],
            self.mode_targets[idx],
            self.reactive_indices[idx],
            self.reactive_cross_edges[idx],
        )


def random_modes_like(mode_target: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Generate random unit vectors per atom respecting the mask."""
    rand = torch.randn_like(mode_target)
    rand = rand / rand.norm(dim=-1, keepdim=True).clamp_min(eps)
    return rand * mask.unsqueeze(-1)


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
        mode_weight = batch.get("mode_weight")
        if mode_weight is not None:
            mode_weight = mode_weight.to(device)

        rand_pred = random_modes_like(mode_target, mode_mask)
        loss_mode = mode_loss_fn(rand_pred, mode_target, mode_mask, mode_weight)
        all_loss_mode.append(loss_mode.item())

        cos_vals = cosine_alignment(rand_pred, mode_target, mode_mask)
        all_cos_vals.append(cos_vals.cpu())

    loss_mode_mean = float(np.mean(all_loss_mode))
    cos_all = torch.cat(all_cos_vals, dim=0)
    mean_abs_cos = cos_all.mean().item()

    print(f"Random baseline mode_loss: {loss_mode_mean:.4f}")
    print(f"Random baseline mean |cos θ|: {mean_abs_cos:.4f}")


def collate(
    samples: Sequence[Tuple[Any, Any, Tuple[np.ndarray, np.ndarray], Tensor, Tensor, list[int], list[Tuple[int, int]]]],
    max_atoms: int,
    cross_edge_cutoff: float,
    center_complex: bool,
    mode_norm_threshold: float,
    weight_mode_by_magnitude: bool,
    cross_edge_mode: str,
):
    g1_list, g2_list, coords_pairs, geom_list, mode_list, reactive_idx_list, reactive_edge_list = zip(*samples)
    bmg1 = BatchMolGraph(list(g1_list))
    bmg2 = BatchMolGraph(list(g2_list))
    geom_target = torch.stack(list(geom_list), dim=0)
    mode_target, mode_mask = pad_modes(list(mode_list), max_atoms=max_atoms)

    with torch.no_grad():
        norms = mode_target.norm(dim=-1)
        move_mask = norms > mode_norm_threshold
        mode_mask = mode_mask & move_mask
        mode_weight = None
        if weight_mode_by_magnitude:
            max_norm = norms.max(dim=1, keepdim=True).values.clamp_min(1e-8)
            weights = norms / max_norm
            mode_weight = weights * mode_mask.float()

    coords_batch = []
    edge_indices = []
    edge_masks = []
    reactive_masks = []
    for (c1, c2), g1, g2, reactive_idxs, edges in zip(coords_pairs, g1_list, g2_list, reactive_idx_list, reactive_edge_list):
        coords_comb = np.concatenate([c1, c2], axis=0)
        if center_complex:
            coords_comb = coords_comb - coords_comb.mean(axis=0, keepdims=True)
        coords_padded = np.zeros((max_atoms, 3), dtype=np.float32)
        n_tot = min(coords_comb.shape[0], max_atoms)
        coords_padded[:n_tot] = coords_comb[:n_tot]
        coords_batch.append(torch.tensor(coords_padded, dtype=torch.float32))
        rmask = torch.zeros((max_atoms,), dtype=torch.bool)
        for idx in reactive_idxs:
            if 0 <= idx < max_atoms:
                rmask[idx] = True
        reactive_masks.append(rmask)

        n1 = c1.shape[0]
        ei1 = torch.tensor(g1.edge_index, dtype=torch.long)
        ei2 = torch.tensor(g2.edge_index, dtype=torch.long) + n1
        cross_edges = []
        if cross_edge_mode == "cutoff" and cross_edge_cutoff > 0:
            c1_t = torch.tensor(c1, dtype=torch.float32)
            c2_t = torch.tensor(c2, dtype=torch.float32)
            dists = torch.cdist(c1_t, c2_t)
            idxs = (dists < cross_edge_cutoff).nonzero(as_tuple=False)
            for i, j in idxs:
                cross_edges.append(torch.tensor([[i.item()], [n1 + j.item()]], dtype=torch.long))
                cross_edges.append(torch.tensor([[n1 + j.item()], [i.item()]], dtype=torch.long))
        elif cross_edge_mode == "reactive":
            for i, j in edges:
                if i < c1.shape[0] and j < c2.shape[0]:
                    cross_edges.append(torch.tensor([[i], [n1 + j]], dtype=torch.long))
                    cross_edges.append(torch.tensor([[n1 + j], [i]], dtype=torch.long))
        ei = torch.cat([ei1, ei2] + cross_edges, dim=1) if cross_edges else torch.cat([ei1, ei2], dim=1)
        edge_indices.append(ei)
        emask = torch.zeros(ei.shape[1], dtype=torch.bool)
        emask[:] = True
        edge_masks.append(emask)

    coords = torch.stack(coords_batch, dim=0)
    max_edges = max(e.shape[1] for e in edge_indices) if edge_indices else 0
    edge_index_b = torch.zeros((len(edge_indices), max_edges, 2), dtype=torch.long)
    edge_mask_b = torch.zeros((len(edge_masks), max_edges), dtype=torch.bool)
    for b, ei in enumerate(edge_indices):
        e = ei.shape[1]
        edge_index_b[b, :e, :] = ei.t()
        edge_mask_b[b, :e] = True

    return {
        "bmgs": [bmg1, bmg2],
        "V_ds": [None, None],
        "geom_target": geom_target,
        "mode_target": mode_target,
        "mode_mask": mode_mask,
        "mode_weight": mode_weight,
        "reactive_mask": torch.stack(reactive_masks, dim=0),
        "coords": coords,
        "edge_index": edge_index_b,
        "edge_mask": edge_mask_b,
    }


def build_splits(
    comp_graphs: Sequence[Tuple[Any, Any]],
    coords_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    geom_targets: Sequence[Tensor],
    mode_targets: Sequence[Tensor],
    reactive_indices: Sequence[list[int]],
    reactive_cross_edges: Sequence[list[Tuple[int, int]]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[TSMultiDataset, TSMultiDataset, TSMultiDataset]:
    n_total = len(comp_graphs)
    n_test = int(test_fraction * n_total)
    n_val = int(val_fraction * n_total)
    n_train = n_total - n_val - n_test

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val :]

    def subset(idxs: list[int]) -> TSMultiDataset:
        return TSMultiDataset(
            [comp_graphs[i] for i in idxs],
            [coords_pairs[i] for i in idxs],
            [geom_targets[i].clone() for i in idxs],
            [mode_targets[i] for i in idxs],
            [reactive_indices[i] for i in idxs],
            [reactive_cross_edges[i] for i in idxs],
        )

    return subset(idx_train), subset(idx_val), subset(idx_test)


def standardize_geom(train_subset: TSMultiDataset, subsets: list[TSMultiDataset]):
    geom_stack = torch.stack([train_subset.geom_targets[i] for i in range(len(train_subset))], dim=0)
    mean = geom_stack.mean(dim=0)
    std = geom_stack.std(dim=0).clamp_min(1e-8)

    def _apply(ds: TSMultiDataset):
        for i in range(len(ds)):
            geom_std = (ds.geom_targets[i] - mean) / std
            ds.geom_targets[i] = geom_std

    for ds in subsets:
        _apply(ds)
    return mean, std


def main(args: argparse.Namespace) -> None:
    ndjson_path = Path(args.ndjson)
    sdf_dir = Path(args.sdf_dir)
    ts_dir = Path(args.ts_mode_dir)
    ts_guess_dir = Path(args.ts_guess_dir) if args.ts_guess_dir is not None else None
    use_ts_coords = ts_guess_dir is not None

    entries = load_entries(ndjson_path, args.limit)
    featurizer = GeometryMolGraphFeaturizer()

    comp_graphs: list[Tuple[Any, Any]] = []
    coords_pairs: list[Tuple[np.ndarray, np.ndarray]] = []
    geom_targets_raw: list[Tensor] = []
    mode_targets: list[Tensor] = []
    reactive_indices_all: list[list[int]] = []
    reactive_cross_edges_all: list[list[Tuple[int, int]]] = []

    skipped = 0
    for entry in entries:
        sdf_file = entry["sdf_file"]
        stem = Path(sdf_file).stem
        sdf_path = sdf_dir / f"{stem}_updated.sdf"
        if not sdf_path.exists():
            print(f"Skipping {sdf_file}: expected updated SDF at {sdf_path} not found.")
            skipped += 1
            continue
        try:
            reactants_raw = load_reactants(sdf_path)
            reactants_dropped = [drop_migrating_hydrogens(m) for m in reactants_raw]
        except ValueError as e:
            print(f"Skipping {sdf_file}: {e}")
            skipped += 1
            continue

        def _coords_from_mols(mols: list[Chem.Mol]) -> list[np.ndarray]:
            coords_local = []
            for mol in mols:
                if mol.GetNumConformers() == 0:
                    raise ValueError("Reactant missing conformer coords.")
                conf = mol.GetConformer()
                coords_local.append(
                    np.array(
                        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
                        dtype=np.float32,
                    )
                )
            return coords_local

        try:
            graphs_drop = [featurizer(m) for m in reactants_dropped]
            coords_pair_drop = _coords_from_mols(reactants_dropped)
        except ValueError as e:
            print(f"Skipping {sdf_file}: {e}")
            skipped += 1
            continue

        ts_mode = load_ts_mode(ts_dir, sdf_file)
        if ts_mode is None:
            skipped += 1
            continue

        use_raw = False
        total_atoms_drop = coords_pair_drop[0].shape[0] + coords_pair_drop[1].shape[0]
        coords_pair_use = coords_pair_drop
        graphs_use = graphs_drop
        if ts_mode.shape[0] != total_atoms_drop:
            try:
                graphs_raw = [featurizer(m) for m in reactants_raw]
                coords_pair_raw = _coords_from_mols(reactants_raw)
                total_atoms_raw = coords_pair_raw[0].shape[0] + coords_pair_raw[1].shape[0]
                if ts_mode.shape[0] == total_atoms_raw:
                    use_raw = True
                    coords_pair_use = coords_pair_raw
                    graphs_use = graphs_raw
                else:
                    print(
                        f"Skipping {sdf_file}: TS mode atoms ({ts_mode.shape[0]}) != reactant atoms "
                        f"({total_atoms_drop}) after drop and ({total_atoms_raw}) before drop."
                    )
                    skipped += 1
                    continue
            except Exception:
                print(
                    f"Skipping {sdf_file}: TS mode atoms ({ts_mode.shape[0]}) mismatch with reactants ({total_atoms_drop})."
                )
                skipped += 1
                continue

        if use_ts_coords:
            try:
                ts_coords = load_ts_guess_xyz(ts_guess_dir, stem)
            except Exception as e:
                print(f"Skipping {sdf_file}: failed to load TS guess: {e}")
                skipped += 1
                continue
            total_atoms_use = coords_pair_use[0].shape[0] + coords_pair_use[1].shape[0]
            if ts_coords.shape[0] != total_atoms_use:
                print(
                    f"Skipping {sdf_file}: TS coords atoms ({ts_coords.shape[0]}) mismatch with selected reactant atoms ({total_atoms_use})."
                )
                skipped += 1
                continue
            n1 = coords_pair_use[0].shape[0]
            coords_pair_use = (ts_coords[:n1], ts_coords[n1:])

        comp_graphs.append((graphs_use[0], graphs_use[1]))
        coords_pairs.append((coords_pair_use[0], coords_pair_use[1]))
        geom_targets_raw.append(torch.tensor(entry["flat_reaction_dmat"], dtype=torch.float32))
        mode_targets.append(torch.from_numpy(ts_mode))
        # reactive indices: donor (*1) from component 1, acceptor (*3) from component 2
        # reactive labels from raw; remap if atoms were dropped
        donor_raw = _label_indices_from_props(
            reactants_raw[0].GetProp("mol_properties") if reactants_raw[0].HasProp("mol_properties") else "",
            ["donor", "donator", "d_atom", "*1"],
        )
        acceptor_raw = _label_indices_from_props(
            reactants_raw[1].GetProp("mol_properties") if reactants_raw[1].HasProp("mol_properties") else "",
            ["acceptor", "a_atom", "*3"],
        )
        if use_raw:
            donor_idxs = donor_raw
            acceptor_idxs = acceptor_raw
        else:
            drop1 = get_drop_indices(reactants_raw[0])
            drop2 = get_drop_indices(reactants_raw[1])
            donor_idxs = remap_indices_after_drop(donor_raw, drop1)
            acceptor_idxs = remap_indices_after_drop(acceptor_raw, drop2)
        n1 = coords_pair_use[0].shape[0]
        n2 = coords_pair_use[1].shape[0]
        donor_idxs = [i for i in donor_idxs if i < n1]
        acceptor_idxs = [j for j in acceptor_idxs if j < n2]
        reactive_combined = donor_idxs + [n1 + j for j in acceptor_idxs]
        reactive_indices_all.append(reactive_combined)
        edges = []
        for i in donor_idxs:
            for j in acceptor_idxs:
                edges.append((i, j))
        reactive_cross_edges_all.append(edges)
        if args.debug_feats:
            print(f"[reactive] {stem}: donor={len(donor_idxs)} acceptor={len(acceptor_idxs)} edges={len(edges)}")

    if args.debug_feats and comp_graphs:
        g0: MolGraph = comp_graphs[0][0]
        print(
            f"[debug] featurizer atom_fdim={featurizer.atom_fdim}, bond_fdim={featurizer.bond_fdim}, "
            f"first graph V={g0.V.shape}, E={g0.E.shape}"
        )

    if len(mode_targets) == 0:
        print("No datapoints found with available TS modes; exiting.")
        return
    print(f"Loaded {len(mode_targets)} datapoints; skipped {skipped} missing/invalid entries.")

    train_ds, val_ds, test_ds = build_splits(
        comp_graphs,
        coords_pairs,
        geom_targets_raw,
        mode_targets,
        reactive_indices_all,
        reactive_cross_edges_all,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )
    geom_mean, geom_std = standardize_geom(train_ds, [train_ds, val_ds, test_ds])
    print(f"Geometry standardization using train split: mean shape {tuple(geom_mean.shape)}, std shape {tuple(geom_std.shape)}")

    hidden_dim = args.hidden_dim
    if args.encoder.lower() == "cmpnn":
        mp_block = CommunicativeMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=hidden_dim)
    elif args.encoder.lower() == "dmpnn":
        from chemprop.nn.message_passing.base import BondMessagePassing as DMPNN

        mp_block = DMPNN(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=hidden_dim)
    else:
        raise ValueError(f"Unknown encoder '{args.encoder}'. Expected one of: cmpnn, dmpnn.")

    if args.shared_enc:
        blocks = [mp_block]
    else:
        # distinct encoders per component
        if args.encoder.lower() == "cmpnn":
            blocks = [
                CommunicativeMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=hidden_dim),
                CommunicativeMessagePassing(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=hidden_dim),
            ]
        else:
            from chemprop.nn.message_passing.base import BondMessagePassing as DMPNN

            blocks = [
                DMPNN(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=hidden_dim),
                DMPNN(d_v=featurizer.atom_fdim, d_e=featurizer.bond_fdim, d_h=hidden_dim),
            ]

    mp = MulticomponentMessagePassing(blocks=blocks, n_components=2, shared=args.shared_enc)
    agg = MeanAggregation()
    encoder = TransitionStateEncoder(mp, agg, batch_norm=args.batch_norm)

    mol_hidden_dim = mp.output_dim
    atom_hidden_dim = blocks[0].output_dim
    max_mode_atoms = max(mt.shape[0] for mt in mode_targets)
    geom_dim = geom_mean.shape[0]

    model = MultiComponentTSLightning(
        encoder=encoder,
        mol_hidden_dim=mol_hidden_dim,
        atom_hidden_dim=atom_hidden_dim,
        n_geom_features=geom_dim,
        geom_hidden_dim=args.geom_hidden_dim or mol_hidden_dim,
        geom_n_layers=args.geom_n_layers,
        mode_hidden_dim=args.mode_hidden_dim,
        mode_n_layers=args.mode_n_layers,
        max_mode_atoms=max_mode_atoms,
        lambda_geom=args.lambda_geom,
        lambda_mode=args.lambda_mode,
        lr=args.lr,
        use_scheduler=args.use_scheduler,
        warmup_epochs=args.warmup_epochs,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
        equivariant_mode_head=args.equivariant_mode_head,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
    )

    collate_fn = partial(
        collate,
        max_atoms=max_mode_atoms,
        cross_edge_cutoff=args.cross_edge_cutoff,
        center_complex=args.center_complex,
        mode_norm_threshold=args.mode_norm_threshold,
        weight_mode_by_magnitude=args.weight_mode_by_magnitude,
        cross_edge_mode=args.cross_edge_mode,
    )

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False) if len(val_ds) > 0 else None
    test_loader = make_loader(test_ds, shuffle=False) if len(test_ds) > 0 else None

    accelerator = "gpu" if args.use_cuda and torch.cuda.is_available() else "cpu"
    logger = CSVLogger(args.log_dir, name="ts_multicomponent", flush_logs_every_n_steps=10)

    ckpt_callback = ModelCheckpoint(
        monitor=args.checkpoint_monitor,
        mode=args.checkpoint_mode,
        save_top_k=args.save_top_k,
        filename="epoch{epoch:03d}-{" + args.checkpoint_monitor + ":.4f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=[ckpt_callback],
        enable_checkpointing=True,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    if test_loader is not None:
        trainer.test(model, test_loader)

    # Optional random baseline on test split
    if test_loader is not None and args.eval_random_baseline:
        print("Evaluating random baseline on test split...")
        evaluate_random_baseline(
            test_loader,
            lambda pred, target, mask, weight=None: model._mode_loss(pred, target, mask, weight),
        )

    # Optional plotting from CSVLogger metrics
    if args.plot_metrics or args.save_loss_plot:
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
                            by_epoch[e] = v
                    if not by_epoch:
                        return [], []
                    xs = sorted(by_epoch.keys())
                    ys = [by_epoch[e] for e in xs]
                    return xs, ys

                with metrics_path.open() as f:
                    rows = list(csv.DictReader(f))

                series = {
                    "train_loss": add_point(rows, "train_loss_epoch"),
                    "val_loss": add_point(rows, "val_loss"),
                    "test_loss": add_point(rows, "test_loss"),
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
                    plt.title("TS multicomponent training losses")
                    if args.save_loss_plot:
                        out_path = Path(logger.log_dir) / "loss_curve.png"
                        plt.savefig(out_path)
                        print(f"Saved loss plot to {out_path}")
                    else:
                        plt.show()
            except ImportError:
                print("matplotlib not installed; skipping plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndjson", type=str, default="examples/ts_molecules.ndjson")
    parser.add_argument("--sdf_dir", type=str, default="DATA/SDF/updated")
    parser.add_argument("--ts_mode_dir", type=str, default="DATA/TSModes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of datapoints (None=all).")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Message passing hidden size.")
    parser.add_argument("--mode_hidden_dim", type=int, default=128, help="Hidden size for TS mode decoder.")
    parser.add_argument("--mode_n_layers", type=int, default=3, help="Number of layers in TS mode decoder.")
    parser.add_argument("--mode_norm_threshold", type=float, default=0.02, help="Minimum ||mode|| (Angstrom) to keep an atom in the mode loss/mask.")
    parser.add_argument("--weight_mode_by_magnitude", action="store_true", help="Weight mode loss per atom by displacement magnitude (normalized per sample).")
    parser.add_argument("--geom_hidden_dim", type=int, default=None, help="Hidden dim for geometry head (defaults to mol hidden).")
    parser.add_argument("--geom_n_layers", type=int, default=1, help="Number of layers in geometry head.")
    parser.add_argument("--lambda_geom", type=float, default=1.0, help="Weight for geometry loss.")
    parser.add_argument("--lambda_mode", type=float, default=0.5, help="Weight for mode loss.")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0, help="Freeze encoder for initial epochs (mode-head warmup).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_scheduler", action="store_true", help="Use Noam-like scheduler.")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--final_lr", type=float, default=1e-4)
    parser.add_argument("--encoder", type=str, default="cmpnn", choices=["cmpnn", "dmpnn"], help="Message passing encoder.")
    parser.add_argument("--shared_enc", action="store_true", help="Share encoder weights across components.")
    parser.add_argument("--batch_norm", action="store_true", help="Enable batch norm in encoder (beware tiny batches).")
    parser.add_argument("--equivariant_mode_head", action="store_true", help="Use EGNN-based TS mode head driven by reactant coords (or TS guesses if ts_guess_dir is set).")
    parser.add_argument("--ts_guess_dir", type=str, default=None, help="Directory containing *_ts_guess.xyz files; if set, EGNN uses these TS coords instead of reactant coords.")
    parser.add_argument("--save_loss_plot", action="store_true", help="Plot train/val/test losses over epochs after run.")
    parser.add_argument("--cross_edge_cutoff", type=float, default=3.0, help="Distance cutoff (Angstrom) for adding cross edges between reactants (0 to disable, used when --cross_edge_mode=cutoff).")
    parser.add_argument("--cross_edge_mode", type=str, default="cutoff", choices=["cutoff", "reactive"], help="How to build cross edges: distance cutoff or reactive-core labels.")
    parser.add_argument("--center_complex", action="store_true", help="Center combined reactant coords before EGNN.")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_random_baseline", action="store_true", help="Evaluate random TS mode baseline on test split.")
    parser.add_argument("--plot_metrics", action="store_true", help="Plot train/val/test losses from CSV logs.")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Directory for CSV logs.")
    parser.add_argument("--checkpoint_monitor", type=str, default="val_loss_mode", help="Metric to monitor for checkpointing.")
    parser.add_argument("--checkpoint_mode", type=str, default="min", choices=["min", "max"], help="Direction for checkpoint monitor.")
    parser.add_argument("--save_top_k", type=int, default=1, help="How many best checkpoints to keep.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path to resume training (Stage B).")
    parser.add_argument("--debug_feats", action="store_true", help="Print feature dimensions for the first graph.")
    args = parser.parse_args()

    main(args)
