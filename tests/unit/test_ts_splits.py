import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))

from ts_hpo import SearchSpace, build_model, build_loaders, fit_scaler_ignore_nan  # noqa: E402
from ts_splits import (  # noqa: E402
    SplitConfig,
    make_split_indices,
    reaction_center_signature,
    donor_acceptor_pair,
)
from chemprop import data, featurizers


class DummyTrial:
    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, low, high, step=1):
        return self.params[name]

    def suggest_float(self, name, low, high, log=False):
        return self.params[name]

    def suggest_categorical(self, name, choices):
        return self.params[name]


def _make_dp(smiles: str, name: str, role_indices: list[int | None], y: np.ndarray, v_f_dim: int):
    mol = Chem.MolFromSmiles(smiles)
    role_order = ["*0", "*1", "*2", "*3", "*4"]
    mol.SetProp("role_order", json.dumps(role_order))
    mol.SetProp("role_indices_ordered", json.dumps(role_indices))
    n_atoms = mol.GetNumAtoms()
    v_f = np.zeros((n_atoms, v_f_dim), dtype=np.float32)
    role_map = dict(zip(role_order, role_indices))
    for col, role in enumerate(["*1", "*2", "*3"]):
        idx = role_map.get(role)
        if idx is not None:
            v_f[idx, col] = 1.0
    return data.MoleculeDatapoint(mol, name=name, y=y, V_f=v_f)


def _toy_dataset(v_f_dim: int = 3):
    ys = [
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.2, 0.3], dtype=np.float32),
        np.array([0.3, 0.4], dtype=np.float32),
        np.array([0.4, 0.5], dtype=np.float32),
        np.array([0.5, 0.6], dtype=np.float32),
        np.array([0.6, 0.7], dtype=np.float32),
    ]
    smiles_list = ["CCO", "CCN", "CCC", "CCS", "COC", "CNC"]
    dps = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        n_atoms = mol.GetNumAtoms()
        role_indices = [
            0,
            1 if n_atoms > 1 else 0,
            2 if n_atoms > 2 else 0,
            2 if n_atoms > 2 else 0,
            0,
        ]
        dps.append(_make_dp(smi, f"mol_{i}", role_indices, ys[i], v_f_dim))
    return dps


def test_reaction_center_split_no_overlap():
    dps = _toy_dataset()
    mols = [d.mol for d in dps]
    cfg = SplitConfig(
        splitter="reaction_center",
        split_sizes=(0.8, 0.1, 0.1),
        seed=7,
        group_kfolds=3,
        group_test_fold=0,
        split_radius=2,
    )
    train_idx, val_idx, test_idx = make_split_indices(mols, cfg, add_adj_roles=False)

    groups = [
        reaction_center_signature(mol, ["*1", "*2", "*3"], radius=2, nbits=2048, use_chirality=True)
        for mol in mols
    ]
    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    test_groups = {groups[i] for i in test_idx}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(dps)


def _train_one_step(dps, split_indices, extra_atom_fdim: int, batch_size: int = 1):
    mols = [d.mol for d in dps]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra_atom_fdim)
    train_dset, val_dset, test_dset, train_loader, _, _ = build_loaders(
        dps, featurizer, batch_size=batch_size, seed=3, split_indices=split_indices
    )

    y_train_raw = train_dset._Y
    scaler = fit_scaler_ignore_nan(y_train_raw)
    train_dset.Y = (y_train_raw - scaler.mean_) / scaler.scale_
    val_dset.Y = (val_dset._Y - scaler.mean_) / scaler.scale_
    test_dset.Y = test_dset._Y

    trial = DummyTrial(
        {
            "hidden_dim": 32,
            "mp_depth": 2,
            "dropout": 0.0,
            "lr": 1e-3,
            "huber_beta": 1.0,
            "ffn_layers": 1,
            "weight_decay": 0.0,
            "agg": "mean",
        }
    )
    model = build_model(
        featurizer,
        scaler,
        y_dim=2,
        trial=trial,
        space=SearchSpace(hidden_min=32, hidden_max=32, depth_min=2, depth_max=2, batch_norm=False),
        x_d_dim=0,
        task_weights=None,
        x_d_transform=None,
    )
    batch = next(iter(train_loader))
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss).item()


def test_build_loaders_and_model_forward():
    dps = _toy_dataset()
    mols = [d.mol for d in dps]
    cfg = SplitConfig(
        splitter="donor_acceptor_pair",
        split_sizes=(0.8, 0.1, 0.1),
        seed=3,
    )
    split_indices = make_split_indices(mols, cfg, add_adj_roles=False)
    _train_one_step(dps, split_indices, extra_atom_fdim=3)


def test_donor_element_holdout_split():
    dps = _toy_dataset()
    mols = [d.mol for d in dps]
    cfg = SplitConfig(
        splitter="donor_element",
        split_sizes=(0.8, 0.1, 0.1),
        seed=11,
        holdout_donor_element="O",
    )
    train_idx, val_idx, test_idx = make_split_indices(mols, cfg, add_adj_roles=False)
    assert len(test_idx) > 0
    assert set(test_idx).isdisjoint(train_idx)
    assert set(test_idx).isdisjoint(val_idx)
    _train_one_step(dps, (train_idx, val_idx, test_idx), extra_atom_fdim=3)


def test_acceptor_element_holdout_split():
    dps = _toy_dataset()
    mols = [d.mol for d in dps]
    cfg = SplitConfig(
        splitter="acceptor_element",
        split_sizes=(0.8, 0.1, 0.1),
        seed=13,
        holdout_acceptor_element="O",
    )
    train_idx, val_idx, test_idx = make_split_indices(mols, cfg, add_adj_roles=False)
    assert len(test_idx) > 0
    assert set(test_idx).isdisjoint(train_idx)
    assert set(test_idx).isdisjoint(val_idx)
    _train_one_step(dps, (train_idx, val_idx, test_idx), extra_atom_fdim=3)


def test_donor_acceptor_pair_split_no_overlap():
    dps = _toy_dataset()
    mols = [d.mol for d in dps]
    cfg = SplitConfig(
        splitter="donor_acceptor_pair",
        split_sizes=(0.8, 0.1, 0.1),
        seed=5,
    )
    train_idx, val_idx, test_idx = make_split_indices(mols, cfg, add_adj_roles=False)
    groups = [donor_acceptor_pair(m) for m in mols]
    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    test_groups = {groups[i] for i in test_idx}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    _train_one_step(dps, (train_idx, val_idx, test_idx), extra_atom_fdim=3)


def test_reaction_center_split_with_adj_roles_trains():
    dps = _toy_dataset(v_f_dim=5)
    mols = [d.mol for d in dps]
    cfg = SplitConfig(
        splitter="reaction_center",
        split_sizes=(0.8, 0.1, 0.1),
        seed=9,
        group_kfolds=3,
        group_test_fold=1,
        split_radius=2,
    )
    train_idx, val_idx, test_idx = make_split_indices(mols, cfg, add_adj_roles=True)
    _train_one_step(dps, (train_idx, val_idx, test_idx), extra_atom_fdim=5)
