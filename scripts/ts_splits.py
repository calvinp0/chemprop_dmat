from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


ROLE_ORDER = ["*0", "*1", "*2", "*3", "*4"]


@dataclass(frozen=True)
class SplitConfig:
    splitter: str
    split_sizes: tuple[float, float, float]
    seed: int
    group_kfolds: int = 5
    group_test_fold: int = 0
    split_radius: int = 2
    split_nbits: int = 2048
    split_use_chirality: bool = True
    holdout_donor_element: str | None = None
    holdout_acceptor_element: str | None = None


def _get_role_map(mol: Chem.Mol) -> dict[str, int | None]:
    if mol.HasProp("role_order") and mol.HasProp("role_indices_ordered"):
        role_order = json.loads(mol.GetProp("role_order"))
        role_indices = json.loads(mol.GetProp("role_indices_ordered"))
        return dict(zip(role_order, role_indices))
    return {role: None for role in ROLE_ORDER}


def _center_fp_onbits(
    mol: Chem.Mol, atom_idx: int | None, radius: int, nbits: int, use_chirality: bool
) -> tuple[int, ...] | tuple[str]:
    if atom_idx is None:
        return "NONE"
    generator = _get_morgan_generator(radius, nbits, use_chirality)
    bv = generator.GetFingerprint(mol, fromAtoms=[atom_idx])
    return tuple(bv.GetOnBits())


@lru_cache(maxsize=None)
def _get_morgan_generator(radius: int, nbits: int, use_chirality: bool):
    return GetMorganGenerator(radius=radius, fpSize=nbits, includeChirality=use_chirality)


def reaction_center_signature(
    mol: Chem.Mol,
    role_keys: Iterable[str],
    radius: int = 2,
    nbits: int = 2048,
    use_chirality: bool = True,
) -> str:
    role_map = _get_role_map(mol)
    parts = []
    for role in role_keys:
        idx = role_map.get(role)
        elem = mol.GetAtomWithIdx(idx).GetSymbol() if idx is not None else "NONE"
        bits = _center_fp_onbits(mol, idx, radius, nbits, use_chirality)
        parts.append((role, elem, bits))
    raw = repr(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def donor_acceptor_pair(mol: Chem.Mol) -> str:
    role_map = _get_role_map(mol)
    donor_idx = role_map.get("*1")
    acceptor_idx = role_map.get("*3")
    donor = mol.GetAtomWithIdx(donor_idx).GetSymbol() if donor_idx is not None else "UNK"
    acceptor = mol.GetAtomWithIdx(acceptor_idx).GetSymbol() if acceptor_idx is not None else "UNK"
    return f"{donor},{acceptor}"


def donor_element(mol: Chem.Mol) -> str:
    role_map = _get_role_map(mol)
    donor_idx = role_map.get("*1")
    return mol.GetAtomWithIdx(donor_idx).GetSymbol() if donor_idx is not None else "UNK"


def acceptor_element(mol: Chem.Mol) -> str:
    role_map = _get_role_map(mol)
    acceptor_idx = role_map.get("*3")
    return mol.GetAtomWithIdx(acceptor_idx).GetSymbol() if acceptor_idx is not None else "UNK"


def _group_split_indices(groups: list[str], sizes: tuple[float, float, float], seed: int):
    test_size = sizes[2]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idxs = list(range(len(groups)))
    train_val_idx, test_idx = next(splitter.split(idxs, groups=groups))

    train_val_groups = [groups[i] for i in train_val_idx]
    val_size = sizes[1] / (sizes[0] + sizes[1]) if (sizes[0] + sizes[1]) > 0 else 0.0
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
    train_idx_rel, val_idx_rel = next(splitter.split(train_val_idx, groups=train_val_groups))
    train_idx = [train_val_idx[i] for i in train_idx_rel]
    val_idx = [train_val_idx[i] for i in val_idx_rel]
    return train_idx, val_idx, test_idx.tolist()


def _group_kfold_test_split(
    groups: list[str],
    sizes: tuple[float, float, float],
    seed: int,
    kfolds: int,
    test_fold: int,
):
    n_groups = len(set(groups))
    if n_groups < kfolds:
        raise ValueError(f"Need >= {kfolds} groups for GroupKFold, have {n_groups}")
    kfold = GroupKFold(n_splits=kfolds)
    idxs = list(range(len(groups)))
    folds = list(kfold.split(idxs, groups=groups))
    test_fold = test_fold % kfolds
    train_val_idx, test_idx = folds[test_fold]
    train_val_groups = [groups[i] for i in train_val_idx]
    val_size = sizes[1] / (sizes[0] + sizes[1]) if (sizes[0] + sizes[1]) > 0 else 0.0
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx_rel, val_idx_rel = next(splitter.split(train_val_idx, groups=train_val_groups))
    train_idx = [train_val_idx[i] for i in train_idx_rel]
    val_idx = [train_val_idx[i] for i in val_idx_rel]
    return train_idx, val_idx, test_idx.tolist()


def _log_group_split(groups: list[str], train_idx, val_idx, test_idx):
    all_groups = set(groups)
    train_groups = set(groups[i] for i in train_idx)
    val_groups = set(groups[i] for i in val_idx)
    test_groups = set(groups[i] for i in test_idx)
    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    total = len(groups)
    print(
        "Split diagnostics:"
        f" samples train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
        f" ({len(train_idx)/total:.2%}/{len(val_idx)/total:.2%}/{len(test_idx)/total:.2%})"
    )
    print(
        "Split diagnostics:"
        f" groups train/val/test = {len(train_groups)}/{len(val_groups)}/{len(test_groups)}"
        f" (total {len(all_groups)})"
    )
    if overlap:
        print(f"Split diagnostics: WARNING group overlap detected ({len(overlap)} groups).")


def make_split_indices(
    mols: list[Chem.Mol],
    config: SplitConfig,
    add_adj_roles: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    role_keys = ROLE_ORDER if add_adj_roles else ["*1", "*2", "*3"]

    if config.splitter == "reaction_center":
        groups = [
            reaction_center_signature(
                mol,
                role_keys,
                radius=config.split_radius,
                nbits=config.split_nbits,
                use_chirality=config.split_use_chirality,
            )
            for mol in mols
        ]
        train_idx, val_idx, test_idx = _group_kfold_test_split(
            groups, config.split_sizes, config.seed, config.group_kfolds, config.group_test_fold
        )
        _log_group_split(groups, train_idx, val_idx, test_idx)
        return train_idx, val_idx, test_idx

    if config.splitter == "donor_acceptor_pair":
        groups = [donor_acceptor_pair(mol) for mol in mols]
        train_idx, val_idx, test_idx = _group_split_indices(groups, config.split_sizes, config.seed)
        _log_group_split(groups, train_idx, val_idx, test_idx)
        return train_idx, val_idx, test_idx

    if config.splitter == "donor_element":
        groups = [donor_element(mol) for mol in mols]
        if config.holdout_donor_element:
            holdout = config.holdout_donor_element
            test_idx = [i for i, g in enumerate(groups) if g == holdout]
            train_val_idx = [i for i, g in enumerate(groups) if g != holdout]
            if not test_idx:
                raise ValueError(f"No samples found for holdout donor element '{holdout}'")
            if not train_val_idx:
                raise ValueError(f"No training samples left after holdout donor element '{holdout}'")
            train_val_groups = [groups[i] for i in train_val_idx]
            val_size = config.split_sizes[1] / (config.split_sizes[0] + config.split_sizes[1])
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=config.seed)
            train_idx_rel, val_idx_rel = next(splitter.split(train_val_idx, groups=train_val_groups))
            train_idx = [train_val_idx[i] for i in train_idx_rel]
            val_idx = [train_val_idx[i] for i in val_idx_rel]
            _log_group_split(groups, train_idx, val_idx, test_idx)
            return train_idx, val_idx, test_idx
        train_idx, val_idx, test_idx = _group_split_indices(groups, config.split_sizes, config.seed)
        _log_group_split(groups, train_idx, val_idx, test_idx)
        return train_idx, val_idx, test_idx

    if config.splitter == "acceptor_element":
        groups = [acceptor_element(mol) for mol in mols]
        if config.holdout_acceptor_element:
            holdout = config.holdout_acceptor_element
            test_idx = [i for i, g in enumerate(groups) if g == holdout]
            train_val_idx = [i for i, g in enumerate(groups) if g != holdout]
            if not test_idx:
                raise ValueError(f"No samples found for holdout acceptor element '{holdout}'")
            if not train_val_idx:
                raise ValueError(f"No training samples left after holdout acceptor element '{holdout}'")
            train_val_groups = [groups[i] for i in train_val_idx]
            val_size = config.split_sizes[1] / (config.split_sizes[0] + config.split_sizes[1])
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=config.seed)
            train_idx_rel, val_idx_rel = next(splitter.split(train_val_idx, groups=train_val_groups))
            train_idx = [train_val_idx[i] for i in train_idx_rel]
            val_idx = [train_val_idx[i] for i in val_idx_rel]
            _log_group_split(groups, train_idx, val_idx, test_idx)
            return train_idx, val_idx, test_idx
        train_idx, val_idx, test_idx = _group_split_indices(groups, config.split_sizes, config.seed)
        _log_group_split(groups, train_idx, val_idx, test_idx)
        return train_idx, val_idx, test_idx

    raise ValueError(f"Unknown splitter '{config.splitter}'.")
