from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import Tensor, nn

from chemprop.data import BatchMolGraph
from chemprop.nn import Aggregation, MessagePassing, MulticomponentMessagePassing
from chemprop.nn.transforms import ScaleTransform


class TransitionStateEncoder(nn.Module):
    """Wraps message passing + aggregation to produce atom-level tensors for TS heads.

    This encoder normalizes (optional batch norm), aggregates to a pooled molecular
    representation, and pads atom representations with a mask so downstream heads can
    operate on dense `[B, N_max, H]` tensors. It supports both single- and multi-component
    message passing (shared or separate blocks via :class:`MulticomponentMessagePassing`).
    """

    def __init__(
        self,
        message_passing: MessagePassing | MulticomponentMessagePassing,
        agg: Aggregation,
        batch_norm: bool = False,
        X_d_transform: ScaleTransform | None = None,
    ) -> None:
        super().__init__()
        self.message_passing = message_passing
        self.agg = agg
        self.bn = (
            nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        )
        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

    def _apply_bn(self, mol_repr: Tensor) -> Tensor:
        """Applies batch norm if enabled and batch size > 1; otherwise returns input."""
        if isinstance(self.bn, nn.BatchNorm1d) and mol_repr.size(0) > 1:
            return self.bn(mol_repr)
        return mol_repr

    def forward(
        self,
        batch: dict[str, Tensor | Sequence[BatchMolGraph] | Sequence[Tensor | None]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Runs message passing and aggregation, returning padded atoms, pooled mols, mask."""

        if isinstance(self.message_passing, MulticomponentMessagePassing):
            return self._forward_multicomponent(batch)
        return self._forward_single(batch)

    def _forward_single(self, batch: dict[str, Tensor | BatchMolGraph | None]) -> Tuple[Tensor, Tensor, Tensor]:
        bmg: BatchMolGraph = batch["bmg"]  # type: ignore[assignment]
        V_d: Tensor | None = batch.get("V_d")  # type: ignore[assignment]
        X_d: Tensor | None = batch.get("X_d")  # type: ignore[assignment]

        atom_repr = self.message_passing(bmg, V_d)  # type: ignore[arg-type]
        mol_repr = self.agg(atom_repr, bmg.batch)
        mol_repr = self._apply_bn(mol_repr)
        if X_d is not None:
            mol_repr = torch.cat((mol_repr, self.X_d_transform(X_d)), dim=1)

        atom_repr_padded, atom_mask = self._pad_atoms(atom_repr, bmg.batch, len(bmg))
        return atom_repr_padded, mol_repr, atom_mask

    def _forward_multicomponent(
        self, batch: dict[str, Tensor | Sequence[BatchMolGraph] | Sequence[Tensor | None]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        bmgs: Sequence[BatchMolGraph] = batch["bmgs"]  # type: ignore[assignment]
        V_ds = batch.get("V_ds")  # type: ignore[assignment]
        X_d: Tensor | None = batch.get("X_d")  # type: ignore[assignment]

        atom_reprs: list[Tensor] = self.message_passing(bmgs, V_ds)  # type: ignore[arg-type]
        mol_reprs = [self.agg(atom_repr, bmg.batch) for atom_repr, bmg in zip(atom_reprs, bmgs)]
        mol_repr = torch.cat(mol_reprs, dim=1)
        mol_repr = self._apply_bn(mol_repr)
        if X_d is not None:
            mol_repr = torch.cat((mol_repr, self.X_d_transform(X_d)), dim=1)

        atom_dim = atom_reprs[0].shape[-1]
        if any(rep.shape[-1] != atom_dim for rep in atom_reprs[1:]):
            raise ValueError(
                "All message-passing blocks must produce the same atom embedding size for the "
                "transition-state mode head. Set consistent hidden sizes or add a projection."
            )

        all_atom_repr = torch.cat(atom_reprs, dim=0)
        all_atom_batches = torch.cat([bmg.batch.to(all_atom_repr.device) for bmg in bmgs], dim=0)
        num_graphs = len(bmgs[0])
        atom_repr_padded, atom_mask = self._pad_atoms(all_atom_repr, all_atom_batches, num_graphs)
        return atom_repr_padded, mol_repr, atom_mask

    @staticmethod
    def _pad_atoms(atom_repr: Tensor, batch_idx: Tensor, num_graphs: int) -> Tuple[Tensor, Tensor]:
        """Pads variable-length atom embeddings to `[B, N_max, H]` with a boolean mask."""

        counts = torch.bincount(batch_idx, minlength=num_graphs)
        max_atoms = int(counts.max().item()) if counts.numel() > 0 else 0
        H = atom_repr.shape[-1]
        device = atom_repr.device

        padded = atom_repr.new_zeros((num_graphs, max_atoms, H))
        mask = torch.zeros((num_graphs, max_atoms), dtype=torch.bool, device=device)

        if max_atoms == 0:
            return padded, mask

        starts = torch.cat(
            (
                torch.tensor([0], device=device, dtype=torch.long),
                counts.cumsum(0, dtype=torch.long)[:-1],
            )
        )

        for i in range(num_graphs):
            n_atoms = int(counts[i].item())
            if n_atoms == 0:
                continue
            s = int(starts[i].item())
            padded[i, :n_atoms] = atom_repr[s : s + n_atoms]
            mask[i, :n_atoms] = True

        return padded, mask
