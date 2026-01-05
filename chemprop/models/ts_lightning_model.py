"""Lightning module for multitask transition-state prediction built on a Chemprop encoder.

The module wraps an existing Chemprop-style encoder and adds two prediction heads:
1) a graph-level geometry head operating on `mol_repr`
2) a per-atom transition-state mode head operating on `atom_repr`
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

from lightning import pytorch as pl
from torch import Tensor, nn, optim
from torch.nn import functional as F

from chemprop.nn import TransitionStateModeHead, EquivariantTSModeHead
from chemprop.nn.ffn import MLP
from chemprop.schedulers import build_NoamLike_LRSched


class TSLightningModel(pl.LightningModule):
    """Multi-task LightningModule for transition-state geometry and mode prediction.

    Parameters
    ----------
    encoder : nn.Module
        Predefined Chemprop encoder returning (atom_repr, mol_repr, atom_mask).
    hidden_dim : int
        Dimensionality of encoder outputs (`H`).
    n_geom_features : int
        Size of the graph-level geometry target vector.
    geom_hidden_dim : int | None, default=None
        Hidden dimension for the geometry head (defaults to encoder hidden_dim if None).
    geom_n_layers : int, default=1
        Number of hidden layers in the geometry head.
    mode_hidden_dim : int, default=64
        Hidden dimension used in the intermediate layers of the transition-state mode head.
    mode_n_layers : int, default=2
        Number of hidden layers in the transition-state mode head.
    lambda_geom : float, default=1.0
        Weight applied to the geometry loss term.
    lambda_mode : float, default=0.5
        Weight applied to the mode-direction loss term.
    lr : float, default=1e-3
        Learning rate used by the optimizer (used directly if scheduler is disabled).
    use_scheduler : bool, default=False
        If True, use Chemprop's Noam-like LR scheduler.
    warmup_epochs : int, default=2
        Warmup epochs for the scheduler (only if `use_scheduler`).
    init_lr : float, default=1e-4
        Initial LR for Noam-like scheduler.
    max_lr : float, default=1e-3
        Peak LR for Noam-like scheduler.
    final_lr : float, default=1e-4
        Final LR for Noam-like scheduler.
    optimizer_cls : type[optim.Optimizer], default=optim.Adam
        Optimizer constructor; defaults to Adam to mirror Chemprop defaults but can be swapped.
    optimizer_kwargs : dict[str, Any] | None, default=None
        Optional kwargs forwarded to the optimizer constructor.
    eps : float, default=1e-8
        Numerical stability constant used during normalization.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        n_geom_features: int,
        geom_hidden_dim: int | None = None,
        geom_n_layers: int = 1,
        mode_hidden_dim: int = 64,
        mode_n_layers: int = 2,
        equivariant_mode_head: bool = False,
        lambda_geom: float = 1.0,
        lambda_mode: float = 0.5,
        lr: float = 1e-3,
        use_scheduler: bool = False,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        optimizer_cls: Type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.save_hyperparameters(
            ignore=["encoder", "optimizer_cls", "optimizer_kwargs"]
        )
        self.lambda_mode = lambda_mode
        self.lambda_geom = lambda_geom
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.eps = eps

        self.geom_head = MLP.build(
            input_dim=hidden_dim,
            output_dim=n_geom_features,
            hidden_dim=geom_hidden_dim or hidden_dim,
            n_layers=geom_n_layers,
            dropout=0.0,
            activation="relu",
        )
        if equivariant_mode_head:
            self.mode_head = EquivariantTSModeHead(
                d_h=hidden_dim, n_layers=mode_n_layers, d_edge=0, d_msg=mode_hidden_dim
            )
        else:
            self.mode_head = TransitionStateModeHead(
                input_dim=hidden_dim, hidden_dim=mode_hidden_dim, n_layers=mode_n_layers
            )
        self.equivariant_mode_head = equivariant_mode_head

    def forward(self, batch: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """Runs the encoder and heads to produce predictions."""
        atom_repr, mol_repr, atom_mask = self.encoder(batch)
        geom_pred = self.geom_head(mol_repr)
        if self.equivariant_mode_head:
            coords: Tensor = batch["coords"]
            edge_index: Tensor = batch["edge_index"]
            edge_mask: Tensor | None = batch.get("edge_mask")
            mode_pred = self.mode_head(coords, atom_repr, edge_index, atom_mask, edge_mask=edge_mask)
        else:
            mode_pred = self.mode_head(atom_repr)
        return geom_pred, mode_pred, atom_mask

    def _mode_loss(self, mode_pred: Tensor, mode_target: Tensor, atom_mask: Tensor) -> Tensor:
        """Masked, sign-invariant cosine loss on per-atom mode directions."""
        valid_mask = atom_mask.float()

        pred_unit = mode_pred / mode_pred.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        target_unit = mode_target / mode_target.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        cosine = (pred_unit * target_unit).sum(dim=-1).abs()
        masked_loss = (1 - cosine) * valid_mask
        normalizer = valid_mask.sum().clamp_min(self.eps)
        return masked_loss.sum() / normalizer

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        geom_pred, mode_pred, atom_mask = self(batch)

        geom_target: Tensor = batch["geom_target"]
        mode_target: Tensor = batch["mode_target"]

        loss_geom = F.mse_loss(geom_pred, geom_target)
        loss_mode = self._mode_loss(mode_pred, mode_target, atom_mask)
        loss_total = self.lambda_geom * loss_geom + self.lambda_mode * loss_mode

        batch_size = geom_target.size(0)
        self.log("train_loss", loss_total, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_loss_geom", loss_geom, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_loss_mode", loss_mode, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss_total

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        geom_pred, mode_pred, atom_mask = self(batch)

        geom_target: Tensor = batch["geom_target"]
        mode_target: Tensor = batch["mode_target"]

        loss_geom = F.mse_loss(geom_pred, geom_target)
        loss_mode = self._mode_loss(mode_pred, mode_target, atom_mask)
        loss_total = self.lambda_geom * loss_geom + self.lambda_mode * loss_mode

        batch_size = geom_target.size(0)
        self.log("val_loss", loss_total, prog_bar=True, batch_size=batch_size)
        self.log("val_loss_geom", loss_geom, batch_size=batch_size)
        self.log("val_loss_mode", loss_mode, batch_size=batch_size)
        return loss_total

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        geom_pred, mode_pred, atom_mask = self(batch)

        geom_target: Tensor = batch["geom_target"]
        mode_target: Tensor = batch["mode_target"]

        loss_geom = F.mse_loss(geom_pred, geom_target)
        loss_mode = self._mode_loss(mode_pred, mode_target, atom_mask)
        loss_total = self.lambda_geom * loss_geom + self.lambda_mode * loss_mode

        batch_size = geom_target.size(0)
        self.log("test_loss", loss_total, prog_bar=True, batch_size=batch_size)
        self.log("test_loss_geom", loss_geom, batch_size=batch_size)
        self.log("test_loss_mode", loss_mode, batch_size=batch_size)

        # mean |cos(theta)| over valid atoms for interpretability
        pred_unit = mode_pred / mode_pred.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        target_unit = mode_target / mode_target.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        cosine = (pred_unit * target_unit).sum(dim=-1).abs()
        mask = atom_mask.bool()
        if mask.any():
            mean_abs_cos = cosine[mask].mean()
            self.log("test_mean_abs_cos_mode", mean_abs_cos, batch_size=batch_size)
        return loss_total

    def configure_optimizers(self):
        opt = self.optimizer_cls(self.parameters(), lr=self.init_lr if self.use_scheduler else self.lr, **self.optimizer_kwargs)

        if not self.use_scheduler:
            return opt

        if self.trainer.train_dataloader is None:
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        return {"optimizer": opt, "lr_scheduler": {"scheduler": lr_sched, "interval": "step"}}
