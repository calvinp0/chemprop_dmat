from abc import abstractmethod

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.nn.ffn import MLP
from chemprop.nn.hparams import HasHParams
from chemprop.nn.metrics import (
    MSE,
    SID,
    BCELoss,
    BinaryAUROC,
    ChempropMetric,
    CrossEntropyLoss,
    DirichletLoss,
    EvidentialLoss,
    MulticlassMCCMetric,
    MVELoss,
    QuantileLoss,
)
from chemprop.nn.transforms import UnscaleTransform
from chemprop.utils import ClassRegistry, Factory

__all__ = [
    "Predictor",
    "PredictorRegistry",
    "RegressionFFN",
    "MveFFN",
    "EvidentialFFN",
    "BinaryClassificationFFNBase",
    "BinaryClassificationFFN",
    "BinaryDirichletFFN",
    "MulticlassClassificationFFN",
    "MulticlassDirichletFFN",
    "SpectralFFN",
    "TransitionStateModeHead",
    "EGNNLayer",
    "EquivariantTSModeHead",
]


class Predictor(nn.Module, HasHParams):
    r"""A :class:`Predictor` is a protocol that defines a differentiable function
    :math:`f` : \mathbb R^d \mapsto \mathbb R^o"""

    input_dim: int
    """the input dimension"""
    output_dim: int
    """the output dimension"""
    n_tasks: int
    """the number of tasks `t` to predict for each input"""
    n_targets: int
    """the number of targets `s` to predict for each task `t`"""
    criterion: ChempropMetric
    """the loss function to use for training"""
    task_weights: Tensor
    """the weights to apply to each task when calculating the loss"""
    output_transform: UnscaleTransform
    """the transform to apply to the output of the predictor"""

    @abstractmethod
    def forward(self, Z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def train_step(self, Z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def encode(self, Z: Tensor, i: int) -> Tensor:
        """Calculate the :attr:`i`-th hidden representation

        Parameters
        ----------
        Z : Tensor
            a tensor of shape ``n x d`` containing the input data to encode, where ``d`` is the
            input dimensionality.
        i : int
            The stop index of slice of the MLP used to encode the input. That is, use all
            layers in the MLP *up to* :attr:`i` (i.e., ``MLP[:i]``). This can be any integer
            value, and the behavior of this function is dependent on the underlying list
            slicing behavior. For example:

            * ``i=0``: use a 0-layer MLP (i.e., a no-op)
            * ``i=1``: use only the first block
            * ``i=-1``: use *up to* the final block

        Returns
        -------
        Tensor
            a tensor of shape ``n x h`` containing the :attr:`i`-th hidden representation, where
            ``h`` is the number of neurons in the :attr:`i`-th hidden layer.
        """
        pass


PredictorRegistry = ClassRegistry[Predictor]()


class _FFNPredictorBase(Predictor, HyperparametersMixin):
    """A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
    underlying :class:`MLP` to map the learned fingerprint to the desired output.
    """

    _T_default_criterion: ChempropMetric
    _T_default_metric: ChempropMetric

    def __init__(
        self,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str | nn.Module = "relu",
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        super().__init__()
        # manually add criterion and output_transform to hparams to suppress lightning's warning
        # about double saving their state_dict values.
        ignore_list = ["criterion", "output_transform", "activation"]
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams["criterion"] = criterion
        self.hparams["output_transform"] = output_transform
        self.hparams["activation"] = activation
        self.hparams["cls"] = self.__class__

        self.ffn = MLP.build(
            input_dim, n_tasks * self.n_targets, hidden_dim, n_layers, dropout, activation
        )
        task_weights = torch.ones(n_tasks) if task_weights is None else task_weights
        self.criterion = criterion or Factory.build(
            self._T_default_criterion, task_weights=task_weights, threshold=threshold
        )
        self.output_transform = output_transform if output_transform is not None else nn.Identity()

    @property
    def input_dim(self) -> int:
        return self.ffn.input_dim

    @property
    def output_dim(self) -> int:
        return self.ffn.output_dim

    @property
    def n_tasks(self) -> int:
        return self.output_dim // self.n_targets

    def forward(self, Z: Tensor) -> Tensor:
        return self.ffn(Z)

    def encode(self, Z: Tensor, i: int) -> Tensor:
        return self.ffn[:i](Z)


@PredictorRegistry.register("regression")
class RegressionFFN(_FFNPredictorBase):
    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def forward(self, Z: Tensor) -> Tensor:
        return self.output_transform(self.ffn(Z))

    train_step = forward


@PredictorRegistry.register("regression-mve")
class MveFFN(RegressionFFN):
    n_targets = 2
    _T_default_criterion = MVELoss

    def forward(self, Z: Tensor) -> Tensor:
        Y = self.ffn(Z)
        mean, var = torch.chunk(Y, self.n_targets, 1)
        var = F.softplus(var)

        mean = self.output_transform(mean)
        if not isinstance(self.output_transform, nn.Identity):
            var = self.output_transform.transform_variance(var)

        return torch.stack((mean, var), dim=2)

    train_step = forward


@PredictorRegistry.register("regression-evidential")
class EvidentialFFN(RegressionFFN):
    n_targets = 4
    _T_default_criterion = EvidentialLoss

    def forward(self, Z: Tensor) -> Tensor:
        Y = self.ffn(Z)
        mean, v, alpha, beta = torch.chunk(Y, self.n_targets, 1)
        v = F.softplus(v)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)

        mean = self.output_transform(mean)
        if not isinstance(self.output_transform, nn.Identity):
            beta = self.output_transform.transform_variance(beta)

        return torch.stack((mean, v, alpha, beta), dim=2)

    train_step = forward


@PredictorRegistry.register("regression-quantile")
class QuantileFFN(RegressionFFN):
    n_targets = 2
    _T_default_criterion = QuantileLoss

    def forward(self, Z: Tensor) -> Tensor:
        lower_bound, upper_bound = torch.chunk(self.ffn(Z), self.n_targets, 1)

        lower_bound = self.output_transform(lower_bound)
        upper_bound = self.output_transform(upper_bound)

        mean = (lower_bound + upper_bound) / 2
        interval = upper_bound - lower_bound

        return torch.stack((mean, interval), dim=2)

    train_step = forward


class BinaryClassificationFFNBase(_FFNPredictorBase):
    pass


@PredictorRegistry.register("classification")
class BinaryClassificationFFN(BinaryClassificationFFNBase):
    n_targets = 1
    _T_default_criterion = BCELoss
    _T_default_metric = BinaryAUROC

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)

        return Y.sigmoid()

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z)


@PredictorRegistry.register("classification-dirichlet")
class BinaryDirichletFFN(BinaryClassificationFFNBase):
    n_targets = 2
    _T_default_criterion = DirichletLoss
    _T_default_metric = BinaryAUROC

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z).reshape(len(Z), -1, 2)

        alpha = F.softplus(Y) + 1

        u = 2 / alpha.sum(-1)
        Y = alpha / alpha.sum(-1, keepdim=True)

        return torch.stack((Y[..., 1], u), dim=2)

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z).reshape(len(Z), -1, 2)

        return F.softplus(Y) + 1


@PredictorRegistry.register("multiclass")
class MulticlassClassificationFFN(_FFNPredictorBase):
    n_targets = 1
    _T_default_criterion = CrossEntropyLoss
    _T_default_metric = MulticlassMCCMetric

    def __init__(
        self,
        n_classes: int,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str | nn.Module = "relu",
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        task_weights = torch.ones(n_tasks) if task_weights is None else task_weights
        super().__init__(
            n_tasks * n_classes,
            input_dim,
            hidden_dim,
            n_layers,
            dropout,
            activation,
            criterion,
            task_weights,
            threshold,
            output_transform,
        )

        self.n_classes = n_classes

    @property
    def n_tasks(self) -> int:
        return self.output_dim // (self.n_targets * self.n_classes)

    def forward(self, Z: Tensor) -> Tensor:
        return super().forward(Z).reshape(Z.shape[0], -1, self.n_classes).softmax(-1)

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z).reshape(Z.shape[0], -1, self.n_classes)


@PredictorRegistry.register("multiclass-dirichlet")
class MulticlassDirichletFFN(MulticlassClassificationFFN):
    _T_default_criterion = DirichletLoss
    _T_default_metric = MulticlassMCCMetric

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().train_step(Z)

        alpha = F.softplus(Y) + 1

        Y = alpha / alpha.sum(-1, keepdim=True)

        return Y

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().train_step(Z)

        return F.softplus(Y) + 1


class _Exp(nn.Module):
    def forward(self, X: Tensor):
        return X.exp()


@PredictorRegistry.register("spectral")
class SpectralFFN(_FFNPredictorBase):
    n_targets = 1
    _T_default_criterion = SID
    _T_default_metric = SID

    def __init__(self, *args, spectral_activation: str | None = "softplus", **kwargs):
        super().__init__(*args, **kwargs)

        match spectral_activation:
            case "exp":
                spectral_activation = _Exp()
            case "softplus" | None:
                spectral_activation = nn.Softplus()
            case _:
                raise ValueError(
                    f"Unknown spectral activation: {spectral_activation}. "
                    "Expected one of 'exp', 'softplus' or None."
                )

        self.ffn.add_module("spectral_activation", spectral_activation)

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        Y = self.ffn.spectral_activation(Y)
        return Y / Y.sum(1, keepdim=True)

    train_step = forward


class TransitionStateModeHead(nn.Module, HasHParams, HyperparametersMixin):
    """Per-atom predictor head that maps atom embeddings to 3D mode vectors."""

    def __init__(
        self,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1 for TransitionStateModeHead.")
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, atom_repr: Tensor) -> Tensor:
        """Predicts per-atom transition-state modes preserving leading dimensions."""

        return self.net(atom_repr)


class EGNNLayer(nn.Module):
    """Lightweight EGNN-style layer updating coords and scalar node features."""

    def __init__(self, d_h: int, d_edge: int = 0, d_msg: int = 64):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * d_h + 1 + d_edge, d_msg),
            nn.SiLU(),
            nn.Linear(d_msg, d_msg),
            nn.SiLU(),
        )
        self.phi_x = nn.Linear(d_msg, 1, bias=False)
        self.phi_h = nn.Sequential(
            nn.Linear(d_h + d_msg, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )

    def forward(
        self,
        x: Tensor,  # [B, N, 3]
        h: Tensor,  # [B, N, d_h]
        edge_index: Tensor,  # [B, E, 2] or [2, E]
        edge_attr: Tensor | None = None,  # [B, E, d_edge] or None
        mask: Tensor | None = None,  # [B, N]
        edge_mask: Tensor | None = None,  # [B, E] optional
    ) -> tuple[Tensor, Tensor]:
        if edge_index.dim() == 2:
            edge_index = edge_index.unsqueeze(0).expand(x.size(0), -1, -1)
        B, E, _ = edge_index.shape
        if E == 0:
            return x, h

        src = edge_index[:, :, 0]  # [B, E]
        dst = edge_index[:, :, 1]  # [B, E]

        x_i = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, 3))
        x_j = torch.gather(x, 1, dst.unsqueeze(-1).expand(-1, -1, 3))
        h_i = torch.gather(h, 1, src.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        h_j = torch.gather(h, 1, dst.unsqueeze(-1).expand(-1, -1, h.size(-1)))

        diff = x_i - x_j
        dist2 = (diff**2).sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [B, E, 1]

        if edge_attr is not None:
            edge_input = torch.cat([h_i, h_j, dist2, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h_i, h_j, dist2], dim=-1)
        edge_input = torch.nan_to_num(edge_input, nan=0.0, posinf=1e4, neginf=-1e4)

        m_ij = self.phi_e(edge_input)  # [B, E, d_msg]
        if edge_mask is not None:
            m_ij = m_ij * edge_mask.float().unsqueeze(-1)
        m_ij = torch.nan_to_num(m_ij, nan=0.0, posinf=1e4, neginf=-1e4)
        m_x = self.phi_x(m_ij)  # [B, E, 1]

        inv_d = (dist2.sqrt() + 1e-8).reciprocal()
        direction = diff * inv_d
        delta_x_ij = m_x * direction  # [B, E, 3]
        delta_x_ij = torch.nan_to_num(delta_x_ij, nan=0.0, posinf=1e4, neginf=-1e4)

        delta_x = x.new_zeros(x.shape)
        delta_h_msg = h.new_zeros((x.size(0), x.size(1), m_ij.size(-1)))

        for b in range(B):
            delta_x[b].index_add_(0, src[b], delta_x_ij[b])
            delta_h_msg[b].index_add_(0, src[b], m_ij[b])

        h_input = torch.cat([h, delta_h_msg], dim=-1)
        h_new = h + self.phi_h(h_input)
        x_new = x + delta_x
        h_new = torch.nan_to_num(h_new, nan=0.0, posinf=1e4, neginf=-1e4)
        x_new = torch.nan_to_num(x_new, nan=0.0, posinf=1e4, neginf=-1e4)

        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            x_new = x * (1 - mask_f) + x_new * mask_f
            h_new = h * (1 - mask_f) + h_new * mask_f

        return x_new, h_new


class EquivariantTSModeHead(nn.Module, HasHParams, HyperparametersMixin):
    """EGNN-based head that predicts per-atom TS displacements."""

    def __init__(self, d_h: int, n_layers: int = 3, d_edge: int = 0, d_msg: int = 64):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1 for EquivariantTSModeHead.")
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.layers = nn.ModuleList([EGNNLayer(d_h=d_h, d_edge=d_edge, d_msg=d_msg) for _ in range(n_layers)])
        self.out = nn.Linear(d_h, 3)

    def forward(
        self,
        coords: Tensor,  # [B, N, 3]
        h: Tensor,  # [B, N, d_h]
        edge_index: Tensor,  # [B, E, 2] or [2, E]
        mask: Tensor,  # [B, N]
        edge_attr: Tensor | None = None,
        edge_mask: Tensor | None = None,
    ) -> Tensor:
        x, h_cur = coords, h
        for layer in self.layers:
            x, h_cur = layer(
                x,
                h_cur,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mask=mask,
                edge_mask=edge_mask,
            )
        return self.out(h_cur)
