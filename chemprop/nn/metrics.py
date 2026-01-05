from abc import abstractmethod

from numpy.typing import ArrayLike
import torch
from torch import Tensor
from torch.nn import functional as F
import torchmetrics
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat

from chemprop.utils.registry import ClassRegistry

__all__ = [
    "ChempropMetric",
    "LossFunctionRegistry",
    "MetricRegistry",
    "MSE",
    "MAE",
    "RMSE",
    "BoundedMixin",
    "BoundedMSE",
    "BoundedMAE",
    "BoundedRMSE",
    "BinaryAccuracy",
    "BinaryAUPRC",
    "BinaryAUROC",
    "BinaryF1Score",
    "BinaryMCCMetric",
    "BoundedMAE",
    "BoundedMSE",
    "BoundedRMSE",
    "MetricRegistry",
    "MulticlassMCCMetric",
    "R2Score",
    "MVELoss",
    "EvidentialLoss",
    "BCELoss",
    "CrossEntropyLoss",
    "BinaryMCCLoss",
    "BinaryMCCMetric",
    "MulticlassMCCLoss",
    "MulticlassMCCMetric",
    "ClassificationMixin",
    "BinaryAUROC",
    "BinaryAUPRC",
    "BinaryAccuracy",
    "BinaryF1Score",
    "DirichletLoss",
    "SID",
    "Wasserstein",
    "QuantileLoss",
]


class ChempropMetric(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, task_weights: ArrayLike = 1.0):
        """
        Parameters
        ----------
        task_weights :  ArrayLike, default=1.0
            the per-task weights of shape `t` or `1 x t`. Defaults to all tasks having a weight of 1.
        """
        super().__init__()
        task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)
        self.register_buffer("task_weights", task_weights)

        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        lt_mask: Tensor | None = None,
        gt_mask: Tensor | None = None,
    ) -> None:
        """Calculate the mean loss function value given predicted and target values

        Parameters
        ----------
        preds : Tensor
            a tensor of shape `b x t x u` (regression with uncertainty), `b x t` (regression without
            uncertainty and binary classification, except for binary dirichlet), or `b x t x c`
            (multiclass classification and binary dirichlet) containing the predictions, where `b`
            is the batch size, `t` is the number of tasks to predict, `u` is the number of values to
            predict for each task, and `c` is the number of classes.
        targets : Tensor
            a float tensor of shape `b x t` containing the target values
        mask : Tensor
            a boolean tensor of shape `b x t` indicating whether the given prediction should be
            included in the loss calculation
        weights : Tensor
            a tensor of shape `b` or `b x 1` containing the per-sample weight
        lt_mask: Tensor
        gt_mask: Tensor
        """
        mask = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask
        weights = (
            torch.ones(targets.shape[0], dtype=torch.float, device=targets.device)
            if weights is None
            else weights
        )
        lt_mask = torch.zeros_like(targets, dtype=torch.bool) if lt_mask is None else lt_mask
        gt_mask = torch.zeros_like(targets, dtype=torch.bool) if gt_mask is None else gt_mask

        L = self._calc_unreduced_loss(preds, targets, mask, weights, lt_mask, gt_mask)
        L = L * weights.view(-1, 1) * self.task_weights * mask

        self.total_loss += L.sum()
        self.num_samples += mask.sum()

    def compute(self):
        return self.total_loss / self.num_samples

    @abstractmethod
    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask) -> Tensor:
        """Calculate a tensor of shape `b x t` containing the unreduced loss values."""

    def extra_repr(self) -> str:
        return f"task_weights={self.task_weights.tolist()}"


LossFunctionRegistry = ClassRegistry[ChempropMetric]()
MetricRegistry = ClassRegistry[ChempropMetric]()


@LossFunctionRegistry.register("mse")
@MetricRegistry.register("mse")
class MSE(ChempropMetric):
    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return F.mse_loss(preds, targets, reduction="none")


@MetricRegistry.register("mae")
@LossFunctionRegistry.register("mae")
class MAE(ChempropMetric):
    def _calc_unreduced_loss(self, preds, targets, *args) -> Tensor:
        return (preds - targets).abs()


@LossFunctionRegistry.register("huber")
@MetricRegistry.register("huber")
class Huber(ChempropMetric):
    def __init__(self, task_weights: ArrayLike = 1.0, beta: float = 1.0):
        super().__init__(task_weights)
        self.beta = beta

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return F.smooth_l1_loss(preds, targets, reduction="none", beta=self.beta)

    def extra_repr(self) -> str:
        return f"task_weights={self.task_weights.tolist()}, beta={self.beta}"


@LossFunctionRegistry.register("rmse")
@MetricRegistry.register("rmse")
class RMSE(MSE):
    def compute(self):
        return (self.total_loss / self.num_samples).sqrt()


class BoundedMixin:
    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask) -> Tensor:
        preds = torch.where((preds < targets) & lt_mask, targets, preds)
        preds = torch.where((preds > targets) & gt_mask, targets, preds)

        return super()._calc_unreduced_loss(preds, targets, mask, weights)


@LossFunctionRegistry.register("bounded-mse")
@MetricRegistry.register("bounded-mse")
class BoundedMSE(BoundedMixin, MSE):
    pass


@LossFunctionRegistry.register("bounded-mae")
@MetricRegistry.register("bounded-mae")
class BoundedMAE(BoundedMixin, MAE):
    pass


@LossFunctionRegistry.register("bounded-rmse")
@MetricRegistry.register("bounded-rmse")
class BoundedRMSE(BoundedMixin, RMSE):
    pass


@MetricRegistry.register("r2")
class R2Score(torchmetrics.R2Score):
    def __init__(self, task_weights: ArrayLike = 1.0, **kwargs):
        """
        Parameters
        ----------
        task_weights :  ArrayLike = 1.0
            .. important::
                Ignored. Maintained for compatibility with :class:`ChempropMetric`
        """
        super().__init__()
        task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)
        self.register_buffer("task_weights", task_weights)

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        super().update(preds[mask], targets[mask])


@LossFunctionRegistry.register("mve")
class MVELoss(ChempropMetric):
    """Calculate the loss using Eq. 9 from [nix1994]_

    References
    ----------
    .. [nix1994] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target
        probability distribution." Proceedings of 1994 IEEE International Conference on Neural
        Networks, 1994 https://doi.org/10.1109/icnn.1994.374138
    """

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        mean, var = torch.unbind(preds, dim=-1)

        L_sos = (mean - targets) ** 2 / (2 * var)
        L_kl = (2 * torch.pi * var).log() / 2

        return L_sos + L_kl


@LossFunctionRegistry.register("evidential")
class EvidentialLoss(ChempropMetric):
    """Calculate the loss using Eqs. 8, 9, and 10 from [amini2020]_. See also [soleimany2021]_.

    References
    ----------
    .. [amini2020] Amini, A; Schwarting, W.; Soleimany, A.; Rus, D.;
        "Deep Evidential Regression" Advances in Neural Information Processing Systems; 2020; Vol.33.
        https://proceedings.neurips.cc/paper_files/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf
    .. [soleimany2021] Soleimany, A.P.; Amini, A.; Goldman, S.; Rus, D.; Bhatia, S.N.; Coley, C.W.;
        "Evidential Deep Learning for Guided Molecular Property Prediction and Discovery." ACS
        Cent. Sci. 2021, 7, 8, 1356-1367. https://doi.org/10.1021/acscentsci.1c00546
    """

    def __init__(self, task_weights: ArrayLike = 1.0, v_kl: float = 0.2, eps: float = 1e-8):
        super().__init__(task_weights)
        self.v_kl = v_kl
        self.eps = eps

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        mean, v, alpha, beta = torch.unbind(preds, dim=-1)

        residuals = targets - mean
        twoBlambda = 2 * beta * (1 + v)

        L_nll = (
            0.5 * (torch.pi / v).log()
            - alpha * twoBlambda.log()
            + (alpha + 0.5) * torch.log(v * residuals**2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        L_reg = (2 * v + alpha) * residuals.abs()

        return L_nll + self.v_kl * (L_reg - self.eps)

    def extra_repr(self) -> str:
        parent_repr = super().extra_repr()
        return parent_repr + f", v_kl={self.v_kl}, eps={self.eps}"


@LossFunctionRegistry.register("bce")
class BCELoss(ChempropMetric):
    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none")


@LossFunctionRegistry.register("ce")
class CrossEntropyLoss(ChempropMetric):
    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        preds = preds.transpose(1, 2)
        targets = targets.long()

        return F.cross_entropy(preds, targets, reduction="none")


@LossFunctionRegistry.register("binary-mcc")
class BinaryMCCLoss(ChempropMetric):
    def __init__(self, task_weights: ArrayLike = 1.0):
        """
        Parameters
        ----------
        task_weights :  ArrayLike, default=1.0
            the per-task weights of shape `t` or `1 x t`. Defaults to all tasks having a weight of 1.
        """
        super().__init__(task_weights)

        self.add_state("TP", default=[], dist_reduce_fx="cat")
        self.add_state("FP", default=[], dist_reduce_fx="cat")
        self.add_state("TN", default=[], dist_reduce_fx="cat")
        self.add_state("FN", default=[], dist_reduce_fx="cat")

    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        *args,
    ):
        mask = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask
        weights = (
            torch.ones(targets.shape[0], dtype=torch.float, device=targets.device)
            if weights is None
            else weights
        )

        if not (0 <= preds.min() and preds.max() <= 1):  # assume logits
            preds = preds.sigmoid()

        TP, FP, TN, FN = self._calc_unreduced_loss(preds, targets.long(), mask, weights, *args)

        self.TP += [TP]
        self.FP += [FP]
        self.TN += [TN]
        self.FN += [FN]

    def _calc_unreduced_loss(self, preds, targets, mask, weights, *args) -> Tensor:
        TP = (targets * preds * weights * mask).sum(0, keepdim=True)
        FP = ((1 - targets) * preds * weights * mask).sum(0, keepdim=True)
        TN = ((1 - targets) * (1 - preds) * weights * mask).sum(0, keepdim=True)
        FN = (targets * (1 - preds) * weights * mask).sum(0, keepdim=True)

        return TP, FP, TN, FN

    def compute(self):
        TP = dim_zero_cat(self.TP).sum(0)
        FP = dim_zero_cat(self.FP).sum(0)
        TN = dim_zero_cat(self.TN).sum(0)
        FN = dim_zero_cat(self.FN).sum(0)

        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-8).sqrt()
        MCC = MCC * self.task_weights
        return 1 - MCC.mean()


@MetricRegistry.register("binary-mcc")
class BinaryMCCMetric(BinaryMCCLoss):
    higher_is_better = True

    def compute(self):
        return 1 - super().compute()


@LossFunctionRegistry.register("multiclass-mcc")
class MulticlassMCCLoss(ChempropMetric):
    """Calculate a soft Matthews correlation coefficient ([mccWiki]_) loss for multiclass
    classification based on the implementataion of [mccSklearn]_
    References
    ----------
    .. [mccWiki] https://en.wikipedia.org/wiki/Phi_coefficient#Multiclass_case
    .. [mccSklearn] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """

    def __init__(self, task_weights: ArrayLike = 1.0):
        """
        Parameters
        ----------
        task_weights :  ArrayLike, default=1.0
            the per-task weights of shape `t` or `1 x t`. Defaults to all tasks having a weight of 1.
        """
        super().__init__(task_weights)

        self.add_state("p", default=[], dist_reduce_fx="cat")
        self.add_state("t", default=[], dist_reduce_fx="cat")
        self.add_state("c", default=[], dist_reduce_fx="cat")
        self.add_state("s", default=[], dist_reduce_fx="cat")

    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        *args,
    ):
        mask = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask
        weights = (
            torch.ones((targets.shape[0], 1), dtype=torch.float, device=targets.device)
            if weights is None
            else weights.view(-1, 1)
        )

        if not (0 <= preds.min() and preds.max() <= 1):  # assume logits
            preds = preds.softmax(2)

        p, t, c, s = self._calc_unreduced_loss(preds, targets.long(), mask, weights, *args)

        self.p += [p]
        self.t += [t]
        self.c += [c]
        self.s += [s]

    def _calc_unreduced_loss(self, preds, targets, mask, weights, *args) -> Tensor:
        device = preds.device
        C = preds.shape[2]
        bin_targets = torch.eye(C, device=device)[targets]
        bin_preds = torch.eye(C, device=device)[preds.argmax(-1)]
        masked_data_weights = weights.unsqueeze(2) * mask.unsqueeze(2)
        p = (bin_preds * masked_data_weights).sum(0, keepdims=True)
        t = (bin_targets * masked_data_weights).sum(0, keepdims=True)
        c = (bin_preds * bin_targets * masked_data_weights).sum(2).sum(0, keepdims=True)
        s = (preds * masked_data_weights).sum(2).sum(0, keepdims=True)

        return p, t, c, s

    def compute(self):
        p = dim_zero_cat(self.p).sum(0)
        t = dim_zero_cat(self.t).sum(0)
        c = dim_zero_cat(self.c).sum(0)
        s = dim_zero_cat(self.s).sum(0)
        s2 = s.square()

        # the `einsum` calls amount to calculating the batched dot product
        cov_ytyp = c * s - torch.einsum("ij,ij->i", p, t)
        cov_ypyp = s2 - torch.einsum("ij,ij->i", p, p)
        cov_ytyt = s2 - torch.einsum("ij,ij->i", t, t)

        x = cov_ypyp * cov_ytyt
        MCC = torch.where(x == 0, torch.tensor(0.0), cov_ytyp / x.sqrt())
        MCC = MCC * self.task_weights

        return 1 - MCC.mean()


@MetricRegistry.register("multiclass-mcc")
class MulticlassMCCMetric(MulticlassMCCLoss):
    higher_is_better = True

    def compute(self):
        return 1 - super().compute()


class ClassificationMixin:
    def __init__(self, task_weights: ArrayLike = 1.0, **kwargs):
        """
        Parameters
        ----------
        task_weights :  ArrayLike = 1.0
            .. important::
                Ignored. Maintained for compatibility with :class:`ChempropMetric`
        """
        super().__init__()
        task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)
        self.register_buffer("task_weights", task_weights)

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        super().update(preds[mask], targets[mask].long())


@MetricRegistry.register("roc")
class BinaryAUROC(ClassificationMixin, torchmetrics.classification.BinaryAUROC):
    pass


@MetricRegistry.register("prc")
class BinaryAUPRC(ClassificationMixin, torchmetrics.classification.BinaryPrecisionRecallCurve):
    def compute(self) -> Tensor:
        p, r, _ = super().compute()
        return auc(r, p)


@MetricRegistry.register("accuracy")
class BinaryAccuracy(ClassificationMixin, torchmetrics.classification.BinaryAccuracy):
    pass


@MetricRegistry.register("f1")
class BinaryF1Score(ClassificationMixin, torchmetrics.classification.BinaryF1Score):
    pass


@LossFunctionRegistry.register("dirichlet")
class DirichletLoss(ChempropMetric):
    """Uses the loss function from [sensoy2018]_ based on the implementation at [sensoyGithub]_

    References
    ----------
    .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
        classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
    .. [sensoyGithub] https://muratsensoy.github.io/uncertainty.html#Define-the-loss-function
    """

    def __init__(self, task_weights: ArrayLike = 1.0, v_kl: float = 0.2):
        super().__init__(task_weights)
        self.v_kl = v_kl

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        targets = torch.eye(preds.shape[2], device=preds.device)[targets.long()]

        S = preds.sum(-1, keepdim=True)
        p = preds / S

        A = (targets - p).square().sum(-1, keepdim=True)
        B = ((p * (1 - p)) / (S + 1)).sum(-1, keepdim=True)

        L_mse = A + B

        alpha = targets + (1 - targets) * preds
        beta = torch.ones_like(alpha)
        S_alpha = alpha.sum(-1, keepdim=True)
        S_beta = beta.sum(-1, keepdim=True)

        ln_alpha = S_alpha.lgamma() - alpha.lgamma().sum(-1, keepdim=True)
        ln_beta = beta.lgamma().sum(-1, keepdim=True) - S_beta.lgamma()

        dg0 = torch.digamma(alpha)
        dg1 = torch.digamma(S_alpha)

        L_kl = ln_alpha + ln_beta + torch.sum((alpha - beta) * (dg0 - dg1), -1, keepdim=True)

        return (L_mse + self.v_kl * L_kl).mean(-1)

    def extra_repr(self) -> str:
        return f"v_kl={self.v_kl}"


@LossFunctionRegistry.register("sid")
class SID(ChempropMetric):
    def __init__(self, task_weights: ArrayLike = 1.0, threshold: float | None = None, **kwargs):
        super().__init__(task_weights, **kwargs)

        self.threshold = threshold

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (preds_norm / targets).log() * preds_norm + (targets / preds_norm).log() * targets

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


@LossFunctionRegistry.register(["earthmovers", "wasserstein"])
class Wasserstein(ChempropMetric):
    def __init__(self, task_weights: ArrayLike = 1.0, threshold: float | None = None):
        super().__init__(task_weights)

        self.threshold = threshold

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return (targets.cumsum(1) - preds_norm.cumsum(1)).abs()

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


@LossFunctionRegistry.register(["quantile", "pinball"])
class QuantileLoss(ChempropMetric):
    def __init__(self, task_weights: ArrayLike = 1.0, alpha: float = 0.1):
        super().__init__(task_weights)
        self.alpha = alpha

        bounds = torch.tensor([-1 / 2, 1 / 2]).view(-1, 1, 1)
        tau = torch.tensor([[alpha / 2, 1 - alpha / 2], [alpha / 2 - 1, -alpha / 2]]).view(
            2, 2, 1, 1
        )

        self.register_buffer("bounds", bounds)
        self.register_buffer("tau", tau)

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        mean, interval = torch.unbind(preds, dim=-1)

        interval_bounds = self.bounds * interval
        pred_bounds = mean + interval_bounds
        error_bounds = targets - pred_bounds
        loss_bounds = (self.tau * error_bounds).amax(0)

        return loss_bounds.sum(0)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"

@LossFunctionRegistry.register("geometry_scaled")
class GeometryScaledLoss(ChempropMetric):
    """
    Columns (t=7): [d1, d2, theta, psi1_sin, psi1_cos, psi2_sin, psi2_cos]
    Targets/preds are standard-scaled before training; this loss inverts scaling to raw space,
    then computes geometry-aware terms.
    """
    def __init__(self, task_weights=1,
                 target_means=None, target_stds=None,
                 angle_unit="deg", huber_delta=1.0,
                 unit_penalty=1e-2, eps=1e-12,
                 scaled_flags=None, col_idx=None):
        super().__init__(task_weights)
        self.eps = eps
        self.huber_delta = huber_delta
        self.unit_penalty = unit_penalty
        self.angle_unit = angle_unit

        # column mapping (sin,cos order per your latest message)
        self.cols = {
            "d1": 0, "d2": 1, "theta": 2,
            "psi1_sin": 3, "psi1_cos": 4,
            "psi2_sin": 5, "psi2_cos": 6,
        }
        if col_idx:
            self.cols.update(col_idx)

        means = torch.zeros(7) if target_means is None else torch.as_tensor(target_means, dtype=torch.float)
        stds  = torch.ones(7)  if target_stds  is None else torch.as_tensor(target_stds,  dtype=torch.float)
        self.register_buffer("y_mean", means.view(1, -1))
        self.register_buffer("y_std",  stds.view(1, -1))

        # IMPORTANT: set True for every column that actually WAS scaled in your dataset
        if scaled_flags is None:
            scaled_flags = [True] * 7
        self.register_buffer("scaled_flags", torch.as_tensor(scaled_flags, dtype=torch.bool).view(1, -1))

    def _inv_scale(self, y_scaled: Tensor) -> Tensor:
        return torch.where(self.scaled_flags, y_scaled * self.y_std + self.y_mean, y_scaled)

    def _to_radians(self, x_raw: Tensor) -> Tensor:
        if self.angle_unit == "deg":      return x_raw * (torch.pi / 180.0)
        if self.angle_unit == "scaled01": return x_raw * torch.pi
        if self.angle_unit == "rad":      return x_raw
        raise ValueError(f"Unknown angle unit: {self.angle_unit}")

    def _huber(self, x: Tensor, delta: float) -> Tensor:
        ax = x.abs()
        return torch.where(ax <= delta, 0.5 * (ax**2) / delta, ax - 0.5 * delta)

    def _pair_alignment_from_raw(self, cos_p: Tensor, sin_p: Tensor,
                                 cos_t: Tensor, sin_t: Tensor) -> Tensor:
        """
        Inputs are already in RAW space (not scaled). We normalize the predicted vector,
        compute cosine alignment, and add a small unit-circle penalty.
        """
        t = torch.stack([cos_p, sin_p], dim=-1)  # [B, 2] predicted (cos, sin)
        u = torch.stack([cos_t, sin_t], dim=-1)  # [B, 2] target    (cos, sin)

        u_p = t / (t.norm(dim=-1, keepdim=True).clamp_min(self.eps))
        align = 1.0 - (u_p * u).sum(dim=-1)                       # [B]
        unit_pen = (t.norm(dim=-1).pow(2) - 1.0).pow(2)           # [B]
        return align + self.unit_penalty * unit_pen

    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *_) -> Tensor:
        # 1) invert scaling ONCE for all columns flagged as scaled
        P_raw = self._inv_scale(preds)
        T_raw = self._inv_scale(targets)

        L = torch.zeros_like(preds)

        # 2) distances (log-space Huber on raw)
        for k in ("d1", "d2"):
            i = self.cols[k]
            z_p = torch.log(P_raw[:, i].clamp_min(1e-12))
            z_t = torch.log(T_raw[:, i].clamp_min(1e-12))
            L[:, i] = self._huber(z_p - z_t, self.huber_delta)

        # 3) theta periodic (compute on raw then to radians)
        i = self.cols["theta"]
        th_p = self._to_radians(P_raw[:, i])
        th_t = self._to_radians(T_raw[:, i])
        L[:, i] = 1.0 - torch.cos(th_p - th_t)

        # 4) ψ1 pair: (sin, cos) -> pass as (cos, sin) to the helper
        i_s1, i_c1 = self.cols["psi1_sin"], self.cols["psi1_cos"]
        pair1 = self._pair_alignment_from_raw(
            P_raw[:, i_c1], P_raw[:, i_s1],
            T_raw[:, i_c1], T_raw[:, i_s1],
        )
        L[:, i_c1] = 0.5 * pair1
        L[:, i_s1] = 0.5 * pair1

        # 5) ψ2 pair
        i_s2, i_c2 = self.cols["psi2_sin"], self.cols["psi2_cos"]
        pair2 = self._pair_alignment_from_raw(
            P_raw[:, i_c2], P_raw[:, i_s2],
            T_raw[:, i_c2], T_raw[:, i_s2],
        )
        L[:, i_c2] = 0.5 * pair2
        L[:, i_s2] = 0.5 * pair2

        return L

@MetricRegistry.register("r2_overall")
class R2Overall(ChempropMetric):
    """
    Overall R² for multi-task regression.
    mode="micro" -> one global R² pooling all tasks (mask-aware).
    mode="macro" -> weight-averaged mean of per-task R².
    """
    higher_is_better = True

    def __init__(self, task_weights: ArrayLike = 1.0, mode: str = "micro", eps: float = 1e-12):
        super().__init__(task_weights)
        assert mode in ("micro", "macro")
        self.mode = mode
        self.eps = eps

        # Lazily sized on first update
        self.add_state("sum_y",  default=torch.tensor([]), dist_reduce_fx="sum")   # [T]
        self.add_state("sum_y2", default=torch.tensor([]), dist_reduce_fx="sum")   # [T]
        self.add_state("rss",    default=torch.tensor([]), dist_reduce_fx="sum")   # [T]
        self.add_state("count",  default=torch.tensor([]), dist_reduce_fx="sum")   # [T]

    # satisfy abstract method; not used for R²
    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask) -> Tensor:
        return torch.zeros_like(targets)

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        # preds/targets/mask: [B, T]
        B, T = targets.shape
        device = targets.device
        if self.sum_y.numel() == 0:
            self.sum_y  = torch.zeros(T, device=device)
            self.sum_y2 = torch.zeros(T, device=device)
            self.rss    = torch.zeros(T, device=device)
            self.count  = torch.zeros(T, device=device)

        m = mask.to(torch.bool)
        p = torch.where(m, preds, torch.nan)
        y = torch.where(m, targets, torch.nan)

        self.rss    += torch.nan_to_num((p - y) ** 2).nansum(dim=0)
        self.sum_y  += torch.nan_to_num(y).nansum(dim=0)
        self.sum_y2 += torch.nan_to_num(y ** 2).nansum(dim=0)
        self.count  += m.float().sum(dim=0)

    def compute(self):
        # per-task stats → per-task R²
        denom = self.count.clamp_min(1.0)
        mu    = self.sum_y / denom
        tss_t = self.sum_y2 - 2 * mu * self.sum_y + denom * (mu ** 2)  # per-task TSS
        r2_t  = 1.0 - (self.rss / tss_t.clamp_min(self.eps))
        r2_t  = torch.where(self.count > 1, r2_t, torch.zeros_like(r2_t))

        if self.mode == "macro":
            w = self.task_weights.view(-1)
            return (r2_t * w).sum() / (w.sum() + 1e-12)

        # micro/global: pool numerators & denominators across tasks
        rss_all = self.rss.sum()
        tss_all = tss_t.sum().clamp_min(self.eps)
        return 1.0 - (rss_all / tss_all)

@MetricRegistry.register("r2_overall")
class R2Overall(ChempropMetric):
    higher_is_better = True

    def __init__(self, task_weights: ArrayLike = 1.0, mode: str = "micro",
                 angle_unit: str = "deg", eps: float = 1e-12, col_idx: dict | None = None):
        super().__init__(task_weights)
        assert mode in ("micro", "macro")
        self.mode, self.eps, self.angle_unit = mode, eps, angle_unit

        self.cols = {"d1":0,"d2":1,"theta":2,"psi1_sin":3,"psi1_cos":4,"psi2_sin":5,"psi2_cos":6}
        if col_idx: self.cols.update(col_idx)

        # lazy-sized states
        self.add_state("rss_d",    default=torch.tensor([]), dist_reduce_fx="sum")   # [2]
        self.add_state("sum_y_d",  default=torch.tensor([]), dist_reduce_fx="sum")   # [2]
        self.add_state("sum_y2_d", default=torch.tensor([]), dist_reduce_fx="sum")   # [2]
        self.add_state("cnt_d",    default=torch.tensor([]), dist_reduce_fx="sum")   # [2]

        self.add_state("rss_theta",     default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_cos_theta", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sin_theta", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cnt_theta",     default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("rss_psi",     default=torch.tensor([]), dist_reduce_fx="sum")  # [2]
        self.add_state("sum_cos_psi", default=torch.tensor([]), dist_reduce_fx="sum")  # [2]
        self.add_state("sum_sin_psi", default=torch.tensor([]), dist_reduce_fx="sum")  # [2]
        self.add_state("cnt_psi",     default=torch.tensor([]), dist_reduce_fx="sum")  # [2]

    def _calc_unreduced_loss(self, *args, **kwargs) -> Tensor:
        return torch.zeros(1)

    def _to_rad(self, x: torch.Tensor) -> torch.Tensor:
        if self.angle_unit == "deg":      return x * (torch.pi / 180.0)
        if self.angle_unit == "scaled01": return x * torch.pi
        if self.angle_unit == "rad":      return x
        raise ValueError(f"Unknown angle_unit={self.angle_unit}")

    @torch.no_grad()
    def update(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        device = targets.device
        if self.rss_d.numel() == 0:
            z = lambda n: torch.zeros(n, device=device, dtype=targets.dtype)
            self.rss_d    = z(2); self.sum_y_d  = z(2); self.sum_y2_d = z(2); self.cnt_d = z(2)
            self.rss_psi  = z(2); self.sum_cos_psi = z(2); self.sum_sin_psi = z(2); self.cnt_psi = z(2)
            # scalars already created; ensure dtypes
            self.rss_theta     = self.rss_theta.to(device=device, dtype=targets.dtype)
            self.sum_cos_theta = self.sum_cos_theta.to(device=device, dtype=targets.dtype)
            self.sum_sin_theta = self.sum_sin_theta.to(device=device, dtype=targets.dtype)
            self.cnt_theta     = self.cnt_theta.to(device=device, dtype=targets.dtype)

        m_d1 = mask[:, self.cols["d1"]].bool()
        m_d2 = mask[:, self.cols["d2"]].bool()
        m_th = mask[:, self.cols["theta"]].bool()
        m_p1 = mask[:, self.cols["psi1_sin"]].bool() & mask[:, self.cols["psi1_cos"]].bool()
        m_p2 = mask[:, self.cols["psi2_sin"]].bool() & mask[:, self.cols["psi2_cos"]].bool()

        # distances
        for gi, key in enumerate(("d1", "d2")):
            idx = self.cols[key]
            m = m_d1 if gi == 0 else m_d2
            if m.any():
                p = preds[m, idx]; y = targets[m, idx]
                self.rss_d[gi]    += (p - y).pow(2).sum()
                self.sum_y_d[gi]  += y.sum()
                self.sum_y2_d[gi] += y.pow(2).sum()
                self.cnt_d[gi]    += m.sum().to(self.rss_d.dtype)

        # theta (circular)
        if m_th.any():
            th_p = self._to_rad(preds[m_th, self.cols["theta"]])
            th_t = self._to_rad(targets[m_th, self.cols["theta"]])
            self.rss_theta     += (1.0 - torch.cos(th_p - th_t)).sum()
            self.sum_cos_theta += torch.cos(th_t).sum()
            self.sum_sin_theta += torch.sin(th_t).sum()
            self.cnt_theta     += m_th.sum().to(self.rss_d.dtype)

        # dihedrals
        def _accum_psi(group_idx: int, s_col: int, c_col: int, m: torch.Tensor):
            if not m.any(): return
            cos_t = targets[m, c_col]; sin_t = targets[m, s_col]
            u_t = torch.stack([cos_t, sin_t], dim=-1)
            cos_p = preds[m, c_col];  sin_p = preds[m, s_col]
            t = torch.stack([cos_p, sin_p], dim=-1)
            u_p = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            self.rss_psi[group_idx]     += (1.0 - (u_p * u_t).sum(dim=-1)).sum()
            self.sum_cos_psi[group_idx] += cos_t.sum()
            self.sum_sin_psi[group_idx] += sin_t.sum()
            self.cnt_psi[group_idx]     += m.sum().to(self.rss_d.dtype)

        _accum_psi(0, self.cols["psi1_sin"], self.cols["psi1_cos"], m_p1)
        _accum_psi(1, self.cols["psi2_sin"], self.cols["psi2_cos"], m_p2)

    @torch.no_grad()
    def compute(self):
        denom_d = self.cnt_d.clamp_min(1.0)
        mu_d    = self.sum_y_d / denom_d
        tss_d   = self.sum_y2_d - 2 * mu_d * self.sum_y_d + denom_d * (mu_d ** 2)

        R_theta = torch.hypot(self.sum_cos_theta, self.sum_sin_theta)
        tss_theta = (self.cnt_theta - R_theta).clamp_min(self.eps)

        R_psi = torch.hypot(self.sum_cos_psi, self.sum_sin_psi)
        tss_psi = (self.cnt_psi - R_psi).clamp_min(self.eps)

        r2_d = torch.where(self.cnt_d > 1, 1.0 - (self.rss_d / tss_d.clamp_min(self.eps)), torch.zeros_like(self.rss_d))
        r2_theta = 1.0 - (self.rss_theta / tss_theta) if (self.cnt_theta > 1) else torch.tensor(0.0, device=self.rss_d.device, dtype=self.rss_d.dtype)
        r2_psi = torch.where(self.cnt_psi > 1, 1.0 - (self.rss_psi / tss_psi), torch.zeros_like(self.rss_psi))

        r2_groups = torch.stack([r2_d[0], r2_d[1], r2_theta, r2_psi[0], r2_psi[1]])

        if self.mode == "macro":
            tw = self.task_weights.view(-1)
            w_d1   = tw[self.cols["d1"]]
            w_d2   = tw[self.cols["d2"]]
            w_th   = tw[self.cols["theta"]]
            w_psi1 = 0.5 * (tw[self.cols["psi1_sin"]] + tw[self.cols["psi1_cos"]])
            w_psi2 = 0.5 * (tw[self.cols["psi2_sin"]] + tw[self.cols["psi2_cos"]])
            w_groups = torch.stack([w_d1, w_d2, w_th, w_psi1, w_psi2])
            return (r2_groups * w_groups).sum() / w_groups.sum().clamp_min(self.eps)

        rss_all = self.rss_d.sum() + self.rss_theta + self.rss_psi.sum()
        tss_all = tss_d.sum() + tss_theta + tss_psi.sum()
        return 1.0 - (rss_all / tss_all.clamp_min(self.eps))
