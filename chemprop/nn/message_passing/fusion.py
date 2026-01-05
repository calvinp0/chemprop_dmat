from typing import Literal
import torch
from torch import Tensor, nn
from chemprop.nn.utils import get_activation_function, Activation

FusionType = Literal["mlp", "gru", "hadamard"]

class _HadamardFusion(nn.Module):
    def __init__(self, d_h: int, activation: str | Activation = "relu"):
        super().__init__()
        self.activation = get_activation_function(activation)
    def forward(self, h_prev: Tensor, msg: Tensor) -> Tensor:
        return self.activation(h_prev * msg)

class _MLPFusion(nn.Module):
    def __init__(self, d_h: int, activation: str | Activation = "relu", bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(2 * d_h, d_h, bias=bias)
        self.activation = get_activation_function(activation)
    def forward(self, h_prev: Tensor, msg: Tensor) -> Tensor:
        return self.activation(self.linear(torch.cat([h_prev, msg], dim=1)))

class _GRUFusion(nn.Module):
    def __init__(self, d_h: int, bias: bool = False):
        super().__init__()
        self.cell = nn.GRUCell(input_size=d_h, hidden_size=d_h, bias=bias)
    def forward(self, h_prev: Tensor, msg: Tensor) -> Tensor:
        return self.cell(msg, h_prev)

class Fusion(nn.Module):
    def __init__(
        self,
        kind: FusionType = "mlp",
        d_h: int = 300,
        activation: str | Activation = "relu",
        bias: bool = False,
        residual: bool | None = None,     # None -> sensible per-mode default
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.fusion = _MLPFusion(d_h, activation=activation, bias=bias)
            res_default = True
        elif kind == "gru":
            self.fusion = _GRUFusion(d_h, bias=bias)
            res_default = False
        elif kind == "hadamard":
            self.fusion = _HadamardFusion(d_h, activation=activation)
            res_default = True
        else:
            raise ValueError(f"Unknown fusion kind: {kind}")
        self.residual = res_default if residual is None else residual
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(d_h) if use_layernorm else nn.Identity()

    def forward(self, h_prev: Tensor, msg: Tensor) -> Tensor:
        h_new = self.fusion(h_prev, msg)     # msg must be d_h
        if self.residual and h_new.shape == h_prev.shape:
            h_new = h_new + h_prev
        h_new = self.norm(h_new)
        return self.dropout(h_new)
