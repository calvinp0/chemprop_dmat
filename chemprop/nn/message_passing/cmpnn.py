import torch
from torch import Tensor, nn

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.nn.utils import Activation, get_activation_function
from chemprop.nn.message_passing.proto import MessagePassing
from chemprop.nn.message_passing.fusion import FusionType, Fusion
from chemprop.nn.transforms import GraphTransform, ScaleTransform

class CommunicativeMessagePassing(MessagePassing, nn.Module):
    """
    CMPNN-Style: Both bond and atom communication using message passing
    """

    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        bias: bool = False,
        d_vd: int | None = None,
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
        fusion_type: FusionType = "mlp",
        residual_updates: bool = True,
    ):
        super().__init__()
        self.hparams = dict(
            cls=self.__class__, d_v=d_v, d_e=d_e, d_h=d_h, depth=depth,
            dropout=dropout, activation=activation, bias=bias, d_vd=d_vd,
            fusion_type=fusion_type, residual_updates=residual_updates
        )

        self.depth = depth
        self.tau = get_activation_function(activation)
        self.dropout = nn.Dropout(dropout)

        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()


        self.Wi_atom = nn.Linear(d_v, d_h, bias=bias) # x_v -> H_a^0
        self.Wi_bond = nn.Linear(d_v + d_e, d_h, bias=bias)  # was nn.Linear(d_e, d_h)

        # Fusion
        self.fuse_atom = Fusion(kind=fusion_type, d_h=d_h, activation=activation, bias=bias,
                                residual=residual_updates, dropout=dropout)
        self.fuse_bond = Fusion(kind=fusion_type, d_h=d_h, activation=activation, bias=bias,
                                residual=residual_updates, dropout=dropout)

        # Inputs
        self.Wh_atom = nn.Linear(d_h, d_h, bias=bias) # atom update
        self.Wh_bond = nn.Linear(d_h, d_h, bias=bias) # bond update


        # Readout
        self.Wo_atom = nn.Linear(d_v + d_h, d_h, bias=bias) # [x_v||H_a^T] -> O_v

        # Extra descriptors
        self.Wd = nn.Linear(d_h + d_vd, d_h + d_vd, bias=True) if d_vd else None
        self.d_h = d_h
        self.d_vd = d_vd


    @staticmethod
    def sum_and_max_into_atoms(H_e: Tensor, edge_index_1: Tensor, num_atoms: int) -> tuple[Tensor, Tensor]:
        """
        Segmented sum and max over incoming edges to each atom.
        """
        E, d = H_e.shape
        idx = edge_index_1.view(-1, 1).expand(E, d)

        atom_sum = torch.zeros(num_atoms, d, device=H_e.device, dtype=H_e.dtype)
        atom_sum.scatter_reduce_(0, idx, H_e, reduce="sum", include_self=False)

        atom_max = torch.full_like(atom_sum, float("-inf"))
        atom_max.scatter_reduce_(0, idx, H_e, reduce="amax", include_self=False)
        atom_max[~torch.isfinite(atom_max)] = 0.0
        return atom_sum, atom_max

    @property
    def output_dim(self) -> int:
        # d_h by default; d_h + d_vd if Wd exists
        return self.Wd.out_features if self.Wd is not None else self.Wo_atom.out_features


    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """
        Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            the batch of :class:`~chemprop.featurizers.molgraph.MolGraph`s to encode
        V_d : Tensor | None, default=None
            an optional tensor of shape `V x d_vd` containing additional descriptors for each vertex
            in the batch. These will be concatenated to the learned vertex descriptors and
            transformed before the readout phase.

        Returns
        -------
        Tensor
            a tensor of shape `V x d_h` or `V x (d_h + d_vd)` containing the hidden representation
            of each vertex in the batch of graphs. The feature dimension depends on whether
            additional vertex descriptors were provided
        """
        bmg = self.graph_transform(bmg)
        V, E = bmg.V, bmg.E
        num_atoms = V.size(0)
        v = bmg.edge_index[0] # source atom per edge
        w = bmg.edge_index[1] # target atom per edge
        rev = bmg.rev_edge_index # reverse edge per edge

        # Initialisation
        H_a0 = self.Wi_atom(V)  # V x d_h
        H_b0 = self.Wi_bond(torch.cat([V[v], bmg.E], dim=1))  # E x d_h
        H_a = self.tau(H_a0)
        H_b = self.tau(H_b0)

        for _ in range(1, self.depth):
            # Bond -> Atoms
            a_sum, a_max = self.sum_and_max_into_atoms(H_b, w, num_atoms)  # (N, d_h), (N, d_h)
            gate = a_sum * a_max  # (N, d_h)
            # Atom update (residual on H_a0)
            m_a = self.Wh_atom(gate)                # project to d_h
            H_a = self.fuse_atom(H_a, m_a)

            # Atoms -> Bonds (exclude reverse) Sum of bonds into source atomv, minus h_{w->v}
            sum_into_src = a_sum[v]
            H_rev = H_b[rev]
            M_b = sum_into_src - H_rev
            m_b = self.Wh_bond(M_b)                 # project to d_h
            H_b = self.fuse_bond(H_b, m_b)

        # Readout
        a_sum, _ = self.sum_and_max_into_atoms(H_b, w, num_atoms)  # (N, d_h), (N, d_h)
        H_atom = self.tau(self.Wo_atom(torch.cat([V, a_sum], dim=1)))  # V x d_h
        H_atom = self.dropout(H_atom)

        if V_d is not None and self.Wd is not None:
            V_d = self.V_d_transform(V_d)
            H_atom = self.Wd(torch.cat([H_atom, V_d], dim=1))      # (N, d_h + d_vd)
            H_atom = self.dropout(H_atom)
        return H_atom
