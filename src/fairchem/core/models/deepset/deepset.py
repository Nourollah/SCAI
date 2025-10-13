"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter
from rich import print

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class DeepSet(torch.nn.Module, GraphModelMixin):
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_rbf: int = 128,
        transform_depth: int = 1,
        cutoff: float = 10.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        
        self.cutoff: float = cutoff

        self.embedding: torch.nn.Module = torch.nn.Embedding(
            100,
            embedding_dim,
            dtype=dtype
        )
        
        self.distance_expansion: torch.nn.Module = GaussianSmearing(
            start=0.0,
            stop=cutoff,
            num_gaussians=num_rbf,
        )

        self.distance_transform: torch.nn.Module = gnn.MLP(
            in_channels=num_rbf,
            hidden_channels=hidden_channels,
            out_channels=num_filters,
            num_layers=transform_depth,
            batch_norm=True,
            dropout=0.0,
            act="relu",
            dtype=dtype,
        )

        self.left_atom_transform: torch.nn.Module = gnn.MLP(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=num_filters,
            num_layers=transform_depth,
            batch_norm=True,
            dropout=0.0,
            act="relu",
            dtype=dtype,
        )
        
        self.right_atom_transform: torch.nn.Module = gnn.MLP(
            in_channels=embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=num_filters,
            num_layers=transform_depth,
            batch_norm=True,
            dropout=0.0,
            act="relu",
            dtype=dtype,
        )

        self.features_interaction: torch.nn.Module = gnn.MLP(
            in_channels=num_filters * 3,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=transform_depth,
            batch_norm=True,
            dropout=0.0,
            act="relu",
            dtype=dtype,
        )
        
        self.output_network: torch.nn.Module = gnn.MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=transform_depth,
            batch_norm=True,
            dropout=0.0,
            act="relu",
            dtype=dtype,
        )


    def forward(self, data):
        """ """
        raise NotImplementedError


@registry.register_model("deepset")
class DeepSetWrap(DeepSet):
    r"""
    Args:
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        use_pbc_single (bool,optional):         Process batch PBC graphs one at a time
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        use_pbc: bool = False,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        otf_graph: bool = False,
        embedding_dim: int = 128,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_rbf: int = 128,
        transform_depth: int = 1,
        cutoff: float = 10.0,
        # readout: str = "add",
    ) -> None:
        self.num_targets = 1
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        # self.reduce = readout
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_rbf=num_rbf,
            transform_depth=transform_depth,
            cutoff=cutoff,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        # This is giving edge_attr, edge_distance, and edge_index
        graph = self.generate_graph(data)


        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_attr = self.distance_expansion(graph.edge_distance)

        h = self.embedding(z)
        
        d_ij_projection = self.distance_transform(edge_attr)
        a_i_projection = self.left_atom_transform(h[graph.edge_index[0]])
        a_j_projection = self.right_atom_transform(h[graph.edge_index[1]])
        
        h_edge_x = torch.cat([d_ij_projection, a_i_projection, a_j_projection], dim=-1)
        h_edge_x = self.features_interaction(h_edge_x)
        h_edge = scatter(h_edge_x, graph.edge_index[0], dim=0, reduce='sum')

        h_edge = self.output_network(h_edge)
        energy = h_edge.sum(dim=0) if batch is None else scatter(h_edge, batch, dim=0)
        
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)
        outputs = {"energy": energy}

        if self.regress_forces:
            forces = (
                -1
                * (
                    torch.autograd.grad(
                        energy,
                        data.pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                    )[0]
                )
            )
            outputs["forces"] = forces

        return outputs

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())