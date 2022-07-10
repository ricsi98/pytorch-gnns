from turtle import forward
import torch
import torch.nn as nn
from gnn_utils import *
from edges import *
from torch.nn.functional import leaky_relu


class GNNLayer:
    
    def forward(self, h: torch.tensor, edges: torch.LongTensor):
        raise NotImplemented


class Sequential(nn.Module, GNNLayer):
    """torch.nn.Sequential like object that supports both torch modules and GNNLayer objects"""

    def __init__(self, *args) -> None:
        super().__init__()
        self.modules = args
        for i, l in enumerate(args):
            assert isinstance(l, nn.Module)
            self.add_module(l._get_name() + f"-{i}", l)

    def forward(self, h: torch.tensor, edges: torch.LongTensor):
        for l in self.modules:
            if isinstance(l, GNNLayer):
                h = l(h, edges)
            else:
                h = l(h)
        return h


class GCNLayer(nn.Module, GNNLayer):
    
    def __init__(self, in_dim: int, out_dim: int, use_self_loops: bool = True, **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, **kwargs)
        self._use_self_loops = use_self_loops

    def forward(self, h: torch.tensor, edges: torch.LongTensor):
        if self._use_self_loops:
            edges = add_self_loops(edges)
        h_ = self.linear(h)
        edge_weights = symmetric_normalization_coefficients(edges)
        agg = aggregate(h_, edges, "sum", edge_weights=edge_weights)
        return agg


class SageLayer(nn.Module, GNNLayer):

    def __init__(self, in_dim: int, out_dim: int, use_self_loops: bool = True, **kwargs) -> None:
        super().__init__()
        self._use_self_loops = use_self_loops
        self.lin_self = nn.Linear(in_dim, out_dim, **kwargs)
        self.lin_neigh = nn.Linear(in_dim, out_dim, **kwargs)

    def forward(self, h: torch.tensor, edges: torch.LongTensor):
        if self._use_self_loops:
            edges = add_self_loops(edges)
        h_self = self.lin_self(h)
        msg = self.lin_neigh(aggregate(h, edges, "mean"))
        return h_self + msg


# avoid exploding coefficients
CLIP_ALPHAS = True

class GATLayer(nn.Module, GNNLayer):

    def __init__(self, in_dim: int, out_dim: int, concatenate: bool = False, heads: int = 1, slope: float = 0.2, use_self_loops: bool = True) -> None:
        super().__init__()
        self._use_self_loops = use_self_loops
        self.heads = heads
        self.concatenate = concatenate
        self.slope = slope

        self.lin_w = [nn.Linear(in_dim, out_dim, bias=False) for _ in range(heads)]
        self.lin_a = [nn.Linear(2*out_dim, 1, bias=False) for _ in range(heads)]

        for i, (w, a) in enumerate(zip(self.lin_w, self.lin_a)):
            self.add_module("w" + str(i), w)
            self.add_module("a" + str(i), a)

    def forward(self, h: torch.tensor, edges: torch.LongTensor):
        h_w = [l(h) for l in self.lin_w]
        # e_ij scores
        h_cat = [torch.concat((h_[edges[0, :]], h_[edges[1, :]]), dim=1) for h_ in h_w]
        e = [
            torch.exp(leaky_relu(a(h_), self.slope))
            for a, h_ in zip(self.lin_a, h_cat)
        ]
        if CLIP_ALPHAS:
            e = [torch.clip(e_, 0.005, 10) for e_ in e]
        # compute edge alphas using softmax
        alpha_deno = [aggregate(e_ij, edges, "sum") for e_ij in e] # softmax denominators per node
        alphas = [e_ij / ad_ij[edges[1, :]] for e_ij, ad_ij in zip(e, alpha_deno)]
        h_next = [aggregate(h_, edges, "sum", a.view(-1)) for h_, a in zip(h_w, alphas)]

        if self.concatenate:
            return torch.concat(h_next, dim=1)
        else:
            return torch.stack(h_next, dim=-1).mean(dim=-1)