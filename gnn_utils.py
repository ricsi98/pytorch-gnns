from enum import Enum
import torch
from collections import Counter
from typing import List
from torch_scatter import scatter


def aggregate(X: torch.tensor, edges: torch.LongTensor, aggr: str, edge_weights: torch.FloatTensor = None) -> torch.tensor:
    if edge_weights is None:
        edge_weights = torch.ones(edges.shape[1], dtype=torch.float)
    expanded = X[edges[0, :]]
    expanded = edge_weights[:, None] * expanded
    return scatter(expanded, edges[1, :], dim=0, reduce=aggr)


"""def aggregate_(X: torch.tensor, edges: torch.LongTensor, aggr: str, edge_weights: torch.FloatTensor = None) -> torch.tensor:
    if aggr is "mean":
        agg_fn = torch.mean
    elif aggr is "sum":
        agg_fn = torch.sum
    elif aggr is "max":
        agg_fn = torch.max
    else:
        raise NotImplemented

    if edge_weights is None:
        edge_weights = torch.ones(edges.shape[1], dtype=torch.float)

    h = torch.empty_like(X)
    for node_id in all_node_id(edges):
        neighbor_mask = edges[1, :] == node_id
        neighbor_ids = edges[0, neighbor_mask]
        neighbor_features = X[neighbor_ids]
        ew = edge_weights[neighbor_mask]
        z = (neighbor_features.transpose(0,1) * ew).transpose(0,1)
        agg = agg_fn(z, axis=0)
        h[node_id, :] = agg[:]
    return h"""
