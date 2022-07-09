import torch
from functools import cached_property
from typing import List
from utils import cache_result_in_param


@cache_result_in_param
def all_node_id(edge_list: torch.LongTensor) -> List[int]:
    from_, to_ = edge_list.detach().cpu().numpy().tolist()
    return sorted(set(from_).union(set(to_)))


@cache_result_in_param
def node_degree_from_edge_list(edge_list: torch.LongTensor) -> torch.LongTensor:
    return torch.unique(edge_list[1, :], sorted=True, return_counts=True)[-1]


@cache_result_in_param
def add_self_loops(edge_list: torch.LongTensor) -> torch.LongTensor:
    v = torch.arange(edge_list.max()+1, dtype=torch.long)
    self_loops = torch.stack((v, v), dim=0)
    return torch.concat((edge_list, self_loops), dim=1)


@cache_result_in_param
def symmetric_normalization_coefficients(edge_list: torch.LongTensor) -> torch.FloatTensor:
    deg = node_degree_from_edge_list(edge_list).float()
    deg_inv_sqrt = deg.pow_(-0.5)
    return deg_inv_sqrt[edge_list[0, :]] * deg_inv_sqrt[edge_list[1, :]]