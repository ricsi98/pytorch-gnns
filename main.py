import torch
import torch.nn as nn
import networkx as nx
import numpy as np

from torch_geometric.nn import GCNConv
from gnn_utils import *
from edges import *
from layers import GATLayer, GCNLayer


g = nx.karate_club_graph()
A = nx.adjacency_matrix(g).todense()
row, col = A.nonzero()
edges = torch.tensor(np.stack((row, col), axis=0))

n_feats, n_nodes = 3, len(g)
X = np.random.normal(0, 1, size=(n_nodes, n_feats))
X_torch = torch.tensor(X).float()

"""
l = GALayer(n_feats, 2, True, 3)
print(l)
print(l(X_torch, edges).shape)"""

print(add_self_loops(edges))