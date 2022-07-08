import networkx as nx
import numpy as np
import torch

from layers import GCNLayer, SageLayer, Sequential
from torch_geometric.nn import GCNConv



@torch.no_grad()
def test_gcn_layer():
    g = nx.karate_club_graph()
    A = nx.adjacency_matrix(g).todense()
    row, col = A.nonzero()
    edges = torch.tensor(np.stack((row, col), axis=0))

    n_feats, n_nodes = 3, len(g)
    X = np.random.normal(0, 1, size=(n_nodes, n_feats))
    X_torch = torch.tensor(X).float()
    
    for use_bias in [True, False]:
        for use_self_loops in [True, False]:
            l1 = GCNLayer(n_feats, n_feats, bias=use_bias, use_self_loops=use_self_loops)
            l2 = GCNConv(n_feats, n_feats, bias=use_bias, add_self_loops=use_self_loops)

            l1.linear.weight[:] = l2.lin.weight[:]
            if use_bias:
                l1.linear.bias[:] = l2.bias[:]

            h1, h2 = l1(X_torch, edges).numpy(), l2(X_torch, edges).numpy()
            assert np.allclose(h1, h2), f"bias={use_bias}, self_loops={use_self_loops}"


def test_gcn_backwards():
    g = nx.karate_club_graph()
    A = nx.adjacency_matrix(g).todense()
    row, col = A.nonzero()
    edges = torch.tensor(np.stack((row, col), axis=0))

    n_feats, n_nodes = 3, len(g)
    X = np.random.normal(0, 1, size=(n_nodes, n_feats))
    X_torch = torch.tensor(X).float()

    l = GCNLayer(n_feats, 1)
    l(X_torch, edges).sum().backward()


def test_sequential():
    g = nx.karate_club_graph()
    A = nx.adjacency_matrix(g).todense()
    row, col = A.nonzero()
    edges = torch.tensor(np.stack((row, col), axis=0))

    n_feats, n_nodes = 3, len(g)
    X = np.random.normal(0, 1, size=(n_nodes, n_feats))
    X_torch = torch.tensor(X).float()

    s = Sequential(
        GCNLayer(n_feats, n_feats), 
        torch.nn.ReLU(), 
        SageLayer(n_feats, n_feats),
        torch.nn.Sigmoid()
    )

    out = s(X_torch, edges)
    x = out.detach().numpy()
    assert (x < 1).all() and (x > 0).all()

if __name__ == '__main__':
    test_sequential()