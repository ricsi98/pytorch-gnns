from gnn_utils import *
import numpy as np
import networkx as nx


def test_aggregate_mean():
    # test aggregate function on circular graph
    k = 10
    features = torch.diag(torch.ones(k)).float()
    adj = torch.tensor([[i, (i + 1) % k] for i in range(k)]).transpose(0,1)
    h = aggregate(features, adj, "mean")
    h_target = torch.zeros_like(features)
    for i in range(k):
        h_target[i, (i-1) % k] = 1
    assert np.allclose(h.numpy(), h_target.numpy())


def test_aggregate_mean_edge_weights():
    k, w = 5, .5
    features = torch.diag(torch.ones(k)).float()
    ew = torch.ones(k) * w
    adj = torch.tensor([[i, (i + 1) % k] for i in range(k)]).transpose(0,1)
    h = aggregate(features, adj, "mean", edge_weights=ew)
    h_target = torch.zeros_like(features)
    for i in range(k):
        h_target[i, (i-1) % k] = w
    assert np.allclose(h.numpy(), h_target.numpy())


def test_symmetric_norm_coefficients_circular():
    k = 5
    edges = torch.tensor([[i, (i + 1) % k] for i in range(k)]).transpose(0,1)
    coeff = symmetric_normalization_coefficients(edges)
    assert np.allclose(np.ones(k), coeff.numpy())


def test_symmetric_norm_coefficients_complete():
    k = 5
    g = nx.complete_graph(5)
    row, col = nx.adjacency_matrix(g).todense().nonzero()
    edges = torch.tensor(np.stack((row, col), axis=0))
    coeff = symmetric_normalization_coefficients(edges)
    assert np.allclose(np.ones(edges.shape[1]) / (k-1), coeff.numpy())


def test_add_self_loops():
    g = nx.karate_club_graph()
    row, col = nx.adjacency_matrix(g).todense().nonzero()
    edges = torch.tensor(np.stack((row, col), axis=0))
    edges_ = [(x,y) for x,y in add_self_loops(edges).transpose(0,1).numpy().tolist()]
    for n in range(len(g)):
        assert (n,n) in edges_