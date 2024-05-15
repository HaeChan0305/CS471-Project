import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

def cycle_adj(g): 
    """
    g: adj matrix with N x N torch tensor (value: 0 or 1)
    return: adj matrix with cycle nodes added
    ex.) g_cyc: [g, boundary_matrix; coboundary_marix, 0]
    """
    cyc = nx.cycle_basis(nx.from_numpy_array(g.cpu().numpy()))
    cyc = [c for c in cyc if len(c) < 7 and 3 <= len(c)]
    num_original_nodes = g.size(0)
    new_size = num_original_nodes + len(cyc)
    g_cyc = torch.zeros((new_size, new_size), device = g.device)
    g_cyc[:g.size(0),:g.size(0)] = g
    for i, cycle in enumerate(cyc):
        for j in range(len(cycle)):
            g_cyc[g.size(0)+i, cycle[j]] = 1
            g_cyc[cycle[j], g.size(0)+i] = 1
    return g_cyc

def initialize_cycle(g_cyc, h):
    """
    g_cyc: adj matrix with (N+cycle) x (N+cycle) torch tensor (value: 0 or 1)
    h: feature matrix with N x D torch tensor
    return: (N+cycle) x D torch tensor with cycle nodes initialized to zeros
   """
    h_cyc = torch.zeros((g_cyc.size(0), h.size(1)), device = h.device)
    h_cyc[:h.size(0), :] = h
    return h_cyc

if __name__ == '__main__':
    adj = torch.tensor([[0, 1, 1, 1],
                        [1, 0, 1, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 0]])
    h = torch.tensor([[1, 1],
                     [2, 2],
                     [3, 3],
                     [4, 4]])
    g_cyc = cycle_adj(adj)
    h_cyc = initialize_cycle(g_cyc, h)
    print(g_cyc, h_cyc)