import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

## Memory Inefficient Version
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

## Memory Efficient Version
def cycle_proc(g, h): 
    """
    g: adj matrix with 2 x edge torch tensor (value: 0 or 1)
    h: feature matrix with N x D torch tensor
    return: 
        adj matrix with cycle nodes added as shape 2 x edge
        h matrix with cycle nodes added as shape (N+Cycle) x D
    ex.) g_cyc: [g, boundary_matrix; coboundary_marix, 0]
    """
    # Find g_cyc [2 x num_edges]
    num_original_nodes = h.size(0)
    edges = g.t().cpu().numpy()
    g_nx = nx.Graph()
    g_nx.add_edges_from(edges)
    cyc = nx.cycle_basis(g_nx)
    cyc = [c for c in cyc if len(c) < 7 and 3 <= len(c)]
    new_size = num_original_nodes + len(cyc)
    g_cyc = g
    for c in range(len(cyc)):
        id = num_original_nodes + c
        for node_c in cyc[c]:
            tensor1 = torch.tensor([[id], [node_c]], device = g.device)
            tensor2 = torch.tensor([[node_c], [id]], device = g.device)
            g_cyc = torch.cat([g_cyc, tensor1, tensor2], dim=1)
    
    # Find h_cyc
    h_cyc = torch.zeros((new_size, h.size(1)), device = h.device)
    h_cyc[:h.size(0), :] = h

    return g_cyc, h_cyc

def initialize_cycle_M(g_cyc, h):
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

    g = torch.tensor([[0, 1],[1, 0],[0,2],[2,0],[0,3],[3,0],[1,2],[2,1]]).T
    print(g.shape)
    g_cyc, h_cyc = cycle_proc(g, h)
    print(g_cyc, h_cyc)