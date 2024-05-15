from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

## Memory Inefficient Version
class CycleProcessor():
    def __init__(self):
        pass
    
    def cycle_adj(self, g): 
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

    def initialize_cycle(self, g_cyc, h):
        """
        g_cyc: adj matrix with (N+cycle) x (N+cycle) torch tensor (value: 0 or 1)
        h: feature matrix with N x D torch tensor
        return: (N+cycle) x D torch tensor with cycle nodes initialized to zeros
    """
        h_cyc = torch.zeros((g_cyc.size(0), h.size(1)), device = h.device)
        h_cyc[:h.size(0), :] = h
        return h_cyc

    ## Memory Efficient Version
    def cycle_proc(self, h, g, start_index): 
        """
        g: adj matrix with 2 x edge torch tensor (value: 0 or 1)
        h: feature matrix with N x D torch tensor
        return: 
            adj matrix with cycle nodes added as shape 2 x edge
            h matrix with cycle nodes added as shape (N+Cycle) x D
        ex.) g_cyc: [g, boundary_matrix; coboundary_marix, 0]
        """
        # Find g_cyc [2 x num_edges]
        edges = g.t().cpu().numpy()
        g_nx = nx.Graph()
        g_nx.add_edges_from(edges)
        cyc = nx.cycle_basis(g_nx)
        cyc = [c for c in cyc if len(c) < 7 and 3 <= len(c)]

        g_cyc = g
        for c in range(len(cyc)):
            id = start_index + c
            for node_c in cyc[c]:
                tensor1 = torch.tensor([[id], [node_c]], device = g.device)
                tensor2 = torch.tensor([[node_c], [id]], device = g.device)
                g_cyc = torch.cat([g_cyc, tensor1, tensor2], dim=1)
        
        # Find h_cyc
        h_cyc = torch.zeros((len(cyc), h.size(1)), device = h.device)

        return h_cyc, g_cyc
    
    def cycle_proc_batch(self, h_concated, g_concated, batch):
        h_seperated = [h_concated[batch == i] for i in range(batch[-1] + 1)]
        g_seperated = [[] for _ in range(batch[-1] + 1)]
        for edge in g_concated.T:
            assert batch[edge[0]] == batch[edge[1]] # both nodes conneted the edge should be in same graph.
            g_seperated[batch[edge[0]]].append(edge.unsqueeze(0))
        g_seperated = [torch.cat(torch_list, dim=0).T for torch_list in g_seperated]
        
        assert len(h_seperated) == len(g_seperated)
        for i, (h, g) in enumerate(zip(h_seperated, g_seperated)):
            h_cyc, g_cyc = self.cycle_proc(h, g, len(h_concated))
            h_concated = torch.cat((h_concated, h_cyc), dim=0)
            g_concated = torch.cat((g_concated, g_cyc), dim=1)
            batch = torch.cat((batch, torch.tensor([i] * len(h_cyc), dtype=torch.int64)))
            
        return h_concated, g_concated, batch
        
    def initialize_cycle_M(self, g_cyc, h):
        """
        g_cyc: adj matrix with (N+cycle) x (N+cycle) torch tensor (value: 0 or 1)
        h: feature matrix with N x D torch tensor
        return: (N+cycle) x D torch tensor with cycle nodes initialized to zeros
    """
        h_cyc = torch.zeros((g_cyc.size(0), h.size(1)), device = h.device)
        h_cyc[:h.size(0), :] = h
        return h_cyc

    def __call__(self, x, edge_index, batch):
        h_cyc, g_cyc, batch = self.cycle_proc_batch(x, edge_index, batch)
        return h_cyc, g_cyc, batch

    # if __name__ == '__main__':
    #     adj = torch.tensor([[0, 1, 1, 1],
    #                         [1, 0, 1, 0],
    #                         [1, 1, 0, 0],
    #                         [1, 0, 0, 0]])
    #     h = torch.tensor([[1, 1],
    #                     [2, 2],
    #                     [3, 3],
    #                     [4, 4]])
    #     g_cyc = cycle_adj(adj)
    #     h_cyc = initialize_cycle(g_cyc, h)
    #     print(g_cyc, h_cyc)

    #     g = torch.tensor([[0, 1],[1, 0],[0,2],[2,0],[0,3],[3,0],[1,2],[2,1]]).T
    #     print(g.shape)
    #     g_cyc, h_cyc = cycle_proc(g, h)
    #     print(g_cyc, h_cyc)