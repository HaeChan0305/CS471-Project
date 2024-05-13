from torch_geometric.nn import GCNConv
from torch_geometric.utils import filter_adj
from torch_geomtric.nn import TopKPooling as topk
from torch.nn import Parameter
import torch.nn as nn
import torch


class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
    
class DualGCN(nn.Module):
    
    def __init__(self, in_dim, out_dim, act, p):
        super(DualGCN, self).__init__()
        self.proj1 = nn.Linear(in_dim, out_dim)
        self.proj2 = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h, mask):
        # mask: 1-d torch showing two types of nodes (original node = (false) vs cycle node = (true))
        # no interconnection between itself
        h = self.drop(h)
        h_n = h[~mask,:] # original node
        h_c = h[mask,:] # cycle node
        g_ntoc = g[mask, :][:, ~mask] # original to cycle
        g_cton = g[~mask, :][:, mask] # cycle to original
        h_n = self.proj1(h_n)
        h_c = self.proj2(h_c)
        h_c = torch.matmul(g_ntoc, h_n)
        h_n = torch.matmul(g_cton, h_c)
        h[mask, :] = h_c
        h[~mask, :] = h_n
        h = self.act(h)
        return h