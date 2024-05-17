from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import TopKPooling
from torch.nn import Parameter
import torch

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
        self.pooling = TopKPooling(in_channels, ratio=self.ratio, nonlinearity=non_linearity)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()
        x, edge_index, edge_attr, batch, perm, score = self.pooling(x, edge_index, edge_attr, batch, attn=score)

        return x, edge_index, edge_attr, batch, perm
    
# class DualGCN(torch.nn.Module):
    
#     def __init__(self, in_dim, out_dim, act, p):
#         super(DualGCN, self).__init__()
#         self.proj1 = nn.Linear(in_dim, out_dim)
#         self.proj2 = nn.Linear(in_dim, out_dim)
#         self.act = act
#         self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

#     def forward(self, g, h, mask):
#         # mask: 1-d torch showing two types of nodes (original node = (false) vs cycle node = (true))
#         # no interconnection between itself
#         h = self.drop(h)
#         h_n = h[~mask,:] # original node
#         h_c = h[mask,:] # cycle node
#         g_ntoc = g[mask, :][:, ~mask] # original to cycle
#         g_cton = g[~mask, :][:, mask] # cycle to original
#         h_n = self.proj1(h_n)
#         h_c = self.proj2(h_c)
#         h_c = torch.matmul(g_ntoc, h_n)
#         h_n = torch.matmul(g_cton, h_c)
#         h[mask, :] = h_c
#         h[~mask, :] = h_n
#         h = self.act(h)
#         return h

# class GraphAttentionLayer(nn.Module):
#     """
#     Graph Attention Layer (GAT) as described in the paper `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

#         This operation can be mathematically described as:

#             e_ij = a(W h_i, W h_j)
#             o_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k(exp(e_ik))     
#             h_i' = o(Σ_j(o_ij W h_j))
            
#             where h_i and h_j are the feature vectors of nodes i and j respectively, W is a learnable weight matrix,
#             a is an attention mechanism that computes the attention coefficients e_ij, and σ is an activation function.

#     """
#     def __init__(self, in_features: int, out_features: int, n_heads: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
#         super(GraphAttentionLayer, self).__init__()

#         self.n_heads = n_heads # Number of attention heads
#         self.concat = concat # wether to concatenate the final attention heads
#         self.dropout = dropout # Dropout rate

#         if concat: # concatenating the attention heads
#             self.out_features = out_features # Number of output features per node
#             assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
#             self.n_hidden = out_features // n_heads
#         else: # averaging output over the attention heads (Used in the main paper)
#             self.n_hidden = out_features

#         #  A shared linear transformation, parametrized by a weight matrix W is applied to every node
#         #  Initialize the weight matrix W 
#         self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))

#         # Initialize the attention weights a
#         self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

#         self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
#         self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients

#         self.reset_parameters() # Reset the parameters


#     def reset_parameters(self):
#         """
#         Reinitialize learnable parameters.
#         """
#         nn.init.xavier_normal_(self.W)
#         nn.init.xavier_normal_(self.a)
    

#     def _get_attention_scores(self, h_transformed: torch.Tensor):
#         """calculates the attention scores e_ij for all pairs of nodes (i, j) in the graph
#         in vectorized parallel form. for each pair of source and target nodes (i, j),
#         the attention score e_ij is computed as follows:

#             e_ij = LeakyReLU(a^T [Wh_i || Wh_j]) 

#             where || denotes the concatenation operation, and a and W are the learnable parameters.

#         Args:
#             h_transformed (torch.Tensor): Transformed feature matrix with shape (n_nodes, n_heads, n_hidden),
#                 where n_nodes is the number of nodes and out_features is the number of output features per node.

#         Returns:
#             torch.Tensor: Attention score matrix with shape (n_heads, n_nodes, n_nodes), where n_nodes is the number of nodes.
#         """
        
#         source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
#         target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])

#         # broadcast add 
#         # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
#         e = source_scores + target_scores.mT
#         return self.leakyrelu(e)

#     def forward(self,  h: torch.Tensor, adj_mat: torch.Tensor):
#         """
#         Performs a graph attention layer operation.

#         Args:
#             h (torch.Tensor): Input tensor representing node features.
#             adj_mat (torch.Tensor): Adjacency matrix representing graph structure.

#         Returns:
#             torch.Tensor: Output tensor after the graph convolution operation.
#         """
#         n_nodes = h.shape[0]

#         # Apply linear transformation to node feature -> W h
#         # output shape (n_nodes, n_hidden * n_heads)
#         h_transformed = torch.mm(h, self.W)
#         h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)

#         # splitting the heads by reshaping the tensor and putting heads dim first
#         # output shape (n_heads, n_nodes, n_hidden)
#         h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)
        
#         # getting the attention scores
#         # output shape (n_heads, n_nodes, n_nodes)
#         e = self._get_attention_scores(h_transformed)

#         # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
#         connectivity_mask = -9e16 * torch.ones_like(e)
#         e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores
        
#         # attention coefficients are computed as a softmax over the rows
#         # for each column j in the attention score matrix e
#         attention = F.softmax(e, dim=-1)
#         attention = F.dropout(attention, self.dropout, training=self.training)

#         # final node embeddings are computed as a weighted average of the features of its neighbors
#         h_prime = torch.matmul(attention, h_transformed)

#         # concatenating/averaging the attention heads
#         # output shape (n_nodes, out_features)
#         if self.concat:
#             h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
#         else:
#             h_prime = h_prime.mean(dim=0)

#         return h_prime