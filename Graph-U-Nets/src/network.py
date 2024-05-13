import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from utils.ops import GCN, GraphUnet, Initializer, norm_g, DualGCN


class GNet(nn.Module):
    def __init__(self, in_dim, n_classes, args):
        super(GNet, self).__init__()
        self.n_act = getattr(nn, args.act_n)()
        self.c_act = getattr(nn, args.act_c)()
        self.s_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        self.s_cyclegcn_first = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        self.s_cyclegcn_last = DualGCN(args.l_dim, args.l_dim, self.n_act, args.drop_n)
        self.g_unet = GraphUnet(
            args.ks, args.l_dim, args.l_dim, args.l_dim, self.n_act,
            args.drop_n)
        self.out_l_1 = nn.Linear(3*args.l_dim*(args.l_num+1), args.h_dim)
        self.out_l_2 = nn.Linear(args.h_dim, n_classes)
        self.out_drop = nn.Dropout(p=args.drop_c)
        Initializer.weights_init(self)

    def forward(self, gs, hs, labels):
        # find boundary matrix and coboundary matrix
        hs = self.embed(gs, hs)
        logits = self.classify(hs)
        return self.metric(logits, labels)

    def embed(self, gs, hs):
        o_hs = []
        for g, h in zip(gs, hs):
            h = self.embed_one(g, h)
            o_hs.append(h)
        hs = torch.stack(o_hs, 0)
        return hs

    def embed_one(self, g, h):
        ## additional step for adding cycle node (g is just adj matrix with scalar values)
        cyc = nx.cycle_basis(nx.from_numpy_array(g.cpu().numpy()))
        # filter those beyond length > k = 7 and below length < k = 3
        cyc = [c for c in cyc if len(c) < 7 and 3 <= len(c)]

        g = norm_g(g)

        # Create new graph including cycle nodes <- appended to the end of the graph
        num_original_nodes = g.size(0)
        new_size = num_original_nodes + len(cyc)
        g_cyc = torch.zeros((new_size, new_size), device = g.device)
        g_cyc[:g.size(0),:g.size(0)] = g
        for i, cycle in enumerate(cyc):
            for j in range(len(cycle)):
                g_cyc[g.size(0)+i, cycle[j]] = 1 # creating coboundary matrix (0,1,3,4) <-> [1 1 0 1 1 0 ...] row vec
                g_cyc[cycle[j], g.size(0)+i] = 1 # creating boundary matrix (0,1,3,4) <-> [1;1;0;1;1;0;...] col vec
        mask = [False] * g.size(0) + [True] * len(cyc)
        mask = torch.tensor(mask, dtype = torch.bool, device = g.device)
        
        # initialize cycle embedding to zeros
        h_cyc = torch.zeros((g.size(0)+len(cyc), h.size(1)), device = h.device)
        h_cyc[:g.size(0), :] = h
        h_cyc[g.size(0):, :] = torch.zeros(len(cyc), h.size(1))

        # cycle gcn
        h_cyc = self.s_cyclegcn_first(g_cyc, h_cyc) # just for initialization of h_cyc
        h_cyc = self.s_cyclegcn_last(g_cyc, h_cyc, mask)
            
        # prun cycle nodes before going into gcn
        h_c = h_cyc[:g.size(0), :]
    
        # weighted sum between h_c and original h for stabiity
        h = self.s_gcn(g, h)
        #print(print(torch.isnan(h_c).any()))
        alpha = 0.5
        h = alpha * h + (1 - alpha) * h_c
        hs = self.g_unet(g, h)
        h = self.readout(hs)
        return h

    def readout(self, hs):
        h_max = [torch.max(h, 0)[0] for h in hs]
        h_sum = [torch.sum(h, 0) for h in hs]
        h_mean = [torch.mean(h, 0) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean)
        return h

    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)

    def metric(self, logits, labels):
        loss = F.nll_loss(logits, labels)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        return loss, acc
