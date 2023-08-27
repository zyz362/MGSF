import torch
import torch.nn as nn
import math
from DGI.gcn import GAT


class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, num_heads, feat_drop, attn_drop, negative_slope, activation, graphAct):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GAT(g, in_feats, n_hidden, num_heads, feat_drop, attn_drop, negative_slope, activation, graphAct)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())  
            features = features[perm]
        features = self.conv(features)
        return features


class DGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, num_heads, feat_drop, attn_drop, negative_slope, gcnAct, graphAct):
        super(DGI, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, num_heads, feat_drop, attn_drop, negative_slope, gcnAct, graphAct)

    def forward(self, features):
        positive = self.encoder(features, corrupt=False)  # features  userembedding
        return positive


