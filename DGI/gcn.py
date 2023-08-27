"""
This code was copied from the GCN implementation in DGL examples.
"""
import torch
import torch.nn as nn
#DGL implement https://github.com/dmlc/dgl
from DGI.graphconv import GATConv

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 activation,
                 dgiGraphAct):
        super(GAT, self).__init__()
        self.g = g
        self.residual = False
        self.layer = GATConv(in_feats, n_hidden, num_heads, feat_drop, attn_drop, negative_slope, self.residual, activation=activation)

    def forward(self, features):
        h = features
        h = self.layer(self.g, h

        return h