import random
import numpy as np
import torch
import torch.nn as nn

class GNNCon(torch.nn.Module):
    def __init__(self,eps=0. , train_eps=True, n_output=512, num_features_xt=78, output_dim=512, dropout=0.2, encoder1=None, encoder2=None):

        super(GNNCon, self).__init__()
        self.num_features_xt = num_features_xt
        self.n_output = n_output
        self.dropout = dropout
        self.output_dim = output_dim

        self.gat = encoder1
        self.gin = encoder2

        # predict head
        self.pre_head = nn.Sequential(
            nn.Linear(output_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, self.n_output)
        )

    def forward(self, data1):
        x, edge_index, batch, x_size, edge_size, edge_attr = data1.x, data1.edge_index, data1.batch, data1.x_size, data1.edge_size, data1.edge_attr

        
        x = x.to('cuda')
        edge_index = edge_index.to('cuda')
        batch = batch.to('cuda')
        edge_attr = edge_attr.to('cuda')

        x1, weight = self.gat(x, edge_index, edge_attr, batch)
        out1 = self.pre_head(x1)

        x2 = self.gin(x, edge_index, edge_attr, batch)
        out2 = self.pre_head(x2)

        len_w = edge_index.shape[1]
        w1 = weight[1]
        ew1 = w1[:len_w]
        xw1 = w1[len_w:]

        return x1, out1, x2, out2, ew1, xw1
