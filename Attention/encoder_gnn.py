import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap
from torch_geometric.nn import Sequential
from torch.nn import Linear

class GINNet(torch.nn.Module):
    def __init__(self, eps=0. , train_eps=True):
    
        super(GINNet, self).__init__()
        
        self.gin1 = GINEConv(nn=Linear(93,512).cuda(), eps=eps, train_eps=train_eps).cuda()
        self.gin2 = GINEConv(nn=Linear(512,1024).cuda(), eps=eps, train_eps=train_eps).cuda()
        self.gin3 = GINEConv(nn=Linear(1024,512).cuda(), eps=eps, train_eps=train_eps).cuda()
        
        self.fc1 = Linear(11,93).cuda()
        self.fc2 = Linear(11,512).cuda()
        self.fc3 = Linear(11,1024).cuda()

    def forward(self, x1, edge_index, edge_attr, batch):
        
        edge_attr1 = self.fc1(edge_attr)
        edge_attr2 = self.fc2(edge_attr)
        edge_attr3 = self.fc3(edge_attr)
                
        x1 = self.gin1(x1, edge_index, edge_attr1)
        x1 = F.elu(x1)

        x1 = self.gin2(x1, edge_index, edge_attr2)
        x1 = F.elu(x1)
        
        x1 = self.gin3(x1, edge_index, edge_attr3)
        x1 = F.elu(x1)
        
        x_mean = gmp(x1, batch)

        return x_mean

    
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=93, output_dim=512, heads=10,edge_dim=11,dropout=0.2):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(num_features_xd, output_dim, heads=10, edge_dim=11, dropout=dropout)
        self.gat2 = GATConv(output_dim * heads, output_dim, heads=1 ,edge_dim=11, dropout=dropout)

    def forward(self, x1, edge_index, edge_attr, batch):
    
        x1, weight = self.gat1(x1, edge_index, edge_attr, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training) 
   
        x1,weight2 = self.gat2(x1, edge_index, edge_attr,return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x_mean = gmp(x1, batch)

        return x_mean, weight2