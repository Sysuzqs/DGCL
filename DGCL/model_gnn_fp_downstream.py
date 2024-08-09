import torch.nn as nn
import torch.nn.functional as F
import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Model_gin_fp(nn.Module):
    def __init__(self, n_output=1, output_dim=512, dropout=0.2, encoder=None):
        super(Model_gin_fp, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.encoder = encoder
        self.dropout = dropout

        self.pre = nn.Sequential(
            nn.Linear(output_dim*2, 512),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_output)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1489,1024),
            nn.Dropout(dropout),
            nn.Linear(1024,output_dim)
        )

    def forward(self, data):
        x, edge_index, batch, y ,edge_attr, fp, w= data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.fps, data.w
        
        fp = fp.reshape(len(fp)//1489,1489)

        x1 = self.encoder(x, edge_index, edge_attr, batch)
        xf = F.normalize(x1)
        
        x_fp = self.fc(fp)
        x_fp = F.normalize(x_fp)
        
        x_sum = torch.cat((xf,x_fp),dim=1)
        
        xc = self.pre(x_sum)
        out = xc.reshape(-1)
        
        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]
        

        return out, y, w, out_loss, y_loss, w_loss

    
class Model_gat_fp(nn.Module):
    def __init__(self, n_output=1, output_dim=512, dropout=0.2, encoder=None):
        super(Model_gat_fp, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.encoder = encoder
        self.dropout = dropout

        self.pre = nn.Sequential(
            nn.Linear(output_dim*2, 512),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_output)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1489,1024),
            nn.Dropout(dropout),
            nn.Linear(1024,output_dim)
        )

    def forward(self, data):
        x, edge_index, batch, y ,edge_attr, fp, w= data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.fps, data.w
        
        fp = fp.reshape(len(fp)//1489,1489)

        x1, g_weight = self.encoder(x, edge_index, edge_attr, batch)
        xf = F.normalize(x1)
        
        x_fp = self.fc(fp)
        x_fp = F.normalize(x_fp)
        
        x_sum = torch.cat((x1,x_fp),dim=1)
 
        xc = self.pre(x_sum)
        out = xc.reshape(-1)

        len_w = edge_index.shape[1]
        weight = g_weight[1]
        edge_weight = weight[:len_w]
        x_weight = weight[len_w:]
        
        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]

        return out, y, w, edge_weight, x_weight, out_loss, y_loss, w_loss

    
class Model_gnn_fp(nn.Module):
    def __init__(self, n_output=1, output_dim=512, dropout=0.2, encoder1=None,encoder2=None):
        super(Model_gnn_fp, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.dropout = dropout
        
        self.pre = nn.Sequential( 
            nn.Linear(output_dim*3, 64),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(64, self.n_output)
        )

        '''
        self.pre = nn.Sequential(
            nn.Linear(output_dim*3, 512),  
            nn.ReLU(),                     
            nn.BatchNorm1d(512),          
            nn.Dropout(dropout),          
            nn.Linear(512, 256),          
            nn.ReLU(),                  
            nn.BatchNorm1d(256),           
            nn.Linear(256, 128),         
            nn.ReLU(),                   
            nn.Dropout(dropout),         
            nn.Linear(128, 64),           
            nn.ReLU(),                   
            nn.Linear(64, self.n_output),
        )
        '''
        
        
        self.fc = nn.Sequential(
            nn.Linear(1489,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024,output_dim)
        )

    def forward(self, data):
        x, edge_index, batch, y ,edge_attr, fp, w = data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.fps, data.w
        
        fp = fp.reshape(len(fp)//1489,1489)

        x1, g_weight = self.encoder1(x, edge_index, edge_attr, batch)
        xf1 = F.normalize(x1)
        
        x2 = self.encoder2(x, edge_index, edge_attr, batch)
        xf2 = F.normalize(x2)
        
        x_fp = self.fc(fp)
        x_fp = F.normalize(x_fp)
        
        xf = torch.cat((x1,x2,x_fp),dim=1)
        
        xc = self.pre(xf)
        out = xc.reshape(-1)

        len_w = edge_index.shape[1]
        weight = g_weight[1]
        edge_weight = weight[:len_w]
        x_weight = weight[len_w:]
        
        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]

        return out, y, w, edge_weight, x_weight, out_loss, y_loss, w_loss