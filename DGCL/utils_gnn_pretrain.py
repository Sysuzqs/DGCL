import os

from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
import pandas as pd
from creat_data_DC import smile_to_graph

class TestbedDataset(InMemoryDataset):
   
    def __init__(self, root='tmp', dataset='', patt= 're',transform=None,
                 pre_transform=None, smile_graph=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.patt = patt
        self.processed_paths[0] = self.processed_paths[0] + self.patt

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(root, self.dataset)
            self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def processed_file_names(self):
        return [self.dataset + self.patt + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, root, dataset):
        data_list = []
        compound_iso_smiles = []
        df = pd.read_csv('data/' + root + '/data/' + dataset + '.csv')
        compound_iso_smiles += list(df['smiles'])
        compound_iso_smiles = set(compound_iso_smiles)  

        count = 0
        for smile in compound_iso_smiles:
            if len(smile) < 4:
                continue
            count = count + 1
            x_size, features, edge_index, edge_features, atoms = smile_to_graph(smile)
            
            num_edge_features = 11
            if edge_features is None:
                edge_features = torch.zeros([len(edge_index),num_edge_features])
                
            GCNData = DATA.Data(x=torch.Tensor(features),
                          edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                          x_size=torch.LongTensor([x_size]),
                          edge_size=torch.LongTensor([len(edge_index)]),
                          edge_attr=torch.Tensor(edge_features)
                               )
            data_list.append(GCNData)   

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]   

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)  
        torch.save((data, slices), self.processed_paths[0]) 
        
        print('Graph construction done. Saving to file.')  