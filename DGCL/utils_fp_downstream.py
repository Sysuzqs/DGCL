import os
from torch_geometric.data import InMemoryDataset
import torch
from creat_data_DC import smile_to_graph
import deepchem as dc
import numpy as np
from rdkit.Chem import AllChem, MACCSkeys
from deepchem.feat.base_classes import MolecularFeaturizer
from pubchemfp import GetPubChemFPs
import pandas as pd
np.set_printoptions(threshold=np.inf)
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolfiles, rdmolops
from deepchem.splits import ScaffoldSplitter
from deepchem.feat import CircularFingerprint
from deepchem.trans import BalancingTransformer
from torch_geometric import data as DATA


class CombinedFingerprintsFeaturizer(MolecularFeaturizer):
    def __init__(self):
        super(CombinedFingerprintsFeaturizer,self).__init__()
        
    def _featurize(self,mol):
        fp = []
        
        pubchem_fp = GetPubChemFPs(mol)
        maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
        erg_fp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
        
        fp = np.concatenate([pubchem_fp, maccs_fp, erg_fp])
        
        return fp

    
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='tmp', dataset='train', task='bbbp',
                 transform=None, pre_transform=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.task = task

        print('Processing data for task {}, dataset {}...'.format(task, dataset))
        self.process(root, task)

        if dataset == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif dataset == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif dataset == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def processed_file_names(self):
        return [self.task + '_train.pt', self.task + '_valid.pt', self.task + '_test.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def process(self, root, task):
        
        csv_file = 'dataset/'+task+'/raw/smiles.csv'
        data = pd.read_csv(csv_file, header=None)

        smiles_all = data.iloc[:, 0].values
        properties_all = data.iloc[:, 1:].values
        smiles_all = [str(smile) for smile in smiles_all]

        smiles = []
        properties = []

        for i, smile in enumerate(smiles_all):
            mol = Chem.MolFromSmiles(smile)

            if mol is not None:
                smiles.append(smile)
                properties.append(properties_all[i])

        smiles = np.array(smiles)
        properties = np.array(properties)
           
        if task != 'bbbp':
            indices = np.arange(len(smiles))
            np.random.shuffle(indices)
            smiles = smiles[indices]
            properties = properties[indices]
       
        comfp_featurizer = CombinedFingerprintsFeaturizer()
        features = comfp_featurizer.featurize(smiles)
        
        n_samples = len(smiles)
        n_tasks = properties.shape[1]
        w = np.ones((n_samples, n_tasks))

        dataset = dc.data.NumpyDataset(X=features, y=properties, w=w, ids=smiles)

        splitter = ScaffoldSplitter()
        train, valid, test = splitter.train_valid_test_split(dataset)

        transformer = BalancingTransformer(dataset=train)
        train = transformer.transform(train)
        valid = transformer.transform(valid)
        test = transformer.transform(test)

        save(self, train, 0)
        save(self, valid, 1)
        save(self, test, 2)

def save(self, dataset, path):
    data_list = []
    label_new = np.nan_to_num(dataset.y,nan=999)
    for i in range(len(dataset)):    
        smile = dataset.ids[i]  
        label = label_new[i]
        weights = dataset.w[i]
        mfp = dataset.X[i]

        x_size, features, edge_index, edge_features, atoms = smile_to_graph(smile)   
        if len(edge_index) > 0:
            edge_index = torch.LongTensor(edge_index).transpose(1, 0)
        else:
            edge_index = torch.LongTensor(edge_index)     
        GCNData = DATA.Data(x=torch.Tensor(features),
                            edge_index=edge_index,
                            y=torch.Tensor(label),
                            edge_attr = torch.Tensor(edge_features),
                            w =torch.Tensor(weights),
                            fps = torch.Tensor(mfp))    

        data_list.append(GCNData)     
    if self.pre_filter is not None:
        data_list = [data for data in data_list if self.pre_filter(data)]   

    if self.pre_transform is not None:
        data_list = [self.pre_transform(data) for data in data_list]   
    print('Graph construction done. Saving to file.')    
    data, slices = self.collate(data_list)      

    torch.save((data, slices), self.processed_paths[path])  


def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
        
def save_RMSEs(RMSEs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, RMSEs)) + '\n')