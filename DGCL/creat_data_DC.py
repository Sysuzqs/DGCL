from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
from rdkit.Chem import rdmolfiles, rdmolops
import networkx as nx
import torch

def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1:]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetFormalCharge(),[-1, -2, 0, 1, 2])+
                    one_of_k_encoding_unk(atom.GetChiralTag(),[0,1,2,3])+
                    one_of_k_encoding_unk(atom.GetHybridization(),[Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])+
                    [atom.GetIsAromatic()]+
                    [atom.GetMass()*0.01])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_edge_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    edge_features = []
    edge_index = []

    for bond in mol.GetBonds():

        bond_type = bond.GetBondType()
        bond_feature = [0, 0, 0, 0, 0, 0]
        if bond_type == rdchem.BondType.SINGLE:
            bond_feature[0] = 1
        elif bond_type == rdchem.BondType.DOUBLE:
            bond_feature[1] = 1
        elif bond_type == rdchem.BondType.TRIPLE:
            bond_feature[2] = 1
        elif bond_type == rdchem.BondType.AROMATIC:
            bond_feature[3] = 1
        elif bond_type == rdchem.BondDir.ENDUPRIGHT:
            bond_feature[4] = 1
        elif bond_type == rdchem.BondDir.ENDDOWNRIGHT:
            bond_feature[5] = 1
        bond_feature = np.array(bond_feature+
                               one_of_k_encoding_unk(bond.GetStereo(),[0,1,2,3,4]))

        bond_index = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        edge_index.append(bond_index)
        
        edge_features.append(bond_feature / sum(bond_feature))

    return edge_features, edge_index


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    atoms = []
    features = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            atoms.append(atom.GetSymbol().lower())
        else:
            atoms.append(atom.GetSymbol())
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    
    edge_features, edge_index = get_edge_features(smile) 

    return c_size, features, edge_index, edge_features, atoms 