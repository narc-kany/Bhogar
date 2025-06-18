import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def mol_to_features(smiles):
    """
    Converts a SMILES string to a list of molecular descriptors (features).
    Returns None if SMILES is invalid or an error occurs.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # print(f"Invalid SMILES: {smiles}") # Uncomment for debugging
            return None
        return [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol)
        ]
    except Exception as e:
        # print(f"Error with SMILES: {smiles} | {e}") # Uncomment for debugging
        return None

class AlzheimerDataset(Dataset):
    """
    A PyTorch Dataset for Alzheimer's drug data.
    """
    def __init__(self, features, labels=None):
        self.X = torch.tensor(features, dtype=torch.float32)
        # Labels are optional, useful for training but not strictly for inference
        self.y = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]