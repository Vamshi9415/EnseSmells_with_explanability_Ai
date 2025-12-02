"""
Dataset module for code smell detection
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class CodeSmellDataset(Dataset):
    """Custom Dataset for Code Smell Detection"""
    
    def __init__(self, embeddings, metrics, labels):
        """
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            metrics: numpy array of shape (n_samples, n_metrics)
            labels: numpy array of shape (n_samples,)
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.metrics = torch.FloatTensor(metrics)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'metrics': self.metrics[idx],
            'label': self.labels[idx]
        }
