"""
Data loading and preprocessing utilities
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import sys
import types

from .dataset import CodeSmellDataset


def load_pickle_with_compatibility(data_path: str):
    """
    Load pickle file with pandas version compatibility
    
    Args:
        data_path: Path to pickle file
        
    Returns:
        DataFrame with loaded data
    """
    # Try pandas read_pickle first (more compatible)
    try:
        df = pd.read_pickle(data_path)
        print("✓ Loaded pickle file using pd.read_pickle")
        return df
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"⚠ pd.read_pickle failed: {e}")
        print("Attempting with standard pickle.load...")
        
        # Fallback: try to patch the missing module
        if 'pandas.core.indexes.numeric' not in sys.modules:
            sys.modules['pandas.core.indexes.numeric'] = types.ModuleType('pandas.core.indexes.numeric')
        
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
        print("✓ Loaded pickle file using pickle.load with compatibility patch")
        return df


def load_and_prepare_data(data_path: str, test_size: float = 0.2, 
                         val_size: float = 0.1, random_state: int = 42) -> Tuple:
    """
    Load data from CSV or PKL and prepare train/val/test splits
    
    Args:
        data_path: Path to the CSV or PKL file
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, embedding_dim, n_metrics)
    """
    print(f"Loading data from {data_path}...")
    
    # Load data based on file extension
    if data_path.endswith('.pkl'):
        df = load_pickle_with_compatibility(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        # Parse string representations to lists
        print("Parsing metrics and embeddings...")
        df['metrics'] = df['metrics'].apply(eval)
        df['embedding'] = df['embedding'].apply(eval)
        print("✓ Loaded and parsed CSV file")
    else:
        raise ValueError("Data file must be .csv or .pkl format")
    
    print(f"\n{'='*60}")
    print(f"DATA LOADED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle different column names (metrics vs metric)
    if 'metric' in df.columns and 'metrics' not in df.columns:
        df['metrics'] = df['metric']
        print("Note: Renamed 'metric' column to 'metrics'")
    
    # Verify required columns
    required_cols = ['embedding', 'metrics', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Found columns: {df.columns.tolist()}")
    
    print(f"\n{'='*60}")
    print(f"PREPARING DATA")
    print(f"{'='*60}")
    
    # Convert to numpy arrays
    embeddings = np.array(df['embedding'].tolist())
    metrics = np.array(df['metrics'].tolist())
    labels = df['label'].values
    
    embedding_dim = embeddings.shape[1]
    n_metrics = metrics.shape[1]
    
    print(f"Total samples: {len(df)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of metrics: {n_metrics}")
    print(f"Label distribution: {np.bincount(labels.astype(int))}")
    
    # First split: separate test set
    X_train_val_emb, X_test_emb, X_train_val_met, X_test_met, y_train_val, y_test = train_test_split(
        embeddings, metrics, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate validation from training
    X_train_emb, X_val_emb, X_train_met, X_val_met, y_train, y_val = train_test_split(
        X_train_val_emb, X_train_val_met, y_train_val, 
        test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    print(f"\n{'='*60}")
    print(f"TRAIN/VAL/TEST SPLIT")
    print(f"{'='*60}")
    print(f"Train set: {len(y_train)} samples ({len(y_train)/len(df)*100:.1f}%)")
    print(f"  - Label 0: {np.sum(y_train == 0)} samples")
    print(f"  - Label 1: {np.sum(y_train == 1)} samples")
    print(f"Validation set: {len(y_val)} samples ({len(y_val)/len(df)*100:.1f}%)")
    print(f"  - Label 0: {np.sum(y_val == 0)} samples")
    print(f"  - Label 1: {np.sum(y_val == 1)} samples")
    print(f"Test set: {len(y_test)} samples ({len(y_test)/len(df)*100:.1f}%)")
    print(f"  - Label 0: {np.sum(y_test == 0)} samples")
    print(f"  - Label 1: {np.sum(y_test == 1)} samples")
    print(f"{'='*60}\n")
    
    # Create datasets
    train_dataset = CodeSmellDataset(X_train_emb, X_train_met, y_train)
    val_dataset = CodeSmellDataset(X_val_emb, X_val_met, y_val)
    test_dataset = CodeSmellDataset(X_test_emb, X_test_met, y_test)
    
    return train_dataset, val_dataset, test_dataset, embedding_dim, n_metrics
