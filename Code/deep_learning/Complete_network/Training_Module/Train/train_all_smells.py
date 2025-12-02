"""
Train EnseSmells model for ALL Embedding types and ALL Code Smell types
using Stratified K-Fold Cross Validation.
Automatically handles Class Imbalance via weighted loss.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset

# Add Complete_network directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
complete_network_dir = os.path.dirname(os.path.dirname(current_dir))
if complete_network_dir not in sys.path:
    sys.path.insert(0, complete_network_dir)

from Ensemble_module.EnseSmells import EnseSmells
from Training_Module.Train.config import TrainingConfig
from Training_Module.Train.data_loader import load_and_prepare_data
from Training_Module.Train.trainer import train_model, validate
from Training_Module.Train.metrics import print_metrics
from Training_Module.Train.utils import save_json, plot_training_history


def train_smell_type_kfold(smell_name: str, embedding_type: str, data_path: str, save_dir: str, config_params: dict):
    """
    Train model using Stratified K-Fold Cross Validation for a specific smell and embedding.
    """
    print("\n" + "#"*80)
    print(f"STARTING 5-FOLD CV | MODEL: {embedding_type} | SMELL: {smell_name}")
    print("#"*80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(config_params['random_seed'])
    np.random.seed(config_params['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config_params['random_seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------
    # Load and Prepare (Merging Train/Val for CV)
    try:
        train_ds, val_ds, test_ds, embedding_dim, n_metrics = load_and_prepare_data(
            data_path, 
            config_params['test_size'], 
            config_params['val_size'], 
            config_params['random_seed']
        )
    except Exception as e:
        print(f"CRITICAL ERROR LOADING DATA: {e}")
        return None

    dev_dataset = ConcatDataset([train_ds, val_ds])
    dev_labels = np.concatenate([train_ds.labels, val_ds.labels])
    
    # K-Fold Initializer 
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config_params['random_seed'])
    
    fold_results = []
    
    # -------------------------------------------------------
    # 2. START K-FOLD LOOP
    # -------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dev_labels)), dev_labels)):
        print(f"\n--- FOLD {fold+1}/{k_folds} ---")
        
        fold_dir = os.path.join(save_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Samplers & Loaders
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dev_dataset, batch_size=config_params['batch_size'], sampler=train_subsampler)
        val_loader = DataLoader(dev_dataset, batch_size=config_params['batch_size'], sampler=val_subsampler)
        
        # Class Weights Calculation
        fold_train_labels = dev_labels[train_idx]
        n_pos = np.sum(fold_train_labels == 1)
        n_neg = np.sum(fold_train_labels == 0)
        weight_value = n_neg / (n_pos + 1e-6)
        pos_weight = torch.tensor([weight_value], dtype=torch.float).to(device)
        
        # Initialize Model
        model = EnseSmells(embedding_dim=embedding_dim, input_metrics=n_metrics)
        model = model.to(device)
        
        # Loss & Optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config_params['lr'], 
            weight_decay=config_params['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )

        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=config_params['epochs'],
            save_dir=fold_dir,
            early_stopping_patience=config_params['early_stopping']
        )
        save_json(history, os.path.join(fold_dir, 'history.json'))
        
        # Evaluate Fold
        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pth'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        _, val_metrics = validate(model, val_loader, criterion, device)
        print(f"    Fold F1: {val_metrics['f1']:.4f}")
        fold_results.append(val_metrics)

    # -------------------------------------------------------
    # 3. AGGREGATE RESULTS
    # -------------------------------------------------------
    avg_metrics = {}
    for key in fold_results[0].keys():
        values = [res[key] for res in fold_results]
        avg_metrics[key] = np.mean(values)
    
    save_json(avg_metrics, os.path.join(save_dir, 'cv_average_metrics.json'))

    # Final Test Set Eval (Using last fold model for approximation)
    print(f"Evaluating on Held-out Test Set...")
    test_loader = DataLoader(test_ds, batch_size=config_params['batch_size'])
    _, test_metrics = validate(model, test_loader, criterion, device)
    save_json(test_metrics, os.path.join(save_dir, 'final_test_metrics.json'))
    
    return avg_metrics


def main():
    # -------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------
    base_data_dir = r"D:\0_final_project\DeepEnsemble\dataset-source\embedding-dataset\combine-Embedding&Metrics"
    # This will create folders like .../Training_Module/code2vec/..., .../Training_Module/CodeBERT/..., etc.
    base_save_root = r"D:\0_final_project\DeepEnsemble\Code\deep_learning\Complete_network\Training_Module"
    
    config_params = {
        'batch_size': 32,
        'epochs': 50,
        'lr': 0.001,
        'weight_decay': 1e-5,
        'early_stopping': 15,
        'test_size': 0.2,
        'val_size': 0.1,
        'random_seed': 42
    }

    # Define the Matrix of Smell Types and Embedding Models
    smell_types = ['DataClass', 'FeatureEnvy', 'GodClass', 'LongMethod']
    
    # Mapping model names to their specific file suffixes based on your directory tree
    # Structure: (ModelName, Filename_Suffix)
    models = {
        'code2vec': '_code2vec_metrics.pkl',
        'CuBERT': '_CuBERT_metrics.pkl',
        'TokenIndexing': '_TokenIndexing_metrics.pkl',
        'CodeBERT': '_CodeBERT_metrics.pkl' # Special handling for CLS inside loop
    }

    # Store all results for final summary
    grand_summary = []

    print("\n" + "="*80)
    print("ENSEMBLE TRAINING SUITE - ALL MODELS & SMELLS")
    print("="*80)

    # -------------------------------------------------------
    # MAIN TRAINING LOOP
    # -------------------------------------------------------
    for model_name, suffix in models.items():
        for smell in smell_types:
            
            # 1. Construct File Name
            # Handle the inconsistency in CodeBERT filenames (CLS vs non-CLS)
            current_suffix = suffix
            if model_name == 'CodeBERT':
                if smell == 'FeatureEnvy':
                    current_suffix = '_CodeBERT_metrics.pkl'
                else:
                    current_suffix = '_CodeBERT_CLS_metrics.pkl'
            
            filename = f"{smell}{current_suffix}"
            data_path = os.path.join(base_data_dir, smell, filename)
            
            # 2. Construct Save Directory
            # Example: .../Training_Module/code2vec/results_dataclass
            save_dir = os.path.join(base_save_root, model_name, f"results_{smell.lower()}")

            # 3. Check Existence
            if not os.path.exists(data_path):
                print(f"\n[SKIP] File not found: {data_path}")
                continue

            # 4. Train
            try:
                metrics = train_smell_type_kfold(
                    smell_name=smell,
                    embedding_type=model_name,
                    data_path=data_path,
                    save_dir=save_dir,
                    config_params=config_params
                )
                
                if metrics:
                    grand_summary.append({
                        'Model': model_name,
                        'Smell': smell,
                        'F1': metrics['f1'],
                        'MCC': metrics['mcc'],
                        'AUC': metrics['auc']
                    })
            except Exception as e:
                print(f"Error training {model_name} on {smell}: {e}")
                import traceback
                traceback.print_exc()

    # -------------------------------------------------------
    # GRAND SUMMARY
    # -------------------------------------------------------
    if grand_summary:
        print("\n" + "="*80)
        print("GRAND SUMMARY (Sorted by F1 Score)")
        print("="*80)
        
        df = pd.DataFrame(grand_summary)
        # Sort by Smell then F1 descending to see best model per smell
        df = df.sort_values(by=['Smell', 'F1'], ascending=[True, False])
        
        print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
        
        # Save summary to CSV
        summary_path = os.path.join(base_save_root, 'final_benchmark_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"\nFull benchmark saved to: {summary_path}")

if __name__ == "__main__":
    main()