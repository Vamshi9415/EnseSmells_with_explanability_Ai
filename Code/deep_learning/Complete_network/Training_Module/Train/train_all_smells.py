"""
Train EnseSmells model for all code smell types using Stratified K-Fold Cross Validation.
Automatically handles Class Imbalance via weighted loss.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
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


def train_smell_type_kfold(smell_name: str, data_path: str, save_dir: str, config_params: dict):
    """
    Train model using Stratified K-Fold Cross Validation.
    """
    print("\n" + "="*80)
    print(f"STARTING 5-FOLD CV FOR: {smell_name}")
    print("="*80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(config_params['random_seed'])
    np.random.seed(config_params['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config_params['random_seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------
    # We load the data. Note: load_and_prepare_data splits into Train/Val/Test.
    # For K-Fold, we will merge Train and Val back together to form a "Dev Set",
    # and keep Test separate for final unseen evaluation.
    train_ds, val_ds, test_ds, embedding_dim, n_metrics = load_and_prepare_data(
        data_path, 
        config_params['test_size'], 
        config_params['val_size'], 
        config_params['random_seed']
    )

    # Merge Train and Val for Cross Validation
    dev_dataset = ConcatDataset([train_ds, val_ds])
    
    # Extract labels from both datasets to use for StratifiedKFold
    # We assume dataset.labels is a tensor/array accessible attribute
    dev_labels = np.concatenate([train_ds.labels, val_ds.labels])
    
    # K-Fold Initializer
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config_params['random_seed'])
    
    fold_results = []
    
    # -------------------------------------------------------
    # 2. START K-FOLD LOOP
    # -------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dev_labels)), dev_labels)):
        print(f"\n{'-'*30} FOLD {fold+1}/{k_folds} {'-'*30}")
        
        # Create Save Dir for this Fold
        fold_dir = os.path.join(save_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # A. Create Samplers and Loaders
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dev_dataset, batch_size=config_params['batch_size'], sampler=train_subsampler)
        val_loader = DataLoader(dev_dataset, batch_size=config_params['batch_size'], sampler=val_subsampler)
        
        # B. Calculate Class Weights for Imbalance (Critical Step)
        # Get labels for this fold's training set
        fold_train_labels = dev_labels[train_idx]
        n_pos = np.sum(fold_train_labels == 1)
        n_neg = np.sum(fold_train_labels == 0)
        
        # Calculate pos_weight: (Negatives / Positives)
        # If dataset is 90 neg, 10 pos -> weight = 9. 
        # The model treats every 1 positive error as 9 negative errors.
        weight_value = n_neg / (n_pos + 1e-6) # epsilon for safety
        pos_weight = torch.tensor([weight_value], dtype=torch.float).to(device)
        
        print(f"  > Imbalance Stats: Neg={n_neg}, Pos={n_pos}")
        print(f"  > Calculated pos_weight: {weight_value:.2f}")

        # C. Initialize Model (Fresh for each fold)
        model = EnseSmells(embedding_dim=embedding_dim, input_metrics=n_metrics)
        model = model.to(device)
        
        # D. Loss & Optimizer
        # IMPORTANT: Using BCEWithLogitsLoss because it's more stable than Sigmoid+BCELoss
        # Ensure EnseSmells.py does NOT have a final Sigmoid layer.
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config_params['lr'], 
            weight_decay=config_params['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # E. Train
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
        
        # Save Fold History
        save_json(history, os.path.join(fold_dir, 'history.json'))
        
        # F. Evaluate Best Model of this Fold
        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pth'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        print(f"  > Fold {fold+1} Validation F1: {val_metrics['f1']:.4f}")
        
        fold_results.append(val_metrics)

    # -------------------------------------------------------
    # 3. AGGREGATE RESULTS
    # -------------------------------------------------------
    print(f"\n{'='*30} CROSS VALIDATION RESULTS {'='*30}")
    avg_metrics = {}
    for key in fold_results[0].keys():
        values = [res[key] for res in fold_results]
        avg_metrics[key] = np.mean(values)
        print(f"Avg {key.capitalize()}: {avg_metrics[key]:.4f} (+/- {np.std(values):.4f})")
    
    # Save Combined Results
    save_json(avg_metrics, os.path.join(save_dir, 'cv_average_metrics.json'))

    # -------------------------------------------------------
    # 4. FINAL TEST EVALUATION (Optional)
    # -------------------------------------------------------
    # Use the model from the best performing fold, or retrain on all Dev data.
    # Here we simply evaluate the model from the LAST fold on the completely unseen Test Set
    # to get a rough idea, but the CV score is the scientifically accurate one.
    print(f"\nEvaluating Last Fold Model on HELD-OUT TEST SET...")
    test_loader = DataLoader(test_ds, batch_size=config_params['batch_size'])
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    print_metrics(test_metrics, "Held-out Test")
    save_json(test_metrics, os.path.join(save_dir, 'final_test_metrics.json'))
    
    return avg_metrics


def main():
    """Main function to train all code smell types"""
    
    # Base paths - UPDATE THESE IF NEEDED
    base_data_dir = r"D:\0_final_project\DeepEnsemble\dataset-source\embedding-dataset\combine-Embedding&Metrics"
    base_save_dir = r"D:\0_final_project\DeepEnsemble\Code\deep_learning\Complete_network\Training_Module\code2vec_kfold"
    
    # Configuration parameters
    config_params = {
        'batch_size': 32,
        'epochs': 50, # Reduced slightly as we run 5 folds
        'lr': 0.001,
        'weight_decay': 1e-5,
        'early_stopping': 15,
        'test_size': 0.2,
        'val_size': 0.1, # Used by loader, but merged for CV
        'random_seed': 42
    }
    
    # Define all code smell types
    smell_types = {
        'DataClass': {
            'data_path': os.path.join(base_data_dir, 'DataClass', 'DataClass_code2vec_metrics.pkl'),
            'save_dir': os.path.join(base_save_dir, 'results_dataclass')
        },
        'FeatureEnvy': {
            'data_path': os.path.join(base_data_dir, 'FeatureEnvy', 'FeatureEnvy_code2vec_metrics.pkl'),
            'save_dir': os.path.join(base_save_dir, 'results_featureenvy')
        },
        'GodClass': {
            'data_path': os.path.join(base_data_dir, 'GodClass', 'GodClass_code2vec_metrics.pkl'),
            'save_dir': os.path.join(base_save_dir, 'results_godclass')
        },
        'LongMethod': {
            'data_path': os.path.join(base_data_dir, 'LongMethod', 'LongMethod_code2vec_metrics.pkl'),
            'save_dir': os.path.join(base_save_dir, 'results_longmethod')
        }
    }
    
    print("\n" + "="*80)
    print("ENSESMELLS - STRATIFIED K-FOLD TRAINING")
    print("="*80)

    # Check files
    available_smells = {}
    for smell_name, paths in smell_types.items():
        if os.path.exists(paths['data_path']):
            available_smells[smell_name] = paths
        else:
            print(f"Skipping {smell_name} (File not found)")

    if not available_smells:
        print("No datasets found.")
        return

    # Loop through smells
    final_summary = {}
    for smell_name, paths in available_smells.items():
        try:
            metrics = train_smell_type_kfold(
                smell_name, 
                paths['data_path'], 
                paths['save_dir'], 
                config_params
            )
            final_summary[smell_name] = metrics
        except Exception as e:
            print(f"Error processing {smell_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("FINAL SUMMARY (AVERAGE ACROSS 5 FOLDS)")
    print("="*80)
    for smell, metrics in final_summary.items():
        print(f"{smell:15} | F1: {metrics['f1']:.4f} | MCC: {metrics['mcc']:.4f} | AUC: {metrics['auc']:.4f}")

if __name__ == "__main__":
    main()