"""
Main training script for EnseSmells model
Modular version with separated concerns
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add Complete_network directory to sys.path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
complete_network_dir = os.path.dirname(os.path.dirname(current_dir))
if complete_network_dir not in sys.path:
    sys.path.insert(0, complete_network_dir)

# Import project modules
from Ensemble_module.EnseSmells import EnseSmells
from .config import parse_arguments, TrainingConfig
from .data_loader import load_and_prepare_data
from .trainer import train_model, validate
from .metrics import print_metrics
from .utils import save_json, plot_training_history

def main():
    """Main training function"""
    # Parse arguments and create config
    args = parse_arguments()
    config = TrainingConfig.from_args(args)
    
    print("\n" + "="*80)
    print("ENSESMELLS TRAINING - CODE SMELL DETECTION")
    print("="*80)
    print(config)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*60}\n")
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset, embedding_dim, n_metrics = load_and_prepare_data(
        config.data_path, config.test_size, config.val_size, config.random_seed
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    print(f"{'='*60}")
    print(f"DATA LOADERS CREATED")
    print(f"{'='*60}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"{'='*60}\n")
    
    # Initialize model
    print(f"{'='*60}")
    print(f"MODEL INITIALIZATION")
    print(f"{'='*60}")
    model = EnseSmells(embedding_dim=embedding_dim, input_metrics=n_metrics)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: EnseSmells")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of metrics: {n_metrics}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*60}\n")
    
    # Loss function and optimizer
    # In train.py, before creating the criterion

    # Calculate positive weight: (Total Negatives / Total Positives)
    n_pos = torch.sum(train_dataset.labels).item()
    n_neg = len(train_dataset) - n_pos
    pos_weight_val = n_neg / (n_pos + 1e-5) # Add epsilon to avoid divide by zero
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    
    print(f"Using Positive Weight: {pos_weight_val:.2f}")
    
    # Switch to Logits Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    # Train model
    print(f"{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}\n")
    
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, config.epochs, config.save_dir, config.early_stopping
    )
    
    # Save training history
    history_path = os.path.join(config.save_dir, 'training_history.json')
    save_json(history, history_path)
    
    # Plot training history
    plot_path = os.path.join(config.save_dir, 'training_history.png')
    try:
        plot_training_history(history, plot_path)
    except Exception as e:
        print(f"⚠ Could not generate plot: {e}")
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(config.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")
    print(f"  Validation F1: {checkpoint['val_f1']:.4f}\n")
    
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, "Test")
    
    # Save test metrics
    test_metrics['test_loss'] = test_loss
    test_metrics_path = os.path.join(config.save_dir, 'test_metrics.json')
    save_json(test_metrics, test_metrics_path)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved to: {config.save_dir}")
    print(f"  - Best model: best_model.pth")
    print(f"  - Training history: training_history.json")
    print(f"  - Test metrics: test_metrics.json")
    if os.path.exists(plot_path):
        print(f"  - Training plot: training_history.png")
    print("="*80)


if __name__ == "__main__":
    main()