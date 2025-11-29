import os
import pickle
import argparse
from pathlib import Path
from typing import Tuple, Dict
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Add Complete_network directory to sys.path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
complete_network_dir = os.path.dirname(os.path.dirname(current_dir))
if complete_network_dir not in sys.path:
    sys.path.insert(0, complete_network_dir)

# Now import using absolute path from Complete_network
from Ensemble_module.EnseSmells import EnseSmells

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

def load_and_prepare_data(data_path: str, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
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
        # Try pandas read_pickle first (more compatible)
        try:
            df = pd.read_pickle(data_path)
            print("Loaded pickle file using pd.read_pickle")
        except Exception as e:
            print(f"pd.read_pickle failed: {e}")
            print("Trying standard pickle.load...")
            with open(data_path, 'rb') as f:
                df = pickle.load(f)
            print("Loaded pickle file using pickle.load")
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        # Parse string representations to lists
        print("Parsing metrics and embeddings...")
        df['metrics'] = df['metrics'].apply(eval)
        df['embedding'] = df['embedding'].apply(eval)
    else:
        raise ValueError("Data file must be .csv or .pkl format")
    
    print(f"DataFrame loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle different column names (metrics vs metric)
    if 'metric' in df.columns and 'metrics' not in df.columns:
        df['metrics'] = df['metric']
    
    # Convert to numpy arrays
    embeddings = np.array(df['embedding'].tolist())
    metrics = np.array(df['metrics'].tolist())
    labels = df['label'].values
    
    embedding_dim = embeddings.shape[1]
    n_metrics = metrics.shape[1]
    
    print(f"Data shape: {len(df)} samples")
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
    
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Validation set: {len(y_val)} samples")
    print(f"Test set: {len(y_test)} samples")
    
    # Create datasets
    train_dataset = CodeSmellDataset(X_train_emb, X_train_met, y_train)
    val_dataset = CodeSmellDataset(X_val_emb, X_val_met, y_val)
    test_dataset = CodeSmellDataset(X_test_emb, X_test_met, y_test)
    
    return train_dataset, val_dataset, test_dataset, embedding_dim, n_metrics
def calculate_metrics(predictions, labels):
    """Calculate evaluation metrics"""
    pred_binary = (predictions > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels, pred_binary),
        'precision': precision_score(labels, pred_binary, zero_division=0),
        'recall': recall_score(labels, pred_binary, zero_division=0),
        'f1': f1_score(labels, pred_binary, zero_division=0),
        'auc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    }
    
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        embeddings = batch['embedding'].to(device)
        metrics = batch['metrics'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        predictions, semantic_attn, structural_attn = model(embeddings, metrics)
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_labels))
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            metrics = batch['metrics'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            # Forward pass
            predictions, semantic_attn, structural_attn = model(embeddings, metrics)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_labels))
    
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, epochs, save_dir, early_stopping_patience=10):
    """
    Complete training loop with validation and model checkpointing
    """
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics
            }
            
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"✓ Saved best model with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train EnseSmells Model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV file containing embeddings, metrics, and labels')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set proportion from training data')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset, embedding_dim, n_metrics = load_and_prepare_data(
        args.data_path, args.test_size, args.val_size, args.random_seed
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print("\nInitializing model...")
    model = EnseSmells(embedding_dim=embedding_dim, input_metrics=n_metrics)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Train model
    print("\nStarting training...")
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args.epochs, args.save_dir, args.early_stopping
    )
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining history saved to {history_path}")
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test AUC-ROC: {test_metrics['auc']:.4f}")
    
    # Save test metrics
    test_metrics_path = os.path.join(args.save_dir, 'test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f"\nTest metrics saved to {test_metrics_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()