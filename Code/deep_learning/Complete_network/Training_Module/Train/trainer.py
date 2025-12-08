"""
Training utilities and functions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import os

from .metrics import calculate_metrics


def train_epoch(model: nn.Module, dataloader, criterion, optimizer, device) -> Tuple[float, Dict]:
    """
    Train for one epoch
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
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


def validate(model: nn.Module, dataloader, criterion, device) -> Tuple[float, Dict]:
    """
    Validate the model
    
    Args:
        model: Neural network model
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
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


def train_model(model: nn.Module, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, epochs: int, save_dir: str, 
                early_stopping_patience: int = 10) -> Dict:
    """
    Complete training loop with validation and model checkpointing
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use (cuda/cpu)
        epochs: Number of training epochs
        save_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Dictionary containing training history
    """
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_mcc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_mcc': []
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
        history['train_mcc'].append(train_metrics['mcc'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_mcc'].append(val_metrics['mcc'])
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['f1']:.4f} | Train MCC: {train_metrics['mcc']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val MCC: {val_metrics['mcc']:.4f}")
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
            print(f"[OK] Saved best model with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return history


def save_checkpoint(model: nn.Module, optimizer, epoch: int, metrics: Dict, 
                   save_path: str):
    """
    Save model checkpoint
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Metrics dictionary
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    
    print(f"[OK] Checkpoint saved to {save_path}")
def load_checkpoint(model: nn.Module, checkpoint_path: str, device):
    """
    Load model checkpoint
    
    Args:
        model: Neural network model
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Loaded model and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[OK] Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")
    return model, checkpoint
