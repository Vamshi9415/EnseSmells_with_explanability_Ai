"""
Utility functions for file I/O, logging, and visualization
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[OK] Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train F1', marker='o')
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined comparison
    axes[1, 1].plot(history['train_loss'], label='Train Loss', alpha=0.6)
    axes[1, 1].plot(history['val_loss'], label='Val Loss', alpha=0.6)
    ax2 = axes[1, 1].twinx()
    ax2.plot(history['val_f1'], label='Val F1', color='green', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    ax2.set_ylabel('F1 Score')
    axes[1, 1].set_title('Loss vs F1 Score')
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Training history plot saved to {save_path}")
    
   # plt.show()


def plot_confusion_matrix(cm, save_path: str = None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Smell', 'Smell'],
                yticklabels=['No Smell', 'Smell'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Confusion matrix plot saved to {save_path}")
    
    # plt.show()


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """
    Create a directory for experiment results
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name (timestamp used if None)
        
    Returns:
        Path to created directory
    """
    from datetime import datetime
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    
    print(f"[OK] Created experiment directory: {exp_dir}")
    return exp_dir
