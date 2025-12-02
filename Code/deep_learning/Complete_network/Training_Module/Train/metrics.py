"""
Evaluation metrics and utilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
)
from typing import Dict


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        predictions: Model predictions (probabilities)
        labels: True labels
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, and auc
    """
    pred_binary = (predictions > 0.5).astype(int)
    
    # Compute standard metrics
    accuracy = accuracy_score(labels, pred_binary)
    precision = precision_score(labels, pred_binary, zero_division=0)
    recall = recall_score(labels, pred_binary, zero_division=0)
    f1 = f1_score(labels, pred_binary, zero_division=0)
    auc = roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0

    # Matthews Correlation Coefficient (robust for imbalanced classes)
    try:
        mcc = matthews_corrcoef(labels, pred_binary)
    except Exception:
        mcc = 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc
    }
    
    return metrics


def get_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        predictions: Model predictions (probabilities)
        labels: True labels
        
    Returns:
        Confusion matrix
    """
    pred_binary = (predictions > 0.5).astype(int)
    return confusion_matrix(labels, pred_binary)


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for the metric names (e.g., "Train", "Val", "Test")
    """
    prefix = f"{prefix} " if prefix else ""
    print(f"{prefix}Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
    print(f"{prefix}Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"{prefix}Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"{prefix}F1 Score:  {metrics.get('f1', 0.0):.4f}")
    print(f"{prefix}AUC-ROC:   {metrics.get('auc', 0.0):.4f}")
    print(f"{prefix}MCC:       {metrics.get('mcc', 0.0):.4f}")
