"""
Configuration and argument parsing
"""

import argparse
from typing import Any
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training"""
    data_path: str
    save_dir: str = './checkpoints'
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 1e-5
    early_stopping: int = 10
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Create config from command line arguments"""
        return cls(
            data_path=args.data_path,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            early_stopping=args.early_stopping,
            test_size=args.test_size,
            val_size=args.val_size,
            random_seed=args.random_seed
        )
    
    def __str__(self):
        """String representation for logging"""
        lines = ["Training Configuration:"]
        lines.append("=" * 60)
        for field, value in self.__dict__.items():
            lines.append(f"  {field:20s}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train EnseSmells Model for Code Smell Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV/PKL file containing embeddings, metrics, and labels')
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
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion (0.0-1.0)')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set proportion from training data (0.0-1.0)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()
