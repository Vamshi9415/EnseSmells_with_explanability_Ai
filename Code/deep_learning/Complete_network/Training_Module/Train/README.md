# EnseSmells Training Module

A modular deep learning framework for code smell detection using ensemble methods.

## 📁 Project Structure

```
Training_Module/code2vec/
├── train.py           # Main training script (entry point)
├── config.py          # Configuration and argument parsing
├── dataset.py         # PyTorch Dataset class
├── data_loader.py     # Data loading and preprocessing
├── trainer.py         # Training and validation loops
├── metrics.py         # Evaluation metrics
├── utils.py           # Utility functions (I/O, plotting)
└── README.md          # This file
```

## 🚀 Quick Start

### Basic Training

```bash
cd D:\0_final_project\DeepEnsemble\Code\deep_learning\Complete_network

python -m Training_Module.code2vec.train 
    --data_path "path/to/data.pkl" 
    --save_dir "./checkpoints" 
    --batch_size 32 
    --epochs 100 
    --lr 0.001 
    --early_stopping 10
```

### All Available Arguments

```bash
python -m Training_Module.Train.train --help
```

cd D:\0_final_project\DeepEnsemble\Code\deep_learning\Complete_network; python -m Training_Module.Train.train_all_smells

**Arguments:**
- `--data_path`: Path to CSV/PKL file with embeddings, metrics, and labels (required)
- `--save_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: L2 regularization (default: 1e-5)
- `--early_stopping`: Early stopping patience in epochs (default: 10)
- `--test_size`: Test set proportion (default: 0.2)
- `--val_size`: Validation set proportion (default: 0.1)
- `--random_seed`: Random seed for reproducibility (default: 42)

## 📦 Module Descriptions

### `train.py` - Main Entry Point
- Orchestrates the entire training process
- Parses command-line arguments
- Initializes model, optimizer, and criterion
- Coordinates data loading, training, and evaluation
- Saves results and generates reports

### `config.py` - Configuration Management
- Defines `TrainingConfig` dataclass
- Parses command-line arguments
- Centralizes all hyperparameters
- Provides clean configuration interface

### `dataset.py` - Dataset Definition
- `CodeSmellDataset`: Custom PyTorch Dataset
- Handles embeddings, metrics, and labels
- Converts data to PyTorch tensors
- Efficient data access via `__getitem__`

### `data_loader.py` - Data Loading & Preprocessing
- `load_pickle_with_compatibility()`: Handles pandas version compatibility
- `load_and_prepare_data()`: Main data loading function
- Supports both CSV and PKL formats
- Automatic train/val/test splitting with stratification
- Data validation and type checking

### `trainer.py` - Training Logic
- `train_epoch()`: Single epoch training
- `validate()`: Model validation/testing
- `train_model()`: Complete training loop with:
  - Early stopping
  - Best model checkpointing
  - Learning rate scheduling
  - Training history tracking
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load model state

### `metrics.py` - Evaluation Metrics
- `calculate_metrics()`: Compute accuracy, precision, recall, F1, AUC
- `get_confusion_matrix()`: Generate confusion matrix
- `print_metrics()`: Formatted metric display

### `utils.py` - Utility Functions
- `save_json()` / `load_json()`: JSON serialization
- `plot_training_history()`: Visualize training curves
- `plot_confusion_matrix()`: Visualize confusion matrix
- `create_experiment_dir()`: Organize experiment outputs

## 📊 Output Files

After training, the following files are saved in `save_dir`:

```
checkpoints/
├── best_model.pth              # Best model checkpoint
├── training_history.json       # Loss, accuracy, F1 per epoch
├── training_history.png        # Training curves visualization
└── test_metrics.json          # Final test set performance
```

### Example `training_history.json`:
```json
{
    "train_loss": [0.693, 0.512, ...],
    "train_acc": [0.501, 0.753, ...],
    "train_f1": [0.498, 0.745, ...],
    "val_loss": [0.685, 0.523, ...],
    "val_acc": [0.513, 0.742, ...],
    "val_f1": [0.511, 0.738, ...]
}
```

### Example `test_metrics.json`:
```json
{
    "accuracy": 0.8523,
    "precision": 0.8412,
    "recall": 0.8634,
    "f1": 0.8521,
    "auc": 0.9123,
    "test_loss": 0.3245
}
```

## 🔧 Extending the Code

### Adding a New Metric

1. Edit `metrics.py`:
```python
def calculate_metrics(predictions, labels):
    # ... existing metrics ...
    metrics['specificity'] = calculate_specificity(labels, pred_binary)
    return metrics
```

### Adding a Custom Loss Function

1. Create `losses.py`:
```python
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Implementation
        pass
```

2. Update `train.py`:
```python
from losses import FocalLoss
# ...
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### Adding Data Augmentation

1. Edit `data_loader.py`:
```python
def augment_embeddings(embeddings):
    # Add noise, dropout, etc.
    noise = np.random.normal(0, 0.01, embeddings.shape)
    return embeddings + noise
```

## 🐛 Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the `Complete_network` directory:
```bash
cd D:\0_final_project\DeepEnsemble\Code\deep_learning\Complete_network
python -m Training_Module.code2vec.train ...
```

### Pickle Version Mismatch
The code automatically handles pandas version compatibility. If issues persist:
```bash
pip install pandas==1.5.3
```

### CUDA Out of Memory
Reduce batch size:
```bash
python -m Training_Module.code2vec.train --batch_size 16 ...
```

### Slow Training
Enable data loader workers (may cause issues on Windows):
```python
# In train.py, modify DataLoader:
train_loader = DataLoader(..., num_workers=4, pin_memory=True)
```

## 📈 Performance Tips

1. **Learning Rate Scheduling**: Already implemented with `ReduceLROnPlateau`
2. **Early Stopping**: Prevents overfitting (default patience: 10 epochs)
3. **Gradient Clipping**: Add in `trainer.py` if gradients explode:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
4. **Mixed Precision Training**: For faster training on newer GPUs:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

## 📝 Citation

If you use this code, please cite:
```bibtex
@software{ensesmells2025,
  title={EnseSmells: Deep Ensemble Learning for Code Smell Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/Vamshi9415/EnseSmells_with_explanability_Ai}
}
```

## 📄 License

This project is part of the EnseSmells research initiative.

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes following the modular structure
3. Test thoroughly
4. Submit pull request

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
