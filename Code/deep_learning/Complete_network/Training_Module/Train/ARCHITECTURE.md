# EnseSmells Modular Architecture

## 🏗️ Module Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          train.py                               │
│                      (Main Entry Point)                         │
│  • Argument parsing                                            │
│  • Orchestrates training flow                                  │
│  • Saves final results                                         │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  config.py   │   │data_loader.py│   │  trainer.py  │
│              │   │              │   │              │
│ • Config     │   │ • Load data  │   │ • train()    │
│   dataclass  │   │ • Preprocess │   │ • validate() │
│ • Argument   │   │ • Split data │   │ • Checkpoint │
│   parser     │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
                            │                   │
                            ▼                   ▼
                   ┌──────────────┐   ┌──────────────┐
                   │  dataset.py  │   │  metrics.py  │
                   │              │   │              │
                   │ • Dataset    │   │ • Accuracy   │
                   │   class      │   │ • Precision  │
                   │ • __getitem__│   │ • Recall     │
                   │              │   │ • F1, AUC    │
                   └──────────────┘   └──────────────┘
                            
                   ┌──────────────┐
                   │   utils.py   │
                   │              │
                   │ • JSON I/O   │
                   │ • Plotting   │
                   │ • Logging    │
                   └──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              External Dependencies                              │
│  • Ensemble_module.EnseSmells  (Model architecture)           │
│  • semantic_module.SemanticModule                             │
│  • structural_module.StructuralModule                         │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Module Responsibilities

### Core Modules

| Module | Purpose | Key Functions | Dependencies |
|--------|---------|---------------|--------------|
| **train.py** | Main entry point | `main()` | All other modules |
| **config.py** | Configuration | `parse_arguments()`, `TrainingConfig` | argparse |
| **dataset.py** | Data container | `CodeSmellDataset` | torch |
| **data_loader.py** | Data I/O | `load_and_prepare_data()` | dataset, pandas |
| **trainer.py** | Training loops | `train_model()`, `validate()` | metrics, torch |
| **metrics.py** | Evaluation | `calculate_metrics()` | sklearn |
| **utils.py** | Utilities | `save_json()`, `plot_training_history()` | matplotlib |

## 🔄 Data Flow

```
Input Data (CSV/PKL)
        │
        ▼
[data_loader.py]
  - Load file
  - Parse embeddings & metrics
  - Train/Val/Test split
        │
        ▼
[dataset.py]
  - Create PyTorch Datasets
  - Convert to tensors
        │
        ▼
[train.py]
  - Create DataLoaders
  - Initialize model
        │
        ▼
[trainer.py]
  - Training loop
  - Forward/Backward pass
  - Call metrics.py
        │
        ▼
[utils.py]
  - Save checkpoints
  - Save history
  - Generate plots
        │
        ▼
Output (Checkpoints, Metrics, Plots)
```

## 🎯 Benefits of Modular Design

### ✅ Advantages

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Reusability**: Modules can be used in other projects independently
3. **Testability**: Easy to write unit tests for each module
4. **Maintainability**: Changes to one module don't affect others
5. **Readability**: Smaller, focused files are easier to understand
6. **Collaboration**: Multiple developers can work on different modules
7. **Extensibility**: Easy to add new features without modifying core logic

### 📊 Code Metrics

**Before Modularization:**
- 1 file, 375 lines
- All logic tightly coupled
- Difficult to test individual components
- Hard to maintain and extend

**After Modularization:**
- 7 files, ~50-150 lines each
- Clear separation of concerns
- Easy to test each module
- Simple to add new features

## 🔧 Common Use Cases

### Use Case 1: Change Dataset Format
**Before**: Modify train.py (risky, affects everything)  
**After**: Only modify `data_loader.py`

### Use Case 2: Add New Metric
**Before**: Search through 375 lines to find metric calculation  
**After**: Add function to `metrics.py`

### Use Case 3: Implement New Training Strategy
**Before**: Refactor large train.py file  
**After**: Modify `trainer.py`, leave other modules unchanged

### Use Case 4: Change Model Architecture
**Before**: Update model initialization in train.py  
**After**: Only update import in `train.py`, modify `Ensemble_module`

### Use Case 5: Custom Data Augmentation
**Before**: Hack into data loading section  
**After**: Add function to `data_loader.py`

## 🧪 Testing Strategy

Each module can be tested independently:

```python
# Test dataset.py
def test_dataset():
    from dataset import CodeSmellDataset
    embeddings = np.random.rand(100, 384)
    metrics = np.random.rand(100, 20)
    labels = np.random.randint(0, 2, 100)
    dataset = CodeSmellDataset(embeddings, metrics, labels)
    assert len(dataset) == 100
    sample = dataset[0]
    assert 'embedding' in sample
    assert 'metrics' in sample
    assert 'label' in sample

# Test metrics.py
def test_metrics():
    from metrics import calculate_metrics
    predictions = np.array([0.2, 0.8, 0.6, 0.3])
    labels = np.array([0, 1, 1, 0])
    metrics = calculate_metrics(predictions, labels)
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 0 <= metrics['accuracy'] <= 1

# Test data_loader.py
def test_data_loader():
    from data_loader import load_and_prepare_data
    train, val, test, emb_dim, n_met = load_and_prepare_data('data.pkl')
    assert len(train) > 0
    assert emb_dim == 384
    assert n_met == 20
```

## 📈 Performance Considerations

### Memory Efficiency
- `dataset.py`: Lazy loading, data kept as tensors
- `data_loader.py`: Batch processing, no full data in memory

### Computation Efficiency  
- `trainer.py`: GPU acceleration, gradient accumulation support
- `metrics.py`: Vectorized operations with numpy/sklearn

### I/O Efficiency
- `utils.py`: Async I/O for saving checkpoints
- `data_loader.py`: Pandas compatibility layer for legacy files

## 🚀 Future Enhancements

### Potential New Modules

1. **logger.py**: Advanced logging with TensorBoard/Weights & Biases
2. **visualize.py**: Advanced visualization (attention maps, embeddings)
3. **callbacks.py**: Custom callbacks for training (like Keras)
4. **schedulers.py**: Custom learning rate schedulers
5. **losses.py**: Custom loss functions (Focal Loss, etc.)
6. **augmentation.py**: Data augmentation strategies
7. **ensemble.py**: Multiple model ensembling strategies

### Code Organization Improvements

```
Training_Module/code2vec/
├── core/
│   ├── __init__.py
│   ├── dataset.py
│   ├── trainer.py
│   └── metrics.py
├── utils/
│   ├── __init__.py
│   ├── io.py
│   ├── plotting.py
│   └── logging.py
├── data/
│   ├── __init__.py
│   ├── loaders.py
│   └── augmentation.py
├── config/
│   ├── __init__.py
│   └── training_config.py
└── train.py
```

## 📚 Additional Resources

- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Python Project Structure](https://docs.python-guide.org/writing/structure/)
- [Clean Code Principles](https://www.oreilly.com/library/view/clean-code-a/9780136083238/)

---

**Created**: November 30, 2025  
**Version**: 1.0  
**Author**: DeepEnsemble Team
