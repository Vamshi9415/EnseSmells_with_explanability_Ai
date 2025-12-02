# train.py
import os
import pickle
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Adjust this relative import if you run differently.
# This file lives at Complete_network/Training_Module/code2vec/train.py
from ...Ensemble_module.EnseSmells import EnseSmells


# -------------------------
# Dataset
# -------------------------
class CodeSmellDataset(Dataset):
    """
    Expects:
      - embeddings: dict(sample_id -> vector) OR list/ndarray of vectors (aligned with CSV order)
      - metrics_df: pd.DataFrame that contains 'sample_id' and 'label' columns plus metric columns
    """
    def __init__(self, embeddings, metrics_df: pd.DataFrame):
        self.metrics_df = metrics_df.reset_index(drop=True)

        # load embeddings in a robust way:
        if isinstance(embeddings, dict):
            # try to map by sample_id
            if 'sample_id' not in metrics_df.columns:
                raise ValueError("CSV has no 'sample_id' column to match dict embeddings.")
            # produce list aligned with metrics_df order
            self.emb_list = []
            for sid in self.metrics_df['sample_id'].values:
                if sid not in embeddings:
                    raise KeyError(f"sample_id {sid} missing in embeddings dict.")
                self.emb_list.append(np.asarray(embeddings[sid], dtype=np.float32))
            self.embeddings = np.stack(self.emb_list)
        else:
            # array/list: assume align by index
            arr = np.asarray(embeddings)
            if len(arr) != len(self.metrics_df):
                raise ValueError(f"Embeddings length ({len(arr)}) != CSV rows ({len(self.metrics_df)}).")
            self.embeddings = arr.astype(np.float32)

        # metrics: drop sample_id and label columns
        if 'label' not in self.metrics_df.columns:
            raise ValueError("CSV has no 'label' column.")
        self.labels = self.metrics_df['label'].astype(np.float32).values
        self.metrics = self.metrics_df.drop(columns=[c for c in ['sample_id', 'label'] if c in self.metrics_df.columns]).values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        metrics = self.metrics[idx]
        label = self.labels[idx]
        return torch.from_numpy(emb), torch.from_numpy(metrics), torch.tensor(label).unsqueeze(0)  # label shape (1,)


# -------------------------
# Utilities
# -------------------------
def load_embeddings(path: str):
    """
    Try to load common formats:
      - .pkl containing list/ndarray or dict
      - .npy
      - .npz (np.load)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    if path.suffix in ('.pkl', '.pickle'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    elif path.suffix == '.npy':
        return np.load(str(path))
    elif path.suffix == '.npz':
        with np.load(str(path)) as data:
            # pick first array found
            keys = list(data.keys())
            return data[keys[0]]
    else:
        # attempt pickle as fallback
        with open(path, 'rb') as f:
            return pickle.load(f)


def collate_fn(batch):
    # batch: list of (emb, metrics, label)
    embs = torch.stack([b[0] for b in batch])
    metrics = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return embs, metrics, labels


# -------------------------
# Train / Eval loops
# -------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for emb, metrics, labels in loader:
        emb = emb.to(device)
        metrics = metrics.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds, _, _ = model(emb, metrics)           # preds shape (batch,1) with sigmoid already
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        # metrics
        predicted = (preds.detach() > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for emb, metrics, labels in loader:
            emb = emb.to(device)
            metrics = metrics.to(device)
            labels = labels.to(device)

            preds, _, _ = model(emb, metrics)
            loss = criterion(preds, labels)

            total_loss += loss.item() * labels.size(0)

            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# -------------------------
# Main
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)

    # 1) load CSV metrics
    csv_path = Path(args.metrics_csv)
    assert csv_path.exists(), f"{csv_path} not found"
    df = pd.read_csv(str(csv_path))
    print(f"Loaded metrics CSV: {len(df)} rows")

    # 2) load embeddings
    embeddings = load_embeddings(args.embeddings)
    print("Loaded embeddings type:", type(embeddings))

    # 3) create dataset and train/val split
    df_train, df_val = train_test_split(df, test_size=args.val_split, stratify=df['label'] if 'label' in df.columns else None, random_state=args.seed)
    train_ds = CodeSmellDataset(embeddings, df_train)
    val_ds = CodeSmellDataset(embeddings, df_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # 4) model init
    # infer dims
    embedding_dim = train_ds.embeddings.shape[1]
    input_metrics = train_ds.metrics.shape[1]
    print(f"Embedding dim: {embedding_dim}, input_metrics: {input_metrics}")

    model = EnseSmells(embedding_dim=embedding_dim, input_metrics=input_metrics).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f} acc: {train_acc:.4f} | val_loss: {val_loss:.4f} acc: {val_acc:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'embedding_dim': embedding_dim,
                'input_metrics': input_metrics
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

    # final save
    final_path = out_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default=r"D:\0_final_project\DeepEnsemble\dataset-source\embedding-dataset\code2vec\GodClass_code2vec_embeddings.pkl", help="path to embeddings (.pkl/.npy/.npz)")
    parser.add_argument("--metrics_csv", type=str, default=r"D:\0_final_project\DeepEnsemble\dataset-source\embedding-dataset\software-metrics\GodClass_code_metrics_values.csv", help="metrics CSV (must have 'label' column and optionally 'sample_id')")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="where to save models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    main(args)
