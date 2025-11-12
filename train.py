# train.py
import argparse
import json
import math
import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from model import MLP

# ---------- dataset ----------
class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, D) numpy, y: (N,) ints
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------- helpers ----------
def load_vectorized_csv(path: str, vector_col='news_vector', impact_col='impact_score'):
    """
    Loads CSV where vector_col contains JSON arrays as strings.
    Returns: X (N,D numpy), y (N numpy int labels), meta (list of tuples date,symbol,impact_str)
    """
    df = pd.read_csv(path, dtype=str)
    if vector_col not in df.columns:
        raise ValueError(f"Vector column '{vector_col}' not found in {path}")
    # parse vectors
    vecs = []
    meta = []
    labels = []
    for _, row in df.iterrows():
        vstr = row[vector_col]
        try:
            v = json.loads(vstr)
            vecs.append(np.array(v, dtype=np.float32))
        except Exception:
            # Try fallback: maybe the CSV cell contains brackets without proper json -> try eval
            try:
                v = eval(vstr)
                vecs.append(np.array(v, dtype=np.float32))
            except Exception:
                # skip invalid row
                continue
        # label parse
        lab = row[impact_col] if impact_col in row.index else ''
        labels.append(lab)
        date = row['date'] if 'date' in row.index else ''
        sym = row['symbol'] if 'symbol' in row.index else ''
        meta.append((date, sym, lab))

    if len(vecs) == 0:
        raise RuntimeError("No vectors parsed from CSV.")

    X = np.vstack(vecs)
    y_raw = np.array(labels)
    # convert y_raw to ints where possible; mark missing/invalid as None
    y = []
    valid_idx = []
    for i, v in enumerate(y_raw):
        try:
            yi = int(v)
            y.append(yi)
            valid_idx.append(i)
        except Exception:
            # skip rows without valid label
            pass

    if len(y) == 0:
        raise RuntimeError("No valid labels found in CSV.")

    X = X[valid_idx]
    meta = [meta[i] for i in valid_idx]
    y = np.array(y, dtype=np.int64)
    return X, y, meta

def build_label_map(y: np.ndarray):
    unique = sorted(list(set(y.tolist())))
    label2idx = {lab: i for i, lab in enumerate(unique)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    y_mapped = np.array([label2idx[v] for v in y], dtype=np.int64)
    return y_mapped, label2idx, idx2label

# ---------- training / evaluation ----------
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device,
                epochs: int = 10,
                lr: float = 1e-3,
                weight_decay: float = 0.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
            total += Xb.size(0)
        avg_loss = running_loss / max(1, total)
        # validation
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                logits = model(Xv)
                predicted = torch.argmax(logits, dim=1)
                preds.append(predicted.cpu().numpy())
                trues.append(yv.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch}/{epochs}  train_loss={avg_loss:.4f}  val_acc={acc:.4f}")
    return model

def evaluate_model(model: nn.Module, loader: DataLoader, device, idx2label):
    model.to(device)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            predicted = torch.argmax(logits, dim=1)
            preds.append(predicted.cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = accuracy_score(trues, preds)
    # convert mapped ints back to original labels for reporting
    mapped_to_label = {mi: lab for mi, lab in idx2label.items()}
    # build target_names for classification_report in order of mapped indices
    target_names = [str(mapped_to_label[i]) for i in range(len(mapped_to_label))]
    creport = classification_report(trues, preds, target_names=target_names, zero_division=0)
    return acc, creport

# ---------- entrypoint ----------
def train_and_evaluate(input_csv: str,
                       out_model_path: str,
                       vector_col: str = 'news_vector',
                       impact_col: str = 'impact_score',
                       hidden_dims=(128,64),
                       batch_size: int = 256,
                       lr: float = 1e-3,
                       epochs: int = 10,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       dropout: float = 0.2,
                       use_batchnorm: bool = True,
                       device_name: str = None) -> Dict[str, Any]:
    # load data
    print(f"Loading data from {input_csv} ...")
    X, y_raw, meta = load_vectorized_csv(input_csv, vector_col=vector_col, impact_col=impact_col)
    print(f"Loaded {len(y_raw)} labeled rows. Feature dim = {X.shape[1]}")
    # build label mapping
    y, label2idx, idx2label = build_label_map(y_raw)
    print("Label mapping (original_label -> mapped_index):", label2idx)

    # train/test split (stratify)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    print(f"Train size: {len(y_train)}  Test size: {len(y_test)}")

    # datasets / loaders
    train_ds = EmbeddingDataset(X_train, y_train)
    test_ds = EmbeddingDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # device
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using device:", device)

    # model
    input_dim = X.shape[1]
    num_classes = len(label2idx)
    print("Model hyperparameters:")
    print("  hidden_dims:", hidden_dims)
    print("  activation: ReLU")
    print("  dropout:", dropout)
    print("  batchnorm:", use_batchnorm)
    print("  lr:", lr)
    print("  batch_size:", batch_size)
    print("  epochs:", epochs)
    print("  optimizer: Adam (weight_decay=0.0)")

    model = MLP(input_dim, hidden_dims=hidden_dims, num_classes=num_classes, activation=nn.ReLU,
                dropout=dropout, use_batchnorm=use_batchnorm)

    # train
    model = train_model(model, train_loader, test_loader, device, epochs=epochs, lr=lr)

    # final evaluation on test set
    acc, creport = evaluate_model(model, test_loader, device, idx2label)
    print("Final Test Accuracy:", acc)
    print("Classification Report:\n", creport)

    # Save checkpoint: include model state, label map, hyperparams
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'label2idx': label2idx,
        'idx2label': idx2label,
        'input_dim': X.shape[1],
        'hidden_dims': hidden_dims,
        'dropout': dropout,
    }
    torch.save(checkpoint, out_model_path)
    print(f"Saved model checkpoint to {out_model_path}")

    return {
        'checkpoint_path': out_model_path,
        'test_accuracy': acc,
        'classification_report': creport,
        'label2idx': label2idx
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input vectorized CSV')
    parser.add_argument('--out', required=True, help='Output model checkpoint (.pth)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[128,64])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    train_and_evaluate(input_csv=args.input,
                       out_model_path=args.out,
                       hidden_dims=tuple(args.hidden_dims),
                       batch_size=args.batch_size,
                       lr=args.lr,
                       epochs=args.epochs,
                       test_size=args.test_size,
                       dropout=args.dropout)
