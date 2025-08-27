#!/usr/bin/env python3
# MNIST classification in PyTorch - preparation for Synergy Project - Time Arrow Prediction

"""
Train an MNIST CNN with PyTorch and save:
- model checkpoints (.pt)
- training/test metrics per epoch (CSV)
- classification report & confusion matrix (CSV)
- per-sample test predictions (CSV)
- sample class probabilities for first 100 test items (CSV)

Example:
    python mnist_cnn_run.py --epochs 5 --batch-size 128 --outdir runs/mnist_csv --amp
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# sklearn is used only to format the report & confusion matrix
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def available_device(pref: str = "cuda") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_csv(rows, header, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


# -----------------------------
# Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Data
# -----------------------------
def get_dataloaders(data_root: str, batch_size: int, num_workers: int, seed: int):
    mean, std = (0.1307,), (0.3081,)
    train_tfms = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=train_tfms)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    return epoch_loss, epoch_acc, y_true, y_pred


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=os.path.expanduser("~/.torch/datasets"))
    ap.add_argument("--outdir", type=str, default="runs/mnist_csv")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision")
    args = ap.parse_args()

    # Prepare output
    outdir = Path(args.outdir)
    (outdir / "artifacts").mkdir(parents=True, exist_ok=True)
    (outdir / "csv").mkdir(parents=True, exist_ok=True)

    # Save config
    cfg = vars(args)
    cfg["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(cfg, outdir / "run_config.json")

    # Seed & device
    set_seed(args.seed)
    device = available_device(args.device)
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(args.data_root, args.batch_size, args.num_workers, args.seed)

    # Model/opt
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)

    # Track metrics
    metrics_rows = [("epoch", "train_loss", "train_acc", "test_loss", "test_acc")]

    best_acc = -1.0
    best_path = outdir / "artifacts" / "model_best.pt"
    last_path = outdir / "artifacts" / "model_last.pt"

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp=args.amp
        )
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1:02d}/{args.epochs} "
              f"| train_loss {train_loss:.4f} acc {train_acc:.2f}% "
              f"| test_loss {test_loss:.4f} acc {test_acc:.2f}%")

        # Save last checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict()}, last_path)

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)

        # Append row
        metrics_rows.append((epoch + 1, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{test_loss:.6f}", f"{test_acc:.4f}"))

    # Write metrics CSV
    write_csv(metrics_rows[1:], metrics_rows[0], outdir / "csv" / "metrics.csv")

    # Final evaluation on test set with best model
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"Best model test acc: {test_acc:.2f}%")

    # Save classification report CSV
    report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
    # Turn report dict into rows
    rep_header = ["label", "precision", "recall", "f1-score", "support"]
    rep_rows = []
    for label, stats in report_dict.items():
        if isinstance(stats, dict):
            rep_rows.append([
                label,
                f"{stats.get('precision', 0.0):.6f}",
                f"{stats.get('recall', 0.0):.6f}",
                f"{stats.get('f1-score', 0.0):.6f}",
                int(stats.get("support", 0)),
            ])
    write_csv(rep_rows, rep_header, outdir / "csv" / "classification_report.csv")

    # Save confusion matrix CSV
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    cm_rows = cm.tolist()
    cm_header = [""] + [f"pred_{i}" for i in range(10)]
    cm_rows_with_idx = [[f"true_{i}"] + row for i, row in enumerate(cm_rows)]
    write_csv(cm_rows_with_idx, cm_header, outdir / "csv" / "confusion_matrix.csv")

    # Save per-sample test predictions CSV
    pred_header = ["index", "true", "pred", "correct"]
    pred_rows = [(i, int(t), int(p), int(t == p)) for i, (t, p) in enumerate(zip(y_true, y_pred))]
    write_csv(pred_rows, pred_header, outdir / "csv" / "predictions_test.csv")

    # Save probabilities for the first 100 test samples (optional visualization source)
    model.eval()
    probs_rows = []
    with torch.no_grad():
        count = 0
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, 10]
            for j in range(probs.shape[0]):
                if count >= 100:
                    break
                probs_rows.append([count, int(targets[j].item())] + [f"{probs[j, k]:.6f}" for k in range(10)])
                count += 1
            if count >= 100:
                break
    probs_header = ["index", "true"] + [str(k) for k in range(10)]
    write_csv(probs_rows, probs_header, outdir / "csv" / "sample_probs_top100.csv")

    # Done
    summary = {
        "best_test_acc": test_acc,
        "epochs": args.epochs,
        "samples_in_test": len(y_true),
        "artifacts": {
            "best_model": str(best_path),
            "last_model": str(last_path),
        },
        "csv_outputs": {
            "metrics": str(outdir / "csv" / "metrics.csv"),
            "classification_report": str(outdir / "csv" / "classification_report.csv"),
            "confusion_matrix": str(outdir / "csv" / "confusion_matrix.csv"),
            "predictions_test": str(outdir / "csv" / "predictions_test.csv"),
            "sample_probs_top100": str(outdir / "csv" / "sample_probs_top100.csv"),
        },
    }
    save_json(summary, outdir / "run_summary.json")
    print("\nSaved files:")
    for k, v in summary["artifacts"].items():
        print(f"  {k}: {v}")
    for k, v in summary["csv_outputs"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

