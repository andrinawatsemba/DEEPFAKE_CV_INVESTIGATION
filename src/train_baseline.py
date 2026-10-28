"""
Run A — Small CNN baseline trained from scratch.

Colab-friendly. Reads metadata CSVs + images from Google Drive.
Outputs: saved model checkpoint, metrics JSON, experiment log row.

Usage (in a Colab cell):
    %run src/train_baseline.py

Edit the CONFIG block below before running.
"""

import os, json, time, argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_metadata, make_dataloaders, compute_metrics,
    ExperimentLog, DEFAULT_MEMBER, DRIVE_ROOT,
)
import numpy as np

# ------------------------------------------------------------------ config
CONFIG = dict(
    member=DEFAULT_MEMBER,          # GW, AW, NG, NC, SM, PN
    sources=["openfake", "community", "reddit"],
    input_size=224,
    batch_size=32,
    epochs=10,
    lr=1e-3,
    augment=False,                  # baseline: no augmentation
    seed=42,
    model_out="reports/small_cnn_baseline.pt",
    metrics_out="reports/small_cnn_baseline_metrics.json",
    log_out="reports/experiment_log.csv",
    run_id="Run_A",
)

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------------------------------------------------------------ model
class SmallCNN(nn.Module):
    """A small convolutional network trained from scratch."""

    def __init__(self, input_size=224):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)


# ------------------------------------------------------------------ train / eval
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_prob = [], []
    for x, y, _ in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        y_prob.extend(prob.tolist())
        y_true.extend(y.numpy().tolist())
    return compute_metrics(y_true, y_prob, threshold)


# ------------------------------------------------------------------ main
def main(cfg):
    root = DRIVE_ROOT
    member = cfg["member"]

    # 1) Load metadata
    meta = load_metadata(root, sources=cfg["sources"], members=[member])
    if meta.empty:
        raise SystemExit(
            f"No metadata found for member {member} at {root}/metadata/. "
            f"Run Week 1 acquisition first."
        )
    print(f"Loaded {len(meta)} rows for member {member}")
    print(meta["label"].value_counts().to_dict())

    # 2) Build dataloaders (use the 'split' column if present)
    if "split" in meta.columns and meta["split"].isin(["train", "val", "test_seen"]).any():
        loaders = make_dataloaders(
            meta, batch_size=cfg["batch_size"], input_size=cfg["input_size"],
            augment_train=cfg["augment"], num_workers=2,
        )
    else:
        from sklearn.model_selection import train_test_split
        groups = meta["image_id"].unique()
        train_g, test_g = train_test_split(groups, test_size=0.2, random_state=cfg["seed"])
        train_g, val_g = train_test_split(train_g, test_size=0.125, random_state=cfg["seed"])
        meta["split"] = "test_seen"
        meta.loc[meta["image_id"].isin(train_g), "split"] = "train"
        meta.loc[meta["image_id"].isin(val_g), "split"] = "val"
        loaders = make_dataloaders(
            meta, batch_size=cfg["batch_size"], input_size=cfg["input_size"],
            augment_train=cfg["augment"], num_workers=2,
        )

    train_loader = loaders["train"]; val_loader = loaders["val"]; test_loader = loaders["test"]
    n_train = len(train_loader.dataset) if train_loader else 0
    n_val = len(val_loader.dataset) if val_loader else 0
    n_test = len(test_loader.dataset) if test_loader else 0
    print(f"train={n_train} val={n_val} test={n_test}")

    # 3) Model, loss, optimizer
    model = SmallCNN(input_size=cfg["input_size"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # 4) Training loop
    best_val_f1 = -1.0
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
              f"val_auc={val_metrics['roc_auc']:.4f} | {time.time()-t0:.1f}s")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            os.makedirs(os.path.dirname(cfg["model_out"]), exist_ok=True)
            torch.save(model.state_dict(), cfg["model_out"])
            print(f"  -> new best model saved (val macro-F1 {best_val_f1:.4f})")

    # 5) Final evaluation on test set
    model.load_state_dict(torch.load(cfg["model_out"], map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== TEST RESULTS ===")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print("  confusion_matrix:", test_metrics["confusion_matrix"])

    # 6) Save metrics JSON
    out = {
        "run_id": cfg["run_id"],
        "member": member,
        "model": "SmallCNN",
        "config": cfg,
        "history": history,
        "test": test_metrics,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    os.makedirs(os.path.dirname(cfg["metrics_out"]), exist_ok=True)
    with open(cfg["metrics_out"], "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics saved to {cfg['metrics_out']}")
    print(f"Model saved to {cfg['model_out']}")

    # 7) Experiment log
    logger = ExperimentLog(cfg["log_out"])
    logger.log(
        run_id=cfg["run_id"], member=member, model_name="SmallCNN",
        dataset=",".join(cfg["sources"]), n_train=n_train, n_val=n_val, n_test=n_test,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        augment=cfg["augment"], input_size=cfg["input_size"],
        metrics=test_metrics, notes="Run A baseline",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", default=CONFIG["member"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--input-size", type=int, default=CONFIG["input_size"])
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg.update(member=args.member, epochs=args.epochs, batch_size=args.batch_size,
               lr=args.lr, augment=args.augment, input_size=args.input_size)
    main(cfg)