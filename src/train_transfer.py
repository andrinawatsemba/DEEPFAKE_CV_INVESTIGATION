"""
Run B — ResNet-50 transfer learning.

Stage 1: freeze backbone, train new classification head.
Stage 2: unfreeze last blocks, fine-tune with a low learning rate.

Colab-friendly. Reads metadata CSVs + images from Google Drive.
Outputs: saved model checkpoint, metrics JSON, experiment log row.

Usage (in a Colab cell):
    %run src/train_transfer.py

Edit the CONFIG block below before running.
"""

import os, json, time, argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

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
    epochs_head=5,                  # stage 1: train head only
    epochs_finetune=5,              # stage 2: fine-tune last blocks
    lr_head=1e-3,
    lr_finetune=1e-5,
    augment=False,                  # set True for Run C-style augmentation test
    seed=42,
    model_out="reports/resnet50_transfer.pt",
    metrics_out="reports/resnet50_transfer_metrics.json",
    log_out="reports/experiment_log.csv",
    run_id="Run_B",
    fallback_resnet18=False,        # set True if Colab GPU is too slow for ResNet-50
)

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if CONFIG["fallback_resnet18"]:
    print("Using ResNet-18 fallback (ResNet-50 too slow on this runtime).")


# ------------------------------------------------------------------ model
def build_model(finetune=False, resnet18=False):
    """Build ResNet-50 (or ResNet-18 fallback) with a new binary head."""
    if resnet18:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
    else:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features

    if not finetune:
        # Stage 1: freeze backbone
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer for binary classification
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return model.to(device)


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

    # 2) Build dataloaders
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
    n_train = len(train_loader.dataset); n_val = len(val_loader.dataset); n_test = len(test_loader.dataset
    )
    print(f"train={n_train} val={n_val} test={n_test}")

    # 3) Model
    model = build_model(finetune=False, resnet18=cfg["fallback_resnet18"])
    criterion = nn.BCEWithLogitsLoss()

    # Stage 1: train head only
    optimizer = optim.Adam(model.fc.parameters(), lr=cfg["lr_head"])
    print("\n=== Stage 1: train classification head (backbone frozen) ===")
    best_val_f1 = -1.0
    for epoch in range(1, cfg["epochs_head"] + 1):
        t0 = time.time()
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vm = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | train_loss={tl:.4f} | val_macro_f1={vm['macro_f1']:.4f} | "
              f"val_auc={vm['roc_auc']:.4f} | {time.time()-t0:.1f}s")
        if vm["macro_f1"] > best_val_f1:
            best_val_f1 = vm["macro_f1"]
            os.makedirs(os.path.dirname(cfg["model_out"]), exist_ok=True)
            torch.save(model.state_dict(), cfg["model_out"])

    # Stage 2: unfreeze last blocks and fine-tune
    print("\n=== Stage 2: fine-tune last blocks ===")
    for name, param in model.named_parameters():
        if name.startswith(("layer3", "layer4", "fc")):
            param.requires_grad = True

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr_finetune"],
    )
    for epoch in range(1, cfg["epochs_finetune"] + 1):
        t0 = time.time()
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vm = evaluate(model, val_loader, device)
        print(f"FT Epoch {epoch:2d} | train_loss={tl:.4f} | val_macro_f1={vm['macro_f1']:.4f} | "
              f"val_auc={vm['roc_auc']:.4f} | {time.time()-t0:.1f}s")
        if vm["macro_f1"] > best_val_f1:
            best_val_f1 = vm["macro_f1"]
            torch.save(model.state_dict(), cfg["model_out"])

    # 4) Final evaluation on test set
    model.load_state_dict(torch.load(cfg["model_out"], map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== TEST RESULTS ===")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print("  confusion_matrix:", test_metrics["confusion_matrix"])

    # 5) Save metrics JSON
    out = {
        "run_id": cfg["run_id"],
        "member": member,
        "model": "ResNet-18" if cfg["fallback_resnet18"] else "ResNet-50",
        "config": cfg,
        "test": test_metrics,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    os.makedirs(os.path.dirname(cfg["metrics_out"]), exist_ok=True)
    with open(cfg["metrics_out"], "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMetrics saved to {cfg['metrics_out']}")
    print(f"Model saved to {cfg['model_out']}")

    # 6) Experiment log
    logger = ExperimentLog(cfg["log_out"])
    logger.log(
        run_id=cfg["run_id"], member=member,
        model_name=out["model"],
        dataset=",".join(cfg["sources"]),
        n_train=n_train, n_val=n_val, n_test=n_test,
        epochs=cfg["epochs_head"] + cfg["epochs_finetune"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr_head"],
        augment=cfg["augment"],
        input_size=cfg["input_size"],
        metrics=test_metrics,
        notes="Run B transfer learning",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", default=CONFIG["member"])
    parser.add_argument("--epochs-head", type=int, default=CONFIG["epochs_head"])
    parser.add_argument("--epochs-finetune", type=int, default=CONFIG["epochs_finetune"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr-head", type=float, default=CONFIG["lr_head"])
    parser.add_argument("--lr-finetune", type=float, default=CONFIG["lr_finetune"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--input-size", type=int, default=CONFIG["input_size"])
    parser.add_argument("--resnet18", action="store_true",
                        help="Use ResNet-18 fallback if ResNet-50 is too slow")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg.update(member=args.member, epochs_head=args.epochs_head,
               epochs_finetune=args.epochs_finetune, batch_size=args.batch_size,
               lr_head=args.lr_head, lr_finetune=args.lr_finetune,
               augment=args.augment, input_size=args.input_size,
               fallback_resnet18=args.resnet18)
    main(cfg)