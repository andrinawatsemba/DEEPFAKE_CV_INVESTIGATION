"""
Shared utilities for the Deepfake CV Investigation project.

Colab-friendly: reads metadata CSVs and images from Google Drive,
no hardcoded personal paths, member-aware.

Current Drive structure (source of truth):
  Root: /content/drive/MyDrive/Deepfake Detection Project/datasets
  Subfolders: public_data/, in_the_wild/, metadata/, audit_outputs/
  Metadata files: public_<source>_<member>.csv  (e.g. public_openfake_AW.csv)
"""

import os, csv, json, glob, math, hashlib, random
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix,
)

# ---------------------------------------------------------------- paths
DEFAULT_MEMBER = "AW"
MEMBERS = ["GW", "AW", "NG", "NC", "SM", "PN"]

# Current Google Drive root
DRIVE_ROOT = "/content/drive/MyDrive/Deepfake Detection Project/datasets"

# Map source name -> folder under root (matches the acquisition notebook)
SOURCE_FOLDERS = {
    "openfake": "public_data/openfake",
    "community": "public_data/community_forensics",
    "reddit": "in_the_wild/openfake_reddit",
}


def metadata_path(root, source, member):
    """Return the path to a member-specific metadata CSV."""
    return os.path.join(root, "metadata", f"public_{source}_{member}.csv")


def load_metadata(root=DRIVE_ROOT, sources=None, members=None):
    """Load and concatenate public metadata CSVs from {root}/metadata/.

    Parameters
    ----------
    root : str
        Drive datasets root folder.
    sources : list[str] | None
        Sources to include (e.g. ['openfake', 'community']). None = all found.
    members : list[str] | None
        Members to include (e.g. ['AW']). None = all found.

    Returns
    -------
    pd.DataFrame with a unified 'full_path' column pointing at the image on Drive.
    """
    meta_dir = os.path.join(root, "metadata")
    files = sorted(glob.glob(os.path.join(meta_dir, "public_*.csv")))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        fname = os.path.basename(f)
        # public_<source>_<member>.csv
        parts = fname.replace("public_", "").replace(".csv", "").split("_")
        if len(parts) < 2:
            continue
        member = parts[-1]
        source = "_".join(parts[:-1])

        if sources and source not in sources:
            continue
        if members and member not in members:
            continue

        df = pd.read_csv(f)
        df["source"] = source
        df["member"] = member
        # The 'path' column is relative to root, e.g. "public_data/openfake/OF_AW_000001.jpg"
        df["full_path"] = df["path"].apply(lambda p: os.path.join(root, str(p)))
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    meta = pd.concat(frames, ignore_index=True)
    return meta


def image_transform(input_size=224, augment=False):
    """Build a torchvision transform pipeline.

    If augment=True, adds flip + colour jitter (used only in Runs C and E).
    Normalisation matches ImageNet to suit the pretrained ResNet-50.
    """
    ops = [
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
    ]
    if augment:
        ops.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        ops.insert(2, transforms.ColorJitter(brightness=0.15, contrast=0.15))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


class ImageLabelDataset(Dataset):
    """PyTorch Dataset from a metadata DataFrame.

    Reads images on the fly from full_path. Label: real=0, ai=1.
    Corrupt/unreadable images return a blank tensor so training does not die.
    """

    def __init__(self, meta, input_size=224, augment=False, label_map=None):
        self.meta = meta.reset_index(drop=True)
        self.transform = image_transform(input_size, augment)
        self.label_map = label_map or {"real": 0, "ai": 1}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        try:
            with PILImage.open(row["full_path"]) as im:
                im = im.convert("RGB")
                x = self.transform(im)
        except Exception:
            x = torch.zeros(3, 224, 224)
        label = self.label_map.get(row["label"], -1)
        return x, label, row["image_id"]


def make_dataloaders(meta, train_split="train", val_split="val",
                     test_split="test_seen", batch_size=32,
                     input_size=224, augment_train=False, num_workers=2):
    """Build train/val/test DataLoaders from a metadata DataFrame that has a
    'split' column."""
    loaders = {}
    for name, split_name in [("train", train_split), ("val", val_split), ("test", test_split)]:
        sub = meta[meta["split"] == split_name]
        if len(sub) == 0:
            loaders[name] = None
            continue
        ds = ImageLabelDataset(sub, input_size=input_size,
                               augment=(augment_train and name == "train"))
        loaders[name] = DataLoader(ds, batch_size=batch_size,
                                   shuffle=(name == "train"),
                                   num_workers=num_workers,
                                   pin_memory=torch.cuda.is_available())
    return loaders


# ---------------------------------------------------------------- metrics
def compute_metrics(y_true, y_prob, threshold=0.5):
    """Return a dict of standard classification metrics."""
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "roc_auc": float(auc),
        "precision_real": float(prec_c[0]),
        "recall_real": float(rec_c[0]),
        "f1_real": float(f1_c[0]),
        "precision_ai": float(prec_c[1]),
        "recall_ai": float(rec_c[1]),
        "f1_ai": float(f1_c[1]),
        "confusion_matrix": cm,
    }


def per_generator_breakdown(meta, y_true, y_prob, threshold=0.5):
    """Metrics broken down by generator (AI images only)."""
    df = meta.copy()
    df["y_true"] = y_true
    df["y_prob"] = y_prob
    df["y_pred"] = (df["y_prob"] >= threshold).astype(int)

    ai = df[df["label"] == "ai"]
    rows = []
    for gen, sub in ai.groupby("generator"):
        m = compute_metrics(sub["y_true"], sub["y_prob"], threshold)
        rows.append({
            "generator": gen,
            "n": len(sub),
            "accuracy": m["accuracy"],
            "precision": m["precision_ai"],
            "recall": m["recall_ai"],
            "f1": m["f1_ai"],
            "roc_auc": m["roc_auc"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiment log
class ExperimentLog:
    """Append-only CSV log of experiment settings and results."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def log(self, run_id, member, model_name, dataset, n_train, n_val, n_test,
            epochs, batch_size, lr, augment, input_size, metrics, notes=""):
        row = {
            "run_id": run_id,
            "member": member,
            "model": model_name,
            "dataset": dataset,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "augment": augment,
            "input_size": input_size,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "roc_auc": metrics.get("roc_auc"),
            "precision_real": metrics.get("precision_real"),
            "recall_real": metrics.get("recall_real"),
            "precision_ai": metrics.get("precision_ai"),
            "recall_ai": metrics.get("recall_ai"),
            "f1_ai": metrics.get("f1_ai"),
            "confusion_matrix": json.dumps(metrics.get("confusion_matrix")),
            "notes": notes,
        }
        write_header = not self._header_written
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        self._header_written = True
        print(f"Logged run {run_id} to {self.path}")