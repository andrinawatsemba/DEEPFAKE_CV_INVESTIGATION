"""
Evaluate a saved model on the test split.

Produces: overall metrics, per-generator breakdown, confusion matrix,
error breakdown by source/label, and shortcut checks (format vs label).

Usage (in a Colab cell):
    %run src/evaluate.py --model reports/resnet50_transfer.pt --model-type resnet50

Edit CONFIG or pass CLI flags.
"""

import os, sys, json, argparse
from datetime import datetime

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_metadata, make_dataloaders, compute_metrics,
    per_generator_breakdown, DEFAULT_MEMBER, DRIVE_ROOT,
)
from train_transfer import build_model as build_resnet
from train_baseline import SmallCNN


# ------------------------------------------------------------------ config
CONFIG = dict(
    member=DEFAULT_MEMBER,
    model_path="reports/resnet50_transfer.pt",
    model_type="resnet50",   # "resnet50", "resnet18", or "smallcnn"
    input_size=224,
    batch_size=32,
    sources=["openfake", "community", "reddit"],
    out_dir="reports",
    run_id="Eval_Final",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ------------------------------------------------------------------ model load
def load_model(model_path, model_type, device):
    if model_type == "smallcnn":
        model = SmallCNN(input_size=CONFIG["input_size"]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    else:
        return build_resnet(finetune=True, resnet18=(model_type == "resnet18")).to(device) \
            if False else _load_resnet(model_path, model_type, device)


def _load_resnet(model_path, model_type, device):
    from torchvision import models
    import torch.nn as nn
    if model_type == "resnet18":
        model = models.resnet18(weights=None)
        in_f = model.fc.in_features
    else:
        model = models.resnet50(weights=None)
        in_f = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ------------------------------------------------------------------ main
def main(cfg):
    root = DRIVE_ROOT
    member = cfg["member"]

    # 1) Load metadata
    meta = load_metadata(root, sources=cfg["sources"], members=[member])
    if meta.empty:
        raise SystemExit(
            f"No metadata found for member {member}. Run Week 1 acquisition first."
        )
    test_meta = meta[meta["split"] == "test_seen"] if "split" in meta.columns else meta
    print(f"Loaded {len(meta)} total rows; {len(test_meta)} in test_seen for member {member}")

    # 2) Build test loader
    from utils import ImageLabelDataset
    from torch.utils.data import DataLoader
    test_ds = ImageLabelDataset(test_meta, input_size=cfg["input_size"], augment=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=2,
                             pin_memory=torch.cuda.is_available())

    # 3) Load model
    model_path = cfg["model_path"]
    if not os.path.exists(model_path):
        raise SystemExit(f"Model checkpoint not found: {model_path}")
    model = load_model(model_path, cfg["model_type"], device)

    # 4) Evaluate
    y_true, y_prob, image_ids = [], [], []
    for x, y, ids in test_loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        y_prob.extend(prob.tolist())
        y_true.extend(y.numpy().tolist())
        image_ids.extend(ids)

    metrics = compute_metrics(y_true, y_prob)
    print("\n=== TEST RESULTS ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print("  confusion_matrix:", metrics["confusion_matrix"])

    # 5) Per-generator breakdown
    gen_df = per_generator_breakdown(test_meta, y_true, y_prob)
    print("\n=== PER-GENERATOR BREAKDOWN (AI images) ===")
    print(gen_df.to_string(index=False))

    # 6) Shortcut check: format vs label
    print("\n=== SHORTCUT CHECK: format share by label ===")
    fmt = test_meta.assign(
        is_jpeg=test_meta["format"].isin(["JPEG", "MPO"])
    ).groupby("label").is_jpeg.mean()
    print(fmt.to_dict())

    # 7) Save outputs
    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = {
        "run_id": cfg["run_id"],
        "member": member,
        "model": model_path,
        "model_type": cfg["model_type"],
        "n_test": len(test_meta),
        "metrics": metrics,
        "per_generator": gen_df.to_dict(orient="records"),
        "shortcut_format_share": fmt.to_dict(),
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    metrics_path = os.path.join(cfg["out_dir"], f"{cfg['run_id']}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    gen_df.to_csv(os.path.join(cfg["out_dir"], f"{cfg['run_id']}_per_generator.csv"), index=False)

    # Save predictions for error analysis
    pred_df = pd.DataFrame({
        "image_id": image_ids,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": (np.array(y_prob) >= 0.5).astype(int),
    })
    pred_df = pred_df.merge(test_meta[["image_id", "source", "label", "generator", "format"]],
                            on="image_id", how="left")
    pred_df.to_csv(os.path.join(cfg["out_dir"], f"{cfg['run_id']}_predictions.csv"), index=False)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Predictions saved to {cfg['out_dir']}/{cfg['run_id']}_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=CONFIG["model_path"])
    parser.add_argument("--model-type", default=CONFIG["model_type"],
                        choices=["resnet50", "resnet18", "smallcnn"])
    parser.add_argument("--member", default=CONFIG["member"])
    parser.add_argument("--input-size", type=int, default=CONFIG["input_size"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--out-dir", default=CONFIG["out_dir"])
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg.update(model_path=args.model, model_type=args.model_type,
               member=args.member, input_size=args.input_size,
               batch_size=args.batch_size, out_dir=args.out_dir)
    main(cfg)