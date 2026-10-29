"""
Scoring that every run and every test set uses.

  predict_table(model, df, root)  -> p_fake for every row (mean of 5 crops)
  evaluate_roles(...)             -> metrics per role, per generator, per format
  python -m src.evaluate --ckpt ... --table ... --root ...   (any table with path,label)
"""

import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

try:                                   # works as `python src/evaluate.py` and as a package
    from data import ImageTable
    from metrics import compute_metrics, per_format, per_generator
    from model import load_checkpoint
except ImportError:                    # pragma: no cover
    from .data import ImageTable
    from .metrics import compute_metrics, per_format, per_generator
    from .model import load_checkpoint


@torch.inference_mode()
def predict_table(model, df, root, device="cpu", batch_size=16, num_workers=0, crop=224):
    """Probability of AI for each row of df. Five crops per image, averaged."""
    model.eval().to(device)
    ds = ImageTable(df, root, train=False, crop=crop)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, ids = [], []
    for x, _, image_id in dl:
        b, n = x.shape[:2]
        logits = model(x.view(b * n, *x.shape[2:]).to(device))
        p = torch.sigmoid(logits).view(b, n).mean(1)
        probs.extend(p.cpu().numpy().tolist())
        ids.extend(image_id)
    out = pd.Series(probs, index=ids)
    return out.loc[df["image_id"].astype(str)].values, ds


def evaluate_roles(model, split, root, roles, threshold=0.5, device="cpu",
                   batch_size=16, num_workers=0):
    """Return (metrics dict by role, predictions DataFrame)."""
    metrics, preds = {}, []
    for role in roles:
        d = split[split.role == role]
        if len(d) == 0:
            continue
        p, _ = predict_table(model, d, root, device, batch_size, num_workers)
        y = d.label.map({"real": 0, "ai": 1}).values
        m = compute_metrics(y, p, threshold)
        if role in ("test_seen", "test_unseen", "wild_test"):
            m["per_generator"] = per_generator(d, p, threshold).to_dict("records")
            m["per_format"] = per_format(d, p, threshold).to_dict("records")
        metrics[role] = m
        o = d[["image_id", "path", "label", "generator", "format", "role"]].copy()
        o["p_fake"] = p
        preds.append(o)
    return metrics, (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def main():
    ap = argparse.ArgumentParser(description="Score a checkpoint on any table with path,label columns")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--table", required=True, help="csv with image_id, path, label (real/ai)")
    ap.add_argument("--root", required=True, help="folder that 'path' is relative to")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="eval_out")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(a.ckpt, dev)
    df = pd.read_csv(a.table)
    if "role" not in df:
        df["role"] = "all"
    for c in ("generator", "format"):
        if c not in df:
            df[c] = "unknown"
    m, p = evaluate_roles(model, df, a.root, df.role.unique(), a.threshold, dev)
    os.makedirs(a.out, exist_ok=True)
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    p.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    print(json.dumps({k: {x: v[x] for x in ("n", "accuracy", "macro_f1", "roc_auc")} for k, v in m.items()}, indent=2))


if __name__ == "__main__":
    main()
