"""Scoring that every run uses, so numbers are comparable."""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)


def compute_metrics(y_true, p_fake, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_fake, dtype=float)
    y_pred = (p >= threshold).astype(int)
    out = {"n": int(len(y_true)), "threshold": float(threshold),
           "accuracy": float(accuracy_score(y_true, y_pred))}
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    out.update(macro_f1=float(np.mean(f1)),
               precision_real=float(pr[0]), recall_real=float(rc[0]), f1_real=float(f1[0]),
               precision_ai=float(pr[1]), recall_ai=float(rc[1]), f1_ai=float(f1[1]))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p)) if len(set(y_true)) == 2 else float("nan")
    except ValueError:
        out["roc_auc"] = float("nan")
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


def best_threshold(y_true, p_fake):
    """Pick the threshold on VALIDATION data only (never on test or campus data)."""
    best, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        f = compute_metrics(y_true, p_fake, t)["macro_f1"]
        if f > best_f1:
            best, best_f1 = float(t), f
    return best


def per_generator(df, p_fake, threshold=0.5):
    """Recall per AI generator (how often each generator is caught)."""
    d = df.copy()
    d["p_fake"] = np.asarray(p_fake)
    d = d[d.label == "ai"]
    rows = [dict(generator=g, n=len(s), recall=float((s.p_fake >= threshold).mean()))
            for g, s in d.groupby("generator")]
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def per_format(df, p_fake, threshold=0.5):
    """Accuracy per file format and class. Catches format shortcuts."""
    d = df.copy()
    d["p_fake"] = np.asarray(p_fake)
    d["correct"] = ((d.p_fake >= threshold).astype(int) == d.label.map({"real": 0, "ai": 1})).astype(int)
    g = d.groupby(["format", "label"]).agg(n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    return g
