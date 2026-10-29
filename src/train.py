"""
ONE training script for Runs A to E. The run is chosen by a config file.

    python src/train.py --config configs/run_B.yaml \
        --split <path>/split.csv --root <image folder> --out <reports folder>

It writes the full evidence bundle to <out>/<run_id>/ :
    config.yaml  metrics.json  predictions.csv  best.pt  training_curves.png
    confusion_matrix.png  failures.csv  failures/ (thumbnails)  shortcut_check.csv  README.md
and appends one row to <out>/experiment_log.csv
"""

import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audit_split import metadata_shortcut_test
from data import ImageTable, load_image, denormalise
from evaluate import evaluate_roles, predict_table
from metrics import best_threshold, compute_metrics
from model import build_model, param_groups, save_checkpoint, set_stage


# ---------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_training_rows(split, cfg, recommended, seed):
    """Pick the images this run trains on.

    * Only role == 'train'.
    * AI side: ONE generator (cfg ai_train: single) or ALL seen generators (cfg ai_train: all).
    * Same number of images per class in every run (n_per_class), so runs differ in ONE thing.
    * Real images are matched to the AI images FORMAT BY FORMAT, so file format
      cannot predict the label inside the training set.
    """
    tr = split[split.role == "train"]
    ai = tr[tr.label == "ai"]
    real = tr[tr.label == "real"]

    if cfg["ai_train"] == "single":
        gen = cfg.get("single_generator", "auto")
        if gen == "auto":
            gen = recommended["single_generator"]
        ai = ai[ai.generator == gen]
        scope = f"single generator: {gen}"
    else:
        scope = "all seen generators"

    n = cfg["n_per_class"]
    if n == "auto":
        n = recommended["n_per_class"]
    n = int(min(n, len(ai), len(real)))
    if n < 10:
        raise SystemExit(f"Only {n} usable images per class. Pull more data or choose another generator.")

    # take n AI images (spread across generators when 'all'), then match reals by format
    if cfg["ai_train"] == "all":
        parts, per = [], max(1, n // max(1, ai.generator.nunique()))
        for g, s in ai.groupby("generator"):
            parts.append(s.sample(min(len(s), per), random_state=seed))
        pick_ai = pd.concat(parts)
        if len(pick_ai) < n:
            rest = ai.drop(pick_ai.index)
            pick_ai = pd.concat([pick_ai, rest.sample(min(len(rest), n - len(pick_ai)), random_state=seed)])
        pick_ai = pick_ai.head(n)
    else:
        pick_ai = ai.sample(n, random_state=seed)

    reals, ais = [], []
    for fmt, s_ai in pick_ai.groupby("format"):
        s_real = real[real.format == fmt]
        k = min(len(s_ai), len(s_real))
        if k:
            ais.append(s_ai.sample(k, random_state=seed))
            reals.append(s_real.sample(k, random_state=seed))
    if not ais:
        raise SystemExit("No format has both real and AI images. The data have a format shortcut: "
                         "see the audit notebook and docs/preprocessing.md.")
    train = pd.concat(ais + reals).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, scope, n


def val_rows(split, train_df, cfg):
    v = split[split.role == "val"]
    gens = set(train_df[train_df.label == "ai"].generator)
    return v[(v.label == "real") | (v.generator.isin(gens))]


# ---------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, opt, device, scaler):
    model.train()
    tot, n = 0.0, 0
    for x, y, _ in loader:
        x = x.squeeze(1).to(device)                  # one random crop per image in training
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += float(loss) * len(y)
        n += len(y)
    return tot / max(1, n)


def val_score(model, vdf, root, device, bs):
    p, _ = predict_table(model, vdf, root, device, bs)
    y = vdf.label.map({"real": 0, "ai": 1}).values
    m = compute_metrics(y, p)
    return m["macro_f1"], p, y


def fit(model, name, cfg, train_df, vdf, root, device, out_dir, workers):
    bs = cfg["batch_size"]
    loader = DataLoader(
        ImageTable(train_df, root, train=True, augment=cfg["augment"], seed=cfg["seed"]),
        batch_size=bs, shuffle=True, num_workers=workers, drop_last=len(train_df) > bs,
        pin_memory=(device == "cuda"))
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    stages = ([(1, cfg["epochs_head"]), (2, cfg["epochs_finetune"])] if name != "tinycnn"
              else [(1, cfg["epochs_scratch"])])
    hist, best, best_epoch, bad, ep = [], -1.0, 0, 0, 0
    ckpt = os.path.join(out_dir, "best.pt")
    for stage, n_ep in stages:
        set_stage(model, stage)
        opt = torch.optim.AdamW(param_groups(model, stage, cfg["lr_head"], cfg["lr_finetune"]),
                                weight_decay=1e-4)
        bad = 0
        for _ in range(n_ep):
            ep += 1
            t0 = time.time()
            tl = run_epoch(model, loader, loss_fn, opt, device, scaler)
            f1, _, _ = val_score(model, vdf, root, device, bs)
            hist.append(dict(epoch=ep, stage=stage, train_loss=tl, val_macro_f1=f1))
            flag = ""
            if f1 > best:
                best, best_epoch, bad = f1, ep, 0
                save_checkpoint(model, ckpt, name)
                flag = "  <- best"
            else:
                bad += 1
            print(f"epoch {ep:2d} (stage {stage}) loss {tl:.4f}  val macro-F1 {f1:.3f}  "
                  f"{time.time() - t0:.0f}s{flag}")
            if bad >= cfg["patience"]:
                print("early stopping in this stage")
                break
    return pd.DataFrame(hist), best, best_epoch, ckpt


# ---------------------------------------------------------------------------
def draw_curves(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(hist.epoch, hist.train_loss, marker="o")
    ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist.epoch, hist.val_macro_f1, marker="o", color="#A64B14")
    ax[1].set_title("validation macro-F1"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def draw_confusions(metrics, path):
    roles = [r for r in ("test_seen", "test_unseen", "wild_test") if r in metrics]
    if not roles:
        return
    fig, ax = plt.subplots(1, len(roles), figsize=(3.6 * len(roles), 3.4))
    ax = np.atleast_1d(ax)
    for a, r in zip(ax, roles):
        cm = np.array(metrics[r]["confusion_matrix"])
        a.imshow(cm, cmap="Oranges")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, int(v), ha="center", va="center")
        a.set_xticks([0, 1]); a.set_xticklabels(["real", "AI"])
        a.set_yticks([0, 1]); a.set_yticklabels(["real", "AI"])
        a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(r)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_failures(preds, root, out_dir, thr, k=24):
    wrong = preds[(preds.p_fake >= thr).astype(int) != preds.label.map({"real": 0, "ai": 1})].copy()
    wrong["confidence_in_wrong_answer"] = np.where(wrong.p_fake >= thr, wrong.p_fake, 1 - wrong.p_fake)
    wrong = wrong.sort_values("confidence_in_wrong_answer", ascending=False)
    wrong.to_csv(os.path.join(out_dir, "failures.csv"), index=False)
    fdir = os.path.join(out_dir, "failures")
    os.makedirs(fdir, exist_ok=True)
    for _, r in wrong.head(k).iterrows():
        try:
            x, _ = load_image(os.path.join(root, r.path), mode="center")
            from PIL import Image
            Image.fromarray(denormalise(x[0])).save(
                os.path.join(fdir, f"{r.role}_{r.label}_p{r.p_fake:.2f}_{r.image_id}.png"))
        except Exception:
            pass
    return len(wrong)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="split.csv from the audit notebook")
    ap.add_argument("--root", required=True, help="folder the 'path' column is relative to")
    ap.add_argument("--out", required=True, help="reports folder (a run folder is created inside)")
    ap.add_argument("--recommended", default=None, help="recommended.json (default: next to split.csv)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--member", default="NA", help="your code: GW, AW, NG, NC, SM, PN")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to test the plumbing")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline tests)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.smoke:
        cfg.update(epochs_head=1, epochs_finetune=1, epochs_scratch=1, n_per_class=min(
            20, cfg["n_per_class"]) if cfg["n_per_class"] != "auto" else 20, batch_size=8)
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Run {cfg['run_id']} | model {cfg['model']} | device {device}")

    split = pd.read_csv(a.split)
    rec_path = a.recommended or os.path.join(os.path.dirname(a.split), "recommended.json")
    recommended = json.load(open(rec_path)) if os.path.exists(rec_path) else {}
    out_dir = os.path.join(a.out, cfg["run_id"])
    os.makedirs(out_dir, exist_ok=True)

    train_df, scope, n = select_training_rows(split, cfg, recommended, cfg["seed"])
    vdf = val_rows(split, train_df, cfg)
    print(f"Training rows: {len(train_df)} ({scope}); validation rows: {len(vdf)}")

    # the shortcut check, on exactly the rows this run trains on
    chk = metadata_shortcut_test(train_df.assign(role="train"))
    chk.to_csv(os.path.join(out_dir, "shortcut_check.csv"), index=False)
    print("Metadata-only check on the training rows (0.5 = no leak):")
    print(chk.to_string(index=False))
    if (chk.verdict == "SHORTCUT").any():
        print("WARNING: file metadata alone predicts the label in this training set. "
              "Do not trust this run. Read docs/preprocessing.md.")

    model = build_model(cfg["model"], pretrained=(cfg["model"] != "tinycnn" and not a.no_pretrained))
    hist, best_f1, best_ep, ckpt = fit(model, cfg["model"], cfg, train_df, vdf, a.root, device, out_dir, a.workers)
    draw_curves(hist, os.path.join(out_dir, "training_curves.png"))

    # reload the best epoch, choose the threshold on VALIDATION only, then score every role
    from model import load_checkpoint
    model, _ = load_checkpoint(ckpt, device)
    pv, _ = predict_table(model, vdf, a.root, device, cfg["batch_size"])
    thr = best_threshold(vdf.label.map({"real": 0, "ai": 1}).values, pv)
    metrics, preds = evaluate_roles(model, split, a.root,
                                    ["val", "test_seen", "test_unseen", "wild_test"],
                                    thr, device, cfg["batch_size"], a.workers)
    metrics["_run"] = dict(run_id=cfg["run_id"], owner=a.member, model=cfg["model"], scope=scope,
                           n_per_class=n, threshold=thr, best_epoch=best_ep, best_val_macro_f1=best_f1,
                           trained_on_images=int(len(train_df)), seed=cfg["seed"])
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    draw_confusions(metrics, os.path.join(out_dir, "confusion_matrix.png"))
    n_wrong = save_failures(preds, a.root, out_dir, thr)
    shutil.copy(a.config, os.path.join(out_dir, "config.yaml"))

    gap = None
    if "test_seen" in metrics and "test_unseen" in metrics:
        gap = metrics["test_seen"]["macro_f1"] - metrics["test_unseen"]["macro_f1"]
    row = dict(run_id=cfg["run_id"], owner=a.member, date=time.strftime("%Y-%m-%d %H:%M"),
               model=cfg["model"], scope=scope, augment=cfg["augment"], n_per_class=n, seed=cfg["seed"],
               val_f1=metrics["val"]["macro_f1"],
               seen_f1=metrics.get("test_seen", {}).get("macro_f1"),
               unseen_f1=metrics.get("test_unseen", {}).get("macro_f1"),
               wild_f1=metrics.get("wild_test", {}).get("macro_f1"),
               seen_minus_unseen=gap, threshold=thr)
    log = os.path.join(a.out, "experiment_log.csv")
    pd.DataFrame([row]).to_csv(log, mode="a", header=not os.path.exists(log), index=False)

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(f"# {cfg['run_id']}\n\nRun by **{a.member}** on {row['date']}.\n\n"
                f"- model: {cfg['model']}; training data: {scope}; {n} images per class; augmentation: {cfg['augment']}\n"
                f"- threshold chosen on validation: {thr:.2f}\n"
                f"- macro-F1  seen generators: {row['seen_f1']}  | unseen generators: {row['unseen_f1']}  | in the wild: {row['wild_f1']}\n"
                f"- wrong predictions: {n_wrong} (see failures.csv and failures/). "
                f"Sort at least 20 of them into categories and write that in analysis.md.\n")
    print("\nDONE. Bundle:", out_dir)
    print(json.dumps({k: round(row[k], 3) if isinstance(row[k], float) else row[k]
                      for k in ("val_f1", "seen_f1", "unseen_f1", "wild_f1")}, indent=1))


if __name__ == "__main__":
    main()
