"""
Audit the pulled data and build split.csv.

Works from the metadata CSVs alone (format, size, file bytes, hashes), so it
does NOT need the images on this machine. The optional near-duplicate check
(perceptual hash) does need the images and is a separate function.

Main entry points
  load_all_meta(meta_dir)        -> one table from every public_*.csv
  format_audit(meta)             -> tables that expose format/size gaps
  metadata_shortcut_test(meta)   -> can a model tell real from AI using NO pixels?
  dedupe(meta, phash=None)       -> drops exact (and near) duplicates
  make_split(meta, ...)          -> adds the 'role' column and writes split.csv
"""

import glob
import json
import os

import numpy as np
import pandas as pd

ROLES = ["train", "val", "test_seen", "test_unseen", "wild_test"]


# ---------------------------------------------------------------------------
def load_all_meta(meta_dir):
    files = sorted(glob.glob(os.path.join(meta_dir, "public_*.csv")))
    if not files:
        raise FileNotFoundError(f"No public_*.csv files in {meta_dir}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["member_file"] = os.path.basename(f)
        frames.append(d)
    meta = pd.concat(frames, ignore_index=True)
    meta["bytes_per_pixel"] = meta["file_bytes"] / (meta["width"] * meta["height"])
    meta["aspect"] = meta["width"] / meta["height"]
    return meta


# ---------------------------------------------------------------------------
def format_audit(meta):
    """Return a dict of small tables. Look at them: gaps here are shortcuts."""
    m = meta.copy()
    out = {}
    out["counts_by_class_source"] = pd.crosstab([m.source, m.role], m.label)
    out["format_by_class"] = pd.crosstab(m.format, m.label)
    out["format_share_by_class"] = pd.crosstab(m.format, m.label, normalize="columns").round(3)
    g = m.groupby("label")[["width", "height", "file_bytes", "bytes_per_pixel"]]
    out["size_summary"] = g.describe().T
    ai = m[m.label == "ai"]
    out["generators"] = ai.groupby(["generator", "format"]).size().unstack(fill_value=0)
    out["real_sources"] = m[m.label == "real"].groupby(["source", "source_detail", "format"]).size().unstack(fill_value=0)
    return out


def metadata_shortcut_test(meta, seed=0, folds=5):
    """Train a small model on metadata ONLY (no pixels).

    If it scores clearly above chance, the data contain a shortcut: a real
    detector could cheat the same way. Returns a table of ROC-AUC values
    (0.5 = no leak, 1.0 = metadata gives the answer away).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    d = meta[meta.role != "wild_test"].copy() if "role" in meta else meta.copy()
    d = d[d.label.isin(["real", "ai"])]
    y = (d.label == "ai").astype(int).values
    if y.sum() < folds or (1 - y).sum() < folds:
        return pd.DataFrame([dict(features="(too few images)", roc_auc=np.nan, verdict="n/a")])

    for c in ("file_bytes", "width", "height"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["file_bytes", "width", "height"])
    y = (d.label == "ai").astype(int).values
    d["log_bytes"] = np.log1p(d["file_bytes"].astype(float))
    d["bpp"] = d["file_bytes"] / (d["width"] * d["height"])
    d["aspect"] = d["width"] / d["height"]
    fmt = pd.get_dummies(d["format"].astype(str), prefix="fmt")
    mode = pd.get_dummies(d["mode"].astype(str), prefix="mode") if "mode" in d else pd.DataFrame(index=d.index)
    sets = {
        "format only": fmt,
        "width + height + aspect": d[["width", "height", "aspect"]],
        "file size + bytes per pixel": d[["log_bytes", "bpp"]],
        "ALL metadata together": pd.concat([fmt, mode, d[["width", "height", "aspect", "log_bytes", "bpp"]]], axis=1),
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for name, X in sets.items():
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=seed)
        p = cross_val_predict(clf, X.values.astype(float), y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        verdict = "SHORTCUT" if auc >= 0.75 else ("suspicious" if auc >= 0.62 else "ok")
        rows.append(dict(features=name, roc_auc=round(float(auc), 3), verdict=verdict))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def compute_phash(meta, root, hash_size=8):
    """Perceptual hash for every image (needs the images). Returns a Series of ints."""
    import imagehash
    from PIL import Image
    out = []
    for p in meta["path"]:
        try:
            with Image.open(os.path.join(root, p)) as im:
                out.append(int(str(imagehash.phash(im.convert("RGB"), hash_size=hash_size)), 16))
        except Exception:
            out.append(-1)
    return pd.Series(out, index=meta.index, name="phash")


def _near_groups(hashes, max_dist=4):
    """Union-find over hashes within Hamming distance max_dist."""
    n = len(hashes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    h = np.array([x if x >= 0 else 0 for x in hashes], dtype=object)
    for i in range(n):
        if hashes[i] < 0:
            continue
        for j in range(i + 1, n):
            if hashes[j] < 0:
                continue
            if bin(int(h[i]) ^ int(h[j])).count("1") <= max_dist:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return [find(i) for i in range(n)]


def dedupe(meta, phash=None, max_dist=4):
    """Drop exact duplicates (same sha256) and, if phash is given, near duplicates.

    When a duplicate exists, the NON-wild copy is kept, so an in-the-wild image
    that also appears in the training pool is the one that is removed.
    """
    m = meta.copy()
    m["_wild"] = (m["role"] == "wild_test").astype(int)
    m = m.sort_values(["_wild", "image_id"]).reset_index(drop=True)
    before = len(m)
    m = m.drop_duplicates("sha256", keep="first")
    n_exact = before - len(m)
    n_near = 0
    if phash is not None:
        lookup = dict(zip(meta["image_id"], phash.values))
        hashes = [lookup.get(i, -1) for i in m["image_id"]]
        grp = _near_groups(hashes, max_dist)
        m["_grp"] = grp
        n0 = len(m)
        m = m.drop_duplicates("_grp", keep="first").drop(columns="_grp")
        n_near = n0 - len(m)
    m = m.drop(columns="_wild").reset_index(drop=True)
    return m, dict(exact_removed=int(n_exact), near_removed=int(n_near), kept=int(len(m)))


# ---------------------------------------------------------------------------
def pick_holdout_generators(meta, holdout_frac=0.2, min_images=8, seed=42):
    """Choose whole AI generators to hide from training ('unseen generators')."""
    d = meta[(meta.label == "ai") & (meta.role != "wild_test")]
    counts = d.generator.value_counts()
    eligible = counts[counts >= min_images].index.tolist()
    n_total = counts.shape[0]
    n_hold = max(1, int(round(holdout_frac * n_total))) if n_total >= 4 else 0
    n_hold = min(n_hold, max(0, len(eligible) - 2))     # always keep >= 2 generators to train on
    rng = np.random.default_rng(seed)
    hold = [str(g) for g in rng.choice(sorted(eligible), size=n_hold, replace=False)] if n_hold else []
    return sorted(hold)


def make_split(meta, holdout_frac=0.2, seed=42, train_frac=0.70, val_frac=0.15):
    """Return (split_table, info). Roles: train, val, test_seen, test_unseen, wild_test."""
    m = meta.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    m["role"] = m["role"].where(m["role"] == "wild_test", "unassigned")

    hold = pick_holdout_generators(m, holdout_frac, seed=seed)
    is_pool = m["role"] == "unassigned"
    unseen_ai = is_pool & (m.label == "ai") & m.generator.isin(hold)
    m.loc[unseen_ai, "role"] = "test_unseen"

    # real images for the unseen test: same number as the unseen AI images
    real_pool = m.index[is_pool & (m.label == "real")].to_numpy()
    n_real_unseen = min(int(unseen_ai.sum()), max(0, len(real_pool) // 4))
    if n_real_unseen > 0:
        pick = rng.choice(real_pool, size=n_real_unseen, replace=False)
        m.loc[pick, "role"] = "test_unseen"

    # everything else: stratified 70 / 15 / 15
    rest = m[m.role == "unassigned"]
    strata = rest.label + "|" + np.where(rest.label == "ai", rest.generator, rest.source_detail.astype(str))
    for _, idx in rest.groupby(strata).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(train_frac * n))
        n_va = int(round(val_frac * n))
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(1, min(n_va, n - n_tr - 1))
        m.loc[idx[:n_tr], "role"] = "train"
        m.loc[idx[n_tr:n_tr + n_va], "role"] = "val"
        m.loc[idx[n_tr + n_va:], "role"] = "test_seen"

    # sanity checks that must always pass
    assert m["image_id"].is_unique, "an image id appears twice"
    assert m["sha256"].is_unique, "duplicate files remain"
    assert not (m.role == "unassigned").any()
    leaked = set(m[m.role == "train"].generator) & set(hold)
    assert not leaked, f"held-out generators leaked into train: {leaked}"

    info = dict(holdout_generators=hold, seed=seed,
                counts=m.groupby(["role", "label"]).size().unstack(fill_value=0).to_dict("index"))
    return m, info


# ---------------------------------------------------------------------------
def recommend(split, min_n=40):
    """Numbers the training configs read: how many images per class every run can use."""
    tr = split[split.role == "train"]
    ai_counts = tr[tr.label == "ai"].generator.value_counts()
    real_n = int((tr.label == "real").sum())
    single = ai_counts.index[0] if len(ai_counts) else None
    n_single = int(ai_counts.iloc[0]) if len(ai_counts) else 0
    n = int(min(n_single, real_n))
    return dict(single_generator=single, n_per_class=n,
                ai_train_by_generator=ai_counts.to_dict(), real_train=real_n,
                warning=("n_per_class is small; pull more images of the top generator"
                         if n < min_n else ""))


def save_all(split, info, rec, tables, shortcut, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    split.to_csv(os.path.join(out_dir, "split.csv"), index=False)
    with open(os.path.join(out_dir, "split_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    with open(os.path.join(out_dir, "recommended.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    shortcut.to_csv(os.path.join(out_dir, "shortcut_test.csv"), index=False)
    for k, t in tables.items():
        t.to_csv(os.path.join(out_dir, f"audit_{k}.csv"))
