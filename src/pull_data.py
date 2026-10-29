"""
Lean, crash-safe puller for public image data (Hugging Face streaming).

WHY THIS VERSION EXISTS
-----------------------
The first puller asked for thousands of images per class but also capped every
generator at a handful of images. To satisfy both, it had to stream through a
huge part of the dataset, and every skipped example still downloads its image
bytes. That is what exhausted memory and crashed laptops.

What is different now
  * Small targets (about 100 to 150 images per class per person).
  * A hard SCAN LIMIT: we stop after looking at MAX_SCAN examples, no matter what.
  * A time limit and a storage limit.
  * Per-generator caps that relax automatically when the stream is not diverse,
    instead of reading forever.
  * Tiny shuffle buffers, no progress bars, short log lines.
  * Resumable: if it stops, run it again and it continues from the saved CSV.

This file is plain Python. It can run in Colab or in a local Jupyter.
"""

import gc
import hashlib
import io
import os
import time
from collections import Counter

import pandas as pd
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Who pulls what (images PER CLASS, so 100 means 100 real + 100 AI).
# AW and SM are lighter because their machines crashed on the old version.
# ---------------------------------------------------------------------------
MEMBERS = ["GW", "AW", "NG", "NC", "SM", "PN"]
SEEDS = {"GW": 11, "AW": 22, "NG": 33, "NC": 44, "SM": 55, "PN": 66}

MEMBER_PLAN = {
    "GW": {"openfake": 150},
    "AW": {"openfake": 100},
    "NG": {"openfake": 150, "reddit": 150},
    "NC": {"openfake": 100, "community": 100},
    "SM": {"community": 100},
    "PN": {"openfake": 100, "community": 100},
}

PILOT_PER_CLASS = 8          # tiny test run: proves that nothing crashes
MAX_SCAN = 2500              # look at no more than this many stream examples per source
MAX_MINUTES = 12             # stop after this long per source
MAX_GB = 0.6                 # stop after this much saved per source

EXT = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp",
       "GIF": "gif", "BMP": "bmp", "TIFF": "tif"}

META_COLS = ["image_id", "path", "label", "source", "source_detail", "generator",
             "original_id", "version", "role", "format", "width", "height", "mode",
             "file_bytes", "sha256", "license", "creator", "pulled_at"]

SPECS = {
    "openfake": dict(
        repo="ComplexDataLab/OpenFake", config="core", split="train",
        image_col="image", decode_feature=True, buffer=10,
        label_fn=lambda ex: {"real": "real", "fake": "ai"}.get(str(ex.get("label")).lower()),
        gen_fn=lambda ex, lab: "real" if lab == "real" else str(ex.get("model")),
        detail_fn=lambda ex: str(ex.get("type")),
        orig_fn=lambda ex: str(ex.get("row_id")),
        skip_fn=lambda ex: False,
        role="train_val", folder="public_data/openfake", prefix="OF",
        license="CC BY-NC 4.0",
        cap={"real": 50, "ai": 25}),          # max images per group (per person)
    "community": dict(
        repo="OwensLab/CommunityForensics-Small", config=None, split="train",
        image_col="image_data", decode_feature=False, buffer=10,
        label_fn=lambda ex: {0: "real", 1: "ai"}.get(int(ex.get("label"))),
        gen_fn=lambda ex, lab: "real" if lab == "real" else str(ex.get("model_name")),
        detail_fn=lambda ex: str(ex.get("real_source") if int(ex.get("label")) == 0
                                 else ex.get("architecture")),
        orig_fn=lambda ex: str(ex.get("image_name")),
        skip_fn=lambda ex: bool(ex.get("nsfw_flag")),
        role="train_val", folder="public_data/community_forensics", prefix="CF",
        license="CC BY-NC-SA 4.0",
        cap={"real": 30, "ai": 10}),
    "reddit": dict(
        repo="ComplexDataLab/OpenFake", config="reddit", split=None,
        image_col="image", decode_feature=True, buffer=10,
        label_fn=lambda ex: {"real": "real", "fake": "ai"}.get(str(ex.get("label")).lower()),
        gen_fn=lambda ex, lab: "real" if lab == "real" else str(ex.get("model")),
        detail_fn=lambda ex: str(ex.get("type")),
        orig_fn=lambda ex: str(ex.get("row_id")),
        skip_fn=lambda ex: False,
        role="wild_test", folder="in_the_wild/openfake_reddit", prefix="OR",
        license="CC BY-NC 4.0",
        cap={"real": 60, "ai": 30}),
}

FOLDERS = ["public_data/openfake", "public_data/community_forensics",
           "in_the_wild/openfake_reddit", "campus/originals", "campus/whatsapp_photo",
           "campus/whatsapp_document", "metadata", "audit_outputs", "reports"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def find_drive_root(candidates=None):
    """Find the shared project folder inside My Drive (after mounting Drive).

    The group used two different names, so we look for both and print what we
    found. Returns the full path, or None if nothing matches.
    """
    base = "/content/drive/MyDrive"
    names = candidates or ["DEEPFAKE_CV_DATA",
                           os.path.join("Deepfake Detection Project", "datasets"),
                           "Deepfake Detection Project"]
    hits = [os.path.join(base, n) for n in names if os.path.isdir(os.path.join(base, n))]
    return hits[0] if hits else None


def make_folders(root):
    for f in FOLDERS:
        os.makedirs(os.path.join(root, f), exist_ok=True)


def get_bytes(v):
    if isinstance(v, dict):
        v = v.get("bytes")
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    raise ValueError(f"unsupported image value type: {type(v)}")


def group_of(label, generator, detail):
    return generator if label == "ai" else detail


def meta_path_for(root, name, member):
    return os.path.join(root, "metadata", f"public_{name}_{member}.csv")


def open_stream(name, seed):
    """Open a Hugging Face stream. Nothing is downloaded until we iterate."""
    from datasets import load_dataset, get_dataset_split_names, Image as HFImage
    spec = SPECS[name]
    split = spec["split"]
    if split is None:
        splits = get_dataset_split_names(spec["repo"], spec["config"])
        print(f"[{name}] splits available: {splits}")
        split = splits[0]
    ds = load_dataset(spec["repo"], name=spec["config"], split=split, streaming=True)
    if spec["decode_feature"]:
        ds = ds.cast_column(spec["image_col"], HFImage(decode=False))
    return ds.shuffle(seed=seed, buffer_size=spec["buffer"])


# ---------------------------------------------------------------------------
# the puller
# ---------------------------------------------------------------------------
def pull_from_stream(stream, name, member, root, n_per_class,
                     max_scan=MAX_SCAN, max_minutes=MAX_MINUTES, max_gb=MAX_GB,
                     flush_every=25, log_every=50, spec=None):
    """Pull images from `stream` (any iterable of dict-like examples).

    Stops at the FIRST of: both classes reached n_per_class, max_scan examples
    looked at, max_minutes elapsed, max_gb saved.
    """
    spec = spec or SPECS[name]
    out_dir = os.path.join(root, spec["folder"])
    os.makedirs(out_dir, exist_ok=True)
    meta_path = meta_path_for(root, name, member)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    counts, per_group, seen, k, used_bytes = Counter(), Counter(), set(), 0, 0
    cap = dict(spec["cap"])

    # resume from the saved CSV
    if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
        try:
            old = pd.read_csv(meta_path, usecols=["sha256", "label", "generator",
                                                  "source_detail", "file_bytes"])
            for r in old.itertuples(index=False):
                counts[r.label] += 1
                per_group[(r.label, group_of(r.label, r.generator, r.source_detail))] += 1
                seen.add(r.sha256)
                used_bytes += int(r.file_bytes)
            k = len(old)
            print(f"[{name}] resuming: {k} images already saved by {member} ({dict(counts)})")
        except Exception as e:
            print(f"[{name}] could not resume ({e}); starting fresh")

    buf, skipped, t0, scanned, first = [], Counter(), time.time(), 0, True
    relaxed = False
    reason = "stream ended"

    def flush():
        if buf:
            need_header = (not os.path.exists(meta_path)) or os.path.getsize(meta_path) == 0
            pd.DataFrame(buf, columns=META_COLS).to_csv(
                meta_path, mode="a", header=need_header, index=False)
            buf.clear()

    for ex in stream:
        scanned += 1
        if scanned > max_scan:
            reason = f"scan limit ({max_scan} examples looked at)"
            break
        if time.time() - t0 > max_minutes * 60:
            reason = f"time limit ({max_minutes} min)"
            break
        if used_bytes > max_gb * 1e9:
            reason = f"storage limit ({max_gb} GB)"
            break
        if all(counts[c] >= n_per_class for c in ("real", "ai")):
            reason = "target reached"
            break
        if first:
            print(f"[{name}] columns: {sorted(ex.keys())}")
            first = False

        # after half the scan budget, double the per-group caps once so a
        # not-very-diverse stream can still fill the target
        if (not relaxed) and scanned > max_scan // 2 and \
                any(counts[c] < n_per_class for c in ("real", "ai")):
            cap = {c: v * 2 for c, v in cap.items()}
            relaxed = True
            print(f"[{name}] stream is not diverse enough; per-group caps doubled -> {cap}")

        label = spec["label_fn"](ex)
        if label is None or spec["skip_fn"](ex):
            skipped["label_or_flag"] += 1
            continue
        if counts[label] >= n_per_class:
            skipped["class_full"] += 1
            continue

        gen, detail = spec["gen_fn"](ex, label), spec["detail_fn"](ex)
        g = (label, group_of(label, gen, detail))
        if per_group[g] >= cap[label]:
            skipped["group_cap"] += 1
            continue

        try:
            raw = get_bytes(ex[spec["image_col"]])
            with PILImage.open(io.BytesIO(raw)) as im:
                fmt, (w, h), mode = im.format, im.size, im.mode
        except Exception:
            skipped["unreadable"] += 1
            continue

        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            skipped["duplicate"] += 1
            continue

        k += 1
        image_id = f"{spec['prefix']}_{member}_{k:06d}"
        fname = f"{image_id}.{EXT.get(fmt, (fmt or 'bin').lower())}"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(raw)                      # original bytes, never re-encoded

        seen.add(sha)
        counts[label] += 1
        per_group[g] += 1
        used_bytes += len(raw)
        buf.append(dict(image_id=image_id, path=os.path.join(spec["folder"], fname),
                        label=label, source=name, source_detail=detail, generator=gen,
                        original_id=spec["orig_fn"](ex), version="orig", role=spec["role"],
                        format=fmt, width=w, height=h, mode=mode, file_bytes=len(raw),
                        sha256=sha, license=spec["license"], creator=member,
                        pulled_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        del raw

        if len(buf) >= flush_every:
            flush()
            gc.collect()
        if k % log_every == 0:
            print(f"[{name}] saved {dict(counts)} | looked at {scanned} | "
                  f"{used_bytes / 1e6:.0f} MB | {int(time.time() - t0)} s")

    flush()
    gc.collect()
    n_groups = len({g for g, v in per_group.items() if v > 0 and g[0] == "ai"})
    return dict(source=name, member=member, counts=dict(counts), looked_at=scanned,
                megabytes=round(used_bytes / 1e6), seconds=round(time.time() - t0),
                ai_groups_seen=n_groups, stopped_because=reason, skipped=dict(skipped))


def run_member(member, root, pilot=True, only=None):
    """Pull everything assigned to `member` in MEMBER_PLAN, one source at a time."""
    assert member in MEMBERS, f"MEMBER must be one of {MEMBERS}"
    make_folders(root)
    plan = MEMBER_PLAN[member]
    results = []
    for name, n in plan.items():
        if only and name != only:
            continue
        n_eff = PILOT_PER_CLASS if pilot else n
        print(f"\n=== {name}: {n_eff} real + {n_eff} AI for {member} "
              f"({'PILOT' if pilot else 'FULL'}) ===")
        stream = open_stream(name, SEEDS[member])
        res = pull_from_stream(
            stream, name, member, root, n_eff,
            max_scan=(300 if pilot else MAX_SCAN),
            max_minutes=(3 if pilot else MAX_MINUTES),
            max_gb=(0.1 if pilot else MAX_GB))
        print(res)
        results.append(res)
        del stream
        gc.collect()
    return results


def summary(root, member=None):
    """Print how many images each metadata file holds."""
    import glob
    pat = f"public_*_{member}.csv" if member else "public_*.csv"
    files = sorted(glob.glob(os.path.join(root, "metadata", pat)))
    if not files:
        print("No metadata files yet.")
        return
    for f in files:
        df = pd.read_csv(f)
        print(f"{os.path.basename(f):32s} rows={len(df):5d}  "
              f"{dict(df.label.value_counts())}  generators={df[df.label == 'ai'].generator.nunique()}"
              f"  formats={dict(df.format.value_counts())}")
