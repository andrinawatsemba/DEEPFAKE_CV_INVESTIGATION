# THE PULL - settings at the top of the script: MEMBER, PILOT = False, ROOT, ONLY = openfake
# ==============================================================================================
# PULL MY SHARE  -  one script for every member (GW, AW, NG, NC, SM, PN).  CPU runtime is fine.
#
# BEFORE YOU RUN (separate cell, then Runtime > Restart session if Colab asks):
#     !pip -q install -U datasets "pyarrow<24" huggingface_hub psutil
# Then mount Drive in its own cell:   from google.colab import drive; drive.mount("/content/drive")
# Then paste THIS whole file into ONE cell, change the 3 lines below, run. Run PILOT = False first.
#
# What it does for you (the problems we found while auditing):
#  1. NO OVERLAP BETWEEN MEMBERS: an image belongs to the member whose slot equals sha256 % 6, so two members can
#     never pull the same image, even if it sits in different files or in both OpenFake core and Reddit.
#  2. FILE-FORMAT AND SIZE SHORTCUTS: images are matched in cells of (file format x size bin). Inside every cell the
#     number of real and AI images is equal, so neither the format nor the image size can tell the classes apart.
#     Cells with only one class (for example AI PNG with no real PNG) are never filled.
#  3. BALANCE: caps per AI generator and per real-image type, so one source cannot dominate.
#  4. COMMUNITY FORENSICS: each file holds ONE class and its real PNGs sit in huge blocks. The script checks, from
#     metadata only, whether a matched sample is feasible within MAX_BLOCK_MB. If not, it pulls that share from
#     OpenFake instead only if you set REDISTRIBUTE = True (default False: each dataset gives its own share).
#  5. RAM SAFETY: nothing is decoded, RAM is checked, blocks are checked against available RAM, CSV is saved every 25 images, and a
#     crashed run resumes where it stopped. Run one member per Colab session; restart before rerunning.
# ==============================================================================================
import os, io, gc, math, time, hashlib, random, warnings
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from PIL import Image as PILImage
warnings.filterwarnings("ignore")
try:
    import psutil
except ImportError:
    psutil = None

# ---------------- EDIT ONLY THESE 3 LINES ----------------
MEMBER = "NC"                                   # GW, AW, NG, NC, SM or PN
PILOT = False                                   # True = tiny test (10 per class). Set False for the real pull.
ROOT = "/content/drive/MyDrive/DEEPFAKE_CV_DATA"  # the SHARED folder (add it to My Drive as a shortcut first)
# ----------------------------------------------------------

MEMBERS = ["GW", "AW", "NG", "NC", "SM", "PN"]            # slot order, the same for everyone
PLAN = {"GW": {"openfake": 150}, "AW": {"openfake": 100}, "NG": {"openfake": 150, "reddit": 150},
        "NC": {"openfake": 100, "community": 100}, "SM": {"community": 100},
        "PN": {"openfake": 100, "community": 100}}          # images PER CLASS (so 100 = 100 real + 100 AI)
REDISTRIBUTE = False       # False = each dataset gives ITS OWN share only (100 real + 100 AI). True = a shortfall is made up from OpenFake
MATCH_REDDIT = False       # Reddit is the in-the-wild TEST set: keep it natural (report the shortcuts, do not match)
MIN_SIDE = 224             # shorter side in pixels (the models crop 224x224 at native resolution)
MAX_BLOCK_MB = 0        # Community Forensics: 0 disables the fixed-size cutoff; row groups are checked against live free RAM.
                           # Larger row groups are attempted only when the dynamic RAM safety check allows them.
HEAD_ROWS = None             # Community Forensics: None = consider rows throughout each block.
ONLY = os.environ.get("PULL_ONLY") or None   # set automatically by the driver below. Leave as is.
MAX_MINUTES = 5 if PILOT else 25
MAX_SCAN = 4000 if PILOT else 40000
RAM_STOP = 85              # stop (and save) when RAM use goes above this percent
K_SLACK = 2                # inside a cell, one class may lead the other by at most this many images
EDGES = [0, .1, .2, .3, .45, .6, 1, 2, 4, 1e9]      # finer bins for small images, where most of the data is (512x512 = 0.26 MP)
BIN_NAMES = ["<0.1MP", "0.1-0.2MP", "0.2-0.3MP", "0.3-0.45MP", "0.45-0.6MP", "0.6-1MP", "1-2MP", "2-4MP", ">4MP"]
META_COLS = ["image_id", "path", "label", "source", "source_detail", "generator", "original_id", "version", "role",
             "format", "width", "height", "mode", "file_bytes", "sha256", "license", "creator", "pulled_at"]
EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
IDX = MEMBERS.index(MEMBER)
SEED = 41 + IDX


def free_memory():
    """Hand freed memory back to the OS (Arrow and glibc keep it otherwise), so the next big step starts clean."""
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def ram_pct():
    return psutil.virtual_memory().percent if psutil else 0.0


def cell_of(fmt, mp):
    b = int(np.clip(np.searchsorted(EDGES, mp, side="right") - 1, 0, len(BIN_NAMES) - 1))
    return f"{fmt}|{BIN_NAMES[b]}"


def meta_path(source):
    return f"{ROOT}/metadata/public_{source}_{MEMBER}.csv"


def load_rows(source, prefix):
    """Resume from our own CSV. Any older CSV (not made by this script) is kept under a new name, never deleted."""
    p = meta_path(source)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p) and os.path.getsize(p) > 400:
        d = pd.read_csv(p)
        if len(d) and d.image_id.astype(str).str.startswith(prefix).all():
            return d.to_dict("records")
        bak = p.replace(".csv", f"_OLD_{int(time.time())}.csv")
        os.replace(p, bak)
        print("  older CSV kept as", os.path.basename(bak), "(not overwritten)")
    return []


def save_rows(source, rows):
    pd.DataFrame(rows, columns=META_COLS).to_csv(meta_path(source), index=False)


def trim_matched(rows, N, prefix):
    """Keep only matched pairs: in every (format x size) cell the same number of real and AI, at most N pairs in total.
    Deletes only the files this script created (ids start with `prefix`)."""
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    df["cell"] = [cell_of(f, w * h / 1e6) for f, w, h in zip(df.format, df.width, df.height)]
    plan, sizes = {}, {}
    for c, g in df.groupby("cell"):
        r, a = g[g.label == "real"].index.tolist(), g[g.label == "ai"].index.tolist()
        k = min(len(r), len(a)); plan[c] = (r[:k], a[:k]); sizes[c] = k
    total = sum(sizes.values())
    while total > N and sizes:
        c = max(sizes, key=sizes.get); sizes[c] -= 1; total -= 1
    keep = []
    for c, (r, a) in plan.items():
        keep += r[:sizes[c]] + a[:sizes[c]]
    for _, row in df.drop(index=keep).iterrows():
        fp = os.path.join(ROOT, row["path"])
        if str(row["image_id"]).startswith(prefix) and os.path.exists(fp):
            os.remove(fp)
    return df.loc[sorted(keep)].drop(columns="cell").to_dict("records")


def report(source, rows, N=None):
    if not rows:
        print(f"[{source}] NOTHING saved."); return
    d = pd.DataFrame(rows)
    d["cell"] = [cell_of(f, w * h / 1e6) for f, w, h in zip(d.format, d.width, d.height)]
    print(f"\n[{source}] saved {len(d)} images -> {meta_path(source)}")
    print(d.groupby(["label", "format"]).size().unstack(fill_value=0).to_string())
    print("(format x size) cells, real vs AI (matched sampling wants equal numbers):")
    print(d.groupby(["cell", "label"]).size().unstack(fill_value=0).to_string())
    print("median width x height / file bytes by class:")
    print(d.groupby("label")[["width", "height", "file_bytes"]].median().astype(int).to_string())
    if N and (d.label == "real").sum() < N:
        print(f"!! only {(d.label == 'real').sum()} real / {(d.label == 'ai').sum()} AI of the {N} per class target")


def collect_stream(stream, source, N, match, folder, prefix, role, licence):
    """Pull from an OpenFake-style stream (examples with image, image_sha256, label, model, type)."""
    rows = load_rows(source, prefix)
    out_dir = os.path.join(ROOT, "public_data", folder); os.makedirs(out_dir, exist_ok=True)
    seen = {r["sha256"] for r in rows}
    k = max([int(str(r["image_id"]).split("_")[-1]) for r in rows] + [0])
    cellcnt = defaultdict(lambda: [0, 0]); gen = Counter(); typ = Counter(); tot = Counter()
    for r in rows:
        cellcnt[cell_of(r["format"], r["width"] * r["height"] / 1e6)][0 if r["label"] == "real" else 1] += 1
        tot[r["label"]] += 1
        (gen if r["label"] == "ai" else typ)[r["generator"] if r["label"] == "ai" else r["source_detail"]] += 1
    gen_cap, type_cap = max(2, math.ceil(0.10 * N)), max(3, math.ceil(0.5 * N))
    done = lambda: (sum(min(v) for v in cellcnt.values()) >= N) if match else (tot["real"] >= N and tot["ai"] >= N)
    skipped, looked, t0, stop = Counter(), 0, time.time(), "stream ended"
    print(f"\n=== {source}: {N} real + {N} AI for {MEMBER} (slot {IDX} of 6){' [PILOT]' if PILOT else ''}"
          f" | matched by format x size: {match} | already have {len(rows)} ===")
    if done():
        print("  already complete."); return rows
    for ex in stream:
        looked += 1
        if looked > MAX_SCAN or time.time() - t0 > MAX_MINUTES * 60:
            stop = "scan/time limit"; break
        if looked % 200 == 0:
            gc.collect()
            if ram_pct() > RAM_STOP:
                stop = f"RAM above {RAM_STOP}% (saved, rerun to resume)"; break
        lab = {"real": "real", "fake": "ai"}.get(str(ex.get("label")).lower())
        if lab is None:
            skipped["bad_label"] += 1; continue
        v = ex.get("image"); raw = None
        h = ex.get("image_sha256")
        try:
            owner = int(str(h)[:8], 16) % 6
        except Exception:
            raw = v.get("bytes") if isinstance(v, dict) else v
            owner = int(hashlib.sha256(raw).hexdigest()[:8], 16) % 6
        if owner != IDX:
            skipped["not_mine"] += 1; continue
        if raw is None:
            raw = v.get("bytes") if isinstance(v, dict) else v
        try:
            im = PILImage.open(io.BytesIO(raw)); fmt, (w, hh), mode = im.format, im.size, im.mode
        except Exception:
            skipped["unreadable"] += 1; continue
        if min(w, hh) < MIN_SIDE:
            skipped["too_small"] += 1; continue
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen:
            skipped["duplicate"] += 1; continue
        g = str(ex.get("model")) if lab == "ai" else "real"
        t = str(ex.get("type")) if lab == "real" else "ai"
        if lab == "ai" and gen[g] >= gen_cap:
            skipped["generator_cap"] += 1; continue
        if lab == "real" and typ[t] >= type_cap:
            skipped["real_type_cap"] += 1; continue
        c = cell_of(fmt, w * hh / 1e6); n_r, n_a = cellcnt[c]
        mine_n, other_n = (n_r, n_a) if lab == "real" else (n_a, n_r)
        if match and (mine_n >= other_n + K_SLACK or mine_n >= math.ceil(0.5 * N) or tot[lab] >= int(1.3 * N)):
            skipped["cell_balance"] += 1; continue
        if not match and tot[lab] >= N:
            skipped["class_full"] += 1; continue
        k += 1; iid = f"{prefix}{MEMBER}_{k:06d}"; rel = f"public_data/{folder}/{iid}.{EXT.get(fmt, 'img')}"
        with open(os.path.join(ROOT, rel), "wb") as f:
            f.write(raw)
        rows.append(dict(image_id=iid, path=rel, label=lab, source=source, source_detail=t, generator=g,
                         original_id=str(ex.get("row_id", h)), version="orig", role=role, format=fmt, width=w,
                         height=hh, mode=mode, file_bytes=len(raw), sha256=sha, license=licence, creator=MEMBER,
                         pulled_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        seen.add(sha); cellcnt[c][0 if lab == "real" else 1] += 1; tot[lab] += 1
        (gen if lab == "ai" else typ)[g if lab == "ai" else t] += 1
        if len(rows) % 25 == 0:
            save_rows(source, rows)
        if done():
            stop = "target reached"; break
    print(f"  stopped: {stop} | looked at {looked} rows in {int(time.time() - t0)} s | skipped: {dict(skipped)}")
    if match:
        rows = trim_matched(rows, N, prefix)
    save_rows(source, rows); report(source, rows, N)
    return rows


def open_hf(config, split):
    from datasets import load_dataset, get_dataset_split_names, Image as HFImage
    if split is None:
        split = get_dataset_split_names("ComplexDataLab/OpenFake", config)[0]
    try:
        ds = load_dataset("ComplexDataLab/OpenFake", name=config, split=split, streaming=True, batch_size=8)
    except TypeError:
        ds = load_dataset("ComplexDataLab/OpenFake", name=config, split=split, streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))            # raw bytes only, no decoding = low RAM
    seed = SEED + int(time.time()) % 100000                        # new seed every run: reruns read NEW rows, not the same first 1800
    print(f"  OpenFake stream seed this run: {seed}")
    return ds.shuffle(seed=seed, buffer_size=10)


def pull_community(N):
    """Matched Community Forensics pull from metadata first. Returns the number of matched pairs saved."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    repo, prefix, source = "OwensLab/CommunityForensics-Small", "CFM_", "community"
    old_rows = load_rows(source, prefix); have = 0
    if old_rows:
        d0 = pd.DataFrame(old_rows); have = int(min((d0.label == "real").sum(), (d0.label == "ai").sum()))
        if have >= N:
            print(f"\n=== community: complete, {have} matched pairs already saved ==="); report(source, old_rows, N); return have
        print(f"\n=== community: have {have} of {N} matched pairs, topping up {N - have} more ===")
    print(f"\n=== community: {N} real + {N} AI for {MEMBER}{' [PILOT]' if PILOT else ''} (metadata check first) ===")
    fs = HfFileSystem()
    files = sorted(x.rfilename for x in HfApi().dataset_info(repo, files_metadata=True).siblings
                   if x.rfilename.endswith(".parquet"))
    mine = files[IDX::6]                                        # my own files: members can never overlap
    print(f"  {len(files)} files, {len(mine)} are mine")
    cand, funnel, starts_of = [], Counter(), {}
    for name in mine:
        with fs.open(f"datasets/{repo}/{name}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
            starts_of[name] = starts
            mb = np.array([pf.metadata.row_group(i).total_byte_size for i in range(pf.num_row_groups)]) / 1e6
            cols = [c for c in pf.schema_arrow.names if c != "image_data"]
            m = pf.read(columns=cols).to_pandas()
        m["row"] = np.arange(len(m)); m["file"] = name
        m["rg"] = np.searchsorted(starts[1:], m["row"].values, side="right"); m["rg_mb"] = mb[m["rg"].values]
        wh = m["resolution"].map(lambda r: (int(r[0]), int(r[1])))
        m["w"] = wh.map(lambda t: t[0]); m["h"] = wh.map(lambda t: t[1]); m["fmt"] = m["format"].astype(str).str.upper()
        for l, n in m.label.value_counts().items():
            funnel[("1 rows in my files", int(l))] += int(n)
        if MAX_BLOCK_MB:
            m = m[m.rg_mb <= MAX_BLOCK_MB]
        for l, n in m.label.value_counts().items():
            funnel[("2 block small enough", int(l))] += int(n)
        nsfw = m["nsfw_flag"].astype(str).str.lower().isin(["true", "1"]) if "nsfw_flag" in m else False
        m = m[
            ~nsfw
            & (np.minimum(m.w, m.h) >= MIN_SIDE)
            & (m.fmt == "PNG")
        ]
        for l, n in m.label.value_counts().items():
            funnel[("3 usable (not NSFW, >=224px, PNG only)", int(l))] += int(n)
        if HEAD_ROWS:
            m = m[(m["row"] - starts[m["rg"].values]) < HEAD_ROWS]
            for l, n in m.label.value_counts().items():
                funnel[("4 within first rows of a block", int(l))] += int(n)
        if len(m):
            m["cell"] = [cell_of(f, a * b / 1e6) for f, a, b in zip(m.fmt, m.w, m.h)]
            cand.append(m)
    print("  FUNNEL (0 = real, 1 = AI):\n" + pd.Series(funnel).unstack().fillna(0).astype(int).sort_index().to_string())
    C = pd.concat(cand, ignore_index=True) if cand else pd.DataFrame()
    pairs = pd.DataFrame()
    if len(C) and C.label.nunique() == 2:
        pairs = C.groupby(["cell", "label"]).size().unstack(fill_value=0)
        pairs["pairs"] = pairs[[0, 1]].min(axis=1)
        print("  cells with BOTH classes (a matched sample can only come from these):")
        print(pairs[pairs.pairs > 0].to_string() if (pairs.pairs > 0).any() else "  none")
    if old_rows and len(C):
        C = C[~C["image_name"].astype(str).isin({str(r["original_id"]) for r in old_rows})]
        pairs = C.groupby(["cell", "label"]).size().unstack(fill_value=0) if len(C) and C.label.nunique() == 2 else pd.DataFrame()
        if len(pairs):
            pairs["pairs"] = pairs[[0, 1]].min(axis=1)
    if len(C):
        # Same 65% RAM safety rule as the block-reading loop below, applied BEFORE the quota is split, so no quota
        # is wasted on cells whose blocks would be skipped anyway. Matching (equal real/AI per cell) is unchanged.
        avail0 = psutil.virtual_memory().available / 1e6 if psutil else 1e9
        n_before = len(C)
        C = C[2.6 * C["rg_mb"] <= 0.65 * avail0]
        print(f"  RAM-feasible rows: {len(C)} of {n_before} (blocks that fit within 65% of {avail0:.0f} MB free)")
        pairs = C.groupby(["cell", "label"]).size().unstack(fill_value=0) if len(C) and C.label.nunique() == 2 else pd.DataFrame()
        if len(pairs):
            pairs["pairs"] = pairs[[0, 1]].min(axis=1)
            print("  cells I can actually fill (both classes, RAM-feasible):\n" + pairs[pairs.pairs > 0].to_string())
    if not len(pairs) or pairs.pairs.sum() == 0:
        print("  -> NO PNG format x size cell holds both real and AI inside my files and the RAM-eligible blocks."
              "\n     A matched Community Forensics sample is not possible here, so it is skipped.")
        return 0
    rs = np.random.RandomState(SEED); quota = {c: 0 for c in pairs.index}; left = N - have
    while left > 0 and any(quota[c] < pairs.pairs[c] for c in quota):
        for c in sorted(quota, key=lambda c: -pairs.pairs[c]):
            if left > 0 and quota[c] < pairs.pairs[c]:
                quota[c] += 1; left -= 1
    gen_cap, picks = max(2, math.ceil(0.10 * N)), []
    for c, q in quota.items():
        if q == 0:
            continue
        a = C[(C.cell == c) & (C.label == 1)].sample(frac=1, random_state=SEED)
        a = a.groupby(a["architecture"].astype(str) + a["model_name"].astype(str), group_keys=False).head(gen_cap).head(q)
        r = C[(C.cell == c) & (C.label == 0)].sample(len(a), random_state=SEED)
        picks += [a, r]
    P = pd.concat(picks, ignore_index=True)
    del cand, C, picks; free_memory()
    out_dir = os.path.join(ROOT, "public_data", "community_forensics"); os.makedirs(out_dir, exist_ok=True)
    rows = []; seen = {r["sha256"] for r in old_rows}
    k = max([int(str(r["image_id"]).split("_")[-1]) for r in old_rows] + [0])
    for (name, rg), g in P.groupby(["file", "rg"]):
        need_mb = 2.6 * float(g.rg_mb.iloc[0]); avail = psutil.virtual_memory().available / 1e6 if psutil else 1e9
        if need_mb > 0.65 * avail:
            print(f"  skip block {name} #{rg}: estimated ~{need_mb:.0f} MB needed; only {avail:.0f} MB free (65% safety limit)"); continue
        print(f"  reading block {name} #{rg} ({float(g.rg_mb.iloc[0]):.0f} MB, {len(g)} rows wanted, RAM {ram_pct():.0f}%)")
        want = {int(r.row - int(starts_of[name][int(rg)])): r for r in g.itertuples()}
        with fs.open(f"datasets/{repo}/{name}", "rb") as fh:
            pf = pq.ParquetFile(fh); pos, last = 0, max(want)
            blobs = {}
            for batch in pf.iter_batches(batch_size=8, row_groups=[int(rg)], columns=["image_data"]):
                colb = batch.column(0)
                for j in range(len(colb)):
                    if pos + j in want:
                        v = colb[j].as_py(); blobs[pos + j] = v.get("bytes") if isinstance(v, dict) else v
                pos += len(colb); del colb
                if pos > last:
                    break
            for li, r in want.items():
                raw = blobs.get(li)
                if raw is None:
                    continue
                sha = hashlib.sha256(raw).hexdigest()
                try:
                    im = PILImage.open(io.BytesIO(raw)); fmt, (w, hh), mode = im.format, im.size, im.mode
                except Exception:
                    continue
                if sha in seen or fmt != r.fmt:
                    continue
                seen.add(sha); k += 1; lab = "real" if r.label == 0 else "ai"
                iid = f"{prefix}{MEMBER}_{k:06d}"; rel = f"public_data/community_forensics/{iid}.{EXT.get(fmt, 'img')}"
                open(os.path.join(ROOT, rel), "wb").write(raw)
                rows.append(dict(image_id=iid, path=rel, label=lab, source=source,
                                 source_detail="real" if lab == "real" else str(r.architecture),
                                 generator="real" if lab == "real" else str(r.model_name), original_id=str(r.image_name),
                                 version="orig", role="train_val", format=fmt, width=w, height=hh, mode=mode,
                                 file_bytes=len(raw), sha256=sha, license="CC BY-NC-SA 4.0", creator=MEMBER,
                                 pulled_at=time.strftime("%Y-%m-%d %H:%M:%S")))
            del blobs; free_memory()
    rows = trim_matched(old_rows + rows, N, prefix); save_rows(source, rows); report(source, rows, N)
    d = pd.DataFrame(rows)
    return int(min((d.label == "real").sum(), (d.label == "ai").sum())) if len(d) else 0


def main():
    plan = {k: (min(v, 10) if PILOT else v) for k, v in PLAN[MEMBER].items()}
    print("MEMBER", MEMBER, "| plan (per class):", plan, "| slot", IDX, "of 6 | seed", SEED, "| pilot:", PILOT)
    if "community" in plan:
        if ONLY in (None, "community"):
            got = pull_community(plan["community"])
            free_memory(); print(f"  RAM in use after community: {ram_pct():.0f}%")
        else:                                              # ONLY = None: use what the community session saved
            d = pd.DataFrame(load_rows("community", "CFM_"))
            got = int(min((d.label == "real").sum(), (d.label == "ai").sum())) if len(d) else 0
            print(f"community: {got} matched pairs already saved by an earlier community session")
        short = plan["community"] - got
        if short > 0:
            print(f"\n>> Community Forensics gave {got} matched pairs, {short} short.")
            if REDISTRIBUTE:
                plan["openfake"] = plan.get("openfake", 0) + short
                print(f">> The {short} missing pairs come from OpenFake instead (OpenFake total now {plan['openfake']}).")
        if ONLY == "community":
            print("\n>> NEXT: Runtime > Disconnect and delete runtime, reconnect, set ONLY = 'openfake' and run again.")
    if plan.get("openfake") and ONLY in (None, "openfake"):
        collect_stream(open_hf("core", "train"), "openfake", plan["openfake"], True, "openfake", "OFM_",
                       "train_val", "CC BY-NC 4.0")
    if plan.get("reddit") and ONLY in (None, "openfake"):
        collect_stream(open_hf("reddit", None), "reddit", plan["reddit"], MATCH_REDDIT, "openfake_reddit", "RDM_",
                       "wild_test", "CC BY-NC 4.0")
    print("\\n=== FINAL TARGET CHECK ===")
    for source, target in plan.items():
        if source not in ("community", "openfake"):
            continue
        prefix = "CFM_" if source == "community" else "OFM_"
        final_rows = load_rows(source, prefix)
        final_df = pd.DataFrame(final_rows)
        real_n = int((final_df.label == "real").sum()) if len(final_df) else 0
        ai_n = int((final_df.label == "ai").sum()) if len(final_df) else 0
        print(f"{source}: real={real_n}/{target}, AI={ai_n}/{target}, total={real_n + ai_n}/{2 * target}")
        if source == "community" and len(final_df) and not (final_df["format"].astype(str).str.upper() == "PNG").all():
            print("  WARNING: Community Forensics CSV contains a non-PNG record.")
        if real_n < target or ai_n < target:
            print("  SHORTFALL: target not reached; inspect the funnel/skip report before rerunning.")
    print("\\nDONE. Upload the CSVs in", ROOT + "/metadata/ (public_*_" + MEMBER + ".csv) to GitHub. Never upload images.")



def pairs_done(source):
    """Matched pairs saved so far for a source, read from its CSV."""
    try:
        d = pd.read_csv(meta_path(source))
        return int(min((d.label == "real").sum(), (d.label == "ai").sum())) if len(d) else 0
    except Exception:
        return 0


def run_stage(source, target, script, max_attempts, patience):
    """Run one source in a FRESH Python process per attempt, so RAM is fully freed between attempts."""
    import subprocess, sys
    best, stale = pairs_done(source), 0
    for attempt in range(1, max_attempts + 1):
        if best >= target:
            break
        print(f"\n########## {source}: attempt {attempt} | have {best}/{target} pairs ##########", flush=True)
        env = dict(os.environ, PULL_CHILD="1", PULL_ONLY=source)
        proc = subprocess.Popen([sys.executable, "-u", script], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        now = pairs_done(source)
        stale = 0 if now > best else stale + 1
        best = max(best, now)
        if stale >= patience:
            print(f"\n>> {source}: no new pairs in {patience} attempts in a row, stopping this source at {best}/{target}.")
            break
    return best


def driver():
    script = globals().get("__file__") or "/content/pull_my_share_NC_modified.py"
    plan = {k: (min(v, 10) if PILOT else v) for k, v in PLAN[MEMBER].items()}
    result = {}
    if "community" in plan:
        result["community"] = run_stage("community", plan["community"], script, max_attempts=4, patience=2)
    if "openfake" in plan:
        result["openfake"] = run_stage("openfake", plan["openfake"], script, max_attempts=80, patience=4)
    print("\n================ ALL DONE ================")
    for source, got in result.items():
        print(f"{source}: {got}/{plan[source]} matched pairs ({2 * got} images)")
    print("Upload ONLY the CSVs in", ROOT + "/metadata/ (public_*_" + MEMBER + ".csv) to GitHub. Never upload images.")


if globals().get("RUN", True):
    if os.environ.get("PULL_CHILD"):
        main()          # one attempt, runs in its own fresh process
    else:
        driver()        # you run this: it launches the attempts itself
