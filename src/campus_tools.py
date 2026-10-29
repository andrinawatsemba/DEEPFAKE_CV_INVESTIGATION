"""
Tools for the UCU Mukono campus set (real phone photos + matched AI images).

Folder layout under the project root (Drive):
    campus/originals/          CAMP_<CODE>_0001.jpg ...   (as taken / as downloaded)
    campus/whatsapp_photo/     same IDs, sent through WhatsApp AS A PHOTO
    campus/whatsapp_document/  same IDs, sent through WhatsApp AS A DOCUMENT

Each member also keeps one small labels file:  metadata/campus_labels_<CODE>.csv
    image_id, label, device_or_tool, prompt_or_scene, contains_person, consent_obtained, notes
(label is real or ai; the label is NEVER put in the file name)

    labels_template(root, code)   -> writes that file with every ID it finds, ready to fill in
    build_log(root)               -> campus_log.csv (3 rows per image) + list of problems
    lock_manifest(root)           -> campus_manifest.csv with SHA-256 of every file (the 'lock')
    test_table(root)              -> table the evaluate/stress code can score
"""

import glob
import hashlib
import os

import pandas as pd
from PIL import Image

VERSIONS = {"original": "campus/originals", "whatsapp_photo": "campus/whatsapp_photo",
            "whatsapp_document": "campus/whatsapp_document"}
LABEL_COLS = ["image_id", "label", "device_or_tool", "prompt_or_scene",
              "contains_person", "consent_obtained", "notes"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root, version):
    d = os.path.join(root, VERSIONS[version])
    out = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                out[os.path.splitext(f)[0]] = os.path.join(VERSIONS[version], f)
    return out


def labels_template(root, code):
    """Write metadata/campus_labels_<CODE>.csv with one empty row per image you uploaded."""
    ids = sorted(i for i in _files(root, "original") if i.startswith(f"CAMP_{code}_"))
    path = os.path.join(root, "metadata", f"campus_labels_{code}.csv")
    if os.path.exists(path):
        old = pd.read_csv(path)
        ids = [i for i in ids if i not in set(old.image_id)]
        new = pd.DataFrame({c: [""] * len(ids) for c in LABEL_COLS})
        new["image_id"] = ids
        pd.concat([old, new]).to_csv(path, index=False)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t = pd.DataFrame({c: [""] * len(ids) for c in LABEL_COLS})
        t["image_id"] = ids
        t.to_csv(path, index=False)
    return path, len(ids)


def build_log(root):
    """Return (log DataFrame, list of problems). Fix every problem before locking."""
    files = {v: _files(root, v) for v in VERSIONS}
    labels = [pd.read_csv(f, dtype=str).fillna("")
              for f in sorted(glob.glob(os.path.join(root, "metadata", "campus_labels_*.csv")))]
    lab = pd.concat(labels, ignore_index=True) if labels else pd.DataFrame(columns=LABEL_COLS)
    lab = lab.drop_duplicates("image_id").set_index("image_id")
    problems, rows = [], []
    ids = sorted(set().union(*[set(d) for d in files.values()]))
    for i in ids:
        for v in VERSIONS:
            if i not in files[v]:
                problems.append(f"{i}: missing version '{v}'")
        if i not in lab.index:
            problems.append(f"{i}: no row in any campus_labels_<CODE>.csv")
            continue
        L = lab.loc[i]
        if L.label not in ("real", "ai"):
            problems.append(f"{i}: label must be 'real' or 'ai' (got '{L.label}')")
        if L.contains_person.lower() in ("yes", "y", "true") and L.consent_obtained.lower() not in ("yes", "y", "true"):
            problems.append(f"{i}: shows a person but consent_obtained is not 'yes'")
        if L.label == "ai" and not L.prompt_or_scene.strip():
            problems.append(f"{i}: AI image without its prompt")
        if not L.device_or_tool.strip():
            problems.append(f"{i}: device_or_tool is empty")
        for v in VERSIONS:
            p = files[v].get(i)
            if not p:
                continue
            try:
                with Image.open(os.path.join(root, p)) as im:
                    fmt, (w, h) = im.format, im.size
            except Exception as e:
                problems.append(f"{i} [{v}]: cannot open ({type(e).__name__}) - iPhone HEIC? use 'Most Compatible'")
                continue
            rows.append(dict(image_id=i, version=v, path=p, label=L.label,
                             creator=i.split("_")[1] if i.count("_") >= 2 else "",
                             device_or_tool=L.device_or_tool, prompt_or_scene=L.prompt_or_scene,
                             contains_person=L.contains_person, consent_obtained=L.consent_obtained,
                             format=fmt, width=w, height=h,
                             file_bytes=os.path.getsize(os.path.join(root, p)), notes=L.notes))
    log = pd.DataFrame(rows)
    if len(log):
        log.to_csv(os.path.join(root, "metadata", "campus_log.csv"), index=False)
    return log, problems


def summary(log):
    o = log[log.version == "original"]
    print("Images per creator and label (originals):")
    print(pd.crosstab(o.creator, o.label))
    print("\nFormat of originals by class (a gap here is the format shortcut, report it):")
    print(pd.crosstab(o.format, o.label))
    print("\nFormat after WhatsApp:")
    print(pd.crosstab([log.version, log.format], log.label))


def lock_manifest(root):
    """Record a SHA-256 for every campus file. After this the set is READ-ONLY."""
    rows = []
    for v in VERSIONS:
        for i, p in _files(root, v).items():
            rows.append(dict(image_id=i, version=v, path=p, sha256=_sha(os.path.join(root, p))))
    m = pd.DataFrame(rows)
    m.to_csv(os.path.join(root, "metadata", "campus_manifest.csv"), index=False)
    return m


def test_table(root, versions=("original", "whatsapp_photo", "whatsapp_document")):
    """Table for scoring: role = campus_<version>."""
    log = pd.read_csv(os.path.join(root, "metadata", "campus_log.csv"))
    t = log[log.version.isin(versions)].copy()
    t["role"] = "campus_" + t.version
    t["generator"] = t.device_or_tool
    return t[["image_id", "path", "label", "role", "generator", "format", "width", "height"]]
