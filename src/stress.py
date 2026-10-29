"""
Damage test images the way real life does, then re-score them.

All functions take and return PIL RGB images and never look at the label.
Also contains the FORMAT-SWAP test: re-save images in the other format. If the
model was reading the file format, accuracy collapses here.
"""

import io

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

try:
    from data import load_image, open_rgb
    from metrics import compute_metrics
except ImportError:                     # pragma: no cover
    from .data import load_image, open_rgb
    from .metrics import compute_metrics


def jpeg(im, q):
    b = io.BytesIO(); im.save(b, format="JPEG", quality=int(q)); b.seek(0)
    return Image.open(b).convert("RGB")


def png(im):
    b = io.BytesIO(); im.save(b, format="PNG"); b.seek(0)
    return Image.open(b).convert("RGB")


def downscale(im, s):
    w, h = im.size
    small = im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def blur(im, r):
    return im.filter(ImageFilter.GaussianBlur(radius=r))


def noise(im, sigma, seed=0):
    a = np.asarray(im).astype(np.float32)
    a = a + np.random.default_rng(seed).normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def brightness(im, f):
    return ImageEnhance.Brightness(im).enhance(f)


CONDITIONS = {
    "original": lambda im: im,
    "jpeg_q90": lambda im: jpeg(im, 90), "jpeg_q70": lambda im: jpeg(im, 70),
    "jpeg_q50": lambda im: jpeg(im, 50), "jpeg_q30": lambda im: jpeg(im, 30),
    "downscale_75": lambda im: downscale(im, .75), "downscale_50": lambda im: downscale(im, .5),
    "downscale_25": lambda im: downscale(im, .25),
    "blur_1": lambda im: blur(im, 1), "blur_2": lambda im: blur(im, 2),
    "noise_5": lambda im: noise(im, 5), "noise_15": lambda im: noise(im, 15),
    "dark_0.6": lambda im: brightness(im, .6), "bright_1.4": lambda im: brightness(im, 1.4),
}
# the format-swap test: JPEG images saved as PNG (lossless copy) and PNG images saved as JPEG q95
FORMAT_SWAP = {"as_png": png, "as_jpeg95": lambda im: jpeg(im, 95)}


@torch.inference_mode()
def p_fake_pil(model, im, device="cpu"):
    x, _ = load_image(im, mode="five")
    return float(torch.sigmoid(model(x.to(device))).mean())


def stress_table(model, df, root, conditions=None, threshold=0.5, device="cpu"):
    """Rows = conditions, columns = metrics. df needs path, label."""
    conditions = conditions or CONDITIONS
    model.eval().to(device)
    y = df.label.map({"real": 0, "ai": 1}).values
    rows = []
    base = [open_rgb(f"{root}/{p}") for p in df.path]
    for name, fn in conditions.items():
        p = np.array([p_fake_pil(model, fn(im), device) for im in base])
        m = compute_metrics(y, p, threshold)
        rows.append(dict(condition=name, accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                         recall_real=m["recall_real"], recall_ai=m["recall_ai"], roc_auc=m["roc_auc"]))
    t = pd.DataFrame(rows)
    t["f1_drop_vs_original"] = t.loc[t.condition == "original", "macro_f1"].values[0] - t.macro_f1 \
        if (t.condition == "original").any() else np.nan
    return t
