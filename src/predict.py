"""
The ONE prediction function. The command line and the web app both call it.

    python src/predict.py path/to/image.jpg --ckpt models/final.pt

predict(image) returns a dict:
    label    : "real", "AI-generated" or "uncertain"
    p_fake   : probability the image is AI-generated (0 to 1)
    heatmap  : PIL image (Grad-CAM overlay on the centre crop) or None
    notes    : list of plain-language warnings
"""

import argparse
import json
import os

import torch

try:
    from data import denormalise, load_image
    from explain import GradCAM, overlay
    from model import load_checkpoint
except ImportError:                     # pragma: no cover
    from .data import denormalise, load_image
    from .explain import GradCAM, overlay
    from .model import load_checkpoint

_CACHE = {}
UNCERTAIN_BAND = 0.10


def _load(ckpt, device):
    key = (os.path.abspath(ckpt), device)
    if key not in _CACHE:
        model, name = load_checkpoint(ckpt, device)
        side = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "final.json")
        thr = json.load(open(side))["threshold"] if os.path.exists(side) else 0.5
        _CACHE[key] = (model, thr)
    return _CACHE[key]


def predict(image, ckpt="models/final.pt", threshold=None, heatmap=True, device=None):
    """image: file path, bytes or PIL image."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, saved_thr = _load(ckpt, device)
    thr = saved_thr if threshold is None else threshold
    notes = []

    x, info = load_image(image, mode="five")
    if info["small"]:
        notes.append(f"This image is small ({info['width']} x {info['height']}); the model works on "
                     f"224 x 224 pixel crops, so the result is less reliable.")
    with torch.inference_mode():
        p = float(torch.sigmoid(model(x.to(device))).mean())

    if isinstance(image, (str, os.PathLike)) and os.path.exists(image):
        size = os.path.getsize(image)
        bpp = size / max(1, info["width"] * info["height"])
        if str(image).lower().endswith((".jpg", ".jpeg")) and bpp < 0.25:
            notes.append("This image looks heavily compressed (for example sent through WhatsApp). "
                         "Compression can hide the traces the model looks for.")
    if abs(p - thr) < UNCERTAIN_BAND:
        notes.append("The score is close to the decision line, so treat this as uncertain.")
    label = "uncertain" if abs(p - thr) < UNCERTAIN_BAND else ("AI-generated" if p >= thr else "real")

    hm = None
    if heatmap:
        try:
            cam_fn = GradCAM(model, model.gradcam_layer())
            xc = x[:1].to(device)
            cam, _ = cam_fn(xc)
            cam_fn.close()
            hm = overlay(denormalise(x[0]), cam)
        except Exception as e:                       # the verdict must never fail because of the picture
            notes.append(f"Heat map unavailable ({type(e).__name__}).")
    return dict(label=label, p_fake=p, heatmap=hm, notes=notes, threshold=thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default="models/final.pt")
    ap.add_argument("--save-heatmap", default=None)
    a = ap.parse_args()
    r = predict(a.image, a.ckpt)
    print(f"{r['label']}  (probability AI-generated: {r['p_fake']:.1%})")
    for n in r["notes"]:
        print(" -", n)
    if a.save_heatmap and r["heatmap"] is not None:
        r["heatmap"].save(a.save_heatmap)
        print("heat map saved to", a.save_heatmap)


if __name__ == "__main__":
    main()
