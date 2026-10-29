"""
Grad-CAM heat maps and the FAITHFULNESS test.

Grad-CAM shows WHERE the model looked, not WHY it decided. The faithfulness test
checks whether the highlighted region really matters to the model:
  hide the most-highlighted region and measure how much the confidence drops;
  do the same for a region of identical shape and area at a random place.
A faithful heat map causes the bigger drop.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from data import load_image
except ImportError:                     # pragma: no cover
    from .data import load_image


class GradCAM:
    """Grad-CAM on one convolutional layer. The model outputs ONE logit (AI)."""

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self._h = [layer.register_forward_hook(self._fwd),
                   layer.register_full_backward_hook(self._bwd)]

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x):
        """x: (1,3,H,W). Returns (cam in 0..1 of shape HxW, p_fake)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logit = self.model(x).view(-1)[0]
            logit.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        rng = cam.max() - cam.min()
        cam = (cam - cam.min()) / rng if rng > 1e-8 else np.zeros_like(cam)
        return cam, float(torch.sigmoid(logit.detach()))

    def close(self):
        for h in self._h:
            h.remove()


def overlay(rgb_uint8, cam, alpha=0.45):
    """Blend a heat map on top of an RGB image. Returns a PIL image."""
    import matplotlib
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    out = (rgb_uint8 * (1 - alpha) + heat * alpha).astype(np.uint8)
    return Image.fromarray(out)


@torch.inference_mode()
def _p(model, x, device):
    return float(torch.sigmoid(model(x.to(device)).view(-1)[0]))


def faithfulness_study(model, df, root, device="cpu", frac=0.2, n_random=5, seed=0):
    """For every image in df (center crop): confidence drop when the top-`frac`
    Grad-CAM pixels are hidden, versus the same mask moved to random places.

    Returns (per-image DataFrame, summary dict).
    """
    rng = np.random.default_rng(seed)
    layer = model.gradcam_layer()
    cam_fn = GradCAM(model, layer)
    rows = []
    model.eval().to(device)
    for r in df.itertuples():
        try:
            x, _ = load_image(f"{root}/{r.path}", mode="center")
        except Exception:
            continue
        x = x[:1].to(device)
        cam, p0 = cam_fn(x)
        pred_ai = p0 >= 0.5
        thresh = np.quantile(cam, 1 - frac)
        mask = cam >= thresh
        def conf(m):
            xm = x.clone()
            xm[:, :, torch.from_numpy(m).to(device)] = 0.0       # 0 = the dataset mean colour
            p = _p(model, xm, device)
            return p if pred_ai else 1 - p
        c0 = p0 if pred_ai else 1 - p0
        drop_cam = c0 - conf(mask)
        drops_rand = []
        for _ in range(n_random):
            dy, dx = rng.integers(0, mask.shape[0]), rng.integers(0, mask.shape[1])
            drops_rand.append(c0 - conf(np.roll(mask, (dy, dx), axis=(0, 1))))
        rows.append(dict(image_id=r.image_id, label=r.label, predicted_ai=bool(pred_ai),
                         correct=bool(pred_ai == (r.label == "ai")), conf_before=c0,
                         drop_cam=drop_cam, drop_random=float(np.mean(drops_rand))))
    cam_fn.close()
    res = pd.DataFrame(rows)
    if res.empty:
        return res, {}
    diff = res.drop_cam - res.drop_random
    boot = [rng.choice(diff.values, size=len(diff)).mean() for _ in range(1000)]
    summary = dict(n=len(res), mean_drop_cam=float(res.drop_cam.mean()),
                   mean_drop_random=float(res.drop_random.mean()),
                   mean_difference=float(diff.mean()),
                   difference_95ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                   share_cam_better=float((diff > 0).mean()),
                   verdict=("heat maps are faithful" if np.quantile(boot, .025) > 0 else
                            "heat maps are NOT clearly better than random"))
    return res, summary
