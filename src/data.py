"""
ONE place that turns an image file into a model input.

Training, evaluation, predict.py and the web app ALL call load_image() from here.
Nobody writes their own image loading code. If this file changes, it changes
for everything at once (see docs/preprocessing.md for the reasons).

The rules
  1. Open with PIL only (never mix PIL and OpenCV: BGR vs RGB bug).
  2. Apply the phone's rotation flag, then drop all metadata.
  3. Convert to 3-channel, 8-bit RGB.
  4. Take 224 x 224 crops at NATIVE resolution. No whole-image resize.
       - training: one random crop (position snapped to the 8-pixel JPEG grid)
       - evaluation and prediction: FIVE crops (centre + 4 corners); the model's
         five probabilities are averaged.
       - images smaller than 224 are padded (reflect) and flagged.
  5. Scale to 0..1, subtract the ImageNet mean, divide by the ImageNet std.
"""

import io
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

CROP = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
GRID = 8                      # JPEG block size: random crops snap to it


# ---------------------------------------------------------------------------
# step 1 to 3: open, rotate, strip, RGB
# ---------------------------------------------------------------------------
def open_rgb(src):
    """Open a path, bytes, file-like object or PIL image as a clean RGB image."""
    if isinstance(src, Image.Image):
        im = src
    elif isinstance(src, (bytes, bytearray)):
        im = Image.open(io.BytesIO(src))
    else:
        im = Image.open(src)
    im = ImageOps.exif_transpose(im)          # fix phone rotation
    im = im.convert("RGB")                    # drops alpha / palette / grayscale
    im.info.clear()                           # drop EXIF and colour profile
    return im


# ---------------------------------------------------------------------------
# step 4: native-resolution crops
# ---------------------------------------------------------------------------
def pad_to_crop(im, crop=CROP):
    """Reflect-pad images smaller than the crop. Returns (image, was_padded)."""
    w, h = im.size
    if w >= crop and h >= crop:
        return im, False
    a = np.asarray(im)
    pad_h, pad_w = max(0, crop - h), max(0, crop - w)
    mode = "reflect" if (h > pad_h and w > pad_w) else "edge"
    a = np.pad(a, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return Image.fromarray(a), True


def _box(x, y, crop):
    return (x, y, x + crop, y + crop)


def center_box(w, h, crop=CROP):
    return _box((w - crop) // 2, (h - crop) // 2, crop)


def five_boxes(w, h, crop=CROP):
    return [center_box(w, h, crop),
            _box(0, 0, crop), _box(w - crop, 0, crop),
            _box(0, h - crop, crop), _box(w - crop, h - crop, crop)]


def random_box(w, h, crop=CROP, rng=random):
    x = rng.randint(0, w - crop)
    y = rng.randint(0, h - crop)
    x, y = x - x % GRID, y - y % GRID         # keep the JPEG grid aligned
    return _box(x, y, crop)


# ---------------------------------------------------------------------------
# step 5: tensor + ImageNet normalisation
# ---------------------------------------------------------------------------
_MEAN_T = torch.tensor(MEAN).view(3, 1, 1)
_STD_T = torch.tensor(STD).view(3, 1, 1)


def to_tensor(im):
    a = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    return (a - _MEAN_T) / _STD_T


def denormalise(t):
    """Inverse of to_tensor, for drawing heat maps. Returns HxWx3 uint8."""
    x = (t.detach().cpu() * _STD_T + _MEAN_T).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# class-blind augmentation (only for Runs C and E, only in training).
# NONE of these functions is allowed to look at the label.
# ---------------------------------------------------------------------------
def jpeg_recompress(im, q_low=40, q_high=95, rng=random):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=rng.randint(q_low, q_high))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def augment_blind(im, rng=random):
    """Imitates WhatsApp and phone processing. Same recipe for real and AI."""
    if rng.random() < 0.5:                                 # re-compress
        im = jpeg_recompress(im, rng=rng)
    if rng.random() < 0.3:                                 # downscale then upscale
        w, h = im.size
        s = rng.uniform(0.5, 0.9)
        small = im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        im = small.resize((w, h), Image.BILINEAR)
    if rng.random() < 0.2:                                 # mild blur
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
    if rng.random() < 0.5:                                 # mirror
        im = ImageOps.mirror(im)
    return im


# ---------------------------------------------------------------------------
# the one function everyone calls
# ---------------------------------------------------------------------------
def load_image(src, mode="five", crop=CROP, augment=False, rng=random):
    """Return (tensor, info).

    mode="five"   : tensor of shape (5, 3, crop, crop)   (evaluation, prediction)
    mode="center" : tensor of shape (1, 3, crop, crop)
    mode="random" : tensor of shape (1, 3, crop, crop)   (training)
    info: dict with original size, whether padding was needed.
    """
    im = open_rgb(src)
    orig_w, orig_h = im.size
    if augment:
        # recompress the WHOLE image (like WhatsApp does), then crop
        im = augment_blind(im, rng=rng)
    im, padded = pad_to_crop(im, crop)
    w, h = im.size
    if mode == "five":
        boxes = five_boxes(w, h, crop)
    elif mode == "center":
        boxes = [center_box(w, h, crop)]
    elif mode == "random":
        boxes = [random_box(w, h, crop, rng)]
    else:
        raise ValueError(f"unknown crop mode: {mode}")
    x = torch.stack([to_tensor(im.crop(b)) for b in boxes])
    return x, dict(width=orig_w, height=orig_h, padded=padded,
                   small=min(orig_w, orig_h) < crop)


# ---------------------------------------------------------------------------
# PyTorch Dataset over a split table
# ---------------------------------------------------------------------------
LABELS = {"real": 0, "ai": 1}


class ImageTable(Dataset):
    """Rows of split.csv -> (crops, label, image_id).

    root    : folder that the 'path' column is relative to (the Drive project folder)
    train   : True -> one random crop per image;  False -> five crops
    augment : class-blind augmentation (training only)
    """

    def __init__(self, df, root, train=False, augment=False, crop=CROP, seed=0):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.train = train
        self.augment = augment and train
        self.crop = crop
        self.rng = random.Random(seed)
        self.failed = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        path = r["path"] if os.path.isabs(str(r["path"])) else os.path.join(self.root, str(r["path"]))
        n = 1 if self.train else 5
        try:
            x, _ = load_image(path, mode="random" if self.train else "five", crop=self.crop,
                              augment=self.augment, rng=self.rng)
        except Exception as e:                               # corrupt file: do not hide it
            self.failed.append((r["image_id"], repr(e)))
            x = torch.zeros(n, 3, self.crop, self.crop)
        return x, torch.tensor(float(LABELS[r["label"]])), str(r["image_id"])
