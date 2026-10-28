"""
predict.py — inference script for the Deepfake CV Investigation project.

Loads a saved model checkpoint and classifies ONE unseen image:
  * binary label (real / AI-generated)
  * confidence score
  * Grad-CAM heat map (screening aid only — not proof of authenticity)

Colab-friendly. No hardcoded personal paths. Reads model from reports/,
writes visualization + JSON to reports/.

Usage (in a Colab cell or terminal):
    %run src/predict.py --image /path/to/image.jpg --model reports/resnet50_transfer.pt

Edit CONFIG or pass CLI flags.
"""

import os, sys, json, argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    image_transform, DEFAULT_MEMBER,
)
from train_baseline import SmallCNN


# ------------------------------------------------------------------ config
CONFIG = dict(
    member=DEFAULT_MEMBER,
    model_path="reports/resnet50_transfer.pt",
    model_type="resnet50",   # "resnet50", "resnet18", or "smallcnn"
    input_size=224,
    image_path=None,         # set to the image you want to classify
    out_dir="reports",
    device=None,             # None = auto (cuda if available)
)

device = torch.device(CONFIG["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
print("Device:", device)


# ------------------------------------------------------------------ Grad-CAM
class GradCAM:
    """Simple Grad-CAM for a target convolutional layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=0):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if logits.dim() > 1:
            score = logits[:, class_idx]
        else:
            score = logits[class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, logits.detach().cpu().numpy()

    def close(self):
        for h in self._hook_handles:
            h.remove()


# ------------------------------------------------------------------ model loaders
def load_smallcnn(model_path, device, input_size):
    model = SmallCNN(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, model.features[-2]  # last conv before adaptive pool


def load_resnet(model_path, model_type, device, input_size):
    if model_type == "resnet18":
        model = models.resnet18(weights=None)
        in_f = model.fc.in_features
    else:
        model = models.resnet50(weights=None)
        in_f = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, model.layer4  # Grad-CAM target: last residual block


def load_model(model_path, model_type, device, input_size):
    if model_type == "smallcnn":
        return load_smallcnn(model_path, device, input_size)
    return load_resnet(model_path, model_type, device, input_size)


# ------------------------------------------------------------------ preprocess
def load_image(image_path, input_size=224):
    tf = image_transform(input_size, augment=False)
    with PILImage.open(image_path) as im:
        im = im.convert("RGB")
        x = tf(im).unsqueeze(0)
    return x


# ------------------------------------------------------------------ main
def main(cfg):
    if cfg["image_path"] is None:
        raise SystemExit("Set --image to the path of an image to classify.")

    image_path = cfg["image_path"]
    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    model_path = cfg["model_path"]
    if not os.path.exists(model_path):
        raise SystemExit(f"Model checkpoint not found: {model_path}")

    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 1) Load model
    model, target_layer = load_model(model_path, cfg["model_type"], device, cfg["input_size"])
    cam = GradCAM(model, target_layer)

    # 2) Load and preprocess image
    x = load_image(image_path, cfg["input_size"]).to(device)

    # 3) Predict + Grad-CAM
    heatmap, logits = cam(x)
    prob_ai = float(1 / (1 + np.exp(-logits[0][0]))) if logits.ndim > 1 else float(1 / (1 + np.exp(-logits[0])))
    label = "ai" if prob_ai >= 0.5 else "real"
    confidence = prob_ai if label == "ai" else 1 - prob_ai

    cam.close()

    # 4) Print result
    print("\n=== SCREENING RESULT ===")
    print(f"Image: {image_path}")
    print(f"Label: {label.upper()}")
    print(f"Confidence: {confidence:.4f} (P(AI)={prob_ai:.4f})")
    print("\nNOTE: This is a screening aid, not proof of authenticity.")
    print("A human fact-checker should review any suspicious image.\n")

    # 5) Save visualization
    with PILImage.open(image_path) as im:
        im = im.convert("RGB").resize((cfg["input_size"], cfg["input_size"]))
        img_arr = np.array(im)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_arr)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM heat map")
    axes[1].axis("off")

    axes[2].imshow(img_arr)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"Prediction: {label.upper()}  |  Confidence: {confidence:.3f}",
                 fontsize=14, color="red" if label == "ai" else "green")

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(cfg["out_dir"], f"prediction_{base}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {out_path}")

    # 6) Save JSON result
    result = {
        "image": image_path,
        "label": label,
        "confidence": confidence,
        "p_ai": prob_ai,
        "model": model_path,
        "member": cfg["member"],
        "screening_aid_notice": "This is a screening aid, not proof of authenticity. Human review required.",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    json_path = os.path.join(cfg["out_dir"], f"prediction_{base}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image to classify")
    parser.add_argument("--model", default=CONFIG["model_path"], help="Path to saved model checkpoint")
    parser.add_argument("--model-type", default=CONFIG["model_type"],
                        choices=["resnet50", "resnet18", "smallcnn"])
    parser.add_argument("--input-size", type=int, default=CONFIG["input_size"])
    parser.add_argument("--out-dir", default=CONFIG["out_dir"])
    parser.add_argument("--member", default=CONFIG["member"])
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg.update(image_path=args.image, model_path=args.model,
               model_type=args.model_type, input_size=args.input_size,
               out_dir=args.out_dir, member=args.member)
    main(cfg)