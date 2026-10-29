"""Models. One logit out: probability of AI = sigmoid(logit). Label 0 real, 1 AI."""

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Run A baseline: a small network trained from scratch."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128),
                                      *block(128, 256), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 64),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1)).squeeze(1)

    def gradcam_layer(self):
        return self.features[-2]        # the last pooling block output (conv features)


def build_model(name="resnet50", pretrained=True):
    """name: tinycnn | resnet50 | resnet18 (fallback if GPU time is short)."""
    from torchvision import models
    if name == "tinycnn":
        return TinyCNN()
    if name in ("resnet50", "resnet18"):
        if name == "resnet50":
            w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = models.resnet50(weights=w)
        else:
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
        return _ResNetWrap(m)
    raise ValueError(f"unknown model: {name}")


class _ResNetWrap(nn.Module):
    """Wraps a torchvision ResNet so the output is one logit of shape (B,)."""

    def __init__(self, m):
        super().__init__()
        self.net = m

    def forward(self, x):
        return self.net(x).squeeze(1)

    def gradcam_layer(self):
        return self.net.layer4

    def set_stage(self, stage):
        """stage 1: only the new head trains. stage 2: layer4 + head train."""
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.net.fc.parameters():
            p.requires_grad = True
        if stage == 2:
            for p in self.net.layer4.parameters():
                p.requires_grad = True

    def param_groups(self, stage, lr_head, lr_backbone):
        if stage == 1:
            return [{"params": self.net.fc.parameters(), "lr": lr_head}]
        return [{"params": self.net.fc.parameters(), "lr": lr_head},
                {"params": self.net.layer4.parameters(), "lr": lr_backbone}]


def set_stage(model, stage):
    if hasattr(model, "set_stage"):
        model.set_stage(stage)
    else:
        for p in model.parameters():
            p.requires_grad = True


def param_groups(model, stage, lr_head, lr_backbone):
    if hasattr(model, "param_groups"):
        return model.param_groups(stage, lr_head, lr_backbone)
    return [{"params": model.parameters(), "lr": lr_head}]


def save_checkpoint(model, path, name):
    torch.save({"model": name, "state_dict": model.state_dict()}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = build_model(ck["model"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck["model"]
