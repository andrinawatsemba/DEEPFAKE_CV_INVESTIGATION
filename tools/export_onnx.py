"""
Convert the frozen model to ONNX so a web page can run it in the visitor's browser.

    python tools/export_onnx.py --ckpt models/final.pt --out web/model
    python tools/export_onnx.py --random resnet50 --out /tmp/test_model     (untrained model, only for testing)

What it writes into --out:
    model.onnx.part0, part1, ...   the model cut into pieces under --chunk-mb (hosts limit file size)
    manifest.json                  part list, threshold, run name, crop size
It also CHECKS that the ONNX model gives the same numbers as the PyTorch model and stops if it does not.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import build_model, load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="models/final.pt (or reports/run_X/best.pt)")
    ap.add_argument("--random", help="build an UNTRAINED model of this name, for tests only")
    ap.add_argument("--out", default="web/model")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--chunk-mb", type=float, default=20)
    a = ap.parse_args()

    if a.random:
        torch.manual_seed(0)
        model, name = build_model(a.random, pretrained=False).eval(), a.random
        run, thr = "untrained-test", 0.5
    else:
        model, name = load_checkpoint(a.ckpt, "cpu")
        side = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "final.json")
        meta = json.load(open(side)) if os.path.exists(side) else {}
        run, thr = meta.get("run", os.path.basename(a.ckpt)), meta.get("threshold", 0.5)
    if a.threshold is not None:
        thr = a.threshold
    model.eval().cpu()

    os.makedirs(a.out, exist_ok=True)
    full = os.path.join(a.out, "model.onnx")
    x = torch.randn(5, 3, 224, 224)
    torch.onnx.export(model, x, full, input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input": {0: "n"}, "logit": {0: "n"}}, opset_version=17, dynamo=False)

    # parity check: same inputs, PyTorch vs ONNX Runtime
    import onnxruntime as ort
    sess = ort.InferenceSession(full, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = model(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"max logit difference PyTorch vs ONNX: {diff:.2e}")
    if diff > 1e-3:
        sys.exit("ONNX output does not match PyTorch. Do not use this export.")

    # cut into parts
    data = open(full, "rb").read()
    size = int(a.chunk_mb * 1024 * 1024)
    parts = []
    for i in range(0, len(data), size):
        fn = f"model.onnx.part{i // size}"
        open(os.path.join(a.out, fn), "wb").write(data[i:i + size])
        parts.append(fn)
    os.remove(full)
    json.dump({"model": name, "run": run, "threshold": thr, "crop": 224, "bytes": len(data), "parts": parts},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"wrote {len(parts)} parts, {len(data) / 1e6:.1f} MB total, threshold {thr}")


if __name__ == "__main__":
    main()
