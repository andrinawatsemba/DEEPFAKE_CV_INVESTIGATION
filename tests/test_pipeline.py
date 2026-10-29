"""
End-to-end smoke test on SYNTHETIC images (no internet, no GPU, about 1 minute on CPU).

    python tests/test_pipeline.py

It builds fake 'public data', then runs: pull -> audit -> split -> train (Run A and B)
-> evaluate -> predict. If this passes, the plumbing is fine and any later problem is
about the data or the model, not the code.
"""
import io
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)

import pull_data as P          # noqa: E402
import audit_split as A        # noqa: E402

rng = np.random.default_rng(0)


def img_bytes(fmt, size, kind):
    base = rng.integers(0, 255, size=(size[1], size[0], 3)).astype("uint8")
    if kind == "ai":                                   # smoother texture
        base = (base * 0.3 + 90).astype("uint8")
    b = io.BytesIO()
    Image.fromarray(base).save(b, format=fmt, quality=90) if fmt == "JPEG" else Image.fromarray(base).save(b, format=fmt)
    return b.getvalue()


def fake_stream(n, gens):
    for i in range(n):
        lab = "real" if i % 2 == 0 else "fake"
        fmt = random.choice(["JPEG", "PNG"])           # same format mix for both classes
        size = random.choice([(300, 260), (260, 300), (340, 280)])
        yield dict(image={"bytes": img_bytes(fmt, size, "ai" if lab == "fake" else "real")},
                   label=lab, model=random.choice(gens) if lab == "fake" else "real",
                   type=random.choice(["a", "b"]), row_id=i)


def main():
    root = tempfile.mkdtemp()
    P.make_folders(root)
    gens = [f"gen{i}" for i in range(6)]
    for member in ("AW", "NG"):
        P.pull_from_stream(fake_stream(600, gens), "openfake", member, root, 60,
                           max_scan=500, log_every=1000)
        for d in os.listdir(os.path.join(root, "public_data", "openfake")):
            pass
    # make the second member's files unique (the synthetic stream is random, so they are)
    P.summary(root)

    meta = A.load_all_meta(os.path.join(root, "metadata"))
    meta, info = A.dedupe(meta)
    print("dedupe:", info)
    print(A.metadata_shortcut_test(meta).to_string(index=False))
    split, sinfo = A.make_split(meta)
    rec = A.recommend(split, min_n=5)
    A.save_all(split, sinfo, rec, A.format_audit(split), A.metadata_shortcut_test(split),
               os.path.join(root, "audit_outputs"))
    print("holdout:", sinfo["holdout_generators"], "| recommended:", rec["single_generator"], rec["n_per_class"])

    split_csv = os.path.join(root, "audit_outputs", "split.csv")
    out = os.path.join(root, "reports")
    for run in ("A", "B"):
        cmd = [sys.executable, os.path.join(SRC, "train.py"), "--config",
               os.path.join(HERE, "..", "configs", f"run_{run}.yaml"),
               "--split", split_csv, "--root", root, "--out", out, "--smoke",
               "--no-pretrained", "--workers", "0", "--member", "TEST"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-900:])
        assert r.returncode == 0, r.stderr[-2000:]
        for f in ("metrics.json", "predictions.csv", "best.pt", "training_curves.png",
                  "confusion_matrix.png", "config.yaml", "README.md", "failures.csv"):
            assert os.path.exists(os.path.join(out, f"run_{run}", f)), f"missing {f} in run_{run}"
    assert os.path.exists(os.path.join(out, "experiment_log.csv"))

    # predict.py on one image
    ck = os.path.join(out, "run_B", "best.pt")
    sample = os.path.join(root, split[split.role == "test_seen"].iloc[0]["path"])
    sys.path.insert(0, SRC)
    import predict as PR
    res = PR.predict(sample, ckpt=ck, threshold=0.5, heatmap=True)
    print({k: (v if k != "heatmap" else type(v).__name__) for k, v in res.items()})
    assert set(["label", "p_fake", "heatmap", "notes"]).issubset(res)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
