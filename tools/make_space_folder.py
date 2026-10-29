"""
Build the folder you upload to a Hugging Face Space.

    python tools/make_space_folder.py                  -> creates space_build/ (real model)
    python tools/make_space_folder.py --placeholder    -> same, with an UNTRAINED model, to test deployment early

Then upload everything inside space_build/ to your Space (Files tab > Add file > Upload files).
Needs models/final.pt and models/final.json (created in notebook 03 after the final model is chosen).
"""
import json
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "space_build")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "src"))
os.makedirs(os.path.join(OUT, "models"))
shutil.copy(os.path.join(ROOT, "app", "app.py"), os.path.join(OUT, "app.py"))
for f in ("data.py", "model.py", "explain.py", "predict.py"):
    shutil.copy(os.path.join(ROOT, "src", f), os.path.join(OUT, "src", f))
if "--placeholder" in sys.argv and not os.path.exists(os.path.join(ROOT, "models", "final.pt")):
    # an UNTRAINED model, only to test that the website deploys. The app shows a TEST VERSION banner.
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from model import build_model, save_checkpoint
    save_checkpoint(build_model("resnet18", pretrained=False), os.path.join(OUT, "models", "final.pt"), "resnet18")
    json.dump({"run": "placeholder", "threshold": 0.5, "placeholder": True},
              open(os.path.join(OUT, "models", "final.json"), "w"))
    print("Built a PLACEHOLDER (untrained) model for a deployment test.")
for f in ("final.pt", "final.json"):
    p = os.path.join(ROOT, "models", f)
    if os.path.exists(os.path.join(OUT, "models", f)):
        continue
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT, "models", f))
    else:
        print(f"WARNING: models/{f} not found. The app will not work until you add it.")
if os.path.isdir(os.path.join(ROOT, "app", "examples")):
    shutil.copytree(os.path.join(ROOT, "app", "examples"), os.path.join(OUT, "examples"))
open(os.path.join(OUT, "requirements.txt"), "w").write(
    "torch\ntorchvision\npillow\nnumpy\nmatplotlib\n")
open(os.path.join(OUT, "README.md"), "w").write(
    "---\ntitle: Real or AI image check\nemoji: 🔍\ncolorFrom: yellow\ncolorTo: red\n"
    "sdk: gradio\nsdk_version: 5.0.0\napp_file: app.py\npinned: false\n---\n\n"
    "Group 2, UCU Computer Vision and Deep Learning capstone.\n")
print("Built:", OUT)
