"""Regenerates the notebooks from this file:  python tools/build_notebooks.py
(Edit the cell text here, not the .ipynb, so the notebooks stay consistent.)"""
import os
import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")

BOOT = '''# CELL 1 - run this first. It downloads the project code into this Colab session.
import os, subprocess, sys
REPO_URL = "https://github.com/andrinawatsemba/DEEPFAKE_CV_INVESTIGATION.git"
CODE = "/content/repo"
if not os.path.isdir(CODE + "/src"):
    for branch in ("main", "gw/lean-pipeline"):          # main first; the branch is only a fallback
        subprocess.run(["rm", "-rf", CODE])
        r = subprocess.run(["git", "clone", "--depth", "1", "-b", branch, REPO_URL, CODE])
        if r.returncode == 0 and os.path.exists(CODE + "/src/colab_setup.py"):
            print("Got the code from branch:", branch)
            break
else:
    subprocess.run(["git", "-C", CODE, "pull", "--ff-only"])
sys.path.insert(0, CODE + "/src")
print("Code ready.")
'''


def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    n.metadata["colab"] = {"provenance": []}
    nbf.write(n, os.path.join(OUT, name))
    print("wrote", name, len(cells), "cells")


# ---------------------------------------------------------------------------
# 01 pull data
# ---------------------------------------------------------------------------
nb([
 ("md", """# 01 - Pull the public data (EVERYONE, in parallel)

**What this does:** streams a *small* number of real and AI-generated images from public datasets on Hugging Face
and saves them to the shared Drive folder, plus a small CSV describing each image.

**Why it is small now:** the old version asked for thousands of images and kept reading the stream to find rare
generators. That is what crashed the laptops. This version has hard limits (images, minutes, storage) and stops by itself.

| Member | Pulls (images per class, so 100 means 100 real + 100 AI) |
|---|---|
| GW | OpenFake 150 |
| AW | OpenFake 100 |
| NG | OpenFake 150 + Reddit (in-the-wild test set) 150 |
| NC | OpenFake 100 + Community Forensics 100 |
| SM | Community Forensics 100 |
| PN | OpenFake 100 + Community Forensics 100 |

**Rules:** set `MEMBER` to your own code. Run cells one at a time (never "Run all"). Do the PILOT first (it takes about 3 minutes),
then set `PILOT = False`. If Colab disconnects, just run the cells again: it resumes."""),
 ("code", BOOT),
 ("code", """# CELL 2 - install (about 1 minute)
!pip -q install -U datasets imagehash psutil"""),
 ("code", """# CELL 3 - EDIT THESE TWO LINES ONLY
MEMBER = "AW"      # your code: GW, AW, NG, NC, SM or PN
PILOT = True       # True = tiny test. After it works, change to False and run again."""),
 ("code", """# CELL 4 - connect Drive and check memory
from colab_setup import setup
ROOT, REPO = setup()
try:
    import psutil
    m = psutil.virtual_memory()
    print(f"RAM free: {m.available/1e9:.1f} GB of {m.total/1e9:.1f} GB")
except Exception:
    pass
try:                                                    # optional: faster downloads if you saved HF_TOKEN in Colab Secrets
    from google.colab import userdata
    import os
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("Using your Hugging Face token.")
except Exception:
    print("No HF_TOKEN secret found (that is fine for public data).")"""),
 ("code", """# CELL 5 - pull (this is the main step)
import pull_data as P
results = P.run_member(MEMBER, ROOT, pilot=PILOT)"""),
 ("code", """# CELL 6 - what did I get?  Check: both classes present, several generators, a mix of formats.
P.summary(ROOT, MEMBER)"""),
 ("md", """## Share your result with the group (GitHub, no images)

Images stay on Drive (they are large and licence-restricted). What goes on GitHub is the small CSV that describes them.
Run the next cell, then upload the downloaded file(s):
GitHub repo > `metadata/` folder > **Add file > Upload files**.
**Only do this after the FULL pull (`PILOT = False`).**"""),
 ("code", """# CELL 7 - download your CSV file(s) to upload to GitHub
import glob, os, shutil
from google.colab import files
mine = sorted(glob.glob(os.path.join(ROOT, "metadata", f"public_*_{MEMBER}.csv")))
print("Files:", [os.path.basename(f) for f in mine])
for f in mine:
    files.download(f)"""),
], "01_pull_data.ipynb")

# ---------------------------------------------------------------------------
# 02 audit and split
# ---------------------------------------------------------------------------
nb([
 ("md", """# 02 - Audit the data and build the split (GW runs this; everyone reads the output)

**Plain words:** before training anything we (1) check the data for cheating routes, (2) remove duplicates, and
(3) decide which images are used for training, for tuning, for testing, and for the *surprise exam* (generators the model has never seen).

**You need:** everyone's `public_*_<CODE>.csv` files in the Drive `metadata/` folder (the pull notebook puts them there).

**Outputs** (saved to Drive `audit_outputs/`): `split.csv`, `recommended.json`, `shortcut_test.csv`, audit tables and charts.
Afterwards upload `split.csv`, `recommended.json`, `shortcut_test.csv` to the GitHub folder `audit_outputs/`."""),
 ("code", BOOT),
 ("code", """# CELL 2 - connect Drive, load everyone's CSVs
!pip -q install -U imagehash
import os, pandas as pd, numpy as np
from colab_setup import setup
import audit_split as A
ROOT, REPO = setup()
META = A.load_all_meta(os.path.join(ROOT, "metadata"))
print(len(META), "images from", META.member_file.nunique(), "metadata files")
display(META.groupby(["member_file", "label"]).size().unstack(fill_value=0))"""),
 ("code", """# CELL 3 - remove duplicates. Exact copies (same SHA-256) are always removed.
# Near-duplicates (the same picture re-saved) need the images and take a few minutes: set RUN_PHASH = True to include them.
RUN_PHASH = False
phash = A.compute_phash(META, ROOT) if RUN_PHASH else None
DATA, dup_info = A.dedupe(META, phash)
print(dup_info)"""),
 ("code", """# CELL 4 - THE AUDIT. Look at every table. A gap between real and AI here is a shortcut the model could cheat with.
import matplotlib.pyplot as plt
tabs = A.format_audit(DATA)
for k in ("counts_by_class_source", "format_by_class", "format_share_by_class", "generators"):
    print("\\n==", k); display(tabs[k])

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
tabs["format_share_by_class"].T.plot.bar(stacked=True, ax=ax[0]); ax[0].set_title("File format share per class")
for lab, c in (("real", "#14202B"), ("ai", "#A64B14")):
    d = DATA[DATA.label == lab]
    ax[1].hist(np.log10(d.width * d.height), bins=30, alpha=.6, label=lab, color=c)
    ax[2].hist(np.log10(d.bytes_per_pixel.clip(1e-3)), bins=30, alpha=.6, label=lab, color=c)
ax[1].set_title("Image size (log10 pixels)"); ax[2].set_title("Bytes per pixel (log10)"); ax[1].legend()
plt.tight_layout(); os.makedirs(os.path.join(ROOT, "audit_outputs"), exist_ok=True)
plt.savefig(os.path.join(ROOT, "audit_outputs", "audit_overview.png"), dpi=130); plt.show()"""),
 ("code", """# CELL 5 - THE SHORTCUT TEST: can a model tell real from AI using NO pixels at all (only format, size, file bytes)?
# ROC-AUC 0.5 = no leak. Above 0.75 = shortcut. Read the plain-language guide below the table.
SHORT = A.metadata_shortcut_test(DATA)
display(SHORT)"""),
 ("md", """**How to read the shortcut table**

* `format only` or `width + height` high: the *file type or picture size* gives the answer away. Fix: balance by format (training does this automatically)
  and use the class-blind augmentation (Runs C and E). Write the number in the report.
* Only `file size + bytes per pixel` high: this can be a *real* signal (AI images are often smoother, so they compress smaller) and not only a file-type leak.
  It is exactly why every run re-checks its own training set, and why the format-swap and WhatsApp tests exist.
* All near 0.5: no metadata leak. Good, but still run the format-swap test later."""),
 ("code", """# CELL 6 - BUILD THE SPLIT
SPLIT, info = A.make_split(DATA, holdout_frac=0.20, seed=42)
REC = A.recommend(SPLIT)
print("Held-out (unseen) generators:", info["holdout_generators"])
display(SPLIT.groupby(["role", "label"]).size().unstack(fill_value=0))
print("\\nRecommended for every run: single generator =", REC["single_generator"], "| images per class =", REC["n_per_class"])
if REC["warning"]:
    print("WARNING:", REC["warning"])"""),
 ("code", """# CELL 7 - save everything to Drive, then download the small files for GitHub
A.save_all(SPLIT, info, REC, tabs, SHORT, os.path.join(ROOT, "audit_outputs"))
import shutil
from google.colab import files
for f in ("split.csv", "recommended.json", "shortcut_test.csv", "split_info.json"):
    shutil.copy(os.path.join(ROOT, "audit_outputs", f), f)
    files.download(f)
print("Saved to Drive/audit_outputs and downloaded. Upload these files to GitHub: audit_outputs/")"""),
], "02_audit_and_split.ipynb")

# ---------------------------------------------------------------------------
# 03 train and evaluate
# ---------------------------------------------------------------------------
nb([
 ("md", """# 03 - Train one run and evaluate it (run owners: AW=A, GW=B, NG=C, NC=D, PN=E)

**Plain words:** every run uses the *same* script and the *same* split. Only the config file differs, so any difference in results is caused by the one thing that changed.

| Run | Model | Fakes from | Augmentation | Question |
|---|---|---|---|---|
| A | small CNN from scratch | one generator | off | baseline |
| B | ResNet-50 (transfer learning) | one generator | off | does transfer learning help? |
| C | ResNet-50 | one generator | on | does augmentation recover robustness? |
| D | ResNet-50 | several generators | off | does generator diversity help on unseen generators? |
| E | ResNet-50 | several generators | on | do the two effects combine? |

**Before you start:** GW has run notebook 02 and `audit_outputs/split.csv` exists on Drive.
**Use a GPU:** Runtime > Change runtime type > T4 GPU. (About 10 to 20 minutes per run.)"""),
 ("code", BOOT),
 ("code", """# CELL 2 - EDIT THESE TWO LINES
RUN = "B"          # A, B, C, D or E  (only the run you own)
MEMBER = "GW"      # your code"""),
 ("code", """# CELL 3 - connect, check GPU, find the split
import os, json, subprocess, torch
from colab_setup import setup
ROOT, REPO = setup()
SPLIT = os.path.join(ROOT, "audit_outputs", "split.csv")
assert os.path.exists(SPLIT), "split.csv not found: ask GW to run notebook 02 first"
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (training will be very slow; change the runtime type)")
print(json.load(open(os.path.join(ROOT, "audit_outputs", "recommended.json"))))"""),
 ("code", """# CELL 4 - SMOKE TEST (about 1 minute). Proves the code works on your session before the long run.
!pip -q install pyyaml
!python {REPO}/src/train.py --config {REPO}/configs/run_{RUN}.yaml --split {SPLIT} --root {ROOT} --out /content/smoke --member {MEMBER} --smoke --workers 2"""),
 ("code", """# CELL 5 - THE REAL RUN. Prints one line per epoch. Keep this tab open; if Colab disconnects, run CELLS 1 to 3 and 5 again.
!python {REPO}/src/train.py --config {REPO}/configs/run_{RUN}.yaml --split {SPLIT} --root {ROOT} --out {ROOT}/reports --member {MEMBER} --workers 2"""),
 ("code", """# CELL 6 - look at the results
from IPython.display import Image, display
import pandas as pd
RUN_DIR = f"{ROOT}/reports/run_{RUN}"
display(Image(f"{RUN_DIR}/training_curves.png"))
display(Image(f"{RUN_DIR}/confusion_matrix.png"))
m = json.load(open(f"{RUN_DIR}/metrics.json"))
print(json.dumps(m["_run"], indent=1))
display(pd.read_csv(f"{ROOT}/reports/experiment_log.csv"))
print(open(f"{RUN_DIR}/shortcut_check.csv").read())"""),
 ("md", """## Error analysis (this is marked: do not skip)

Open `failures.csv` and the pictures in `failures/`. Look at **at least 20** wrong predictions. Sort them into categories
(for example: heavily compressed, unusual scene, text in the image, a generator the model never saw, very smooth real photo).
Write the counts and what you think causes each. The next cell saves your notes into the run folder."""),
 ("code", """# CELL 7 - look at the worst failures (most confident wrong answers)
import glob
from PIL import Image as PI
import matplotlib.pyplot as plt
fs = sorted(glob.glob(f"{RUN_DIR}/failures/*.png"))[:12]
fig, ax = plt.subplots(3, 4, figsize=(12, 9))
for a, f in zip(ax.ravel(), fs):
    a.imshow(PI.open(f)); a.set_title(os.path.basename(f)[:34], fontsize=7); a.axis("off")
plt.show()
display(pd.read_csv(f"{RUN_DIR}/failures.csv").head(20))"""),
 ("code", """# CELL 8 - WRITE YOUR ERROR ANALYSIS here, then run the cell (edit the text, keep the structure)
ANALYSIS = '''
# Error analysis, Run {run}, by {member}

## What the numbers say
(2 sentences: seen vs unseen generators, and in the wild)

## Failure categories (at least 20 wrong predictions)
| Category | Count | Example image ids | Likely cause |
|---|---|---|---|
| e.g. heavily compressed real photos called AI | 7 | ... | compression removes fine detail |

## Most dangerous error for a user
(real called AI, or AI called real? who is harmed?)

## What I would try next (one testable idea)
'''.format(run=RUN, member=MEMBER)
open(f"{RUN_DIR}/analysis.md", "w").write(ANALYSIS)
print("saved", f"{RUN_DIR}/analysis.md")"""),
 ("md", """## Share with the group (GitHub, no weights, no images)
The next cell zips the small result files (numbers, tables, charts). Upload the zip's contents to GitHub: `reports/run_<X>/`."""),
 ("code", """# CELL 9 - download the small files for GitHub
import shutil, glob
from google.colab import files
small = [f for f in glob.glob(f"{RUN_DIR}/*") if os.path.isfile(f) and not f.endswith(".pt")]
os.makedirs(f"/content/upload_run_{RUN}", exist_ok=True)
for f in small:
    shutil.copy(f, f"/content/upload_run_{RUN}")
shutil.copy(f"{ROOT}/reports/experiment_log.csv", f"/content/upload_run_{RUN}")
shutil.make_archive(f"/content/run_{RUN}_results", "zip", f"/content/upload_run_{RUN}")
files.download(f"/content/run_{RUN}_results.zip")"""),
 ("md", """## GW only: choose the final model
After A to E are done, GW compares them (validation numbers first, never the campus set) and picks one. Then run the next cell once."""),
 ("code", """# CELL 10 (GW ONLY) - freeze the final model for the web app
FINAL_RUN = "run_E"      # change to the run you chose; write the reason in docs/final_model.md
import shutil, json
os.makedirs(f"{ROOT}/models", exist_ok=True)
shutil.copy(f"{ROOT}/reports/{FINAL_RUN}/best.pt", f"{ROOT}/models/final.pt")
thr = json.load(open(f"{ROOT}/reports/{FINAL_RUN}/metrics.json"))["_run"]["threshold"]
json.dump({"run": FINAL_RUN, "threshold": thr}, open(f"{ROOT}/models/final.json", "w"))
print("final.pt and final.json are in Drive/models. Threshold:", thr)"""),
], "03_train_and_evaluate.ipynb")

# ---------------------------------------------------------------------------
# 04 campus set
# ---------------------------------------------------------------------------
nb([
 ("md", """# 04 - The UCU Mukono campus set: label, check, lock (SM and PN lead; everyone adds their own images)

Read `docs/campus_protocol.md` first (how to take the photos and make the AI images).

**Steps for each member**
1. Put your files in the shared Drive folder: `campus/originals/` (name them `CAMP_<CODE>_0001.jpg`, `CAMP_<CODE>_0002.jpg`, ...),
   and the WhatsApp versions in `campus/whatsapp_photo/` and `campus/whatsapp_document/` with the **same names**.
2. Run CELL 3 to create your labels file, open it in Drive (right-click > Open with > Google Sheets), and fill every row.
3. Run CELL 4 until it reports **no problems**.

**Locking (SM or GW, only when everyone is finished):** CELL 6. After that nobody touches the campus files."""),
 ("code", BOOT),
 ("code", """# CELL 2 - connect Drive
from colab_setup import setup
import campus_tools as C, pandas as pd
ROOT, REPO = setup()
MEMBER = "SM"   # your code"""),
 ("code", """# CELL 3 - create YOUR labels file (one row per image you uploaded)
path, n = C.labels_template(ROOT, MEMBER)
print(n, "new rows written to", path)
print("Fill: label (real or ai), device_or_tool, prompt_or_scene (AI only), contains_person (yes/no), consent_obtained (yes/no)")"""),
 ("code", """# CELL 4 - CHECK. Fix every line it prints, then run it again until it says no problems.
log, problems = C.build_log(ROOT)
print(len(problems), "problem(s)")
for p in problems:
    print(" -", p)
if len(log):
    C.summary(log)"""),
 ("md", """**Read the format table above.** Phone photos are usually JPEG and AI tools often give PNG. That gap is a shortcut risk, so it is reported in the data statement.
The WhatsApp versions exist to test that: after WhatsApp both classes look alike on the outside."""),
 ("code", """# CELL 6 - LOCK THE SET (SM or GW, once, when all problems are fixed and quotas are met)
assert not problems, "Fix the problems first"
manifest = C.lock_manifest(ROOT)
print(len(manifest), "files recorded in metadata/campus_manifest.csv. The campus set is now READ-ONLY.")"""),
], "04_campus_set.ipynb")

# ---------------------------------------------------------------------------
# 05 robustness, format swap, explain, campus test
# ---------------------------------------------------------------------------
nb([
 ("md", """# 05 - Robustness (NC), format-swap (NC), explainability (SM), campus test (NG)

Run only the part that is yours. Each part loads a trained model from `reports/run_X/best.pt` (or `models/final.pt`).

* **Part A (NC):** how much does accuracy fall under compression, resizing, blur, noise, darkness?
* **Part B (NC):** format-swap test: re-save images in the other format. If accuracy collapses, the model was reading file types.
* **Part C (SM):** Grad-CAM heat maps and the *faithfulness* test (does hiding the highlighted part really change the answer more than hiding a random part?).
* **Part D (NG, ONCE):** the locked UCU Mukono campus set, in three versions (original, WhatsApp photo, WhatsApp document)."""),
 ("code", BOOT),
 ("code", """# CELL 2 - connect, load a model and the test images
import os, json, torch, pandas as pd, numpy as np
from colab_setup import setup
from model import load_checkpoint
ROOT, REPO = setup()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN = "run_E"                     # which run's model to examine
model, _ = load_checkpoint(f"{ROOT}/reports/{RUN}/best.pt", DEVICE)
THR = json.load(open(f"{ROOT}/reports/{RUN}/metrics.json"))["_run"]["threshold"]
SPLIT = pd.read_csv(f"{ROOT}/audit_outputs/split.csv")
TEST = SPLIT[SPLIT.role.isin(["test_seen", "test_unseen"])]
SAMPLE = pd.concat([g.sample(min(len(g), 75), random_state=0) for _, g in TEST.groupby("label")])   # 150 images keeps it fast
print(len(SAMPLE), "test images; threshold", THR)"""),
 ("md", "## Part A (NC): robustness under damage"),
 ("code", """# CELL 3 - Part A. Takes a few minutes.
import stress, matplotlib.pyplot as plt
T = stress.stress_table(model, SAMPLE, ROOT, threshold=THR, device=DEVICE)
display(T.round(3))
os.makedirs(f"{ROOT}/reports/robustness", exist_ok=True)
T.to_csv(f"{ROOT}/reports/robustness/{RUN}_stress.csv", index=False)
ax = T.set_index("condition").macro_f1.plot.barh(figsize=(7, 5), color="#A64B14")
ax.set_xlabel("macro-F1"); ax.set_title(f"{RUN}: accuracy under damage"); plt.tight_layout()
plt.savefig(f"{ROOT}/reports/robustness/{RUN}_stress.png", dpi=130); plt.show()"""),
 ("md", "## Part B (NC): the format-swap test"),
 ("code", """# CELL 4 - Part B. JPEG images saved losslessly as PNG, and everything saved as high-quality JPEG.
FS = stress.stress_table(model, SAMPLE, ROOT, conditions={"original": stress.CONDITIONS["original"], **stress.FORMAT_SWAP},
                         threshold=THR, device=DEVICE)
display(FS.round(3))
FS.to_csv(f"{ROOT}/reports/robustness/{RUN}_format_swap.csv", index=False)
print("If macro_f1 drops a lot for as_png or as_jpeg95, the model was using the file format.")"""),
 ("md", "## Part C (SM): Grad-CAM and the faithfulness test"),
 ("code", """# CELL 5 - Part C. Heat maps for 6 images.
from explain import GradCAM, overlay
from data import load_image, denormalise
import matplotlib.pyplot as plt
rows = SAMPLE.sample(6, random_state=1)
fig, ax = plt.subplots(2, 6, figsize=(16, 6))
cam_fn = GradCAM(model, model.gradcam_layer())
for j, r in enumerate(rows.itertuples()):
    x, _ = load_image(f"{ROOT}/{r.path}", mode="center")
    cam, p = cam_fn(x.to(DEVICE))
    ax[0, j].imshow(denormalise(x[0])); ax[0, j].set_title(f"{r.label}", fontsize=9); ax[0, j].axis("off")
    ax[1, j].imshow(overlay(denormalise(x[0]), cam)); ax[1, j].set_title(f"p(AI)={p:.2f}", fontsize=9); ax[1, j].axis("off")
cam_fn.close(); plt.tight_layout(); plt.show()"""),
 ("code", """# CELL 6 - THE FAITHFULNESS TEST (100 images: 50 real + 50 AI). A few minutes.
from explain import faithfulness_study
FSAMPLE = pd.concat([g.sample(min(len(g), 50), random_state=2) for _, g in TEST.groupby("label")])
per_image, summary = faithfulness_study(model, FSAMPLE, ROOT, DEVICE, frac=0.2, n_random=5)
print(json.dumps(summary, indent=1))
os.makedirs(f"{ROOT}/reports/explainability", exist_ok=True)
per_image.to_csv(f"{ROOT}/reports/explainability/{RUN}_faithfulness.csv", index=False)
json.dump(summary, open(f"{ROOT}/reports/explainability/{RUN}_faithfulness.json", "w"), indent=2)
plt.figure(figsize=(5, 4))
plt.bar(["hide Grad-CAM region", "hide random region"], [summary["mean_drop_cam"], summary["mean_drop_random"]], color=["#A64B14", "#56636E"])
plt.ylabel("mean drop in confidence"); plt.title("Does the heat map point at what matters?"); plt.tight_layout()
plt.savefig(f"{ROOT}/reports/explainability/{RUN}_faithfulness.png", dpi=130); plt.show()
print(summary.get("verdict"))"""),
 ("md", """## Part D (NG): the campus test, ONCE
This opens the locked set. **Only run it with the final model, after the model is chosen.** Do not change the model and run it again to improve the score."""),
 ("code", """# CELL 7 - Part D. Set the flag to True only when you are sure.
I_UNDERSTAND_THIS_RUNS_ONCE = False
if I_UNDERSTAND_THIS_RUNS_ONCE:
    import campus_tools as C
    from evaluate import evaluate_roles
    CAMP = C.test_table(ROOT)
    m, p = evaluate_roles(model, CAMP, ROOT, sorted(CAMP.role.unique()), THR, DEVICE)
    os.makedirs(f"{ROOT}/reports/campus_test", exist_ok=True)
    json.dump(m, open(f"{ROOT}/reports/campus_test/{RUN}_metrics.json", "w"), indent=2)
    p.to_csv(f"{ROOT}/reports/campus_test/{RUN}_predictions.csv", index=False)
    display(pd.DataFrame({k: {x: v[x] for x in ("n", "accuracy", "macro_f1", "recall_real", "recall_ai")} for k, v in m.items()}).T.round(3))
else:
    print("Not run. Set I_UNDERSTAND_THIS_RUNS_ONCE = True when the final model is frozen.")"""),
], "05_robustness_explain_campus.ipynb")

# ---------------------------------------------------------------------------
# 06 web app
# ---------------------------------------------------------------------------
nb([
 ("md", """# 06 - Build the web app folder (PN leads, AW backs up)

**Plain words:** the website is just a small Gradio app that calls the same `predict()` function as the command line.
This notebook builds the folder you upload to a free **Hugging Face Space** (the public link).

**Do it twice**
1. **Now, with a PLACEHOLDER model** (untrained, the page says TEST VERSION). This proves the deployment works early, so there are no surprises at the end.
2. **Later, with the real final model** (GW finishes notebook 03 CELL 10 first).

**One-time:** create a free account at huggingface.co, then *New > Space*: name it, SDK **Gradio**, hardware **CPU basic (free)**, visibility **Public**."""),
 ("code", BOOT),
 ("code", """# CELL 2 - connect Drive and pick the mode
!pip -q install gradio
import os, shutil, subprocess
from colab_setup import setup
ROOT, REPO = setup()
USE_PLACEHOLDER = True      # True = untrained test model. Change to False once models/final.pt exists on Drive."""),
 ("code", """# CELL 3 - build the folder
os.makedirs(f"{REPO}/models", exist_ok=True)
if not USE_PLACEHOLDER:
    for f in ("final.pt", "final.json"):
        shutil.copy(f"{ROOT}/models/{f}", f"{REPO}/models/{f}")
args = ["python", f"{REPO}/tools/make_space_folder.py"] + (["--placeholder"] if USE_PLACEHOLDER else [])
print(subprocess.run(args, capture_output=True, text=True).stdout)
shutil.make_archive("/content/space_build", "zip", f"{REPO}/space_build")
print(os.listdir(f"{REPO}/space_build"))"""),
 ("code", """# CELL 4 - test the app function here BEFORE uploading (use any photo from Drive)
import sys, glob
sys.path.insert(0, f"{REPO}/space_build")
import importlib, app as webapp
importlib.reload(webapp)
sample = sorted(glob.glob(f"{ROOT}/public_data/openfake/*"))[:1]
print(webapp.analyse(sample[0])[0] if sample else "put a photo path here")"""),
 ("code", """# CELL 5 - download the folder as a zip
from google.colab import files
files.download("/content/space_build.zip")"""),
 ("md", """## Upload to Hugging Face (step by step)
1. Unzip the download on your computer.
2. Open your Space > **Files** tab > **Add file > Upload files**. Drag **everything inside** the unzipped folder (`app.py`, `README.md`, `requirements.txt`, `src/`, `models/`). The `.pt` file is big, so Hugging Face handles it automatically.
3. Click **Commit changes**. The Space shows *Building*, then *Running* (2 to 5 minutes).
4. Open the public link on your phone, on mobile data, in a private window. Upload 5 different photos.
5. Paste the link in the group chat and into the README.

**Before the examination:** repeat step 4 the day before. Free Spaces go to sleep when unused, so the first request can be slow: open it early."""),
], "06_build_web_app.ipynb")
