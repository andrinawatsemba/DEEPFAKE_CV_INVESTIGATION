# Can We Trust AI-Image Detectors on Ugandan Content?

Group 2, UCU Computer Vision and Deep Learning capstone.
A real vs AI-generated image classifier, studied for robustness and explainability, deployed as a public website.

* **Public data:** OpenFake, OpenFake Reddit (in-the-wild test only), Community Forensics-Small.
* **Local data:** a pilot set of phone photos and matched AI images around **UCU Mukono**, saved three ways (original, WhatsApp photo, WhatsApp document), locked for the final test.
* **Models:** Run A small CNN (baseline) and Runs B to E ResNet-50 transfer learning, one controlled change at a time.
* **Pipeline:** data > audit and split > preprocess > train > evaluate > robustness and explainability > one `predict()` function > web app.

## Open the notebooks (click, then File > Save a copy in Drive)
| # | Notebook | Who | Open in Colab |
|---|---|---|---|
| 01 | Pull public data (lean version) | everyone | [open](https://colab.research.google.com/github/andrinawatsemba/DEEPFAKE_CV_INVESTIGATION/blob/main/notebooks/01_pull_data.ipynb) |
| 02 | Audit and split | GW | [open](https://colab.research.google.com/github/andrinawatsemba/DEEPFAKE_CV_INVESTIGATION/blob/main/notebooks/02_audit_and_split.ipynb) |
| 03 | Train and evaluate one run | run owners | [open](https://colab.research.google.com/github/andrinawatsemba/DEEPFAKE_CV_INVESTIGATION/blob/main/notebooks/03_train_and_evaluate.ipynb) |
| 04 | Campus set: label, check, lock | SM, PN, everyone | [open](https://colab.research.google.com/github/andrinawatsemba/DEEPFAKE_CV_INVESTIGATION/blob/main/notebooks/04_campus_set.ipynb) |
| 05 | Robustness, format-swap, explainability, campus test | NC, SM, NG | [open](https://colab.research.google.com/github/andrinawatsemba/DEEPFAKE_CV_INVESTIGATION/blob/main/notebooks/05_robustness_explain_campus.ipynb) |
| 06 | Build the web app folder | PN, AW | [open](https://colab.research.google.com/github/andrinawatsemba/DEEPFAKE_CV_INVESTIGATION/blob/main/notebooks/06_build_web_app.ipynb) |

## Who does what (everyone works in parallel and uploads to GitHub)
| Member | Pulls | Owns | Run |
|---|---|---|---|
| GW | OpenFake 150 per class | audit, split, `src/data.py`, `src/train.py`, final model choice, integration | B |
| AW | OpenFake 100 | repo, README, `predict.py` checks, reproducibility test, comparison table | A |
| NG | OpenFake 150 + Reddit 150 | `src/evaluate.py`, error analysis, campus test (once) | C |
| NC | OpenFake 100 + Community Forensics 100 | data statement, robustness, format-swap test | D |
| SM | Community Forensics 100 | campus real photos, consent and privacy, Grad-CAM, faithfulness test | none |
| PN | OpenFake 100 + Community Forensics 100 | campus AI images, web app on Hugging Face | E |

"150 per class" means 150 real + 150 AI. These numbers are about ten times smaller than the first version, on purpose (the first version crashed laptops).

## How to upload your work to GitHub (no command line needed)
1. Go to the repo on github.com (sign in with **your own account**).
2. Open the folder where your file belongs (`metadata/`, `audit_outputs/`, `reports/run_X/`, `docs/`).
3. **Add file > Upload files**, drag your file(s) in.
4. At the bottom choose **Create a new branch for this commit** and name it `<code>/<what>` (for example `nc/run-d-results`), then **Propose changes** and **Create pull request**.
5. Someone else reviews and merges it. Never edit `main` directly.
Never upload images, passwords, tokens or `.pt` model files. (`.gitignore` blocks most of them.)

## Run the code on your own machine (optional)
```
pip install -r requirements.txt
python tests/test_pipeline.py          # ~1 minute, synthetic images, proves the plumbing works
python src/predict.py some_image.jpg --ckpt models/final.pt
python app/app.py                      # the web app, locally
```

## Where things are
```
src/       pull_data.py  audit_split.py  data.py  model.py  train.py  evaluate.py
           metrics.py  stress.py  explain.py  predict.py  campus_tools.py  colab_setup.py
configs/   run_A.yaml ... run_E.yaml      (the ONLY difference between runs)
notebooks/ 01 to 06 are current; the older week*.ipynb are first drafts, not used
app/       app.py                         (web app)       tools/  build_notebooks.py, make_space_folder.py
docs/      preprocessing.md  campus_protocol.md  data_statement.md
metadata/  small CSVs (no images)         audit_outputs/  split.csv and audit results
reports/   one folder per run             tests/          test_pipeline.py
```

## Rules
* **Real work, real names.** Log only runs you ran. Never run something under another member's code.
* **One change per run.** B to E differ from the previous run in exactly one thing.
* **Never tune on the campus set.** It is the surprise exam, opened once.
* **One image loader.** Always `src/data.py: load_image()`.
* **Label honestly:** pilot, subset, seed, true counts. No number goes in a document before a run produced it.
* **AI tools:** allowed for explanations, debugging and drafts. Keep `docs/ai_use_log.md` (tool, purpose, how you checked it). Never submit what you cannot explain and defend.
* **Raw images and tokens never go on GitHub.** Only code, configs, docs and small result files.
