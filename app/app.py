"""
Public web app (Gradio). A thin layer: all the thinking is in src/predict.py.

Run locally:      python app/app.py
On Hugging Face:  tools/make_space_folder.py builds the folder to upload.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "src"), os.path.join(HERE, "..", "src")):
    if os.path.isdir(cand):
        sys.path.insert(0, cand)
        break

import gradio as gr                                     # noqa: E402
from predict import predict                             # noqa: E402

CKPT = next((p for p in (os.path.join(HERE, "models", "final.pt"),
                         os.path.join(HERE, "..", "models", "final.pt")) if os.path.exists(p)),
            os.path.join(HERE, "models", "final.pt"))
MAX_MB = 25


def _is_placeholder():
    import json
    side = os.path.join(os.path.dirname(CKPT), "final.json")
    try:
        return bool(json.load(open(side)).get("placeholder"))
    except Exception:
        return False


PLACEHOLDER = _is_placeholder()

NOTICE = (
    "**This is a screening aid, not proof.** It was trained on public images and checked on a small set "
    "of phone photos taken around UCU Mukono (a pilot from one area). It looks at still images only "
    "(not video, not face swaps). Heavy compression, for example after WhatsApp, makes it less reliable. "
    "A human should always double-check important decisions. Images you upload are **not stored**."
)


def analyse(path):
    if not path:
        return "Please upload an image first.", None, None, ""
    if os.path.getsize(path) > MAX_MB * 1024 * 1024:
        return f"That file is larger than {MAX_MB} MB. Please upload a smaller one.", None, None, ""
    if not os.path.exists(CKPT):
        return "The model file is missing on this server (models/final.pt).", None, None, ""
    try:
        r = predict(path, ckpt=CKPT, heatmap=True)
    except Exception:
        return ("Sorry, I could not read that file as an image. Try a JPEG, PNG or WebP photo.",
                None, None, "")
    p = r["p_fake"]
    verdict = {"real": "Likely real", "AI-generated": "Likely AI-generated",
               "uncertain": "Uncertain"}[r["label"]]
    headline = f"## {verdict}\nProbability that this image is AI-generated: **{p:.0%}**"
    notes = "\n".join(f"- {n}" for n in r["notes"]) or "No warnings for this image."
    return headline, {"AI-generated": p, "Real": 1 - p}, r["heatmap"], notes


with gr.Blocks(title="Real or AI? Image check") as demo:
    gr.Markdown("# Real or AI-generated? An image check\n"
                "Upload a photo. The model gives a verdict, how sure it is, and where it looked.")
    if PLACEHOLDER:
        gr.Markdown("## TEST VERSION\nThis model is **not trained**. The answers are meaningless. "
                    "It only proves the website works.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="filepath", label="Your image")
            btn = gr.Button("Check this image", variant="primary")
        with gr.Column():
            out_head = gr.Markdown()
            out_bar = gr.Label(label="Confidence", num_top_classes=2)
            out_map = gr.Image(label="Where the model looked (heat map)")
            out_notes = gr.Markdown()
    gr.Markdown(NOTICE)
    ex_dir = os.path.join(HERE, "examples")
    if os.path.isdir(ex_dir):
        files = [os.path.join(ex_dir, f) for f in sorted(os.listdir(ex_dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))][:4]
        if files:
            gr.Examples(files, inputs=inp)
    btn.click(analyse, inputs=inp, outputs=[out_head, out_bar, out_map, out_notes])

if __name__ == "__main__":
    demo.launch()
