# Preprocessing and the file-type shortcut

One function, `load_image()` in `src/data.py`, turns any image into a model input. Training, evaluation,
`predict.py` and the web app all call it. **Nobody writes their own image loading code.**

## The five fixed steps
1. Open with **PIL only** (mixing PIL and OpenCV flips colour order, a silent bug).
2. Apply the phone's rotation flag, then **drop all metadata** (EXIF, colour profile).
3. Convert to **3-channel, 8-bit RGB** (grayscale gets 3 copies, transparency is dropped).
4. Take **224 x 224 crops at native resolution. No whole-image resize**, because resizing destroys the fine traces generators leave.
   * training: one random crop, position snapped to the 8-pixel JPEG grid
   * evaluation and prediction: five crops (centre and four corners), the five probabilities are averaged
   * images smaller than 224 are reflect-padded and flagged ("small image" warning)
5. Scale to 0..1, subtract the ImageNet mean (0.485, 0.456, 0.406), divide by the ImageNet std (0.229, 0.224, 0.225).
   ResNet-50 was pretrained with these numbers. The small CNN uses them too so Runs A and B differ only in the model.

## Why file types are dangerous
If most real images are JPEG and most AI images are PNG, a model can score well by learning the *format*, the file size
or the image shape, and never look at the picture. It would then fail on images it has not seen (the lecturer's).

## What we do about it (four steps)
| Step | What | Where |
|---|---|---|
| 1 Detect | Table of class against format, size, bytes per pixel. **Metadata-only test:** a small model sees NO pixels, only format, size and file bytes. ROC-AUC near 0.5 = no leak, above 0.75 = shortcut. | notebook 02, `audit_split.metadata_shortcut_test` |
| 2a Balance | Every run's training set is built so that, **format by format, real and AI counts are equal**. The check is repeated on exactly the rows each run trains on (`shortcut_check.csv`). | `train.py: select_training_rows` |
| 2b Class-blind augmentation | Runs C and E re-compress with random JPEG quality, downscale-then-upscale, blur, mirror. The recipe never looks at the label. | `data.augment_blind` |
| 3 Prove it is gone | Format-swap test (re-save as the other format), per-format accuracy, and the campus three-version test (original, WhatsApp photo, WhatsApp document). | notebook 05, `stress.py`, `evaluate.per_format` |
| 4 Report it | Same model on unbalanced vs balanced data; a gap on the campus set is direct evidence of the shortcut. | report |

## Caveats we say out loud
* A PNG fake compressed once and an already-JPEG real compressed twice can leave a "double compression" fingerprint. That is why we balance first, then augment, and why the format-swap test is the judge.
* `file size + bytes per pixel` can be high for a *real* reason: AI images are often smoother, so they compress smaller. A high value there does not always mean a pure metadata leak. The other rows (format only, width and height) are the cleaner shortcut signals.
* The threshold ("how sure before we say AI") is chosen on the **validation** images only, never on test or campus images.
