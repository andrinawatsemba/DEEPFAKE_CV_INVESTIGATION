# Data statement (NC owns this file; fill every TODO from real counts, never from guesses)

## Public sources
| Source | What we use it for | Licence | Access |
|---|---|---|---|
| OpenFake (`ComplexDataLab/OpenFake`, config `core`) | training, validation, seen-generator test | CC BY-NC 4.0 | streamed from Hugging Face |
| OpenFake Reddit (config `reddit`) | **in-the-wild test only**, never trained on | CC BY-NC 4.0 | streamed from Hugging Face |
| Community Forensics-Small (`OwensLab/CommunityForensics-Small`) | training and validation, more generators | CC BY-NC-SA 4.0 | streamed from Hugging Face |
| TrueFake | TODO NC: can it be downloaded? yes/no, link or error, licence. Drop it if unclear. | TODO | TODO |

Non-commercial licences: this is a university project and nothing here is sold.

## How labels were obtained
Labels (real / AI) and generator names come from the dataset providers. We did not label by hand. TODO NC: note any known label noise.

## What we actually pulled (from `audit_outputs/`)
TODO NC after notebook 02: total images, per class, per source, per generator, held-out generators, duplicates removed.

## The campus set (UCU Mukono)
Real phone photos and matched AI images, three versions each. Protocol: `docs/campus_protocol.md`. Pilot from one area, locked, opened once.
TODO SM and PN: counts per class, phones and tools used, format table (JPEG vs PNG), consent summary (no names).

## Risks
* **Shortcut learning** (format, size, compression): see `docs/preprocessing.md`.
* **Leakage:** held-out generators, duplicates removed (exact and near), Reddit set never trained on, campus set locked.
* **Small local set:** one area, tens of images per person. We report uncertainty and do not claim it represents all of Uganda.
* **Licence:** non-commercial; no redistribution of images; only CSV metadata is on GitHub.
* **Privacy:** consent for any identifiable person; campus photos stay in the private Drive.
