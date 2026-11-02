# run_F

Run by **GW** on 2026-11-02 13:08.

- model: resnet50; training data: all seen generators; 400 images per class; augmentation: True
- threshold chosen on validation: 0.55
- macro-F1  seen generators: 0.7322725418196654  | unseen generators: 0.7262820512820514  | in the wild: 0.6637190185577282
- wrong predictions: 319 (see failures.csv and failures/). Sort at least 20 of them into categories and write that in analysis.md.
