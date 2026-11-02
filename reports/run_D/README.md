# run_D

Run by **GW** on 2026-11-02 10:42.

- model: resnet50; training data: all seen generators; 100 images per class; augmentation: False
- threshold chosen on validation: 0.54
- macro-F1  seen generators: 0.6317684680084916  | unseen generators: 0.6386644290258748  | in the wild: 0.673131401176727
- wrong predictions: 391 (see failures.csv and failures/). Sort at least 20 of them into categories and write that in analysis.md.
