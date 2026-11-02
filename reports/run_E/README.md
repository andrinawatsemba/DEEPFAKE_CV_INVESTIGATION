# run_E

Run by **GW** on 2026-11-02 10:53.

- model: resnet50; training data: all seen generators; 100 images per class; augmentation: True
- threshold chosen on validation: 0.54
- macro-F1  seen generators: 0.6191441624091044  | unseen generators: 0.658203125  | in the wild: 0.7190599226004446
- wrong predictions: 365 (see failures.csv and failures/). Sort at least 20 of them into categories and write that in analysis.md.
