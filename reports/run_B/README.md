# run_B

Run by **GW** on 2026-11-02 09:49.

- model: resnet50; training data: narrow set of 4 generator(s): stable-diffusion-v1-5, GigaGAN, midjourney-6, dalle-3; 100 images per class; augmentation: False
- threshold chosen on validation: 0.62
- macro-F1  seen generators: 0.6546170365068003  | unseen generators: 0.5831173147224992  | in the wild: 0.48735475051264526
- wrong predictions: 415 (see failures.csv and failures/). Sort at least 20 of them into categories and write that in analysis.md.
