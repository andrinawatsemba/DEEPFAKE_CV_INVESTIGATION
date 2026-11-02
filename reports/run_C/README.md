# run_C

Run by **GW** on 2026-11-02 10:30.

- model: resnet50; training data: narrow set of 4 generator(s): stable-diffusion-v1-5, GigaGAN, midjourney-6, dalle-3; 100 images per class; augmentation: True
- threshold chosen on validation: 0.56
- macro-F1  seen generators: 0.598316894398697  | unseen generators: 0.6021422772865215  | in the wild: 0.5037447085639857
- wrong predictions: 416 (see failures.csv and failures/). Sort at least 20 of them into categories and write that in analysis.md.
