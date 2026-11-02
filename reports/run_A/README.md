# run_A

Run by **AW** on 2026-11-02 11:02.

- model: tinycnn; training data: narrow set of 4 generator(s): stable-diffusion-v1-5, GigaGAN, midjourney-6, dalle-3; 100 images per class; augmentation: False
- threshold chosen on validation: 0.50
- macro-F1  seen generators: 0.5958440721649485  | unseen generators: 0.5982062780269057  | in the wild: 0.6104070574537638
- wrong predictions: 435 (see failures.csv and failures/). Sort at least 20 of them into categories and write that in analysis.md.
