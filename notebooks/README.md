# notebooks/

**Use these (numbered, current):** `01_pull_data` > `02_audit_and_split` > `03_train_and_evaluate` > `04_campus_set` > `05_robustness_explain_campus` > `06_build_web_app`.
They are described in the main README, with Colab links.

**Do NOT use these (first draft, kept for reference):** `proof_of_concept_pilot`, `week1_data_acquisition`, `week2_small_model_training`,
`week3_kampala_controlled`, `week4_evaluation_robustness`, `week5_explainability`, `week6_final_inference`.
They use code that was replaced because it:

* resized whole images to 128 or 224 pixels (our plan uses native-resolution crops),
* trained on the in-the-wild Reddit set (that set is for testing only: leakage),
* pulled thousands of images per person and crashed laptops,
* had no held-out generator split and no format-shortcut check,
* still said "Kampala" (the local set is now photographed around UCU Mukono).
