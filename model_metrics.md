# Model Metrics Summary

All reported accuracies come from `training_data_clean.csv` after preprocessing and grouped splitting.

Split protocol (same as `sample.py`):
1. Load `training_data_clean.csv` via `load_dataframe`.
2. Build model dataframe with `unique_id`, target `Painting`, and selected features.
3. Drop only rows where `Painting` is missing.
4. Call `get_splits_with_test(df_model, target_col="Painting", n_splits=5, test_size=0.2, seed=42)`.
5. Use `df_train_pool` for CV/tuning and `df_test` for final held-out evaluation.

Final test accuracy is measured on held-out 20% split from `training_data_clean.csv`.

## Latest run results (2026-03-29)

| Model | Script | CV metric | Holdout test metric | Notes |
|---|---|---|---|---|
| Gaussian Naive Bayes | `naive_bayes/run_gnb.py` | CV accuracy mean=0.6778, std=0.0124 | Accuracy=0.6903, Macro Precision=0.6966, Macro Recall=0.6903, Macro F1=0.6511 | Best `var_smoothing=1e-12` |
| GDA (QDA) | `gda/run_gda.py` | CV accuracy mean=0.6667, std=0.0124 | Accuracy=0.6755, Macro Precision=0.6658, Macro Recall=0.6755, Macro F1=0.6384 | Iterative tuning selected `reg_param=0.345833` |
| Neural Net (MLP + optional TF-IDF text) | `neural_network.py` | CV val accuracy mean=0.8804, std=0.0069 | Accuracy=0.8997, Macro Precision=0.8996, Macro Recall=0.8997, Macro F1=0.8991 | Run used `use_text=True`; includes holdout classification report |

## Where model metrics are documented
- Naive Bayes: `naive_bayes/doc.md`
- GDA: `gda/doc.md`
- Combined summary: `model_metrics.md`
