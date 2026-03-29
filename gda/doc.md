# GDA Model Notes (Numeric + Text Features)

## Goal
Build and communicate a Gaussian Discriminant Analysis baseline that follows the same split and leakage-control protocol as other team models so results are directly comparable.

## Files In This Folder
- `doc.md`: GDA documentation and tuning notes.
- `run_gda.py`: iterative GDA training and evaluation script.

## Why GDA (QDA form)
- The current default uses numeric features plus text features (`text_all`).
- GDA (implemented via `QuadraticDiscriminantAnalysis`) models class-conditional Gaussians with class-specific covariance.
- It is a strong generative baseline to compare against Gaussian Naive Bayes.

## Data + Split Protocol
- Input file: `training_data_clean.csv`
- Target: `Painting`
- Group key: `unique_id`
- Split function: `get_splits_with_test(...)` from `split_data.py`
- Configuration: 5-fold CV on train pool, 20% holdout test, seed 42

### Leakage Controls
- Rows with missing target are dropped.
- Missing feature values are imputed with train medians only:
  - For CV: medians from each train fold
  - For holdout test: medians from full train pool
- Test set is used only once, after model selection.

## Feature Set
Numeric:
1. On a scale of 1-10, how intense is the emotion conveyed by the artwork?
2. This art piece makes me feel sombre.
3. This art piece makes me feel content.
4. This art piece makes me feel calm.
5. This art piece makes me feel uneasy.
6. How many prominent colours do you notice in this painting?
7. How many objects caught your eye in the painting?

Text:
- `text_all` (TF-IDF on train fold only, then reduced with TruncatedSVD)

## How To Run
From repository root:

```bash
python3 gda/run_gda.py
```

## Fine-Tuning Guide
The GDA script uses iterative coarse-to-fine tuning of `reg_param`.

Hyperparameter to tune:
- `reg_param`: covariance regularization amount
  - lower values: less bias, can overfit or become numerically unstable
  - higher values: more stable, may underfit

Current tuning flow in `run_gda.py`:
1. Start from a coarse initial grid.
2. Run CV and select best `reg_param`.
3. Build a narrower grid around that winner.
4. Repeat for configurable rounds.

CLI tuning controls:
- `--n-rounds` (default 5)
- `--n-points` (default 13)
- `--no-text` (disable text features)
- `--tfidf-max-features` (default 10000)
- `--text-svd-components` (default 400)

Examples:

```bash
python3 gda/run_gda.py
python3 gda/run_gda.py --n-rounds 5 --n-points 13
python3 gda/run_gda.py --no-text
```

## Iterative Run Snapshot (2026-03-29)

| Setting | Best reg_param | CV Mean | CV Std | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|---:|
| Numeric + text (current default, 5 rounds, 13 points, 10000 TF-IDF, 400 SVD) | 0.108333 | 0.8233 | 0.0164 | 0.8466 | 0.8435 |
| Numeric only (previous baseline, 3 rounds, 9 points) | 0.345833 | 0.6667 | 0.0124 | 0.6755 | 0.6384 |


## Notes For Comparison Fairness
- Keep split seed, test size, and folds unchanged across all models.
- Keep feature columns identical for first-pass model comparison.
