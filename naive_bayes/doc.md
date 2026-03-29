# Naive Bayes Model Notes (Numeric Features)

## Goal
Build and communicate a Gaussian Naive Bayes baseline that follows the same split and leakage-control protocol as other team models so results are directly comparable.

## Files In This Folder
- `run_gnb.py`: trains and evaluates Gaussian Naive Bayes using the shared CV and holdout flow.
- `doc.md`: Naive Bayes documentation.

## Why Gaussian Naive Bayes
- Our first-pass model uses numeric features only.
- GaussianNB is a simple probabilistic baseline that is easy to explain to a group audience.
- It is fast to train and gives class probabilities for interpretation.

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

## Feature Set (Numeric)
1. On a scale of 1–10, how intense is the emotion conveyed by the artwork?
2. This art piece makes me feel sombre.
3. This art piece makes me feel content.
4. This art piece makes me feel calm.
5. This art piece makes me feel uneasy.
6. How many prominent colours do you notice in this painting?
7. How many objects caught your eye in the painting?

## How To Run
From repository root:

```bash
python3 naive_bayes/run_gnb.py
```

For GDA documentation and script, see `gda/doc.md` and `gda/run_gda.py`.

## Output You Should Record
- Fold accuracies for the selected setting
- CV mean accuracy and CV standard deviation
- Holdout test metrics:
  - Accuracy
  - Macro precision
  - Macro recall
  - Macro F1
- Classification report by class

## Results Table Template
Fill this after running the script.

| Metric | Value |
|---|---:|
| Best var_smoothing |  |
| CV Mean Accuracy |  |
| CV Std |  |
| Test Accuracy |  |
| Test Macro Precision |  |
| Test Macro Recall |  |
| Test Macro F1 |  |

## Initial Baseline Run (2026-03-28)

From running `python3 naive_bayes/run_gnb.py`:

| Metric | Value |
|---|---:|
| Best var_smoothing | 1e-12 |
| CV Fold Accuracy (1-5) | 0.6926, 0.6852, 0.6630, 0.6629, 0.6852 |
| CV Mean Accuracy | 0.6778 |
| CV Std | 0.0124 |
| Test Accuracy | 0.6903 |
| Test Macro Precision | 0.6966 |
| Test Macro Recall | 0.6903 |
| Test Macro F1 | 0.6511 |

Class-level note: `The Starry Night` has lower recall than the other two classes in this first run. This is a useful talking point when comparing Naive Bayes versus KNN.

## Suggested Group Presentation Flow
1. Problem framing: 3-class painting classification from survey responses.
2. Data protocol: shared grouped split and leakage-safe imputation.
3. Model idea: Naive Bayes assumptions and why this is a strong baseline.
4. Results: CV stability plus final test performance.
5. Comparison: where NB is stronger/weaker versus KNN baseline.
6. Next step: optional text feature extension with TF-IDF + MultinomialNB.

## Notes For Comparison Fairness
- Keep split seed, test size, and folds unchanged across all models.
- Keep feature columns identical for first-pass model comparison.
