# Training workflow (splits, imputation, multiple models)

When four people each implement a model, **use the same split and imputation rules** so results are comparable.

## 1. Where the data comes from

1. Raw: `training_data_202601.csv`
2. Cleaning: run `python cleaning.py` → `training_data_clean.csv` (logic in `data_cleaning_utils.py`)

## 2. Splits: `split_data.py`

- Function: `get_splits_with_test(df, target_col="Painting", n_splits=5, test_size=0.2, seed=42)`
- It returns three values:
  - **`df_trainval`**: training pool (used for K-fold cross-validation)
  - **`df_test`**: held-out test set (**not** used for tuning; evaluate once at the end)
  - **`splits`**: a `list` of length `n_splits`; each item is `(train_idx, val_idx)` with indices into **`df_trainval`** (0-based row positions)
- **Grouped by respondent**: rows with the same `unique_id` belong to one group. `GroupShuffleSplit` + `StratifiedGroupKFold` keep a person from appearing in both train and test, or in both train and validation within the same fold.

**Do not `dropna()` on all features before splitting**—that breaks “three paintings per person” groups. Only drop rows where the label **`Painting`** is missing (see `sample.py` / `training_utils.py`).

## 3. Missing-value imputation (numeric features)

Rule: **compute medians on the training part of the current fold only**, then impute that fold’s train and val. For the final hold-out evaluation, use **medians from the full `train_pool`** to fill both `train_pool` and `test`, so test statistics never leak into training.

Reference implementations:

- `impute_with_train_median(train_fold, val_fold, feature_cols)` in `sample.py`
- Same helper plus `evaluate_cv` / `evaluate_holdout` in `training_utils.py`

`pandas` `fillna(med)` **only replaces NaN**; observed values are unchanged.

## 4. Adding your own model (follow `sample.py`)

Minimal steps:

1. `load_dataframe("training_data_clean.csv")` (or `training_utils.load_and_prepare()`)
2. Match `sample.py`: Likert via `extract_rating`, define `feature_cols`, build `df_model` (include `unique_id`), drop only rows with missing `Painting`
3. `df_train_pool, df_test, splits = get_splits_with_test(df_model, ...)`
4. **CV**: for each hyperparameter (e.g. `k`) or fixed settings, loop over `splits`:
   - `train_fold = df_train_pool.iloc[train_idx]`
   - `val_fold = df_train_pool.iloc[val_idx]`
   - `X_train, X_val = impute_with_train_median(train_fold, val_fold, feature_cols)`
   - `model.fit(X_train, y_train)`, score on `X_val`
5. **Final test**: impute `train_pool` and `test` with `train_pool` medians, fit on all of `train_pool`, score on `test` (same as the end of `sample.py`)

## 5. Optional scripts

| File | Role |
|------|------|
| `sample.py` | KNN + CV for `k` + test (**best starting point**) |
| `training_utils.py` | Shared helpers: prepare data, `make_splits`, imputation, `evaluate_cv` / `evaluate_holdout` |
| `run_knn.py` / `run_logreg.py` / `run_svm.py` / `run_gnb.py` | Run one model each; all use the same `training_utils` |

## 6. Text features (if your model uses `text_all`)

- Use **`fillna("")`** for missing text, not the median.
- Put TF-IDF (etc.) in a **Pipeline** and **`fit` on train only** each fold, then `transform` val (avoids leakage). You can add a separate `*_text.py` that mirrors the structure of `training_utils`.

## 7. Checklist before submission

- [ ] Everyone uses the same `test_size`, `seed`, and `n_splits` (or reads the same saved split files)
- [ ] No feature `dropna()` before split that breaks groups
- [ ] Imputation statistics come only from train (per fold or full `train_pool`), never from `test`
- [ ] `test` is used only for the final evaluation (or as required by the course)
