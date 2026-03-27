"""
Same flow as project_baseline.py (Likert parsing, same features, same KNN + k search).
Only change: data split uses split_data.get_splits_with_test instead of train_test_split.
"""
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from split_data import get_splits_with_test, load_dataframe


def extract_rating(response):
    if pd.isna(response):
        return None
    match = re.match(r"^(\d+)", str(response))
    return int(match.group(1)) if match else None


def impute_with_train_median(train_df, val_df, feature_cols):
    med = train_df[feature_cols].median(numeric_only=True)
    x_train = train_df[feature_cols].fillna(med).to_numpy()
    x_val = val_df[feature_cols].fillna(med).to_numpy()
    return x_train, x_val


def main():
    # 1. Load Data (same file as baseline; load_dataframe can fall back to raw + clean)
    df = load_dataframe("training_data_clean.csv")

    # 2. Define Features and Target
    target_col = "Painting"
    feature_cols = [
        "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
        "This art piece makes me feel sombre.",
        "This art piece makes me feel content.",
        "This art piece makes me feel calm.",
        "This art piece makes me feel uneasy.",
        "How many prominent colours do you notice in this painting?",
        "How many objects caught your eye in the painting?",
    ]

    # 3. Preprocessing (same as project_baseline.py)
    likert_cols = [
        "This art piece makes me feel sombre.",
        "This art piece makes me feel content.",
        "This art piece makes me feel calm.",
        "This art piece makes me feel uneasy.",
    ]

    df_processed = df.copy()
    for col in likert_cols:
        df_processed[col] = df_processed[col].apply(extract_rating)

    # Keep all rows before split (except missing target) so group integrity is preserved.
    df_model = df_processed[["unique_id", target_col] + feature_cols].copy()
    df_model = df_model[df_model[target_col].notna()].reset_index(drop=True)

    # 4. Split: use split_data (hold-out test + CV folds on train pool)
    # test_size=0.30 matches baseline's 30% test set.
    df_train_pool, df_test, splits = get_splits_with_test(
        df_model,
        target_col=target_col,
        n_splits=5,
        test_size=0.2,
        seed=42,
    )

    print(f"Data Splits: train_pool={len(df_train_pool)}, test={len(df_test)}, folds={len(splits)}")

    # 5. Find Best k using CV (average validation accuracy across folds)
    best_k = 1
    best_cv_mean = -1.0
    best_cv_std = 0.0

    for k in range(1, 31):
        fold_val_accs = []
        for train_idx, val_idx in splits:
            train_fold = df_train_pool.iloc[train_idx]
            val_fold = df_train_pool.iloc[val_idx]
            X_train, X_val = impute_with_train_median(train_fold, val_fold, feature_cols)
            y_train = train_fold[target_col].to_numpy()
            y_val = val_fold[target_col].to_numpy()

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            fold_val_accs.append(knn.score(X_val, y_val))

        cv_mean = float(np.mean(fold_val_accs))
        cv_std = float(np.std(fold_val_accs))

        if cv_mean > best_cv_mean:
            best_cv_mean = cv_mean
            best_cv_std = cv_std
            best_k = k

    print(f"Best k found: {best_k} | CV mean={best_cv_mean:.4f} std={best_cv_std:.4f}")

    # 6. Retrain on full train_pool and evaluate on held-out test set
    pool_med = df_train_pool[feature_cols].median(numeric_only=True)
    X_pool = df_train_pool[feature_cols].fillna(pool_med).to_numpy()
    y_pool = df_train_pool[target_col].to_numpy()
    X_test = df_test[feature_cols].fillna(pool_med).to_numpy()
    y_test = df_test[target_col].to_numpy()
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_pool, y_pool)
    test_acc = final_knn.score(X_test, y_test)

    print(f"Test Accuracy (trained on full train_pool): {test_acc:.4f}")


if __name__ == "__main__":
    main()
