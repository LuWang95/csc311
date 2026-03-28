"""
Gaussian Naive Bayes baseline using the shared split and imputation protocol.

Usage:
    python naive_bayes/run_gnb.py
"""

import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB

# Ensure project-root imports work even when running this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


def evaluate_var_smoothing_grid(
    df_train_pool: pd.DataFrame,
    splits,
    feature_cols: List[str],
    target_col: str,
    smoothing_values: List[float],
) -> Dict[str, object]:
    best = {
        "var_smoothing": None,
        "cv_mean": -1.0,
        "cv_std": 0.0,
        "fold_scores": [],
    }

    for smoothing in smoothing_values:
        fold_scores = []

        for train_idx, val_idx in splits:
            train_fold = df_train_pool.iloc[train_idx]
            val_fold = df_train_pool.iloc[val_idx]

            x_train, x_val = impute_with_train_median(train_fold, val_fold, feature_cols)
            y_train = train_fold[target_col].to_numpy()
            y_val = val_fold[target_col].to_numpy()

            model = GaussianNB(var_smoothing=smoothing)
            model.fit(x_train, y_train)
            preds = model.predict(x_val)
            fold_scores.append(accuracy_score(y_val, preds))

        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))

        if cv_mean > best["cv_mean"]:
            best = {
                "var_smoothing": smoothing,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "fold_scores": fold_scores,
            }

    return best


def main():
    df = load_dataframe("training_data_clean.csv")

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

    likert_cols = [
        "This art piece makes me feel sombre.",
        "This art piece makes me feel content.",
        "This art piece makes me feel calm.",
        "This art piece makes me feel uneasy.",
    ]

    df_processed = df.copy()
    for col in likert_cols:
        df_processed[col] = df_processed[col].apply(extract_rating)

    df_model = df_processed[["unique_id", target_col] + feature_cols].copy()
    df_model = df_model[df_model[target_col].notna()].reset_index(drop=True)

    df_train_pool, df_test, splits = get_splits_with_test(
        df_model,
        target_col=target_col,
        n_splits=5,
        test_size=0.2,
        seed=42,
    )

    print("=== Gaussian Naive Bayes (numeric features) ===")
    print(
        f"Rows: total={len(df_model)}, train_pool={len(df_train_pool)}, test={len(df_test)}, folds={len(splits)}"
    )

    # Lightweight hyperparameter search to stay comparable with other model scripts.
    smoothing_values = [10 ** e for e in range(-12, -5)]
    best = evaluate_var_smoothing_grid(
        df_train_pool=df_train_pool,
        splits=splits,
        feature_cols=feature_cols,
        target_col=target_col,
        smoothing_values=smoothing_values,
    )

    print("\nCV results for best setting")
    print(f"Best var_smoothing: {best['var_smoothing']:.0e}")
    for i, score in enumerate(best["fold_scores"], start=1):
        print(f"Fold {i} accuracy: {score:.4f}")
    print(f"CV mean accuracy: {best['cv_mean']:.4f}")
    print(f"CV std: {best['cv_std']:.4f}")

    train_pool_med = df_train_pool[feature_cols].median(numeric_only=True)
    x_pool = df_train_pool[feature_cols].fillna(train_pool_med).to_numpy()
    y_pool = df_train_pool[target_col].to_numpy()
    x_test = df_test[feature_cols].fillna(train_pool_med).to_numpy()
    y_test = df_test[target_col].to_numpy()

    final_model = GaussianNB(var_smoothing=best["var_smoothing"])
    final_model.fit(x_pool, y_pool)
    test_preds = final_model.predict(x_test)

    test_acc = accuracy_score(y_test, test_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test,
        test_preds,
        average="macro",
        zero_division=0,
    )

    print("\nHoldout test results")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test precision (macro): {precision_macro:.4f}")
    print(f"Test recall (macro): {recall_macro:.4f}")
    print(f"Test F1 (macro): {f1_macro:.4f}")

    print("\nClassification report (holdout test)")
    print(classification_report(y_test, test_preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
