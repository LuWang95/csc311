"""
Gaussian Discriminant Analysis baseline using the shared split and imputation protocol.

Implementation note:
- This script uses Quadratic Discriminant Analysis (QDA) from scikit-learn,
  which is the class-conditional Gaussian model with class-specific covariance.

Usage:
    python3 gda/run_gda.py
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

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


def evaluate_reg_param_grid(
    df_train_pool: pd.DataFrame,
    splits,
    feature_cols: List[str],
    target_col: str,
    reg_param_values: List[float],
) -> Dict[str, object]:
    best = {
        "reg_param": None,
        "cv_mean": -1.0,
        "cv_std": 0.0,
        "fold_scores": [],
    }

    results = []

    for reg_param in reg_param_values:
        fold_scores = []

        for train_idx, val_idx in splits:
            train_fold = df_train_pool.iloc[train_idx]
            val_fold = df_train_pool.iloc[val_idx]

            x_train, x_val = impute_with_train_median(train_fold, val_fold, feature_cols)
            y_train = train_fold[target_col].to_numpy()
            y_val = val_fold[target_col].to_numpy()

            model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
            try:
                model.fit(x_train, y_train)
                preds = model.predict(x_val)
                fold_scores.append(accuracy_score(y_val, preds))
            except ValueError:
                # Skip settings that are numerically unstable for this fold.
                fold_scores.append(np.nan)

        fold_scores_np = np.array(fold_scores, dtype=float)
        valid_scores = fold_scores_np[np.isfinite(fold_scores_np)]
        if len(valid_scores) == 0:
            continue

        cv_mean = float(np.mean(valid_scores))
        cv_std = float(np.std(valid_scores))
        results.append({"reg_param": reg_param, "cv_mean": cv_mean, "cv_std": cv_std})

        if cv_mean > best["cv_mean"]:
            best = {
                "reg_param": reg_param,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "fold_scores": fold_scores,
            }

    best["results"] = sorted(results, key=lambda r: r["reg_param"])
    return best


def make_refined_grid(center: float, previous_grid: List[float], n_points: int = 9) -> List[float]:
    if len(previous_grid) < 2:
        return [center]

    span = max(previous_grid) - min(previous_grid)
    # Shrink search region each round for coarse-to-fine tuning.
    half_width = max(span / 6.0, 1e-4)
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    grid = np.linspace(low, high, n_points)
    return sorted({round(float(v), 6) for v in grid} | {round(float(center), 6)})


def iterative_tune_reg_param(
    df_train_pool: pd.DataFrame,
    splits,
    feature_cols: List[str],
    target_col: str,
    initial_grid: List[float],
    n_rounds: int = 3,
    n_points: int = 9,
) -> Dict[str, object]:
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")

    current_grid = sorted({round(float(v), 6) for v in initial_grid if 0.0 <= float(v) <= 1.0})
    if not current_grid:
        raise ValueError("initial_grid must contain at least one value in [0, 1]")

    history = []
    overall_best = {
        "reg_param": None,
        "cv_mean": -1.0,
        "cv_std": 0.0,
        "fold_scores": [],
        "round": None,
    }

    for round_idx in range(1, n_rounds + 1):
        round_best = evaluate_reg_param_grid(
            df_train_pool=df_train_pool,
            splits=splits,
            feature_cols=feature_cols,
            target_col=target_col,
            reg_param_values=current_grid,
        )
        if round_best["reg_param"] is None:
            break

        history.append(
            {
                "round": round_idx,
                "grid": current_grid,
                "best_reg_param": round_best["reg_param"],
                "cv_mean": round_best["cv_mean"],
                "cv_std": round_best["cv_std"],
                "results": round_best["results"],
            }
        )

        if round_best["cv_mean"] > overall_best["cv_mean"]:
            overall_best = {
                "reg_param": round_best["reg_param"],
                "cv_mean": round_best["cv_mean"],
                "cv_std": round_best["cv_std"],
                "fold_scores": round_best["fold_scores"],
                "round": round_idx,
            }

        # Refine around the current round winner for next round.
        current_grid = make_refined_grid(
            center=float(round_best["reg_param"]),
            previous_grid=current_grid,
            n_points=n_points,
        )

    overall_best["history"] = history
    return overall_best


def main():
    parser = argparse.ArgumentParser(description="Run iterative GDA (QDA) tuning and evaluation.")
    parser.add_argument("--n-rounds", type=int, default=3, help="Number of iterative tuning rounds.")
    parser.add_argument(
        "--n-points",
        type=int,
        default=9,
        help="Number of refinement points per round (coarse-to-fine grid density).",
    )
    args = parser.parse_args()

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

    print("=== Gaussian Discriminant Analysis (QDA, numeric features) ===")
    print(
        f"Rows: total={len(df_model)}, train_pool={len(df_train_pool)}, test={len(df_test)}, folds={len(splits)}"
    )

    # Iterative QDA regularization search for stability and generalization.
    initial_reg_param_grid = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1]
    best = iterative_tune_reg_param(
        df_train_pool=df_train_pool,
        splits=splits,
        feature_cols=feature_cols,
        target_col=target_col,
        initial_grid=initial_reg_param_grid,
        n_rounds=args.n_rounds,
        n_points=args.n_points,
    )

    if best["reg_param"] is None:
        raise RuntimeError("No valid GDA configuration found. Try increasing reg_param values.")

    print("\nIterative tuning summary")
    for round_info in best["history"]:
        print(
            f"Round {round_info['round']}: best reg_param={round_info['best_reg_param']} "
            f"| cv_mean={round_info['cv_mean']:.4f} | cv_std={round_info['cv_std']:.4f}"
        )

    print("\nCV results for selected setting")
    print(f"Best reg_param: {best['reg_param']} (from round {best['round']})")
    for i, score in enumerate(best["fold_scores"], start=1):
        score_str = "nan" if not np.isfinite(score) else f"{score:.4f}"
        print(f"Fold {i} accuracy: {score_str}")
    print(f"CV mean accuracy: {best['cv_mean']:.4f}")
    print(f"CV std: {best['cv_std']:.4f}")

    train_pool_med = df_train_pool[feature_cols].median(numeric_only=True)
    x_pool = df_train_pool[feature_cols].fillna(train_pool_med).to_numpy()
    y_pool = df_train_pool[target_col].to_numpy()
    x_test = df_test[feature_cols].fillna(train_pool_med).to_numpy()
    y_test = df_test[target_col].to_numpy()

    final_model = QuadraticDiscriminantAnalysis(reg_param=best["reg_param"])
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
