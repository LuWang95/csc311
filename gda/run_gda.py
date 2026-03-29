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
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# Ensure project-root imports work even when running this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from split_data import get_splits_with_test, load_dataframe


TEXT_COL = "text_all"


def extract_rating(response):
    if pd.isna(response):
        return None
    match = re.match(r"^(\d+)", str(response))
    return int(match.group(1)) if match else None


def build_features_with_train_fit(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    feature_cols: List[str],
    use_text: bool,
    tfidf_max_features: int,
    text_svd_components: int,
):
    # Numeric path: fold-safe median imputation + scaling fit only on train.
    med = train_df[feature_cols].median(numeric_only=True)
    x_train_num = train_df[feature_cols].fillna(med).to_numpy(dtype=np.float32)
    x_other_num = other_df[feature_cols].fillna(med).to_numpy(dtype=np.float32)
    num_scaler = StandardScaler()
    x_train_num = num_scaler.fit_transform(x_train_num)
    x_other_num = num_scaler.transform(x_other_num)

    if not use_text or TEXT_COL not in train_df.columns:
        return x_train_num.astype(np.float32), x_other_num.astype(np.float32)

    # Text path: fit TF-IDF and SVD on train only, then transform other.
    train_text = train_df[TEXT_COL].fillna("").astype(str)
    other_text = other_df[TEXT_COL].fillna("").astype(str)

    min_df = 1 if len(train_df) < 80 else 2
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
    )
    x_train_text_sparse = vectorizer.fit_transform(train_text)
    x_other_text_sparse = vectorizer.transform(other_text)

    n_vocab = x_train_text_sparse.shape[1]
    n_comp = min(text_svd_components, max(0, n_vocab - 1))

    if n_comp >= 2:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        x_train_text = svd.fit_transform(x_train_text_sparse)
        x_other_text = svd.transform(x_other_text_sparse)
    else:
        x_train_text = x_train_text_sparse.toarray()
        x_other_text = x_other_text_sparse.toarray()

    text_scaler = StandardScaler()
    x_train_text = text_scaler.fit_transform(x_train_text)
    x_other_text = text_scaler.transform(x_other_text)

    x_train = np.hstack([x_train_num, x_train_text]).astype(np.float32)
    x_other = np.hstack([x_other_num, x_other_text]).astype(np.float32)
    return x_train, x_other


def evaluate_reg_param_grid(
    df_train_pool: pd.DataFrame,
    splits,
    feature_cols: List[str],
    target_col: str,
    reg_param_values: List[float],
    use_text: bool,
    tfidf_max_features: int,
    text_svd_components: int,
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

            x_train, x_val = build_features_with_train_fit(
                train_df=train_fold,
                other_df=val_fold,
                feature_cols=feature_cols,
                use_text=use_text,
                tfidf_max_features=tfidf_max_features,
                text_svd_components=text_svd_components,
            )
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
    use_text: bool,
    tfidf_max_features: int,
    text_svd_components: int,
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
            use_text=use_text,
            tfidf_max_features=tfidf_max_features,
            text_svd_components=text_svd_components,
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
    parser.add_argument("--n-rounds", type=int, default=5, help="Number of iterative tuning rounds.")
    parser.add_argument(
        "--n-points",
        type=int,
        default=13,
        help="Number of refinement points per round (coarse-to-fine grid density).",
    )
    parser.add_argument("--no-text", action="store_true", help="Disable text features and use numeric-only GDA.")
    parser.add_argument("--tfidf-max-features", type=int, default=10000)
    parser.add_argument("--text-svd-components", type=int, default=400)
    args = parser.parse_args()
    use_text = not args.no_text

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

    if TEXT_COL in df_processed.columns:
        df_processed[TEXT_COL] = df_processed[TEXT_COL].fillna("").astype(str)

    model_cols = ["unique_id", target_col] + feature_cols
    if use_text and TEXT_COL in df_processed.columns:
        model_cols.append(TEXT_COL)

    df_model = df_processed[model_cols].copy()
    df_model = df_model[df_model[target_col].notna()].reset_index(drop=True)

    df_train_pool, df_test, splits = get_splits_with_test(
        df_model,
        target_col=target_col,
        n_splits=5,
        test_size=0.2,
        seed=42,
    )

    mode = "numeric+text" if use_text and TEXT_COL in df_model.columns else "numeric-only"
    print(f"=== Gaussian Discriminant Analysis (QDA, {mode}) ===")
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
        use_text=use_text and TEXT_COL in df_model.columns,
        tfidf_max_features=args.tfidf_max_features,
        text_svd_components=args.text_svd_components,
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

    x_pool, x_test = build_features_with_train_fit(
        train_df=df_train_pool,
        other_df=df_test,
        feature_cols=feature_cols,
        use_text=use_text and TEXT_COL in df_model.columns,
        tfidf_max_features=args.tfidf_max_features,
        text_svd_components=args.text_svd_components,
    )
    y_pool = df_train_pool[target_col].to_numpy()
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
