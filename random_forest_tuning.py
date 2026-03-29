"""
Random Forest hyperparameter tuning via 5-fold CV.
Run this first to find best params, then update random_forest_final.py.

Usage:
    python random_forest/tune_rf.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from split_data import get_splits_with_test, load_dataframe

TARGET    = "Painting"
TEXT_COL  = "text_all"
TFIDF_MAX = 3000

FEATURE_COLS = [
    "On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?",
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
]
LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

def extract_rating(x):
    if pd.isna(x): return None
    m = re.match(r"^(\d+)", str(x))
    return int(m.group(1)) if m else None

def build_features(train_df, other_df, fit_objs=None):
    med = train_df[FEATURE_COLS].median(numeric_only=True)
    x0 = train_df[FEATURE_COLS].fillna(med).to_numpy(np.float32)
    x1 = other_df[FEATURE_COLS].fillna(med).to_numpy(np.float32)
    if fit_objs is None:
        sc = StandardScaler()
        x0 = sc.fit_transform(x0)
        vec = TfidfVectorizer(max_features=TFIDF_MAX, ngram_range=(1,2), min_df=2, sublinear_tf=True)
        t0 = vec.fit_transform(train_df[TEXT_COL]).toarray().astype(np.float32)
        fit_objs = (sc, vec)
    else:
        sc, vec = fit_objs
        x0 = sc.transform(x0)
        t0 = vec.transform(train_df[TEXT_COL]).toarray().astype(np.float32)
    x1 = sc.transform(x1)
    t1 = vec.transform(other_df[TEXT_COL]).toarray().astype(np.float32)
    return np.hstack([x0, t0]), np.hstack([x1, t1]), fit_objs

def main():
    df = load_dataframe("training_data_clean.csv")
    for col in LIKERT_COLS:
        df[col] = df[col].apply(extract_rating)

    cols = ["unique_id", TARGET] + FEATURE_COLS + [TEXT_COL]
    df_model = df[cols].loc[df[TARGET].notna()].reset_index(drop=True)
    df_model[TEXT_COL] = df_model[TEXT_COL].fillna("").astype(str)

    df_pool, _, splits = get_splits_with_test(
        df_model, target_col=TARGET, n_splits=5, test_size=0.2, seed=42
    )

    n_estimators_opts = [100, 200]
    max_depth_opts    = [10, 15, None]
    min_leaf_opts     = [3, 5]

    best_score, best_params = -1, None

    for n_est in n_estimators_opts:
        for max_d in max_depth_opts:
            for min_leaf in min_leaf_opts:
                fold_f1s = []
                for train_idx, val_idx in splits:
                    tr, va = df_pool.iloc[train_idx], df_pool.iloc[val_idx]
                    X_tr, X_va, _ = build_features(tr, va)
                    clf = RandomForestClassifier(
                        n_estimators=n_est, max_depth=max_d, min_samples_leaf=min_leaf,
                        max_features="sqrt", class_weight="balanced",
                        random_state=42, n_jobs=-1
                    )
                    clf.fit(X_tr, tr[TARGET].to_numpy())
                    preds = clf.predict(X_va)
                    fold_f1s.append(f1_score(va[TARGET].to_numpy(), preds, average="macro"))

                avg_f1 = float(np.mean(fold_f1s))
                print(
                    f"n_estimators={n_est}, max_depth={str(max_d):<5}, "
                    f"min_samples_leaf={min_leaf}, avg_macro_f1={avg_f1:.4f}"
                )
                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_params = (n_est, max_d, min_leaf)

    print("\nBest parameters found:")
    print(f"  n_estimators     = {best_params[0]}")
    print(f"  max_depth        = {best_params[1]}")
    print(f"  min_samples_leaf = {best_params[2]}")
    print(f"  Best CV Macro F1 = {best_score:.4f}")

if __name__ == "__main__":
    main()