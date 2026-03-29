"""
Random Forest final model — trains with best hyperparams, evaluates on
holdout test set, and saves rf_model.json to project root for pred.py.

Usage:
    python random_forest/random_forest_final.py
"""

import json
import re
import sys
from pathlib import Path
from sklearn.tree import _tree

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# ── UPDATE THESE after running tune_rf.py ────────────────────────────────────
BEST_PARAMS = {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 3}
# ─────────────────────────────────────────────────────────────────────────────

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
        fit_objs = (sc, vec, med)
    else:
        sc, vec, med = fit_objs
        x0 = sc.transform(x0)
        t0 = vec.transform(train_df[TEXT_COL]).toarray().astype(np.float32)
    x1 = sc.transform(x1)
    t1 = vec.transform(other_df[TEXT_COL]).toarray().astype(np.float32)
    return np.hstack([x0, t0]), np.hstack([x1, t1]), fit_objs

def export_tree(tree):
    t = tree.tree_
    def recurse(node):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            return {"leaf": True, "value": t.value[node][0].tolist()}
        return {"leaf": False, "feature": int(t.feature[node]),
                "threshold": float(t.threshold[node]),
                "left": recurse(t.children_left[node]),
                "right": recurse(t.children_right[node])}
    return recurse(0)

def main():
    df = load_dataframe("training_data_clean.csv")
    for col in LIKERT_COLS:
        df[col] = df[col].apply(extract_rating)

    cols = ["unique_id", TARGET] + FEATURE_COLS + [TEXT_COL]
    df_model = df[cols].loc[df[TARGET].notna()].reset_index(drop=True)
    df_model[TEXT_COL] = df_model[TEXT_COL].fillna("").astype(str)

    df_pool, df_test, splits = get_splits_with_test(
        df_model, target_col=TARGET, n_splits=5, test_size=0.2, seed=42
    )

    print("=== Random Forest Final ===")
    print(f"total={len(df_model)}, train_pool={len(df_pool)}, test={len(df_test)}")
    print(f"Best params: {BEST_PARAMS}")

    # ── CV with best params ───────────────────────────────────────────────────
    fold_scores = []
    for train_idx, val_idx in splits:
        tr, va = df_pool.iloc[train_idx], df_pool.iloc[val_idx]
        X_tr, X_va, _ = build_features(tr, va)
        clf = RandomForestClassifier(**BEST_PARAMS, max_features="sqrt",
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_tr, tr[TARGET].to_numpy())
        preds = clf.predict(X_va)
        fold_scores.append(accuracy_score(va[TARGET].to_numpy(), preds))

    print("\nCV results")
    for i, s in enumerate(fold_scores, 1):
        print(f"  Fold {i} accuracy: {s:.4f}")
    print(f"  CV mean: {np.mean(fold_scores):.4f}  std: {np.std(fold_scores):.4f}")

    # ── Final model on full pool → evaluate on test ───────────────────────────
    X_pool, X_test, fit_objs = build_features(df_pool, df_test)
    y_pool = df_pool[TARGET].to_numpy()
    y_test = df_test[TARGET].to_numpy()

    le = LabelEncoder()
    y_pool_enc = le.fit_transform(y_pool)

    final_clf = RandomForestClassifier(**BEST_PARAMS, max_features="sqrt",
                                       class_weight="balanced", random_state=42, n_jobs=-1)
    final_clf.fit(X_pool, y_pool)
    test_preds = final_clf.predict(X_test)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
    print("\nHoldout test results")
    print(f"  Accuracy:          {accuracy_score(y_test, test_preds):.4f}")
    print(f"  Precision (macro): {prec:.4f}")
    print(f"  Recall (macro):    {rec:.4f}")
    print(f"  F1 (macro):        {f1:.4f}")
    print("\nClassification report")
    print(classification_report(y_test, test_preds, digits=4, zero_division=0))

    # ── Save rf_model.json for pred.py ────────────────────────────────────────
    sc, vec, med = fit_objs
    forest_data = {
        "trees":           [export_tree(e) for e in final_clf.estimators_],
        "n_classes":       int(len(le.classes_)),
        "classes":         le.classes_.tolist(),
        "tfidf_vocab":     {k: int(v) for k, v in vec.vocabulary_.items()},
        "tfidf_idf":       vec.idf_.tolist(),
        "tfidf_max":       TFIDF_MAX,
        "scaler_mean":     sc.mean_.tolist(),
        "scaler_scale":    sc.scale_.tolist(),
        "feature_medians": med.tolist(),
    }

    out_path = ROOT_DIR / "rf_model.json"
    with open(out_path, "w") as f:
        json.dump(forest_data, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved rf_model.json ({size_mb:.2f} MB)")
    if size_mb > 9.5:
        print("WARNING: approaching 10 MB limit!")

if __name__ == "__main__":
    main()