"""
One-off export: train MLP on full train_pool with fixed hyperparameters (matching
your Chosen-by-CV line) and save numpy artifacts for pred_example.py (numpy/pandas only).

Requires: torch, scikit-learn (not needed at prediction time).

Usage (from project root):
  python export_mlp_params.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from neural_network import (
    FEATURE_COLS,
    LIKERT,
    MLP,
    TARGET,
    TEXT,
    fit_epochs,
    likert_num,
    prep_df,
    set_seed,
)
from split_data import get_splits_with_test, load_dataframe

# Chosen-by-CV hyperparameters (edit if you retrain with different values)
LR = 0.003
HIDDEN = 32
N_HIDDEN_LAYERS = 1
REFIT_EPOCHS = 21
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
SEED = 42
TFIDF_MAX_FEATURES = 5000
USE_TEXT = True

EXPORT_DIR = Path(__file__).resolve().parent / "mlp_export"


def featurize_and_save_artifacts(train_df: pd.DataFrame, other_df: pd.DataFrame, tfidf_max: int):
    """Same logic as neural_network.featurize but returns scaler, med, vectorizer."""
    med = train_df[FEATURE_COLS].median(numeric_only=True)
    x0 = train_df[FEATURE_COLS].fillna(med).to_numpy(np.float32)
    x1 = other_df[FEATURE_COLS].fillna(med).to_numpy(np.float32)
    sc = StandardScaler()
    x0, x1 = sc.fit_transform(x0), sc.transform(x1)
    vectorizer = None
    if USE_TEXT and TEXT in train_df.columns:
        min_df = 1 if len(train_df) < 80 else 2
        vectorizer = TfidfVectorizer(
            max_features=tfidf_max,
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=True,
        )
        train_text = train_df[TEXT].fillna("").astype(str)
        other_text = other_df[TEXT].fillna("").astype(str)
        t0 = vectorizer.fit_transform(train_text)
        t1 = vectorizer.transform(other_text)
        x0 = np.hstack([x0, t0.toarray()]).astype(np.float32)
        x1 = np.hstack([x1, t1.toarray()]).astype(np.float32)
    else:
        x0, x1 = x0.astype(np.float32), x1.astype(np.float32)
    return x0, x1, med.to_numpy(np.float64), sc.mean_.astype(np.float64), sc.scale_.astype(np.float64), vectorizer


def validate_numpy_tfidf(vectorizer: TfidfVectorizer, texts, atol=1e-5):
    """Ensure pred_example's TF-IDF matches sklearn on sample strings."""
    from pred_example import tfidf_matrix_numpy  # noqa: circular import at runtime

    ref = vectorizer.transform(texts).toarray().astype(np.float64)
    terms = vectorizer.get_feature_names_out()
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    got = tfidf_matrix_numpy(texts, terms, idf)
    if not np.allclose(ref, got, atol=atol):
        diff = np.abs(ref - got).max()
        raise AssertionError(f"TF-IDF mismatch max abs diff={diff}")
    print("TF-IDF numpy vs sklearn: OK (max diff check passed).")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = prep_df(load_dataframe("training_data_clean.csv"), use_text=USE_TEXT)
    pool, _test, _splits = get_splits_with_test(
        df, target_col=TARGET, n_splits=5, test_size=0.2, seed=SEED
    )
    le = LabelEncoder()
    le.fit(pool[TARGET])

    x_pool, _, med, sc_mean, sc_scale, vec = featurize_and_save_artifacts(
        pool, pool, TFIDF_MAX_FEATURES
    )
    y_pool = le.transform(pool[TARGET])
    n_class = len(le.classes_)
    in_dim = x_pool.shape[1]

    if in_dim != 5007:
        print(f"Warning: input_dim={in_dim} (expected 5007 with default data/settings).")

    # TF-IDF sanity check
    sample = pool[TEXT].fillna("").astype(str).head(20).tolist()
    validate_numpy_tfidf(vec, sample)

    model = MLP(in_dim, n_class, HIDDEN, N_HIDDEN_LAYERS, DROPOUT).to(device)
    set_seed(SEED)
    fit_epochs(
        model,
        x_pool,
        y_pool,
        REFIT_EPOCHS,
        BATCH_SIZE,
        LR,
        device,
        WEIGHT_DECAY,
    )
    model.eval()
    sd = model.state_dict()
    W0 = sd["net.0.weight"].detach().cpu().numpy().astype(np.float32)
    b0 = sd["net.0.bias"].detach().cpu().numpy().astype(np.float32)
    W1 = sd["net.3.weight"].detach().cpu().numpy().astype(np.float32)
    b1 = sd["net.3.bias"].detach().cpu().numpy().astype(np.float32)

    terms = np.asarray(vec.get_feature_names_out(), dtype=object)
    idf = np.asarray(vec.idf_, dtype=np.float64)
    classes = np.asarray(le.classes_, dtype=object)
    feature_cols = np.asarray(FEATURE_COLS, dtype=object)

    out = EXPORT_DIR / "mlp_export.npz"
    np.savez_compressed(
        out,
        W0=W0,
        b0=b0,
        W1=W1,
        b1=b1,
        median=med,
        scaler_mean=sc_mean,
        scaler_scale=sc_scale,
        tfidf_terms=terms,
        tfidf_idf=idf,
        classes=classes,
        feature_cols=feature_cols,
        use_text=np.array([USE_TEXT]),
        input_dim=np.array([in_dim]),
        n_classes=np.array([n_class]),
        lr=np.array([LR]),
        hidden=np.array([HIDDEN]),
        n_hidden_layers=np.array([N_HIDDEN_LAYERS]),
        refit_epochs=np.array([REFIT_EPOCHS]),
        dropout=np.array([DROPOUT]),
        weight_decay=np.array([WEIGHT_DECAY]),
    )
    print(f"Saved: {out}")
    print(f"  input_dim={in_dim}  n_classes={n_class}  classes={list(classes)}")


if __name__ == "__main__":
    main()
