"""
Offline prediction using exported MLP weights (numpy only at inference).

1) Train & export once (needs torch + sklearn):
     python export_mlp_params.py
   This writes mlp_export/mlp_export.npz next to this file.

2) predict_all() loads that bundle and only uses numpy + pandas (+ stdlib re).

Course `pred.py` should expose predict_all(filename) the same way.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# sklearn default token pattern for word analyzer (min 2 word chars)
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

TARGET = "Painting"
LIKERT = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]
TEXT = "text_all"


def likert_num(x):
    if pd.isna(x):
        return None
    m = re.match(r"^(\d+)", str(x))
    return int(m.group(1)) if m else None


def tfidf_matrix_numpy(texts: list, terms: np.ndarray, idf: np.ndarray) -> np.ndarray:
    """
    Match sklearn TfidfVectorizer(sublinear_tf=True, norm='l2') on fitted vocabulary.
    terms[i] is i-th column name; idf same length as terms.
    """
    vocab = {str(terms[i]): i for i in range(len(terms))}
    n_terms = len(terms)
    n_doc = len(texts)
    out = np.zeros((n_doc, n_terms), dtype=np.float64)
    for di, raw in enumerate(texts):
        tokens = _TOKEN_RE.findall(str(raw).lower())
        counts: dict[str, float] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0.0) + 1.0
        for j in range(len(tokens) - 1):
            bg = f"{tokens[j]} {tokens[j + 1]}"
            counts[bg] = counts.get(bg, 0.0) + 1.0
        row = np.zeros(n_terms, dtype=np.float64)
        for term, c in counts.items():
            idx = vocab.get(term)
            if idx is None or c <= 0:
                continue
            # sklearn sublinear_tf: 1 + log(tf)
            row[idx] = 1.0 + np.log(c)
        row *= idf
        nrm = np.linalg.norm(row)
        if nrm > 0:
            row /= nrm
        out[di] = row
    return out


def _load_bundle():
    path = Path(__file__).resolve().parent / "mlp_export" / "mlp_export.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python export_mlp_params.py (from project root)."
        )
    return np.load(path, allow_pickle=True)


def prep_df_predict(df: pd.DataFrame, use_text: bool, feature_cols: list) -> pd.DataFrame:
    """Like training prep but keeps rows when TARGET is missing (test file)."""
    d = df.copy()
    for c in LIKERT:
        if c in d.columns:
            d[c] = d[c].apply(likert_num)
    cols = ["unique_id"] + list(feature_cols)
    if use_text:
        if TEXT not in d.columns:
            d[TEXT] = ""
        cols.append(TEXT)
    else:
        cols = [c for c in cols if c != TEXT]
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    out = d[cols].copy()
    if TEXT in out.columns:
        out[TEXT] = out[TEXT].fillna("").astype(str)
    return out


def featurize_rows(
    rows: pd.DataFrame,
    median: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    terms: np.ndarray,
    idf: np.ndarray,
    feature_cols: list,
    use_text: bool,
) -> np.ndarray:
    med_series = pd.Series(median, index=feature_cols)
    x = rows[feature_cols].fillna(med_series).to_numpy(dtype=np.float32)
    x = (x - scaler_mean) / scaler_scale
    if use_text:
        texts = rows[TEXT].fillna("").astype(str).tolist()
        tf = tfidf_matrix_numpy(texts, terms, idf).astype(np.float32)
        x = np.hstack([x, tf])
    return x


def mlp_forward(X: np.ndarray, W0: np.ndarray, b0: np.ndarray, W1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Eval mode: no dropout. Matches PyTorch Linear stacking."""
    h = np.maximum(0.0, X @ W0.T + b0)
    logits = h @ W1.T + b1
    return logits


def predict_all(filename: str | Path) -> list:
    """
    Load CSV (same schema as training / cleaned data), return list of Painting class strings.
    """
    bundle = _load_bundle()
    W0 = bundle["W0"]
    b0 = bundle["b0"]
    W1 = bundle["W1"]
    b1 = bundle["b1"]
    median = bundle["median"]
    scaler_mean = bundle["scaler_mean"]
    scaler_scale = bundle["scaler_scale"]
    terms = bundle["tfidf_terms"]
    idf = bundle["tfidf_idf"]
    classes = bundle["classes"]
    feature_cols = list(bundle["feature_cols"])
    use_text = bool(bundle["use_text"][0])

    df = pd.read_csv(filename)
    rows = prep_df_predict(df, use_text, feature_cols)
    X = featurize_rows(rows, median, scaler_mean, scaler_scale, terms, idf, feature_cols, use_text)
    logits = mlp_forward(X, W0, b0, W1, b1)
    pred_idx = np.argmax(logits, axis=1)
    return [str(classes[i]) for i in pred_idx]


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "training_data_clean.csv"
    preds = predict_all(csv_path)
    print(f"predictions ({len(preds)} rows), first 5:", preds[:5])
