"""
PyTorch MLP: numeric features (scaled + median imputed) + optional TF-IDF on text_all.
Same group splits as sample.py. TF-IDF is fit on train only each fold / on train_pool for holdout.
Requires: pip install torch scikit-learn
"""
import argparse
import random
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from split_data import get_splits_with_test, load_dataframe

TARGET_COL = "Painting"
FEATURE_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
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

TEXT_COL = "text_all"


def extract_rating(response):
    if pd.isna(response):
        return None
    match = re.match(r"^(\d+)", str(response))
    return int(match.group(1)) if match else None


def prepare_model_dataframe(df: pd.DataFrame, use_text: bool) -> pd.DataFrame:
    df_processed = df.copy()
    for col in LIKERT_COLS:
        df_processed[col] = df_processed[col].apply(extract_rating)
    cols = ["unique_id", TARGET_COL] + FEATURE_COLS
    if use_text and TEXT_COL in df_processed.columns:
        cols.append(TEXT_COL)
    df_model = df_processed[cols].copy()
    df_model = df_model[df_model[TARGET_COL].notna()].reset_index(drop=True)
    if use_text and TEXT_COL in df_model.columns:
        df_model[TEXT_COL] = df_model[TEXT_COL].fillna("").astype(str)
    return df_model


def impute_median(
    train_df: pd.DataFrame, other_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    med = train_df[feature_cols].median(numeric_only=True)
    x_train = train_df[feature_cols].fillna(med).to_numpy(dtype=np.float32)
    x_other = other_df[feature_cols].fillna(med).to_numpy(dtype=np.float32)
    return x_train, x_other


def _tfidf_for_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_text = train_df[text_col].fillna("").astype(str)
    val_text = val_df[text_col].fillna("").astype(str)
    min_df = 1 if len(train_df) < 80 else 2
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
    )
    tf_tr = vectorizer.fit_transform(train_text)
    tf_va = vectorizer.transform(val_text)
    return tf_tr.toarray().astype(np.float32), tf_va.toarray().astype(np.float32)


def _tfidf_for_holdout(
    train_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_text = train_pool[text_col].fillna("").astype(str)
    test_text = test_df[text_col].fillna("").astype(str)
    min_df = 1 if len(train_pool) < 80 else 2
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
    )
    tf_pool = vectorizer.fit_transform(train_text)
    tf_te = vectorizer.transform(test_text)
    return tf_pool.toarray().astype(np.float32), tf_te.toarray().astype(np.float32)


def arrays_for_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    le: LabelEncoder,
    feature_cols: List[str],
    target_col: str,
    use_text: bool,
    tfidf_max_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_val = impute_median(train_df, val_df, feature_cols)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    if use_text and TEXT_COL in train_df.columns:
        xt_tr, xt_va = _tfidf_for_fold(train_df, val_df, TEXT_COL, tfidf_max_features)
        x_train = np.hstack([x_train, xt_tr]).astype(np.float32)
        x_val = np.hstack([x_val, xt_va]).astype(np.float32)
    y_train = le.transform(train_df[target_col])
    y_val = le.transform(val_df[target_col])
    return x_train, x_val, y_train, y_val


def arrays_for_holdout(
    train_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    le: LabelEncoder,
    feature_cols: List[str],
    target_col: str,
    use_text: bool,
    tfidf_max_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_pool, x_test = impute_median(train_pool, test_df, feature_cols)
    scaler = StandardScaler()
    x_pool = scaler.fit_transform(x_pool)
    x_test = scaler.transform(x_test)
    if use_text and TEXT_COL in train_pool.columns:
        xt_pool, xt_te = _tfidf_for_holdout(train_pool, test_df, TEXT_COL, tfidf_max_features)
        x_pool = np.hstack([x_pool, xt_pool]).astype(np.float32)
        x_test = np.hstack([x_test, xt_te]).astype(np.float32)
    y_pool = le.transform(train_pool[target_col])
    y_test = le.transform(test_df[target_col])
    return x_pool, x_test, y_pool, y_test


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_dict_cpu(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_state_dict(model: nn.Module, state: dict, device: torch.device):
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    hidden: int = 64,
    patience: Optional[int] = None,
    min_delta: float = 0.0,
) -> Tuple[float, int]:
    """
    Train with optional early stopping on validation **loss** (lower is better).
    If patience is None, run all `epochs` with no early stopping.
    Always restores weights from the epoch with lowest val loss when ES is enabled.
    Returns (val_accuracy, best_epoch) where best_epoch is the 1-based epoch index
    with lowest val loss (or total epochs if ES disabled).
    """
    in_dim = x_train.shape[1]
    model = MLP(in_dim, num_classes, hidden=hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train).long(),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    y_val_t = torch.from_numpy(y_val).long().to(device)
    x_val_t = torch.from_numpy(x_val).to(device)

    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    best_epoch_1based = 1
    stalled = 0
    use_es = patience is not None and patience > 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        if not use_es:
            continue

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = _state_dict_cpu(model)
            best_epoch_1based = epoch + 1
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break

    if use_es and best_state is not None:
        _load_state_dict(model, best_state, device)

    model.eval()
    with torch.no_grad():
        logits = model(x_val_t)
        pred = logits.argmax(dim=1).cpu().numpy()
    val_acc = float((pred == y_val).mean())
    epochs_report = best_epoch_1based if use_es else max(epochs, 1)
    return val_acc, epochs_report


def train_epochs_on_data(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="PyTorch MLP with group CV + hold-out test.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping: stop if val loss does not improve for this many epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum val-loss improvement to count as progress.",
    )
    parser.add_argument(
        "--final-monitor-size",
        type=float,
        default=0.1,
        help="Fraction of train_pool held out (stratified) to tune epochs for final refit on full train_pool.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Disable text_all TF-IDF; use numeric features only.",
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=3000,
        help="Max TF-IDF vocabulary size (per fold / holdout fit).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_dataframe("training_data_clean.csv")
    use_text = not args.no_text
    df_model = prepare_model_dataframe(df, use_text=use_text)

    df_train_pool, df_test, splits = get_splits_with_test(
        df_model,
        target_col=TARGET_COL,
        n_splits=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
    )

    le = LabelEncoder()
    le.fit(df_train_pool[TARGET_COL])

    num_classes = len(le.classes_)

    print(
        f"train_pool={len(df_train_pool)} test={len(df_test)} folds={len(splits)} "
        f"classes={num_classes} use_text={use_text} device={device}"
    )

    es_patience = args.patience if args.patience > 0 else None

    fold_accs = []
    for train_idx, val_idx in splits:
        train_fold = df_train_pool.iloc[train_idx]
        val_fold = df_train_pool.iloc[val_idx]
        x_tr, x_va, y_tr, y_va = arrays_for_fold(
            train_fold,
            val_fold,
            le,
            FEATURE_COLS,
            TARGET_COL,
            use_text,
            args.tfidf_max_features,
        )
        acc, _ = train_model(
            x_tr,
            y_tr,
            x_va,
            y_va,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            hidden=args.hidden,
            patience=es_patience,
            min_delta=args.min_delta,
        )
        fold_accs.append(acc)

    print(f"CV val acc: mean={np.mean(fold_accs):.4f} std={np.std(fold_accs):.4f}")

    x_pool, x_te, y_pool, y_test = arrays_for_holdout(
        df_train_pool,
        df_test,
        le,
        FEATURE_COLS,
        TARGET_COL,
        use_text,
        args.tfidf_max_features,
    )

    # Hold out a stratified slice of train_pool to pick epoch count via early stopping, then refit on full pool.
    x_fit, x_mon, y_fit, y_mon = train_test_split(
        x_pool,
        y_pool,
        test_size=args.final_monitor_size,
        stratify=y_pool,
        random_state=args.seed,
    )
    _, best_epochs = train_model(
        x_fit,
        y_fit,
        x_mon,
        y_mon,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        hidden=args.hidden,
        patience=es_patience,
        min_delta=args.min_delta,
    )
    best_epochs = max(best_epochs, 1)

    in_dim = x_pool.shape[1]
    print(f"Input dim (numeric + optional TF-IDF): {in_dim}")

    model = MLP(in_dim, num_classes, hidden=args.hidden).to(device)
    train_epochs_on_data(
        model,
        x_pool,
        y_pool,
        epochs=best_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_te).to(device))
        test_pred = logits.argmax(dim=1).cpu().numpy()
    test_acc = float((test_pred == y_test).mean())
    print(f"Final refit epochs (from monitor ES): {best_epochs}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
