"""
PyTorch MLP: numeric (median + scale) + optional TF-IDF on text_all.
Group splits via split_data. CV uses early stopping; final model uses a small
monitor split on train_pool to pick epoch count, then refits on ALL train_pool
before evaluating test (same idea as the longer original script).

Requires: torch, scikit-learn
"""
import argparse
import random
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from split_data import get_splits_with_test, load_dataframe

TARGET = "Painting"
FEATURE_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
]
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


def prep_df(raw: pd.DataFrame, use_text: bool) -> pd.DataFrame:
    d = raw.copy()
    for c in LIKERT:
        d[c] = d[c].apply(likert_num)
    cols = ["unique_id", TARGET] + FEATURE_COLS
    if use_text and TEXT in d.columns:
        cols.append(TEXT)
    out = d[cols].loc[d[TARGET].notna()].reset_index(drop=True)
    if TEXT in out.columns:
        out[TEXT] = out[TEXT].fillna("").astype(str)
    return out


def featurize(train_df: pd.DataFrame, other_df: pd.DataFrame, use_text: bool, tfidf_max: int):
    """Median + scale on train_df; TF-IDF fit on train_df only."""
    med = train_df[FEATURE_COLS].median(numeric_only=True)
    x0 = train_df[FEATURE_COLS].fillna(med).to_numpy(np.float32)
    x1 = other_df[FEATURE_COLS].fillna(med).to_numpy(np.float32)
    sc = StandardScaler()
    x0, x1 = sc.fit_transform(x0), sc.transform(x1)
    if use_text and TEXT in train_df.columns:
        min_df = 1 if len(train_df) < 80 else 2
        vec = TfidfVectorizer(
            max_features=tfidf_max,
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=True,
        )
        t0 = vec.fit_transform(train_df[TEXT])
        t1 = vec.transform(other_df[TEXT])
        x0 = np.hstack([x0, t0.toarray()]).astype(np.float32)
        x1 = np.hstack([x1, t1.toarray()]).astype(np.float32)
    else:
        x0, x1 = x0.astype(np.float32), x1.astype(np.float32)
    return x0, x1


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_class: int, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, n_class),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_with_val(
    x_tr,
    y_tr,
    x_va,
    y_va,
    n_class: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    device: torch.device,
    patience: Optional[int],
    min_delta: float,
) -> Tuple[float, int]:
    """
    Returns (val_accuracy, best_epoch).
    If patience is set, early-stop on val loss and restore best weights.
    best_epoch = epoch index (1-based) with lowest val loss when ES used, else epochs.
    """
    model = MLP(x_tr.shape[1], n_class, hidden).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long()),
        batch_size=min(batch_size, len(x_tr)),
        shuffle=True,
    )
    xv = torch.from_numpy(x_va).to(device)
    yv = torch.from_numpy(y_va).long().to(device)

    use_es = patience is not None and patience > 0
    best_loss, best_sd = float("inf"), None
    best_ep, stalled = 1, 0

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = crit(model(xv), yv).item()
        if not use_es:
            continue
        if vloss < best_loss - min_delta:
            best_loss = vloss
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep + 1
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break

    if use_es and best_sd is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_sd.items()})

    model.eval()
    with torch.no_grad():
        pred = model(xv).argmax(1).cpu().numpy()
    acc = float((pred == y_va).mean())
    return acc, (best_ep if use_es else max(epochs, 1))


def fit_epochs(model: nn.Module, x, y, epochs, batch_size, lr, device):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long()),
        batch_size=min(batch_size, len(x)),
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15, help="0 = no early stopping")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument(
        "--final-monitor-size",
        type=float,
        default=0.1,
        help="Stratified fraction of train_pool for picking refit epoch count.",
    )
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--tfidf-max-features", type=int, default=3000)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_text = not args.no_text
    es_pat = args.patience if args.patience > 0 else None

    df = prep_df(load_dataframe("training_data_clean.csv"), use_text)
    pool, test, splits = get_splits_with_test(
        df, target_col=TARGET, n_splits=args.n_splits, test_size=args.test_size, seed=args.seed
    )
    le = LabelEncoder()
    le.fit(pool[TARGET])
    n_class = len(le.classes_)

    print(f"pool={len(pool)} test={len(test)} folds={len(splits)} classes={n_class} use_text={use_text}")

    fold_acc = []
    for tr_i, va_i in splits:
        tr, va = pool.iloc[tr_i], pool.iloc[va_i]
        x_tr, x_va = featurize(tr, va, use_text, args.tfidf_max_features)
        y_tr, y_va = le.transform(tr[TARGET]), le.transform(va[TARGET])
        acc, _ = train_with_val(
            x_tr,
            y_tr,
            x_va,
            y_va,
            n_class,
            args.epochs,
            args.batch_size,
            args.lr,
            args.hidden,
            device,
            es_pat,
            args.min_delta,
        )
        fold_acc.append(acc)
    print(f"CV val acc: mean={np.mean(fold_acc):.4f} std={np.std(fold_acc):.4f}")

    x_pool, x_te = featurize(pool, test, use_text, args.tfidf_max_features)
    y_pool, y_te = le.transform(pool[TARGET]), le.transform(test[TARGET])

    x_fit, x_mon, y_fit, y_mon = train_test_split(
        x_pool,
        y_pool,
        test_size=args.final_monitor_size,
        stratify=y_pool,
        random_state=args.seed,
    )
    _, best_ep = train_with_val(
        x_fit,
        y_fit,
        x_mon,
        y_mon,
        n_class,
        args.epochs,
        args.batch_size,
        args.lr,
        args.hidden,
        device,
        es_pat,
        args.min_delta,
    )
    best_ep = max(best_ep, 1)

    model = MLP(x_pool.shape[1], n_class, args.hidden).to(device)
    fit_epochs(model, x_pool, y_pool, best_ep, args.batch_size, args.lr, device)

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_te).to(device)).argmax(1).cpu().numpy()
    test_acc = float((pred == y_te).mean())
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_te,
        pred,
        average="macro",
        zero_division=0,
    )

    print(f"Refit epochs (from monitor ES): {best_ep}  input_dim={x_pool.shape[1]}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test precision (macro): {precision_macro:.4f}")
    print(f"Test recall (macro): {recall_macro:.4f}")
    print(f"Test F1 (macro): {f1_macro:.4f}")
    print("\nClassification report (holdout test)")
    print(classification_report(y_te, pred, target_names=le.classes_, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
