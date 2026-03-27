import re

import numpy as np
import pandas as pd

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

NUM_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
]

PRICE_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"

MULTI_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]

TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]


def normalize_na(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.replace("\u00A0", " ").replace("&nbsp;", " ").strip()
        if s == "" or s.lower() in {"na", "n/a", "none", "null"}:
            return np.nan
        return s
    return x


def parse_likert(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    m = re.match(r"^\s*([1-5])\s*[-–]\s*", str(x))
    return float(m.group(1)) if m else np.nan


def parse_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    if "million" in s:
        m = re.search(r"(\d+(\.\d+)?)", s)
        return float(m.group(1)) * 1_000_000 if m else np.nan
    if "thousand" in s:
        m = re.search(r"(\d+(\.\d+)?)", s)
        return float(m.group(1)) * 1_000 if m else np.nan
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def normalize_multiselect(x):
    if pd.isna(x):
        return np.nan
    parts = [p.strip().lower() for p in str(x).split(",") if p.strip()]
    parts = sorted(set(parts))
    return ",".join(parts) if parts else np.nan


def clean_text(x):
    if pd.isna(x):
        return ""
    s = str(x).lower().replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_dataframe(df):
    df = df.map(normalize_na)

    for c in LIKERT_COLS:
        df[c] = df[c].apply(parse_likert)

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[PRICE_COL] = df[PRICE_COL].apply(parse_price)

    for c in MULTI_COLS:
        df[c] = df[c].apply(normalize_multiselect)

    for c in TEXT_COLS:
        df[c] = df[c].apply(clean_text)

    df["text_all"] = (
        df[TEXT_COLS].agg(" ".join, axis=1) + " " + df[MULTI_COLS].fillna("").agg(" ".join, axis=1)
    ).str.strip()

    label_map = {}
    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "Painting" in df.columns:
        label_map = {name: i for i, name in enumerate(sorted(df["Painting"].dropna().unique()))}
        y = df["Painting"].map(label_map).astype("Int64")

    return df.copy(), y, label_map


def build_cleaning_report(df_raw, df_clean):
    rows = []
    all_cols = sorted(set(df_raw.columns).union(set(df_clean.columns)))
    for col in all_cols:
        raw_na = float(df_raw[col].isna().mean()) if col in df_raw.columns else np.nan
        clean_na = float(df_clean[col].isna().mean()) if col in df_clean.columns else np.nan
        rows.append(
            {
                "column": col,
                "missing_rate_raw": raw_na,
                "missing_rate_clean": clean_na,
                "missing_rate_delta": clean_na - raw_na,
            }
        )

    report_df = pd.DataFrame(rows).sort_values("missing_rate_clean", ascending=False)

    summary = {
        "n_rows": int(len(df_clean)),
        "n_columns_raw": int(df_raw.shape[1]),
        "n_columns_clean": int(df_clean.shape[1]),
        "avg_missing_rate_raw": float(df_raw.isna().mean().mean()),
        "avg_missing_rate_clean": float(df_clean.isna().mean().mean()),
    }
    return report_df, summary
