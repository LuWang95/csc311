import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# ========= Load =========
df = pd.read_csv("training_data_202601.csv")

# ========= Cleaning (same spirit as yours, but kept compact for EDA) =========
def normalize_na(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.replace("\u00A0", " ").replace("&nbsp;", " ").strip()
        if s == "" or s.lower() in {"na", "n/a", "none", "null"}:
            return np.nan
        return s
    return x

df = df.map(normalize_na)

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

def parse_likert(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    m = re.match(r"^\s*([1-5])\s*[-–]\s*", str(x))
    return float(m.group(1)) if m else np.nan

for c in LIKERT_COLS:
    df[c] = df[c].apply(parse_likert)

NUM_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
]
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

PRICE_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"

def parse_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()

    # optional: handle million/thousand words
    if "million" in s:
        m = re.search(r"(\d+(\.\d+)?)", s)
        return float(m.group(1)) * 1_000_000 if m else np.nan
    if "thousand" in s:
        m = re.search(r"(\d+(\.\d+)?)", s)
        return float(m.group(1)) * 1_000 if m else np.nan

    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

df[PRICE_COL] = df[PRICE_COL].apply(parse_price)

MULTI_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]

def normalize_multiselect(x):
    if pd.isna(x):
        return np.nan
    parts = [p.strip().lower() for p in str(x).split(",") if p.strip()]
    parts = sorted(set(parts))
    return ",".join(parts) if parts else np.nan

for c in MULTI_COLS:
    df[c] = df[c].apply(normalize_multiselect)

TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]

def clean_text(x):
    if pd.isna(x):
        return ""
    s = str(x).lower().replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

for c in TEXT_COLS:
    df[c] = df[c].apply(clean_text)

df["text_all"] = (
    df[TEXT_COLS].agg(" ".join, axis=1) + " " +
    df[MULTI_COLS].fillna("").agg(" ".join, axis=1)
).str.strip()

df["text_len_words"] = df["text_all"].apply(lambda s: 0 if not isinstance(s, str) or s == "" else len(s.split()))
df["price_log"] = np.log1p(df[PRICE_COL])

# ========= Useful Tables to print/save =========
# Missingness per column
missing = df.isna().mean().sort_values(ascending=False)
missing.to_csv("eda_missingness.csv")

# Class balance table
class_counts = df["Painting"].value_counts(dropna=False)
class_props = df["Painting"].value_counts(normalize=True, dropna=False)
pd.DataFrame({"count": class_counts, "proportion": class_props}).to_csv("eda_class_balance.csv")

# ========= Plots (saved as PNG) =========

# 1) Class distribution
plt.figure()
df["Painting"].value_counts().plot(kind="bar")
plt.title("Class Distribution (Painting)")
plt.xlabel("Painting")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig_class_distribution.png", dpi=200)
plt.close()

# 2) Text length distribution
plt.figure()
df["text_len_words"].hist(bins=30)
plt.title("Text Length Distribution (words) for text_all")
plt.xlabel("Number of words")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig_text_length_hist.png", dpi=200)
plt.close()

# 3) Text length by class (boxplot)
plt.figure()
classes = [c for c in df["Painting"].dropna().unique()]
data = [df.loc[df["Painting"] == c, "text_len_words"].values for c in classes]
plt.boxplot(data, labels=classes, vert=True)
plt.title("Text Length (words) by Class")
plt.ylabel("Number of words")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("fig_text_length_by_class.png", dpi=200)
plt.close()

# 4) Price raw distribution (clip for visibility)
plt.figure()
price_clip = df[PRICE_COL].clip(upper=df[PRICE_COL].quantile(0.99))  # clip top 1% for a readable plot
price_clip.dropna().hist(bins=30)
plt.title("Price Distribution (clipped at 99th percentile)")
plt.xlabel("Price (CAD)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig_price_hist_clipped.png", dpi=200)
plt.close()

# 5) Price log distribution
plt.figure()
df["price_log"].dropna().hist(bins=30)
plt.title("Log(1+Price) Distribution")
plt.xlabel("log(1 + price)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig_price_log_hist.png", dpi=200)
plt.close()

# 6) Likert distributions (one figure per column)
for c in LIKERT_COLS:
    plt.figure()
    vc = df[c].value_counts().sort_index()
    # ensure 1..5 all appear
    idx = [1, 2, 3, 4, 5]
    counts = [vc.get(i, 0) for i in idx]
    plt.bar(idx, counts)
    plt.title(f"Likert Distribution: {c}")
    plt.xlabel("Score (1-5)")
    plt.ylabel("Count")
    plt.xticks(idx)
    plt.tight_layout()
    out = "fig_likert_" + re.sub(r"[^a-zA-Z0-9]+", "_", c).strip("_") + ".png"
    plt.savefig(out, dpi=200)
    plt.close()

print("Saved:")
print("  Tables: eda_missingness.csv, eda_class_balance.csv")
print("  Figures: fig_class_distribution.png, fig_text_length_hist.png, fig_text_length_by_class.png, fig_price_hist_clipped.png, fig_price_log_hist.png")
print("  Likert figs: fig_likert_*.png")