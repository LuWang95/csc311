import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
from data_cleaning_utils import clean_dataframe, LIKERT_COLS, PRICE_COL

# ========= Load =========
clean_path = Path("training_data_clean.csv")
if clean_path.exists():
    df = pd.read_csv(clean_path)
else:
    df_raw = pd.read_csv("training_data_202601.csv")
    df, _, _ = clean_dataframe(df_raw)

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