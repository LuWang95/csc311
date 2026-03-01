import pandas as pd
import numpy as np
import re

df = pd.read_csv("training_data_202601.csv")


# ===== 1) 基础：统一缺失值 + 去掉 nbsp =====
def normalize_na(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.replace("\u00A0", " ")  # NBSP
        s = s.replace("&nbsp;", " ")  # 有时是HTML实体
        s = s.strip()
        if s == "" or s.lower() in {"na", "n/a", "none", "null"}:
            return np.nan
        return s
    return x


df = df.map(normalize_na)

# ===== 2) Likert 量表列：把 "4 - Agree" 变成 4 =====
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

# ===== 3) 纯数值列：强制转数字（错误变 NaN）=====
NUM_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
]
for c in NUM_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ===== 4) 钱：把 "$5" / "300 dollars." / "0" 变成数字 =====
PRICE_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"


def parse_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    # 提取第一个数字（可含小数）
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


df[PRICE_COL] = df[PRICE_COL].apply(parse_price)

# ===== 5) multi-select（逗号分隔）：规范化成排序后的 token 串 =====
MULTI_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]


def normalize_multiselect(x):
    if pd.isna(x):
        return np.nan
    parts = [p.strip().lower() for p in str(x).split(",") if p.strip() != ""]
    parts = sorted(set(parts))
    return ",".join(parts) if parts else np.nan


for c in MULTI_COLS:
    df[c] = df[c].apply(normalize_multiselect)

# ===== 6) 文本列：统一小写、去多空格、保留字母数字标点 =====
TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]


def clean_text(x):
    if pd.isna(x):
        return ""
    s = str(x).lower()
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


for c in TEXT_COLS:
    df[c] = df[c].apply(clean_text)

# ===== 7) label：Painting -> 0/1/2 =====
label_map = {name: i for i, name in enumerate(sorted(df["Painting"].dropna().unique()))}
y = df["Painting"].map(label_map).astype("Int64")

# ===== 8) 拼一个总文本（常用做 baseline）=====
# 把 multi-select 也当文本特征（会很有用）
df["text_all"] = (
        df[TEXT_COLS].agg(" ".join, axis=1) + " " +
        df[MULTI_COLS].fillna("").agg(" ".join, axis=1)
).str.strip()

# ===== 9) 输出清洗后的表 =====
df_clean = df.copy()
df_clean.to_csv("training_data_clean.csv", index=False)
print("label_map =", label_map)
print("df_clean shape:", df_clean.shape)
print("y missing:", y.isna().mean())
print(df_clean[["unique_id", "Painting", "text_all"]].head(3))
