"""
Run pred_example on the entire training_data_clean.csv and report accuracy vs Painting.

Note: this file includes both the ~80% pool used to train the exported model and the
~20% hold-out test — so the overall accuracy mixes seen (train) and unseen (test) rows.
"""
from pathlib import Path

import pandas as pd

from pred_example import predict_all

DATA = Path(__file__).resolve().parent / "training_data_clean.csv"


def main():
    preds = predict_all(str(DATA))
    df = pd.read_csv(DATA)
    if "Painting" not in df.columns:
        print(f"Rows: {len(preds)} (no Painting column — cannot compute accuracy)")
        print("First 5 preds:", preds[:5])
        return
    y = df["Painting"].astype(str)
    acc = (pd.Series(preds) == y).mean()
    print(f"File: {DATA}")
    print(f"Rows: {len(preds)}")
    print(f"Accuracy (full CSV vs Painting): {acc:.4f}")
    print("(Mix of pool rows [seen in export training] and hold-out test rows.)")


if __name__ == "__main__":
    main()
