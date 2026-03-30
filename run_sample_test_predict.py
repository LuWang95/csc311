"""
Hold-out test subset (same GroupShuffleSplit as neural_network, seed=42, test_size=0.2),
save rows without labels, run pred_example.predict_all, report accuracy.

Requires: pandas, numpy; pred_example needs mlp_export/mlp_export.npz (run export_mlp_params.py).
"""
from pathlib import Path

import pandas as pd

from neural_network import FEATURE_COLS, TARGET, TEXT, prep_df
from pred_example import predict_all
from split_data import get_splits_with_test, load_dataframe

OUT_NO_LABEL = Path(__file__).resolve().parent / "sample_test_no_label.csv"
OUT_WITH_LABEL = Path(__file__).resolve().parent / "sample_holdout_test_with_label.csv"


def main():
    df = prep_df(load_dataframe("training_data_clean.csv"), use_text=True)
    _pool, df_test, _splits = get_splits_with_test(
        df, target_col=TARGET, n_splits=5, test_size=0.2, seed=42
    )
    y_true = df_test[TARGET].astype(str).tolist()

    # Features only (what you would submit for blind grading)
    feature_cols = ["unique_id"] + FEATURE_COLS + [TEXT]
    df_test[feature_cols].to_csv(OUT_NO_LABEL, index=False)
    df_test.to_csv(OUT_WITH_LABEL, index=False)

    preds = predict_all(str(OUT_NO_LABEL))
    acc = sum(p == t for p, t in zip(preds, y_true)) / len(y_true)

    print(f"Hold-out test rows: {len(df_test)}")
    print(f"Saved (no label): {OUT_NO_LABEL}")
    print(f"Saved (with label, for reference): {OUT_WITH_LABEL}")
    print(f"Accuracy on this subset: {acc:.4f}")
    print("\nFirst 10 rows: pred vs true")
    for i in range(min(10, len(preds))):
        ok = preds[i] == y_true[i]
        print(f"  {'OK ' if ok else 'BAD'} pred={preds[i]!r}  true={y_true[i]!r}")


if __name__ == "__main__":
    main()
