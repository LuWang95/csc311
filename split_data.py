import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from data_cleaning_utils import clean_dataframe


def load_dataframe(input_path="training_data_clean.csv"):
    path = Path(input_path)
    if path.exists():
        return pd.read_csv(path)

    raw_path = Path("training_data_202601.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    df_raw = pd.read_csv(raw_path)
    df_clean, _, _ = clean_dataframe(df_raw)
    return df_clean

# df_trainval and df_test are the CV set and the reserved test set. Split is a list of tuples, each containing the indices of the training and validation sets for each fold.
def get_splits_with_test(
    df,
    target_col="Painting",
    n_splits=5,
    test_size=0.2,
    seed=42,
):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df_valid = df[df[target_col].notna()].reset_index(drop=True)
    y = df_valid[target_col]
    groups = _resolve_groups(df_valid)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df_valid, y, groups=groups))

    df_trainval = df_valid.iloc[trainval_idx].reset_index(drop=True)
    df_test = df_valid.iloc[test_idx].reset_index(drop=True)

    y_trainval = df_trainval[target_col]
    groups_trainval = _resolve_groups(df_trainval)
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df_trainval, y_trainval, groups=groups_trainval))
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df_trainval, y_trainval, groups=groups_trainval))

    return df_trainval, df_test, splits


def _resolve_groups(df: pd.DataFrame):

    if "unique_id" in df.columns:
        return df["unique_id"]

    raise ValueError(
        "Cannot infer groups. Provide unique_id column for group-aware split."
    )