import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

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

# df_trainval and df_test are the CV set and the reserved test set.
def get_splits_with_test(
    df,
    target_col="Painting",
    n_splits=5,
    test_size=0.2,
    seed=42
):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")


    df_valid = df[df[target_col].notna()].reset_index(drop=True)
    y = df_valid[target_col]

    
    trainval_idx, test_idx = train_test_split(
        df_valid.index,
        test_size=test_size,
        stratify=y,
        random_state=seed
    )

    df_trainval = df_valid.iloc[trainval_idx].reset_index(drop=True)
    df_test = df_valid.iloc[test_idx].reset_index(drop=True)

   
    y_trainval = df_trainval[target_col]

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    splits = list(splitter.split(df_trainval, y_trainval))

    return df_trainval, df_test, splits