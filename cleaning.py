import pandas as pd

from data_cleaning_utils import clean_dataframe

df_raw = pd.read_csv("training_data_202601.csv")
df_clean, y, label_map = clean_dataframe(df_raw)

df_clean.to_csv("training_data_clean.csv", index=False)
print("label_map =", label_map)
print("df_clean shape:", df_clean.shape)
print("y missing:", y.isna().mean())
print(df_clean[["unique_id", "Painting", "text_all"]].head(3))
