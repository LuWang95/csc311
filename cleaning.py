import pandas as pd

from data_cleaning_utils import build_cleaning_report, clean_dataframe

df_raw = pd.read_csv("training_data_202601.csv")
df_clean, y, label_map = clean_dataframe(df_raw)
report_df, report_summary = build_cleaning_report(df_raw, df_clean)

df_clean.to_csv("training_data_clean.csv", index=False)
report_df.to_csv("cleaning_report.csv", index=False)

with open("cleaning_report_summary.txt", "w", encoding="utf-8") as f:
    for k, v in report_summary.items():
        f.write(f"{k}: {v}\n")

print("label_map =", label_map)
print("df_clean shape:", df_clean.shape)
print("y missing:", y.isna().mean())
print("Saved: cleaning_report.csv, cleaning_report_summary.txt")
print(df_clean[["unique_id", "Painting", "text_all"]].head(3))
