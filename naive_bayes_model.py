import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

from split_data import load_dataframe, get_splits_with_test


def run_naive_bayes():
    df = load_dataframe()

    df_trainval, df_test, splits = get_splits_with_test(df, target_col="Painting")

    fold_f1s = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = df_trainval.iloc[train_idx]
        val_df = df_trainval.iloc[val_idx]

        X_train = train_df["text_all"].fillna("")
        y_train = train_df["Painting"]

        X_val = val_df["text_all"].fillna("")
        y_val = val_df["Painting"]

        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        model = MultinomialNB(alpha=1.0)
        model.fit(X_train_vec, y_train)

        preds = model.predict(X_val_vec)

        f1 = f1_score(y_val, preds, average="macro")
        fold_f1s.append(f1)

        print(f"Fold {fold_num} Macro F1:", f1)

    print("Average Macro F1:", sum(fold_f1s) / len(fold_f1s))

    # Train final model on full trainval
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_trainval_vec = vectorizer.fit_transform(df_trainval["text_all"].fillna(""))
    y_trainval = df_trainval["Painting"]

    model = MultinomialNB(alpha=1.0)
    model.fit(X_trainval_vec, y_trainval)

    # Evaluate on test set
    X_test_vec = vectorizer.transform(df_test["text_all"].fillna(""))
    y_test = df_test["Painting"]

    test_preds = model.predict(X_test_vec)

    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print("Test Macro F1:", f1_score(y_test, test_preds, average="macro"))


if __name__ == "__main__":
    run_naive_bayes()
