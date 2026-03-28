from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from split_data import load_dataframe, get_splits_with_test


def run_naive_bayes_tuning():
    df = load_dataframe()
    df_trainval, df_test, splits = get_splits_with_test(df, target_col="Painting")

    # Hyperparameter grid
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    max_features_options = [1000, 2000, 5000]
    ngram_options = [(1, 1), (1, 2)]

    best_score = -1
    best_params = None

    for alpha in alphas:
        for max_features in max_features_options:
            for ngram_range in ngram_options:
                fold_f1s = []

                for train_idx, val_idx in splits:
                    train_df = df_trainval.iloc[train_idx]
                    val_df = df_trainval.iloc[val_idx]

                    X_train = train_df["text_all"].fillna("")
                    y_train = train_df["Painting"]

                    X_val = val_df["text_all"].fillna("")
                    y_val = val_df["Painting"]

                    vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range
                    )

                    X_train_vec = vectorizer.fit_transform(X_train)
                    X_val_vec = vectorizer.transform(X_val)

                    model = MultinomialNB(alpha=alpha)
                    model.fit(X_train_vec, y_train)

                    preds = model.predict(X_val_vec)
                    f1 = f1_score(y_val, preds, average="macro")
                    fold_f1s.append(f1)

                avg_f1 = sum(fold_f1s) / len(fold_f1s)

                print(
                    f"alpha={alpha}, "
                    f"max_features={max_features}, "
                    f"ngram_range={ngram_range}, "
                    f"avg_macro_f1={avg_f1:.4f}"
                )

                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_params = (alpha, max_features, ngram_range)

    print("\nBest parameters found:")
    print(f"alpha={best_params[0]}")
    print(f"max_features={best_params[1]}")
    print(f"ngram_range={best_params[2]}")
    print(f"Best CV Macro F1={best_score:.4f}")


if __name__ == "__main__":
    run_naive_bayes_tuning()
