from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from split_data import load_dataframe, get_splits_with_test


def run_naive_bayes_final():
    df = load_dataframe()
    df_trainval, df_test, splits = get_splits_with_test(df, target_col="Painting")

    # Best hyperparameters from tuning
    alpha = 0.5
    max_features = 5000
    ngram_range = (1, 1)

    # Train final model on full trainval
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    X_trainval_vec = vectorizer.fit_transform(df_trainval["text_all"].fillna(""))
    y_trainval = df_trainval["Painting"]

    model = MultinomialNB(alpha=alpha)
    model.fit(X_trainval_vec, y_trainval)

    # Evaluate on test set
    X_test_vec = vectorizer.transform(df_test["text_all"].fillna(""))
    y_test = df_test["Painting"]

    test_preds = model.predict(X_test_vec)

    print("Final Naive Bayes Results")
    print("Test Accuracy:", accuracy_score(y_test, test_preds))
    print("Test Macro F1:", f1_score(y_test, test_preds, average="macro"))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))


if __name__ == "__main__":
    run_naive_bayes_final()
