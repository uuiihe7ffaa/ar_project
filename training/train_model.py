import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = "training/dataset.csv"
MODEL_PATH = "training/model.pkl"


def build_input_text(row):
    place_type = str(row["place_type"]).strip().lower()
    text = str(row["text"]).strip()
    return f"{place_type}. {text}"


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    required_columns = {"place_type", "text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"В dataset.csv не хватает колонок: {missing}")

    X = df.apply(build_input_text, axis=1)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nМодель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    main()