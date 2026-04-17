import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


DATA_PATH = "training/dataset.csv"
MODEL_PATH = "training/model.pkl"
MLB_PATH = "training/mlb.pkl"


def build_input_text(row):
    place_type = str(row["place_type"]).strip().lower()
    text = str(row["text"]).strip()
    return f"{place_type}. {text}"


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")

    required_columns = {"place_type", "text", "labels"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"В dataset.csv не хватает колонок: {missing}")

    X = df.apply(build_input_text, axis=1)
    y = df["labels"].apply(lambda x: [label.strip() for label in str(x).split(",")])

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    model.fit(X_train, Y_train)

    score = model.score(X_test, Y_test)
    print("Test score:", score)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(mlb, MLB_PATH)

    print(f"Модель сохранена в {MODEL_PATH}")
    print(f"MultiLabelBinarizer сохранён в {MLB_PATH}")


if __name__ == "__main__":
    main()