import joblib
import numpy as np

MODEL_PATH = "training/model.pkl"
MLB_PATH = "training/mlb.pkl"

model = joblib.load(MODEL_PATH)
mlb = joblib.load(MLB_PATH)


def predict_aspects(place_type: str, text: str, threshold: float = 0.5, max_labels: int = 2):
    input_text = f"{place_type.strip().lower()}. {text.strip()}"

    probabilities = model.predict_proba([input_text])[0]

    pairs = list(zip(mlb.classes_, probabilities))
    pairs.sort(key=lambda x: x[1], reverse=True)

    predicted_labels = [label for label, prob in pairs if prob >= threshold]

    if not predicted_labels:
        predicted_labels = [pairs[0][0]]

    predicted_labels = predicted_labels[:max_labels]

    prob_dict = {
        label: float(prob) for label, prob in pairs
    }

    return predicted_labels, prob_dict