import joblib

MODEL_PATH = "training/model.pkl"

model = joblib.load(MODEL_PATH)


def predict_aspect(place_type: str, text: str):
    input_text = f"{place_type.strip().lower()}. {text.strip()}"

    prediction = model.predict([input_text])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([input_text])[0]
        classes = model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    else:
        prob_dict = {}

    return prediction, prob_dict