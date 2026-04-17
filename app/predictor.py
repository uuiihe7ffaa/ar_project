import joblib

MODEL_PATH = "training/model.pkl"
model = joblib.load(MODEL_PATH)


def predict_aspect(place_type: str, text: str):
    input_text = f"{place_type.strip().lower()}. {text.strip()}"

    prediction = model.predict([input_text])[0]
    prediction = str(prediction)

    probabilities = model.predict_proba([input_text])[0]
    classes = model.classes_

    prob_dict = {}
    for cls, prob in zip(classes, probabilities):
        prob_dict[str(cls)] = float(prob)

    return prediction, prob_dict