from fastapi import FastAPI
from app.schemas import ReviewRequest
from app.predictor import predict_aspects

app = FastAPI(
    title="Aspect Review Analyzer API",
    description="API для multi-label анализа отзывов по аспектам",
    version="3.0"
)


@app.get("/")
def root():
    return {"message": "API работает"}


@app.post("/analyze")
def analyze_review(request: ReviewRequest):
    aspects, probabilities = predict_aspects(request.place_type, request.text)

    return {
        "place_type": request.place_type,
        "text": request.text,
        "predicted_aspects": aspects,
        "probabilities": probabilities
    }