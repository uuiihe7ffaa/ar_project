from fastapi import FastAPI
from app.schemas import ReviewRequest
from app.predictor import predict_aspect

app = FastAPI(
    title="Aspect Review Analyzer API",
    description="API для определения основного аспекта пользовательского отзыва",
    version="4.0"
)


@app.get("/")
def root():
    return {"message": "API работает"}


@app.post("/analyze")
def analyze_review(request: ReviewRequest):
    aspect, probabilities = predict_aspect(request.place_type, request.text)

    return {
        "place_type": request.place_type,
        "text": request.text,
        "predicted_aspect": aspect,
        "probabilities": probabilities
    }