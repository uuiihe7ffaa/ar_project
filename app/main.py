from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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
    try:
        aspect, probabilities = predict_aspect(request.place_type, request.text)

        payload = {
            "place_type": str(request.place_type),
            "text": str(request.text),
            "predicted_aspect": str(aspect),
            "probabilities": {str(k): float(v) for k, v in probabilities.items()}
        }

        return JSONResponse(content=payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))