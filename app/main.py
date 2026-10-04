from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.model_service import predict


app = FastAPI(
    title="SLR Prediction API",
    description="API for S.L.R prediction using stacking ensemble",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: dict


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "SLR Stacking Ensemble",
        "model_loaded": True
    }

@app.get("/")
def root():
    return {
        "message": "SLR Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/predict")
def make_prediction(request: PredictionRequest):

    try:

        prediction = predict(
            request.features
        )

        return {
            "success": True,
            "prediction": prediction,
            "unit": "mm"
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )