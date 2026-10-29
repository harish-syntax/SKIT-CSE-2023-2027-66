"""Prediction router: LFT prediction endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/predict", tags=["prediction"])


class PredictionRequest(BaseModel):
    features: dict


def _load_predictor():
    # Imported lazily so the app still starts even if the model fails to load.
    try:
        import predictor  # backend/predictor.py
        return predictor
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Predictor unavailable: {e}")


@router.get("/health")
def predict_health():
    """Simple check that the prediction router is registered."""
    return {"status": "prediction router ok"}


@router.post("/")
def predict(request: PredictionRequest):
    """Run the trained model on the given features."""
    predictor = _load_predictor()
    fn = getattr(predictor, "predict", None)
    if fn is None:
        raise HTTPException(status_code=501, detail="predictor.predict not found")
    try:
        result = fn(request.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"prediction": result}
