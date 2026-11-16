from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import logging
import math

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features import prepare_inference_features


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained model once when the application starts.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Sri Lanka House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the HTML, CSS and JavaScript from the frontend folder.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class HouseInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    bedrooms: int = Field(ge=1)
    bathrooms: int = Field(ge=1)
    house_size: float = Field(gt=0, description="House size in square feet")
    land_size_perches: float = Field(gt=0, description="Land size in perches")

    location: str = Field(min_length=1)
    exact_area: str = Field(min_length=1)
    geo_region: str = Field(min_length=1)
    membership_level: str = Field(min_length=1)

    is_verified: bool
    is_member: bool
    is_authorized_dealer: bool
    posted_year: int = Field(ge=1900)

    @field_validator("posted_year")
    @classmethod
    def check_posted_year(cls, year: int) -> int:
        if year > datetime.now().year:
            raise ValueError("posted_year cannot be in the future")
        return year


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.post("/predict")
def predict_price(house: HouseInput, request: Request):
    try:
        features = prepare_inference_features(house.model_dump())
        predicted_log_price = request.app.state.model.predict(features)[0]
        predicted_price = float(np.expm1(predicted_log_price))

        if not math.isfinite(predicted_price) or predicted_price < 0:
            raise ValueError("The model returned an invalid price")

    except Exception as error:
        logger.exception("House price prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to predict the price. Please try again.",
        ) from error

    return {
        "predicted_price_lkr": round(predicted_price),
        "predicted_price_millions": round(predicted_price / 1_000_000, 2),
        "currency": "LKR",
    }
