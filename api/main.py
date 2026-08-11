from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------
# Find project root and model path
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"


# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------

app = FastAPI(
    title="Sri Lanka House Price Prediction API"
)


# ---------------------------------------------------
# Define user input format
# ---------------------------------------------------

class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: int

    house_size: float

    # user enters land size in perches
    land_size_perches: float

    location: str
    exact_area: str
    geo_region: str
    membership_level: str

    is_verified: bool
    is_member: bool
    is_authorized_dealer: bool

    posted_year: int


# ---------------------------------------------------
# Home route
# ---------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }

