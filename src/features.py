"""Shared house-price features for training and FastAPI inference.

The model expects land_size in square feet (not perches) and a full set
of 13 features. The existing fitted scikit-learn Pipeline handles scaling
and one-hot encoding, so these operations must NOT be repeated here.
"""

from datetime import datetime
from typing import Any, Mapping

import pandas as pd


SQUARE_FEET_PER_PERCH = 272.25
PERCHES_PER_ACRE = 160.0

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "house_size",
    "land_size",
    "house_size_per_land_size",
    "years_since_posted",
]
BOOLEAN_FEATURES = [
    "is_verified",
    "is_member",
    "is_authorized_dealer",
]
CATEGORICAL_FEATURES = ["geo_region", "membership_level"]
HIGH_CARD_FEATURES = ["location", "exact_area"]

MODEL_FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + BOOLEAN_FEATURES
    + CATEGORICAL_FEATURES
    + HIGH_CARD_FEATURES
)


def land_size_text_to_perches(value: Any) -> float | None:
    """Parse dataset values such as '10.0 perches' or '0.5 acres'.

    The input dataset stores land sizes as text; API clients instead supply
    a numeric `land_size_perches` value.
    """
    if not isinstance(value, str):
        return None

    text = value.strip()
    if "acres" in text:
        return float(text.replace("acres", "").replace(",", "").strip()) * PERCHES_PER_ACRE
    if "perches" in text:
        return float(text.replace("perches", "").replace(",", "").strip())
    return None


def perches_to_square_feet(perches: float) -> float:
    """Convert land area from perches to square feet for the trained model."""
    return float(perches) * SQUARE_FEET_PER_PERCH


def _boolean_as_model_int(value: Any) -> int:
    """Match the original training rule: str(value).lower() == 'true'."""
    return int(str(value).lower() == "true")


def prepare_training_features(
    cleaned_houses: pd.DataFrame,
    *,
    reference_year: int | None = None,
) -> pd.DataFrame:
    """Add model features after dataset cleaning and sqft conversion.

    Required existing columns: house_size, land_size (square feet),
    posted_date (pandas datetime), and three boolean seller flags.
    Does not drop rows or change outlier filtering in train.py.
    """
    frame = cleaned_houses.copy()
    year = datetime.now().year if reference_year is None else reference_year

    frame["years_since_posted"] = year - frame["posted_date"].dt.year
    frame["house_size_per_land_size"] = frame["house_size"] / frame["land_size"]

    for column in BOOLEAN_FEATURES:
        frame[column] = frame[column].map(_boolean_as_model_int)

    return frame


def prepare_inference_features(
    house: Mapping[str, Any],
    *,
    reference_year: int | None = None,
) -> pd.DataFrame:
    """Transform one validated API request to the model's exact 13 columns.

    Required fields follow the existing api/main.py HouseInput schema.
    FastAPI/Pydantic should validate incoming types and sensible ranges.
    """
    year = datetime.now().year if reference_year is None else reference_year
    land_size_sqft = perches_to_square_feet(house["land_size_perches"])
    house_size_sqft = float(house["house_size"])

    if land_size_sqft <= 0 or house_size_sqft <= 0:
        raise ValueError("land_size_perches and house_size must be positive")

    row = {
        "bedrooms": int(house["bedrooms"]),
        "bathrooms": int(house["bathrooms"]),
        "house_size": house_size_sqft,
        "land_size": land_size_sqft,
        "house_size_per_land_size": house_size_sqft / land_size_sqft,
        "years_since_posted": year - int(house["posted_year"]),
        "is_verified": _boolean_as_model_int(house["is_verified"]),
        "is_member": _boolean_as_model_int(house["is_member"]),
        "is_authorized_dealer": _boolean_as_model_int(house["is_authorized_dealer"]),
        "geo_region": house["geo_region"],
        "membership_level": house["membership_level"],
        "location": house["location"],
        "exact_area": house["exact_area"],
    }

    return pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)
