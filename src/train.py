from pathlib import Path
from datetime import datetime
import ast

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow
import mlflow.sklearn

RANDOM_STATE = 67

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "properties.csv"
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

mlflow.set_experiment("house-price-prediction")
mlflow.set_experiment("house-price-prediction")

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ---------------------------------------------------
# Keep only houses for sale
# ---------------------------------------------------

houses_df = df[df["category"] == "Houses For Sale"].copy()

houses_df = houses_df[
    houses_df["type"] == "for_sale"
]

print("Houses dataset shape:", houses_df.shape)


# ---------------------------------------------------
# Extract values from properties column
# ---------------------------------------------------

def parse_props(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}


houses_df["properties"] = houses_df["properties"].apply(parse_props)


houses_df["bedrooms"] = houses_df["properties"].apply(
    lambda x: x.get("Bedrooms")
)

houses_df["bathrooms"] = houses_df["properties"].apply(
    lambda x: x.get("Bathrooms")
)

houses_df["house_size"] = houses_df["properties"].apply(
    lambda x: x.get("House size")
)

houses_df["land_size"] = houses_df["properties"].apply(
    lambda x: x.get("Land size")
)


# ---------------------------------------------------
# Clean bedrooms and bathrooms
# ---------------------------------------------------

houses_df["bedrooms"] = pd.to_numeric(
    houses_df["bedrooms"]
    .str.replace("+", "", regex=False),
    errors="coerce"
)

houses_df["bathrooms"] = pd.to_numeric(
    houses_df["bathrooms"]
    .str.replace("+", "", regex=False),
    errors="coerce"
)


# ---------------------------------------------------
# Clean price
# ---------------------------------------------------

houses_df["price"] = pd.to_numeric(
    houses_df["price"]
    .str.replace("Rs", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

houses_df = houses_df.dropna(subset=["price"])


# ---------------------------------------------------
# Clean house size
# ---------------------------------------------------

houses_df["house_size"] = pd.to_numeric(
    houses_df["house_size"]
    .str.replace("sqft", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)


# ---------------------------------------------------
# Convert land size into perches
# ---------------------------------------------------

def convert_land_size(x):

    if pd.isna(x):
        return None

    x = x.strip()

    if "acres" in x:

        num = float(
            x.replace("acres", "")
            .replace(",", "")
            .strip()
        )

        return num * 160

    elif "perches" in x:

        num = float(
            x.replace("perches", "")
            .replace(",", "")
            .strip()
        )

        return num

    return None


houses_df["land_size"] = houses_df["land_size"].apply(
    convert_land_size
)


# remove missing important values
houses_df = houses_df.dropna(
    subset=[
        "bedrooms",
        "bathrooms",
        "house_size",
        "land_size"
    ]
)


# remove zero sizes
houses_df = houses_df[
    (houses_df["house_size"] != 0) &
    (houses_df["land_size"] != 0)
]


# ---------------------------------------------------
# Basic outlier cleaning
# ---------------------------------------------------

houses_df = houses_df[
    (houses_df["price"] >= 100000) &
    (houses_df["price"] < 9e11)
]

houses_df = houses_df[
    houses_df["house_size"] <= 20000
]


# ---------------------------------------------------
# Convert land perches -> square feet
# ---------------------------------------------------

houses_df["land_size"] = (
    houses_df["land_size"] * 272.25
)

houses_df = houses_df[
    houses_df["land_size"] <= 27225
]


# same 1% - 99% filtering used during experimentation
for col in ["price", "house_size", "land_size"]:

    lower = houses_df[col].quantile(0.01)
    upper = houses_df[col].quantile(0.99)

    houses_df = houses_df[
        (houses_df[col] >= lower) &
        (houses_df[col] <= upper)
    ]


# ---------------------------------------------------
# Log target
# ---------------------------------------------------

houses_df["log_price"] = np.log1p(
    houses_df["price"]
)


# ---------------------------------------------------
# Exact area
# ---------------------------------------------------

def parse_area(x):

    try:
        return ast.literal_eval(x)

    except:
        return {}


houses_df["area"] = houses_df["area"].apply(
    parse_area
)

houses_df["exact_area"] = houses_df["area"].apply(
    lambda x: x.get("name")
)


# ---------------------------------------------------
# Date feature
# ---------------------------------------------------

houses_df["posted_date"] = pd.to_datetime(
    houses_df["posted_date"],
    errors="coerce"
)

houses_df = houses_df.dropna(
    subset=["posted_date"]
)

houses_df["years_since_posted"] = (
    datetime.now().year
    - houses_df["posted_date"].dt.year
)


# ---------------------------------------------------
# Feature engineering
# ---------------------------------------------------

houses_df["house_size_per_land_size"] = (
    houses_df["house_size"]
    / houses_df["land_size"]
)


# ---------------------------------------------------
# Boolean columns
# ---------------------------------------------------

boolean_features = [
    "is_verified",
    "is_member",
    "is_authorized_dealer"
]

for col in boolean_features:

    houses_df[col] = (
        houses_df[col]
        .astype(str)
        .str.lower()
        .eq("true")
        .astype(int)
    )


# ---------------------------------------------------
# Feature groups
# ---------------------------------------------------

numeric_features = [
    "bedrooms",
    "bathrooms",
    "house_size",
    "land_size",
    "house_size_per_land_size",
    "years_since_posted"
]

categorical_features = [
    "geo_region",
    "membership_level"
]

high_card_features = [
    "location",
    "exact_area"
]


feature_cols = (
    numeric_features
    + boolean_features
    + categorical_features
    + high_card_features
)


# ---------------------------------------------------
# X and y
# ---------------------------------------------------

X = houses_df[feature_cols]

y = houses_df["log_price"]

# original price only for evaluation
y_real = houses_df["price"]


print("Final dataset shape:", X.shape)


# ---------------------------------------------------
# Train / test split
# ---------------------------------------------------

X_train, X_test, y_train, y_test, _, y_test_real = train_test_split(
    X,
    y,
    y_real,
    test_size=0.2,
    random_state=RANDOM_STATE
)


# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            StandardScaler(),
            numeric_features
        ),

        (
            "bool",
            "passthrough",
            boolean_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "cat_highcard",
            OneHotEncoder(
                handle_unknown="ignore",
                max_categories=30
            ),
            high_card_features
        )
    ]
)


# ---------------------------------------------------
# Final Random Forest
# ---------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


final_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])


# ---------------------------------------------------
# Train
# ---------------------------------------------------

with mlflow.start_run():

    print("\nTraining model...")

    final_model.fit(X_train, y_train)

    print("Training completed.")


    # predictions
    y_pred_log = final_model.predict(X_test)


    # log-space metrics
    rmse_log = np.sqrt(
        mean_squared_error(y_test, y_pred_log)
    )

    mae_log = mean_absolute_error(
        y_test,
        y_pred_log
    )

    r2_log = r2_score(
        y_test,
        y_pred_log
    )


    # convert predictions back to real prices
    y_pred_real = np.expm1(y_pred_log)


    # real-price metrics
    rmse = np.sqrt(
        mean_squared_error(
            y_test_real,
            y_pred_real
        )
    )

    mae = mean_absolute_error(
        y_test_real,
        y_pred_real
    )

    r2 = r2_score(
        y_test_real,
        y_pred_real
    )


    # log model parameters
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "max_features": 1.0,
        "random_state": RANDOM_STATE
    })


    # log evaluation metrics
    mlflow.log_metrics({
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_price": rmse,
        "mae_price": mae,
        "r2_price": r2
    })


    print("\nFinal Model Evaluation")

    print(
        f"Log-space -> "
        f"RMSE: {rmse_log:.4f} | "
        f"MAE: {mae_log:.4f} | "
        f"R2: {r2_log:.4f}"
    )

    print(
        f"Price-space -> "
        f"RMSE: Rs {rmse:,.0f} | "
        f"MAE: Rs {mae:,.0f} | "
        f"R2: {r2:.4f}"
    )


    # save normal joblib model
    MODEL_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(
        final_model,
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)


    # also log model inside MLflow
    mlflow.sklearn.log_model(
        sk_model=final_model,
        name="house_price_model"
    )

