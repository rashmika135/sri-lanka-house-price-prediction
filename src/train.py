from pathlib import Path
from datetime import datetime
import ast

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "properties.csv"
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"


# load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# keep only house sale records
houses_df = df[df["category"] == "Houses For Sale"].copy()
houses_df = houses_df[houses_df["type"] == "for_sale"]
print("Houses dataset shape:", houses_df.shape)

# convert properties column from string to dictionary
def parse_props(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}

houses_df["properties"] = houses_df["properties"].apply(parse_props)
# extract important fetures from properties column
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

houses_df.head(5)
houses_df.columns




# convert bedrooms and bathrooms to numbers
houses_df["bedrooms"] = pd.to_numeric(
    houses_df["bedrooms"].str.replace("+", "", regex=False)
)

houses_df["bathrooms"] = pd.to_numeric(
    houses_df["bathrooms"].str.replace("+", "", regex=False)
)


houses_df["bedrooms"].dtype, houses_df["bathrooms"].dtype

#Cleaning price column and converting in to numeric
houses_df["price"] = pd.to_numeric(
    houses_df["price"]
    .str.replace("Rs", "", regex=False)
    .str.replace(",", "", regex=False),
    errors="coerce"
)

houses_df["price"].isnull().sum()

houses_df = houses_df.dropna(subset=["price"])
houses_df["price"].isnull().sum()

#cleanin house_size column and converting in to numeric
houses_df["house_size"] = (
    houses_df["house_size"]
    .str.replace("sqft", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

houses_df["house_size"].head()
houses_df["house_size"].isnull().sum()

#changing acres to perches to
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

houses_df["land_size"].head()
houses_df["land_size"].isnull().sum()

    # convert perches to square feet
houses_df["land_size"] = houses_df["land_size"] * 272.25
houses_df["land_size"].head()

#turn price column in to a log target
houses_df["log_price"] = np.log1p(houses_df["price"])

# Create years_since_posted
houses_df["posted_date"] = pd.to_datetime(
    houses_df["posted_date"],
    errors="coerce"
)

houses_df = houses_df.dropna(subset=["posted_date"])

houses_df["years_since_posted"] = (
    datetime.now().year - houses_df["posted_date"].dt.year
)

houses_df.drop(columns=["posted_date"], inplace=True)

houses_df.drop(columns=["posted_date"], inplace=True)

# create house_size_per_land_size
houses_df["house_size_per_land_size"] = houses_df["house_size"] / houses_df["land_size"]

#Creating exact area column
def parse_area(x):
    try:
        return ast.literal_eval(x)
    except Exception as e:
        print(e)
        return {}


houses_df["area"] = houses_df["area"].apply(parse_area)

houses_df["exact_area"] = houses_df["area"].apply(
    lambda x: x.get("name")
)


# preprocessing
#convert boolean column to 0 and 1
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
    
for col in boolean_features:
    print(col, houses_df[col].unique())
    

numeric_features = [
    "bedrooms",
    "bathrooms",
    "house_size",
    "land_size",
    "house_size_per_land_size",
    "years_since_posted"
]

boolean_features = [
    "is_verified",
    "is_member",
    "is_authorized_dealer"
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

X = houses_df[feature_cols]

y = houses_df["log_price"]

y_real = houses_df["price"]

RANDOM_STATE=67
X_train, X_test, y_train, y_test, _, y_test_real = train_test_split(
    X,
    y,
    y_real,
    test_size=0.2,
    random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),

        ("bool", "passthrough", boolean_features),

        ("cat",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features),

        ("cat_highcard",
         OneHotEncoder(
             handle_unknown="ignore",
             max_categories=30
         ),
         high_card_features),
    ]
)


#model development
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

print("Training model...")

final_model.fit(X_train, y_train)

print("Training completed.")

y_pred_log = final_model.predict(X_test)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
mae_log = mean_absolute_error(y_test, y_pred_log)
r2_log = r2_score(y_test, y_pred_log)

y_pred_real = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)

print("\nFinal Model Evaluation")

print(
    f"Log-space -> RMSE: {rmse_log:.4f} | "
    f"MAE: {mae_log:.4f} | "
    f"R2: {r2_log:.4f}"
)

print(
    f"Price-space -> RMSE: Rs {rmse:,.0f} | "
    f"MAE: Rs {mae:,.0f} | "
    f"R2: {r2:.4f}"
)

MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump(final_model, MODEL_PATH)

print("Model saved to:", MODEL_PATH)

