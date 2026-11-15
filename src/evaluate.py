from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import MODEL_FEATURE_COLUMNS


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"
REPORT_PATH = BASE_DIR / "reports" / "evaluation_metrics.json"


def evaluate_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    required_columns = MODEL_FEATURE_COLUMNS + ["price"]
    missing_columns = [col for col in required_columns if col not in test_data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in test dataset: {missing_columns}")

    X_test = test_data[MODEL_FEATURE_COLUMNS]
    actual_prices = test_data["price"]

    predicted_log_prices = model.predict(X_test)
    predicted_prices = np.expm1(predicted_log_prices)
    actual_log_prices = np.log1p(actual_prices)

    metrics = {
        "test_samples": len(X_test),
        "log_space": {
            "rmse": float(np.sqrt(mean_squared_error(actual_log_prices, predicted_log_prices))),
            "mae": float(mean_absolute_error(actual_log_prices, predicted_log_prices)),
            "r2": float(r2_score(actual_log_prices, predicted_log_prices)),
},
        "price_space": {
            "rmse": float(np.sqrt(mean_squared_error(actual_prices, predicted_prices))),
            "mae": float(mean_absolute_error(actual_prices, predicted_prices)),
            "r2": float(r2_score(actual_prices, predicted_prices)),},}

    print(f"\nEvaluated {metrics['test_samples']:,} test houses")
    print("\nLog-space evaluation")
    print(f"RMSE: {metrics['log_space']['rmse']:.4f}")
    print(f"MAE:  {metrics['log_space']['mae']:.4f}")
    print(f"R2:   {metrics['log_space']['r2']:.4f}")

    print("\nPrice-space evaluation")
    print(f"RMSE: Rs {metrics['price_space']['rmse']:,.0f}")
    print(f"MAE:  Rs {metrics['price_space']['mae']:,.0f}")
    print(f"R2:   {metrics['price_space']['r2']:.4f}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nEvaluation report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_model()
