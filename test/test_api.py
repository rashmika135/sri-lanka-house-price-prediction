
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from src.features import MODEL_FEATURE_COLUMNS


class FakeModel:
    def predict(self, features):
        assert list(features.columns) == MODEL_FEATURE_COLUMNS
        assert features.iloc[0]["land_size"] == 2722.5
        return np.array([np.log1p(25_000_000)])


class TestHousePriceAPI(unittest.TestCase):
    def setUp(self):
        self.model_patch = patch("app.main.joblib.load", return_value=FakeModel())
        self.path_patch = patch("app.main.MODEL_PATH", Path(__file__))
        self.model_patch.start()
        self.path_patch.start()

        self.client = TestClient(app)
        self.client.__enter__()

        self.house = {
            "bedrooms": 3,
            "bathrooms": 2,
            "house_size": 1800,
            "land_size_perches": 10,
            "location": "Nugegoda",
            "exact_area": "Colombo",
            "geo_region": "LK-11",
            "membership_level": "free",
            "is_verified": False,
            "is_member": True,
            "is_authorized_dealer": False,
            "posted_year": 2023,
        }

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.path_patch.stop()
        self.model_patch.stop()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_loaded"], True)

    def test_predict(self):
        response = self.client.post("/predict", json=self.house)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_price_lkr"], 25_000_000)
        self.assertEqual(response.json()["predicted_price_millions"], 25.0)
        self.assertEqual(response.json()["currency"], "LKR")

    def test_invalid_land_size(self):
        house = {**self.house, "land_size_perches": 0}
        response = self.client.post("/predict", json=house)
        self.assertEqual(response.status_code, 422)

    def test_future_posted_year(self):
        house = {**self.house, "posted_year": datetime.now().year + 1}
        response = self.client.post("/predict", json=house)
        self.assertEqual(response.status_code, 422)

    def test_missing_feature(self):
        house = self.house.copy()
        del house["location"]
        response = self.client.post("/predict", json=house)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
