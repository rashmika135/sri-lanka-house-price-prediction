"""Fast regression checks for the shared training/inference features."""

import unittest

import pandas as pd

from src.features import (
    MODEL_FEATURE_COLUMNS,
    land_size_text_to_perches,
    perches_to_square_feet,
    prepare_inference_features,
    prepare_training_features,
)


class TestSharedFeatures(unittest.TestCase):
    def setUp(self):
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

    def test_land_units(self):
        self.assertEqual(land_size_text_to_perches("0.5 acres"), 80.0)
        self.assertEqual(land_size_text_to_perches("10.0 perches"), 10.0)
        self.assertEqual(perches_to_square_feet(10), 2722.5)

    def test_api_feature_order_and_values(self):
        row = prepare_inference_features(self.house, reference_year=2026)
        self.assertEqual(list(row.columns), MODEL_FEATURE_COLUMNS)
        self.assertEqual(row.loc[0, "land_size"], 2722.5)
        self.assertAlmostEqual(row.loc[0, "house_size_per_land_size"], 1800 / 2722.5)
        self.assertEqual(row.loc[0, "years_since_posted"], 3)
        self.assertEqual(row.loc[0, "is_member"], 1)

    def test_training_inference_feature_parity(self):
        training_row = {
            **{k: v for k, v in self.house.items() if k not in ("land_size_perches", "posted_year")},
            "land_size": 2722.5,
            "posted_date": pd.Timestamp("2023-02-01"),
        }
        training_features = prepare_training_features(
            pd.DataFrame([training_row]), reference_year=2026
        )[MODEL_FEATURE_COLUMNS]
        api_features = prepare_inference_features(self.house, reference_year=2026)
        pd.testing.assert_frame_equal(training_features, api_features, check_dtype=False)

    def test_invalid_size(self):
        house = {**self.house, "land_size_perches": 0}
        with self.assertRaises(ValueError):
            prepare_inference_features(house)


if __name__ == "__main__":
    unittest.main()
