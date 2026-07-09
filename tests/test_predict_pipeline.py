import numpy as np
import pandas as pd

from src.pipeline import predict_pipeline
from src.pipeline.predict_pipeline import PredictPipeline


class DummyPreprocessor:
    def transform(self, features):
        return features


class DummyModel:
    def predict(self, data_scaled):
        return np.array([-12.5, 42.25, 118.9])


def test_predict_clips_scores_to_valid_math_score_range(monkeypatch):
    def fake_load_obj(file_path):
        if file_path.endswith("model.pkl"):
            return DummyModel()
        return DummyPreprocessor()

    monkeypatch.setattr(predict_pipeline, "load_obj", fake_load_obj)

    features = pd.DataFrame({"reading score": [0, 50, 100]})
    preds = PredictPipeline().predict(features)

    np.testing.assert_array_equal(preds, np.array([0, 42.25, 100]))
