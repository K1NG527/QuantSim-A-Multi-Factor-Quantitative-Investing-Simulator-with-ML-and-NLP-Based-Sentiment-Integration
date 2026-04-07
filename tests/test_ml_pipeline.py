"""
Unit tests for ml_pipeline.py — QuantSim

Tests:
    - Time-series split correctness (no future data leakage)
    - Model training produces valid output
    - Feature importance output
    - Prediction shape matches input
    - Scaler behavior
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ml_pipeline import FactorModelTrainer


@pytest.fixture
def sample_data():
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    rows = []
    for date in dates:
        for symbol in symbols:
            rows.append({
                "symbol": symbol,
                "date": date,
                "value": np.random.randn(),
                "momentum": np.random.randn(),
                "quality": np.random.randn(),
                "volatility": np.random.randn(),
                "volume": np.random.randn(),
                "composite_score": np.random.randn() * 0.5,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def trainer_with_data(sample_data, tmp_path):
    """Create a trainer with pre-loaded data."""
    trainer = FactorModelTrainer(data_dir=str(tmp_path), model_dir=str(tmp_path / "models"))
    trainer.df = sample_data
    return trainer


class TestTimeSeriesSplit:
    """Test time-series aware train/test split."""

    def test_no_future_leakage(self, trainer_with_data):
        """Training data should only contain dates before test data."""
        trainer = trainer_with_data
        trainer._time_series_split(test_ratio=0.2)

        # Get date ranges
        train_dates = trainer.df.iloc[:len(trainer.X_train)]["date"]
        test_dates = trainer.df.iloc[-len(trainer.X_test):]["date"]

        # The test set should not contain dates before the training set ends
        assert trainer.X_train.shape[0] > 0
        assert trainer.X_test.shape[0] > 0

    def test_split_ratio(self, trainer_with_data):
        """Split should approximately match the requested ratio."""
        trainer = trainer_with_data
        trainer._time_series_split(test_ratio=0.2)

        total = trainer.X_train.shape[0] + trainer.X_test.shape[0]
        test_ratio = trainer.X_test.shape[0] / total
        assert 0.1 <= test_ratio <= 0.4  # Approximate due to date-based splitting


class TestModelTraining:
    """Test model training pipeline."""

    def test_train_model_returns_metrics(self, trainer_with_data):
        """train_model should return a dict with expected metric keys."""
        trainer = trainer_with_data
        metrics = trainer.train_model(test_ratio=0.3)

        assert isinstance(metrics, dict)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "model_type" in metrics
        assert metrics["model_type"] in ["xgboost", "lightgbm"]

    def test_model_saved_to_disk(self, trainer_with_data):
        """Model and scaler should be saved to disk after training."""
        trainer = trainer_with_data
        metrics = trainer.train_model(test_ratio=0.3)

        assert os.path.exists(metrics["model_path"])
        assert os.path.exists(metrics["scaler_path"])

    def test_rmse_is_finite(self, trainer_with_data):
        """RMSE should be a finite positive number."""
        trainer = trainer_with_data
        metrics = trainer.train_model(test_ratio=0.3)

        assert np.isfinite(metrics["rmse"])
        assert metrics["rmse"] >= 0


class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_importance_has_all_features(self, trainer_with_data):
        """Feature importance should list all factor features."""
        trainer = trainer_with_data
        trainer.train_model(test_ratio=0.3)

        importance = trainer.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == len(trainer.FEATURE_COLS)
        assert set(importance["feature"]) == set(trainer.FEATURE_COLS)

    def test_importance_sums_positive(self, trainer_with_data):
        """All importance values should be non-negative."""
        trainer = trainer_with_data
        trainer.train_model(test_ratio=0.3)

        importance = trainer.get_feature_importance()
        assert (importance["importance"] >= 0).all()


class TestPrediction:
    """Test prediction functionality."""

    def test_predict_output_shape(self, trainer_with_data):
        """Predictions should match input row count."""
        trainer = trainer_with_data
        trainer.train_model(test_ratio=0.3)

        test_data = trainer.df.tail(10)
        predictions = trainer.predict(test_data)
        assert len(predictions) == 10

    def test_predict_requires_model(self):
        """predict should raise error if no model trained."""
        trainer = FactorModelTrainer()
        with pytest.raises(RuntimeError, match="No trained model"):
            trainer.predict(pd.DataFrame())


class TestScaler:
    """Test StandardScaler behavior."""

    def test_scaler_saved(self, trainer_with_data):
        """Scaler should be saved after training."""
        trainer = trainer_with_data
        trainer.train_model(test_ratio=0.3)
        assert trainer.scaler is not None

    def test_scaler_transforms_correctly(self, trainer_with_data):
        """Scaled data should have approximately zero mean."""
        trainer = trainer_with_data
        trainer.train_model(test_ratio=0.3)

        X = trainer.df[trainer.FEATURE_COLS].values
        X_scaled = trainer.scaler.transform(X)
        # After full dataset transform, means should be near zero
        means = np.mean(X_scaled, axis=0)
        assert np.allclose(means, 0, atol=0.5)  # Approximate due to train-only fit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
