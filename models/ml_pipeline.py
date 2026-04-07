"""
ML Pipeline — Production-Grade Factor Model Trainer for EquiSense.

Upgrades over original:
    - XGBoost as primary model (fallback: LightGBM)
    - Time-series aware train/test split (by date, no data leakage)
    - StandardScaler for feature normalization
    - Model + scaler saved to disk via joblib
    - Predicted returns + feature importance output
    - Configurable paths (no hardcoded OS-specific paths)
    - Comprehensive logging and error handling

Usage:
    trainer = FactorModelTrainer(data_dir="data")
    trainer.prepare_dataset()
    trainer.train_model()
    trainer.plot_feature_importance()
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER — XGBoost primary, LightGBM fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _get_model_class():
    """
    Attempt to load XGBoost; fall back to LightGBM if unavailable.
    Returns (ModelClass, model_name).
    """
    try:
        from xgboost import XGBRegressor
        logger.info("Using XGBoost as ML model")
        return XGBRegressor, "xgboost"
    except ImportError:
        logger.warning("XGBoost not found, falling back to LightGBM")
    try:
        from lightgbm import LGBMRegressor
        logger.info("Using LightGBM as ML model")
        return LGBMRegressor, "lightgbm"
    except ImportError:
        logger.error("Neither XGBoost nor LightGBM available — install one: pip install xgboost")
        raise ImportError("Install xgboost or lightgbm: pip install xgboost lightgbm")


# ═══════════════════════════════════════════════════════════════════════════════
# FACTOR MODEL TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class FactorModelTrainer:
    """
    Production-grade ML pipeline for factor-based stock scoring.

    Attributes:
        data_dir:        Base directory for data files
        model_dir:       Directory to save models
        factor_files:    Dict mapping factor name -> CSV path
        composite_file:  Path to composite scores CSV (labels)
        feature_cols:    List of feature column names
        df:              Prepared dataset
        model:           Trained ML model
        scaler:          Fitted StandardScaler
    """

    FEATURE_COLS = ["value", "momentum", "quality", "volatility", "volume"]

    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models",
        factor_files: Optional[dict] = None,
        composite_file: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            data_dir:        Base directory for data files
            model_dir:       Directory to save trained models
            factor_files:    Dict mapping factor name -> relative CSV path
            composite_file:  Path to composite scores CSV
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Default factor files — relative paths
        base = self.data_dir / "processed_not_normalized" / "factor_scores"
        self.factor_files = factor_files if factor_files is not None else {
            "value": str(base / "value_scores.csv"),
            "momentum": str(base / "momentum_scores.csv"),
            "quality": str(base / "quality_scores.csv"),
            "volatility": str(base / "volatility_scores.csv"),
            "volume": str(base / "volume_scores.csv"),
        }
        self.composite_file = composite_file or str(base / "composite_scores.csv")

        self.df: Optional[pd.DataFrame] = None
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.model_name: str = ""

        # Results storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.feature_importances_: Optional[np.ndarray] = None

    # ───────────────────────────────────────────────────────────────────────
    # DATA LOADING
    # ───────────────────────────────────────────────────────────────────────

    def _load_factors(self) -> Optional[list]:
        """Load individual factor CSV files."""
        dfs = []
        for factor, fpath in self.factor_files.items():
            try:
                df = pd.read_csv(fpath)
                if df.shape[1] > 2:
                    df = df.rename(columns={df.columns[-1]: factor})
                else:
                    logger.warning(f"{fpath} does not have expected column structure — skipping")
                    continue
                dfs.append(df)
                logger.debug(f"Loaded {factor} factor: {len(df)} rows")
            except FileNotFoundError:
                logger.error(f"Factor file not found: {fpath}")
                return None
            except Exception as e:
                logger.error(f"Error reading {fpath}: {e}")
                return None
        return dfs

    def _merge_factors(self, factor_dfs: list) -> pd.DataFrame:
        """Merge factor dataframes on symbol + date."""
        if not factor_dfs:
            return pd.DataFrame()
        merged = factor_dfs[0]
        for df in factor_dfs[1:]:
            merged = pd.merge(merged, df, on=["symbol", "date"], how="inner")
        logger.info(f"Merged {len(factor_dfs)} factors: {len(merged)} rows")
        return merged

    def _load_labels(self) -> Optional[pd.DataFrame]:
        """Load composite score labels."""
        try:
            df = pd.read_csv(self.composite_file)
            logger.info(f"Loaded labels: {len(df)} rows")
            return df
        except FileNotFoundError:
            logger.error(f"Labels file not found: {self.composite_file}")
            return None
        except Exception as e:
            logger.error(f"Error reading labels: {e}")
            return None

    # ───────────────────────────────────────────────────────────────────────
    # DATASET PREPARATION
    # ───────────────────────────────────────────────────────────────────────

    def prepare_dataset(self) -> pd.DataFrame:
        """
        Load, merge, and clean all factor data + labels.

        Returns:
            Cleaned DataFrame with factor features + composite_score target.
            Also stored in self.df.

        Example:
            >>> trainer = FactorModelTrainer()
            >>> df = trainer.prepare_dataset()
            >>> print(df.shape, df.columns.tolist())
        """
        logger.info("Preparing dataset...")
        factor_dfs = self._load_factors()
        if factor_dfs is None:
            self.df = pd.DataFrame()
            return self.df

        features = self._merge_factors(factor_dfs)
        labels = self._load_labels()

        if labels is None or features.empty or labels.empty:
            logger.warning("Features or labels empty after loading")
            self.df = pd.DataFrame()
            return self.df

        # Merge features with labels
        df = pd.merge(features, labels, on=["symbol", "date"], how="inner")
        df = df.dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Round extremely large values
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if "composite_score" in numeric_cols:
            numeric_cols.remove("composite_score")
        for col in numeric_cols:
            large_mask = df[col].abs() > 1e6
            if large_mask.any():
                df.loc[large_mask, col] = df.loc[large_mask, col].round(2)
                logger.debug(f"Rounded large values in '{col}'")

        self.df = df.copy()
        logger.info(f"Dataset prepared: {len(self.df)} rows, "
                     f"{len(self.FEATURE_COLS)} features, "
                     f"date range: {df['date'].min()} to {df['date'].max()}")
        return self.df

    # ───────────────────────────────────────────────────────────────────────
    # TIME-SERIES TRAIN/TEST SPLIT
    # ───────────────────────────────────────────────────────────────────────

    def _time_series_split(self, test_ratio: float = 0.2):
        """
        Split data by date to prevent data leakage.
        Earlier dates → training, later dates → testing.

        Args:
            test_ratio: Fraction of dates to use for testing (default 0.2)
        """
        dates = self.df["date"].sort_values().unique()
        split_idx = int(len(dates) * (1 - test_ratio))
        cutoff_date = dates[split_idx]

        train_mask = self.df["date"] < cutoff_date
        test_mask = self.df["date"] >= cutoff_date

        X = self.df[self.FEATURE_COLS]
        y = self.df["composite_score"]

        self.X_train = X[train_mask].values
        self.X_test = X[test_mask].values
        self.y_train = y[train_mask].values
        self.y_test = y[test_mask].values

        logger.info(f"Time-series split: train={self.X_train.shape[0]} rows "
                     f"(before {cutoff_date}), test={self.X_test.shape[0]} rows "
                     f"(from {cutoff_date})")

    # ───────────────────────────────────────────────────────────────────────
    # TRAINING
    # ───────────────────────────────────────────────────────────────────────

    def train_model(self, test_ratio: float = 0.2, **model_kwargs) -> dict:
        """
        Train XGBoost (or LightGBM fallback) model with time-series split.

        Steps:
            1. Time-series train/test split
            2. Feature scaling with StandardScaler
            3. Model training
            4. Evaluation (RMSE, MAE, R²)
            5. Save model + scaler to disk

        Args:
            test_ratio:    Fraction of dates for testing (default 0.2)
            **model_kwargs: Additional model hyperparameters

        Returns:
            Dict with metrics: rmse, mae, r2, model_path, model_type

        Example:
            >>> trainer = FactorModelTrainer()
            >>> trainer.prepare_dataset()
            >>> metrics = trainer.train_model()
            >>> print(f"R²: {metrics['r2']:.4f}")
        """
        if self.df is None or self.df.empty:
            logger.error("No data available. Run prepare_dataset() first.")
            return {}

        # Step 1: Time-series split
        self._time_series_split(test_ratio=test_ratio)

        # Step 2: Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Step 3: Get model class and train
        ModelClass, self.model_name = _get_model_class()

        default_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "random_state": 42,
        }
        if self.model_name == "xgboost":
            default_params["n_jobs"] = -1
            default_params["tree_method"] = "hist"
        elif self.model_name == "lightgbm":
            default_params["n_jobs"] = -1
            default_params["verbose"] = -1

        default_params.update(model_kwargs)

        logger.info(f"Training {self.model_name} model...")
        start_time = time.time()
        self.model = ModelClass(**default_params)
        self.model.fit(X_train_scaled, self.y_train)
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f}s")

        # Step 4: Evaluate
        self.y_pred = self.model.predict(X_test_scaled)
        rmse = float(np.sqrt(mean_squared_error(self.y_test, self.y_pred)))
        mae = float(mean_absolute_error(self.y_test, self.y_pred))
        r2 = float(r2_score(self.y_test, self.y_pred))

        logger.info(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

        # Feature importance
        self.feature_importances_ = self.model.feature_importances_

        # Step 5: Save
        model_path = self.model_dir / f"{self.model_name}_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "model_type": self.model_name,
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "train_samples": len(self.y_train),
            "test_samples": len(self.y_test),
            "training_time_seconds": round(train_time, 2),
        }

    # ───────────────────────────────────────────────────────────────────────
    # PREDICTION
    # ───────────────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.

        Args:
            X: DataFrame with factor features (value, momentum, quality,
               volatility, volume)

        Returns:
            Array of predicted composite scores

        Example:
            >>> preds = trainer.predict(new_data[trainer.FEATURE_COLS])
        """
        if self.model is None:
            raise RuntimeError("No trained model. Run train_model() first.")
        if self.scaler is None:
            raise RuntimeError("No scaler. Run train_model() first.")

        X_scaled = self.scaler.transform(X[self.FEATURE_COLS].values)
        return self.model.predict(X_scaled)

    def load_model(self, model_path: str = None, scaler_path: str = None):
        """
        Load a previously saved model and scaler from disk.

        Args:
            model_path:  Path to .pkl model file
            scaler_path: Path to .pkl scaler file
        """
        model_path = model_path or str(self.model_dir / "xgboost_model.pkl")
        scaler_path = scaler_path or str(self.model_dir / "scaler.pkl")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Loaded scaler from {scaler_path}")

    # ───────────────────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE
    # ───────────────────────────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
            DataFrame with columns: feature, importance (sorted descending)
        """
        if self.model is None:
            raise RuntimeError("No trained model. Run train_model() first.")

        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.FEATURE_COLS,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ───────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ───────────────────────────────────────────────────────────────────────

    def plot_actual_vs_predicted(self, save_path: str = None):
        """Plot actual vs predicted composite scores scatter plot."""
        if self.y_test is None or self.y_pred is None:
            logger.error("No predictions. Run train_model() first.")
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(self.y_test, self.y_pred, alpha=0.5, s=20, color="#2196F3")
        lims = [
            min(self.y_test.min(), self.y_pred.min()),
            max(self.y_test.max(), self.y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", lw=2, label="Perfect prediction")
        ax.set_xlabel("Actual Composite Score", fontsize=12)
        ax.set_ylabel("Predicted Composite Score", fontsize=12)
        ax.set_title("Actual vs. Predicted Composite Scores", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_feature_importance(self, save_path: str = None):
        """Plot feature importance bar chart."""
        if self.model is None:
            logger.error("No trained model.")
            return

        importance_df = self.get_feature_importance()

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#FF6F61", "#6B5B95", "#88B04B", "#F7CAC9", "#92A8D1"]
        ax.barh(
            importance_df["feature"],
            importance_df["importance"],
            color=colors[:len(importance_df)],
        )
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Feature Importances ({self.model_name.upper()})", fontsize=14)
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_residuals(self, save_path: str = None):
        """Plot residual distribution."""
        if self.y_test is None or self.y_pred is None:
            logger.error("No predictions. Run train_model() first.")
            return

        residuals = self.y_test - self.y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals vs predicted
        axes[0].scatter(self.y_pred, residuals, alpha=0.5, s=15, color="#FF6F61")
        axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.7)
        axes[0].set_xlabel("Predicted", fontsize=11)
        axes[0].set_ylabel("Residual", fontsize=11)
        axes[0].set_title("Residuals vs Predicted", fontsize=13)
        axes[0].grid(True, alpha=0.3)

        # Residual histogram
        axes[1].hist(residuals, bins=50, color="#6B5B95", alpha=0.7, edgecolor="white")
        axes[1].set_xlabel("Residual", fontsize=11)
        axes[1].set_ylabel("Frequency", fontsize=11)
        axes[1].set_title("Residual Distribution", fontsize=13)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE INTEGRATION — Store predictions
# ═══════════════════════════════════════════════════════════════════════════════

def store_predictions_in_db(df: pd.DataFrame, model_version: str = "xgboost_v1") -> int:
    """
    Store model predictions in the database ModelPredictions table.

    Args:
        df: DataFrame with columns: symbol, date, predicted_return + factor features
        model_version: String identifier for the model version

    Returns:
        Number of records stored
    """
    try:
        import json
        from utils.db_models import get_session, ModelPrediction

        session = get_session()
        count = 0
        for _, row in df.iterrows():
            features = {col: float(row[col]) for col in FactorModelTrainer.FEATURE_COLS if col in row}
            pred = ModelPrediction(
                ticker=row.get("symbol", ""),
                date=str(row.get("date", "")),
                predicted_return=float(row.get("predicted_return", 0)),
                model_version=model_version,
                features_json=json.dumps(features),
            )
            session.add(pred)
            count += 1

        session.commit()
        session.close()
        logger.info(f"Stored {count} predictions in database")
        return count
    except Exception as e:
        logger.error(f"Failed to store predictions: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    trainer = FactorModelTrainer(
        data_dir="data",
        model_dir="models",
    )

    # Prepare dataset
    df = trainer.prepare_dataset()
    if df.empty:
        print("No data found — check your data directory paths")
    else:
        # Train model
        metrics = trainer.train_model()
        print("\n=== Training Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # Feature importance
        print("\n=== Feature Importance ===")
        print(trainer.get_feature_importance().to_string(index=False))

        # Save plots
        trainer.plot_actual_vs_predicted(save_path="models/actual_vs_predicted.png")
        trainer.plot_feature_importance(save_path="models/feature_importance.png")
        trainer.plot_residuals(save_path="models/residuals.png")
