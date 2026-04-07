"""
SHAP Explainability Module — QuantSim

Provides per-prediction and global factor explainability using SHAP values.

Features:
    - TreeExplainer for XGBoost/LightGBM models
    - Per-stock SHAP values showing top contributing factors
    - Global feature importance aggregation
    - Saves explainability.json and visual plots
    - Integrates with the ML pipeline's trained model

Usage:
    from scripts.explainability import explain_model
    results = explain_model(trainer)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(model, X: np.ndarray, feature_names: list) -> dict:
    """
    Compute SHAP values using TreeExplainer.

    Args:
        model:         Trained XGBoost or LightGBM model
        X:             Feature array (scaled)
        feature_names: List of feature column names

    Returns:
        Dict with 'shap_values' (np.ndarray), 'base_value' (float),
        'feature_names' (list)

    Example:
        >>> result = compute_shap_values(model, X_test, ["value", "momentum", ...])
        >>> print(result["shap_values"].shape)
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap")
        raise ImportError("Install shap: pip install shap")

    logger.info(f"Computing SHAP values for {X.shape[0]} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return {
        "shap_values": shap_values,
        "base_value": float(explainer.expected_value)
            if isinstance(explainer.expected_value, (int, float, np.floating))
            else float(explainer.expected_value[0]),
        "feature_names": feature_names,
        "X": X,
    }


def get_top_factors_per_sample(shap_values: np.ndarray, feature_names: list,
                               top_k: int = 3) -> list[dict]:
    """
    For each sample, identify the top-k contributing factors.

    Args:
        shap_values:   Array of SHAP values (n_samples x n_features)
        feature_names: List of feature names
        top_k:         Number of top factors to return

    Returns:
        List of dicts, one per sample, each with 'top_factors' list

    Example:
        >>> factors = get_top_factors_per_sample(shap_vals, features, top_k=3)
        >>> print(factors[0])
        {'top_factors': [{'feature': 'momentum', 'shap_value': 0.45}, ...]}
    """
    results = []
    for i in range(shap_values.shape[0]):
        sample_shap = shap_values[i]
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]
        top_factors = [
            {
                "feature": feature_names[idx],
                "shap_value": round(float(sample_shap[idx]), 6),
                "direction": "positive" if sample_shap[idx] > 0 else "negative",
            }
            for idx in sorted_idx
        ]
        results.append({"sample_index": i, "top_factors": top_factors})
    return results


def get_global_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Compute aggregate (mean absolute) SHAP feature importance.

    Args:
        shap_values:   Array of SHAP values (n_samples x n_features)
        feature_names: List of feature names

    Returns:
        DataFrame with columns: feature, mean_abs_shap, rank

    Example:
        >>> importance = get_global_importance(shap_vals, features)
        >>> print(importance)
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.round(mean_abs, 6),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_shap_bar(shap_values: np.ndarray, feature_names: list,
                  save_path: str = None):
    """
    Generate SHAP global feature importance bar chart.

    Args:
        shap_values:  Array of SHAP values
        feature_names: Feature names
        save_path:    Path to save PNG (None = display)
    """
    importance = get_global_importance(shap_values, feature_names)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E91E63", "#9C27B0", "#2196F3", "#4CAF50", "#FF9800"]
    ax.barh(
        importance["feature"],
        importance["mean_abs_shap"],
        color=colors[:len(importance)],
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("SHAP Global Feature Importance", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"SHAP bar chart saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_shap_summary(shap_result: dict, save_path: str = None):
    """
    Generate SHAP beeswarm/summary plot.

    Args:
        shap_result: Dict from compute_shap_values()
        save_path:   Path to save PNG (None = display)
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed — cannot generate summary plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_result["shap_values"],
        features=shap_result["X"],
        feature_names=shap_result["feature_names"],
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Impact Summary", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"SHAP summary plot saved to {save_path}")
    else:
        plt.show()
    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_explainability_json(global_importance: pd.DataFrame,
                             per_sample_factors: list,
                             base_value: float,
                             output_path: str = "data/explainability.json") -> str:
    """
    Save explainability results to JSON file.

    Args:
        global_importance:  DataFrame from get_global_importance()
        per_sample_factors: List from get_top_factors_per_sample()
        base_value:         SHAP base value
        output_path:        Path to save JSON

    Returns:
        Path to saved JSON file

    Example:
        >>> path = save_explainability_json(importance_df, factors, 0.5)
        >>> print(f"Saved to {path}")
    """
    output = {
        "metadata": {
            "model_type": "XGBoost/LightGBM",
            "explainer": "SHAP TreeExplainer",
            "num_samples_analyzed": len(per_sample_factors),
        },
        "global_feature_importance": global_importance.to_dict(orient="records"),
        "base_value": base_value,
        "per_sample_top_factors": per_sample_factors[:100],  # Cap to first 100
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Explainability JSON saved to {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER EXPLAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def explain_model(trainer, output_dir: str = "data",
                  plot_dir: str = "models") -> dict:
    """
    Run full explainability pipeline on a trained FactorModelTrainer.

    Steps:
        1. Compute SHAP values on test set
        2. Extract per-sample top factors
        3. Compute global importance
        4. Save JSON + plots

    Args:
        trainer:    Trained FactorModelTrainer instance (must have model + test data)
        output_dir: Directory for JSON output
        plot_dir:   Directory for plot PNGs

    Returns:
        Dict with: global_importance (DataFrame), per_sample_factors (list),
        json_path (str), plot_paths (dict)

    Example:
        >>> from models.ml_pipeline import FactorModelTrainer
        >>> trainer = FactorModelTrainer()
        >>> trainer.prepare_dataset()
        >>> trainer.train_model()
        >>> results = explain_model(trainer)
        >>> print(results["global_importance"])
    """
    if trainer.model is None:
        raise RuntimeError("No trained model in trainer. Run train_model() first.")
    if trainer.X_test is None:
        raise RuntimeError("No test data. Run train_model() first.")

    feature_names = trainer.FEATURE_COLS
    X_test = trainer.X_test  # Already scaled by the trainer

    # Scale test data if scaler exists
    if trainer.scaler is not None:
        X_test_scaled = trainer.scaler.transform(X_test) if X_test.ndim == 2 else X_test
    else:
        X_test_scaled = X_test

    # Step 1: Compute SHAP values
    shap_result = compute_shap_values(trainer.model, X_test_scaled, feature_names)

    # Step 2: Per-sample factors
    per_sample = get_top_factors_per_sample(
        shap_result["shap_values"], feature_names, top_k=3
    )

    # Step 3: Global importance
    global_imp = get_global_importance(shap_result["shap_values"], feature_names)

    # Step 4: Save outputs
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    json_path = save_explainability_json(
        global_imp, per_sample, shap_result["base_value"],
        output_path=os.path.join(output_dir, "explainability.json"),
    )

    bar_path = os.path.join(plot_dir, "shap_bar.png")
    summary_path = os.path.join(plot_dir, "shap_summary.png")
    plot_shap_bar(shap_result["shap_values"], feature_names, save_path=bar_path)
    plot_shap_summary(shap_result, save_path=summary_path)

    logger.info("Explainability pipeline complete")

    return {
        "global_importance": global_imp,
        "per_sample_factors": per_sample,
        "json_path": json_path,
        "plot_paths": {"bar": bar_path, "summary": summary_path},
        "shap_result": shap_result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from models.ml_pipeline import FactorModelTrainer

    trainer = FactorModelTrainer(data_dir="data", model_dir="models")
    df = trainer.prepare_dataset()

    if not df.empty:
        trainer.train_model()
        results = explain_model(trainer)

        print("\n=== Global Feature Importance (SHAP) ===")
        print(results["global_importance"].to_string(index=False))
        print(f"\nJSON saved to: {results['json_path']}")
        print(f"Plots saved to: {results['plot_paths']}")
    else:
        print("No data — check data directory paths")
