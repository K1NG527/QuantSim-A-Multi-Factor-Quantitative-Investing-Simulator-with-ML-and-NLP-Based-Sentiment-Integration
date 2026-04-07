"""
Unit tests for backtester.py — QuantSim

Tests:
    - Metrics calculation correctness
    - Transaction cost deduction
    - Slippage model
    - Rebalancing frequency filtering
    - Drawdown calculation
    - Benchmark comparison
"""

import os
import sys

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.backtester import (
    calculate_metrics,
    calculate_drawdown,
    calculate_daily_returns,
    calculate_portfolio_returns,
    _filter_rebalance_dates,
    _apply_slippage,
    compare_to_benchmark,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_returns():
    """Create a sample return series."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    return returns


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    prices = pd.DataFrame({
        "AAPL": 150 + np.cumsum(np.random.randn(100) * 2),
        "MSFT": 300 + np.cumsum(np.random.randn(100) * 3),
    }, index=dates)
    return prices


@pytest.fixture
def sample_weights():
    """Create a sample portfolio weights DataFrame."""
    dates = pd.date_range("2023-01-01", periods=4, freq="ME")
    rows = []
    for d in dates:
        rows.append({"rebalance_date": d, "symbol": "AAPL", "weight": 0.5})
        rows.append({"rebalance_date": d, "symbol": "MSFT", "weight": 0.5})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateMetrics:
    """Test performance metrics calculation."""

    def test_returns_correct_keys(self, sample_returns):
        """Metrics dict should have all expected keys."""
        metrics = calculate_metrics(sample_returns)
        expected_keys = [
            "cumulative_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "max_drawdown", "sortino_ratio", "calmar_ratio",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_max_drawdown_is_negative(self, sample_returns):
        """Max drawdown should be non-positive."""
        metrics = calculate_metrics(sample_returns)
        assert metrics["max_drawdown"] <= 0

    def test_sharpe_is_finite(self, sample_returns):
        """Sharpe ratio should be a finite number."""
        metrics = calculate_metrics(sample_returns)
        assert np.isfinite(metrics["sharpe_ratio"])

    def test_empty_returns(self):
        """Empty returns should produce NaN metrics."""
        metrics = calculate_metrics(pd.Series(dtype=float))
        assert np.isnan(metrics["cumulative_return"])

    def test_positive_only_returns(self):
        """All positive returns should yield positive cumulative return."""
        returns = pd.Series([0.01] * 100, index=pd.date_range("2023-01-01", periods=100))
        metrics = calculate_metrics(returns)
        assert metrics["cumulative_return"] > 0
        assert metrics["max_drawdown"] == 0  # No drawdown with only positive returns


class TestCalculateDrawdown:
    """Test drawdown calculation."""

    def test_drawdown_shape(self, sample_returns):
        """Drawdown series should have same length as returns."""
        dd = calculate_drawdown(sample_returns)
        assert len(dd) == len(sample_returns)

    def test_drawdown_always_non_positive(self, sample_returns):
        """All drawdown values should be <= 0."""
        dd = calculate_drawdown(sample_returns)
        assert (dd <= 0 + 1e-10).all()


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSACTION COSTS AND SLIPPAGE
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransactionCosts:
    """Test transaction cost deduction."""

    def test_costs_reduce_returns(self, sample_prices, sample_weights):
        """Portfolio returns with costs should be lower than without."""
        daily_returns = calculate_daily_returns(sample_prices)

        returns_no_cost = calculate_portfolio_returns(
            sample_weights.copy(), daily_returns, transaction_cost=0.0
        )
        returns_with_cost = calculate_portfolio_returns(
            sample_weights.copy(), daily_returns, transaction_cost=0.01
        )

        if len(returns_no_cost) > 0 and len(returns_with_cost) > 0:
            cum_no_cost = (1 + returns_no_cost).prod()
            cum_with_cost = (1 + returns_with_cost).prod()
            assert cum_with_cost <= cum_no_cost + 1e-10


class TestSlippage:
    """Test slippage model."""

    def test_slippage_reduces_returns(self, sample_returns):
        """Slippage should reduce cumulative returns."""
        rebalance_dates = [sample_returns.index[0], sample_returns.index[50]]

        adjusted = _apply_slippage(sample_returns.copy(), slippage_bps=10, rebalance_dates=rebalance_dates)
        cum_original = (1 + sample_returns).prod()
        cum_adjusted = (1 + adjusted).prod()
        assert cum_adjusted <= cum_original + 1e-10

    def test_zero_slippage_unchanged(self, sample_returns):
        """Zero slippage should not change returns."""
        adjusted = _apply_slippage(sample_returns.copy(), slippage_bps=0, rebalance_dates=[])
        pd.testing.assert_series_equal(adjusted, sample_returns)


# ═══════════════════════════════════════════════════════════════════════════════
# REBALANCING FREQUENCY
# ═══════════════════════════════════════════════════════════════════════════════

class TestRebalanceFrequency:
    """Test rebalance date filtering."""

    def test_daily_returns_all_dates(self):
        """Daily frequency should return all dates."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B").tolist()
        filtered = _filter_rebalance_dates(dates, "daily")
        assert len(filtered) == len(dates)

    def test_monthly_reduces_dates(self):
        """Monthly frequency should have fewer dates than daily."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B").tolist()
        filtered = _filter_rebalance_dates(dates, "monthly")
        assert len(filtered) < len(dates)
        assert len(filtered) >= 3  # ~5 months of data

    def test_quarterly_reduces_further(self):
        """Quarterly should have fewer dates than monthly."""
        dates = pd.date_range("2023-01-01", periods=252, freq="B").tolist()
        monthly = _filter_rebalance_dates(dates, "monthly")
        quarterly = _filter_rebalance_dates(dates, "quarterly")
        assert len(quarterly) <= len(monthly)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkComparison:
    """Test benchmark comparison functionality."""

    def test_comparison_output_keys(self, sample_returns):
        """Benchmark comparison should return expected keys."""
        benchmark = pd.Series(
            np.random.randn(252) * 0.01,
            index=sample_returns.index,
        )
        result = compare_to_benchmark(sample_returns, benchmark)

        assert "alpha" in result
        assert "beta" in result
        assert "tracking_error" in result
        assert "information_ratio" in result

    def test_beta_around_one_for_same_returns(self):
        """Beta should be ~1 when portfolio = benchmark."""
        returns = pd.Series(
            [0.01, -0.005, 0.02, -0.01, 0.005],
            index=pd.date_range("2023-01-01", periods=5),
        )
        result = compare_to_benchmark(returns, returns)
        assert abs(result["beta"] - 1.0) < 0.01

    def test_no_common_dates(self):
        """Should return empty dict when no common dates."""
        r1 = pd.Series([0.01], index=pd.DatetimeIndex(["2023-01-01"]))
        r2 = pd.Series([0.01], index=pd.DatetimeIndex(["2024-01-01"]))
        result = compare_to_benchmark(r1, r2)
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
