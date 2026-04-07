"""
Backtester — Production-grade backtesting engine for EquiSense.

Improvements over original:
    - Transaction costs (configurable, default 0.1%)
    - Slippage model (configurable basis points)
    - Rebalancing frequency (daily/monthly/quarterly)
    - Benchmark comparison (S&P 500)
    - All existing functions preserved with new optional parameters

Usage:
    returns, metrics = run_backtest(
        "portfolio.csv", "prices.csv",
        transaction_cost=0.001, slippage_bps=5,
        rebalance_freq="monthly"
    )
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_price_data(price_file: str) -> pd.DataFrame:
    """
    Load and pivot price data from CSV.

    Args:
        price_file: Path to CSV with columns: date, symbol, close

    Returns:
        Pivoted DataFrame: DatetimeIndex x symbols, values = close prices

    Example:
        >>> prices = load_price_data("data/daily_prices_clean.csv")
        >>> print(prices.shape)
    """
    prices = pd.read_csv(price_file)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(by=["date"]).drop_duplicates(
        subset=["date", "symbol"], keep="last"
    )
    return prices.pivot(index="date", columns="symbol", values="close")


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from price data.

    Args:
        prices: Pivoted price DataFrame (date x symbols)

    Returns:
        DataFrame of daily returns, same shape as input
    """
    return prices.pct_change(fill_method=None).fillna(0)


# ═══════════════════════════════════════════════════════════════════════════════
# REBALANCING SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_rebalance_dates(rebalance_dates: list, freq: str) -> list:
    """
    Filter rebalance dates based on frequency.

    Args:
        rebalance_dates: Sorted list of datetime rebalance points
        freq: "daily", "monthly", or "quarterly"

    Returns:
        Filtered list of rebalance dates
    """
    if freq == "daily" or not rebalance_dates:
        return rebalance_dates

    filtered = [rebalance_dates[0]]
    for dt in rebalance_dates[1:]:
        prev = filtered[-1]
        if freq == "monthly":
            if dt.month != prev.month or dt.year != prev.year:
                filtered.append(dt)
        elif freq == "quarterly":
            prev_q = (prev.month - 1) // 3
            curr_q = (dt.month - 1) // 3
            if curr_q != prev_q or dt.year != prev.year:
                filtered.append(dt)

    logger.info(f"Rebalancing: {len(rebalance_dates)} -> {len(filtered)} dates ({freq})")
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# SLIPPAGE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_slippage(returns: pd.Series, slippage_bps: float,
                    rebalance_dates: list) -> pd.Series:
    """
    Apply slippage costs at rebalance points.

    Simulates execution impact: at each rebalance, a slippage penalty is
    applied proportional to the defined basis points.

    Args:
        returns:         Portfolio return series
        slippage_bps:    Slippage in basis points (e.g., 5 = 0.05%)
        rebalance_dates: List of rebalance datetimes

    Returns:
        Returns series with slippage deducted at rebalance points
    """
    if slippage_bps <= 0:
        return returns

    slippage_frac = slippage_bps / 10000.0
    adjusted = returns.copy()
    for dt in rebalance_dates:
        # Find the nearest trading day after rebalance
        mask = adjusted.index > dt
        if mask.any():
            first_day = adjusted.index[mask][0]
            adjusted.loc[first_day] -= slippage_frac

    return adjusted


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO RETURNS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_portfolio_returns(
    portfolio_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    transaction_cost: float = 0.0,
    slippage_bps: float = 0.0,
    rebalance_freq: str = "daily",
) -> pd.Series:
    """
    Calculate portfolio returns with transaction costs and slippage.

    Args:
        portfolio_weights: DataFrame with columns: rebalance_date, symbol, weight
        daily_returns:     DataFrame of daily returns (date x symbols)
        transaction_cost:  Proportional cost per rebalance (default 0.0, e.g. 0.001 = 0.1%)
        slippage_bps:      Slippage in basis points (default 0.0)
        rebalance_freq:    "daily", "monthly", or "quarterly"

    Returns:
        Series of daily portfolio returns

    Example:
        >>> returns = calculate_portfolio_returns(
        ...     weights_df, daily_returns,
        ...     transaction_cost=0.001, slippage_bps=5,
        ...     rebalance_freq="monthly"
        ... )
    """
    portfolio_returns = pd.Series(index=daily_returns.index, dtype=float)

    if not pd.api.types.is_datetime64_any_dtype(portfolio_weights["rebalance_date"]):
        portfolio_weights["rebalance_date"] = pd.to_datetime(
            portfolio_weights["rebalance_date"]
        )

    portfolio_weights = portfolio_weights.sort_values("rebalance_date")
    rebalance_dates = (
        portfolio_weights["rebalance_date"].drop_duplicates().sort_values().tolist()
    )

    # Apply rebalance frequency filter
    rebalance_dates = _filter_rebalance_dates(rebalance_dates, rebalance_freq)
    n = len(rebalance_dates)

    prev_weights = None  # Track previous weights for turnover calculation

    for i, rebalance_date in enumerate(rebalance_dates):
        if i < n - 1:
            next_rebalance = rebalance_dates[i + 1]
            period_mask = (daily_returns.index > rebalance_date) & (
                daily_returns.index <= next_rebalance
            )
        else:
            period_mask = daily_returns.index > rebalance_date

        period_dates = daily_returns.index[period_mask]
        group = portfolio_weights[
            portfolio_weights["rebalance_date"] == rebalance_date
        ]
        weights = group.set_index("symbol")["weight"]
        valid_stocks = weights.index.intersection(daily_returns.columns)
        weights = weights[valid_stocks]

        if len(valid_stocks) == 0:
            continue

        weights = weights / weights.sum()

        # Calculate turnover for transaction costs
        if transaction_cost > 0 and prev_weights is not None:
            common = weights.index.intersection(prev_weights.index)
            turnover = 0.0
            if len(common) > 0:
                turnover = (weights.reindex(common, fill_value=0) -
                            prev_weights.reindex(common, fill_value=0)).abs().sum()
            # Add new positions and removed positions
            new_positions = weights.index.difference(prev_weights.index)
            removed_positions = prev_weights.index.difference(weights.index)
            turnover += weights.reindex(new_positions, fill_value=0).abs().sum()
            turnover += prev_weights.reindex(removed_positions, fill_value=0).abs().sum()
            tc_cost = turnover * transaction_cost
        else:
            tc_cost = transaction_cost if transaction_cost > 0 else 0.0

        prev_weights = weights.copy()

        if len(period_dates) > 0:
            returns_slice = daily_returns.loc[period_dates, valid_stocks]
            returns_slice = returns_slice[weights.index]
            period_returns = returns_slice.dot(weights)

            # Deduct transaction cost on first day of period
            if tc_cost > 0 and len(period_returns) > 0:
                period_returns.iloc[0] -= tc_cost

            portfolio_returns.loc[period_dates] = period_returns

    portfolio_returns = portfolio_returns.dropna()

    # Apply slippage
    if slippage_bps > 0:
        portfolio_returns = _apply_slippage(
            portfolio_returns, slippage_bps, rebalance_dates
        )

    return portfolio_returns


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculate portfolio performance metrics.

    Args:
        returns:        Series of daily portfolio returns
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Dict with: cumulative_return, annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio

    Example:
        >>> metrics = calculate_metrics(portfolio_returns)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    if len(returns) == 0:
        return {k: np.nan for k in [
            "cumulative_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "max_drawdown", "sortino_ratio", "calmar_ratio"
        ]}

    cumulative_returns = (1 + returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1]
    n_days = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1
    annualized_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_vol
        if annualized_vol != 0 else np.nan
    )

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Sortino ratio (uses downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino_ratio = (
        (annualized_return - risk_free_rate) / downside_std
        if downside_std and downside_std != 0 else np.nan
    )

    # Calmar ratio
    calmar_ratio = (
        annualized_return / abs(max_drawdown)
        if max_drawdown != 0 else np.nan
    )

    return {
        "cumulative_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
    }


def calculate_drawdown(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from returns.

    Args:
        returns: Series of daily returns

    Returns:
        Series of drawdown values
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    return (cumulative - peak) / peak


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def load_benchmark(benchmark_file: str = "data/sp500.csv") -> pd.Series:
    """
    Load S&P 500 benchmark data.

    Args:
        benchmark_file: Path to S&P 500 CSV (date index, Close column)

    Returns:
        Series of daily benchmark returns

    Example:
        >>> bench_returns = load_benchmark("data/sp500.csv")
        >>> bench_metrics = calculate_metrics(bench_returns)
    """
    try:
        sp500 = pd.read_csv(benchmark_file, header=2, parse_dates=True, index_col=0)
        sp500.columns = ["Close"]
        sp500 = sp500.sort_index()
        returns = sp500["Close"].pct_change().dropna()
        logger.info(f"Loaded benchmark: {len(returns)} days")
        return returns
    except Exception as e:
        logger.error(f"Failed to load benchmark: {e}")
        return pd.Series(dtype=float)


def compare_to_benchmark(portfolio_returns: pd.Series,
                         benchmark_returns: pd.Series) -> dict:
    """
    Compare portfolio performance against benchmark.

    Args:
        portfolio_returns: Series of daily portfolio returns
        benchmark_returns: Series of daily benchmark returns

    Returns:
        Dict with portfolio_metrics, benchmark_metrics, alpha, beta,
        information_ratio, tracking_error

    Example:
        >>> comparison = compare_to_benchmark(pf_returns, bench_returns)
        >>> print(f"Alpha: {comparison['alpha']:.4f}")
    """
    # Align dates
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) == 0:
        logger.warning("No common dates between portfolio and benchmark")
        return {}

    pf = portfolio_returns.loc[common_idx]
    bm = benchmark_returns.loc[common_idx]

    pf_metrics = calculate_metrics(pf)
    bm_metrics = calculate_metrics(bm)

    # Beta (covariance / variance)
    covariance = pf.cov(bm)
    bm_variance = bm.var()
    beta = covariance / bm_variance if bm_variance != 0 else np.nan

    # Alpha (Jensen's alpha)
    alpha = pf_metrics["annualized_return"] - (beta * bm_metrics["annualized_return"])

    # Tracking error
    excess = pf - bm
    tracking_error = excess.std() * np.sqrt(252)

    # Information ratio
    info_ratio = (
        (pf_metrics["annualized_return"] - bm_metrics["annualized_return"]) / tracking_error
        if tracking_error != 0 else np.nan
    )

    return {
        "portfolio_metrics": pf_metrics,
        "benchmark_metrics": bm_metrics,
        "alpha": alpha,
        "beta": beta,
        "tracking_error": tracking_error,
        "information_ratio": info_ratio,
        "common_days": len(common_idx),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    portfolio_file: str,
    price_file: str,
    transaction_cost: float = 0.0,
    slippage_bps: float = 0.0,
    rebalance_freq: str = "daily",
    benchmark_file: Optional[str] = None,
) -> tuple:
    """
    Run a complete backtest.

    Args:
        portfolio_file:   CSV with columns: rebalance_date, symbol, weight
        price_file:       CSV with columns: date, symbol, close
        transaction_cost: Proportional cost per rebalance (e.g. 0.001 = 0.1%)
        slippage_bps:     Slippage in basis points (e.g. 5)
        rebalance_freq:   Rebalancing frequency: "daily", "monthly", "quarterly"
        benchmark_file:   Optional path to benchmark CSV for comparison

    Returns:
        Tuple of (portfolio_returns: Series, metrics: dict)
        If benchmark_file provided, metrics includes benchmark comparison

    Example:
        >>> returns, metrics = run_backtest(
        ...     "portfolio.csv", "prices.csv",
        ...     transaction_cost=0.001, slippage_bps=5,
        ...     rebalance_freq="monthly",
        ...     benchmark_file="data/sp500.csv"
        ... )
    """
    portfolio_weights = pd.read_csv(portfolio_file)
    prices = load_price_data(price_file)
    daily_returns = calculate_daily_returns(prices)

    portfolio_returns = calculate_portfolio_returns(
        portfolio_weights, daily_returns,
        transaction_cost=transaction_cost,
        slippage_bps=slippage_bps,
        rebalance_freq=rebalance_freq,
    )

    metrics = calculate_metrics(portfolio_returns)

    # Add benchmark comparison if available
    if benchmark_file:
        bench = load_benchmark(benchmark_file)
        if not bench.empty:
            comparison = compare_to_benchmark(portfolio_returns, bench)
            metrics["benchmark_comparison"] = comparison

    logger.info(
        f"Backtest complete: {len(portfolio_returns)} days, "
        f"Sharpe={metrics.get('sharpe_ratio', 'N/A'):.2f}, "
        f"MaxDD={metrics.get('max_drawdown', 'N/A'):.2%}"
    )

    return portfolio_returns, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    prices_path = "data/processed/clean_dataset/daily_prices_clean.csv"
    portfolio_path = "data/processed/portfolio_weight/equal_weight_portfolio.csv"

    returns, metrics = run_backtest(
        portfolio_path, prices_path,
        transaction_cost=0.001,
        slippage_bps=5,
        rebalance_freq="monthly",
        benchmark_file="data/sp500.csv",
    )

    print("=== Backtest Results ===")
    for k, v in metrics.items():
        if k != "benchmark_comparison":
            print(f"  {k}: {v}")

    if "benchmark_comparison" in metrics:
        comp = metrics["benchmark_comparison"]
        print(f"\n=== Benchmark Comparison ===")
        print(f"  Alpha:             {comp.get('alpha', 'N/A')}")
        print(f"  Beta:              {comp.get('beta', 'N/A')}")
        print(f"  Information Ratio: {comp.get('information_ratio', 'N/A')}")
        print(f"  Tracking Error:    {comp.get('tracking_error', 'N/A')}")