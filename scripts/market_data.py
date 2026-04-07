"""
Market Data Layer — Production-grade module for EquiSense.

Sources:
    Primary:  Alpha Vantage (daily prices, fundamentals, technical indicators)
    Backup:   yfinance (automatic fallback if Alpha Vantage fails)

Features:
    - File-based caching (data/cache/) with configurable TTL
    - Clean pandas DataFrame output
    - Graceful error handling with logging
    - Backward-compatible enrich_stock / enrich_holdings API
"""

import os
import json
import math
import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"
CACHE_DIR = Path(os.path.dirname(__file__)).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache TTL in seconds
PRICE_CACHE_TTL = 86400       # 24 hours
FUNDAMENTAL_CACHE_TTL = 604800  # 7 days
INDICATOR_CACHE_TTL = 86400    # 24 hours


# ═══════════════════════════════════════════════════════════════════════════════
# CACHING LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def _cache_key(prefix: str, *args) -> str:
    """Generate a filesystem-safe cache key."""
    raw = f"{prefix}_{'_'.join(str(a) for a in args)}"
    hashed = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{prefix}_{hashed}"


def _read_cache(key: str, ttl_seconds: int) -> Optional[pd.DataFrame]:
    """Read from file cache if not expired. Returns DataFrame or None."""
    cache_file = CACHE_DIR / f"{key}.parquet"
    meta_file = CACHE_DIR / f"{key}.meta.json"

    if not cache_file.exists() or not meta_file.exists():
        return None

    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
        cached_at = datetime.fromisoformat(meta["cached_at"])
        if (datetime.now() - cached_at).total_seconds() > ttl_seconds:
            logger.debug(f"Cache expired for {key}")
            return None
        df = pd.read_parquet(cache_file)
        logger.info(f"Cache hit: {key} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"Cache read failed for {key}: {e}")
        return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    """Write DataFrame to file cache with metadata."""
    try:
        cache_file = CACHE_DIR / f"{key}.parquet"
        meta_file = CACHE_DIR / f"{key}.meta.json"
        df.to_parquet(cache_file, index=True)
        with open(meta_file, "w") as f:
            json.dump({"cached_at": datetime.now().isoformat(), "rows": len(df)}, f)
        logger.debug(f"Cache written: {key} ({len(df)} rows)")
    except Exception as e:
        logger.warning(f"Cache write failed for {key}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHA VANTAGE — PRIMARY SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

def _av_request(params: dict, timeout: int = 15) -> dict:
    """Make an Alpha Vantage API request with error handling."""
    if not ALPHA_VANTAGE_KEY or ALPHA_VANTAGE_KEY == "your_key_here":
        raise ValueError("ALPHA_VANTAGE_KEY not configured in .env")
    params["apikey"] = ALPHA_VANTAGE_KEY
    resp = requests.get(BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Check for API error messages
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
        raise ConnectionError("Alpha Vantage API rate limit reached")
    return data


def _av_get_price_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch daily price data from Alpha Vantage."""
    try:
        data = _av_request({
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
        })
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return None

        records = []
        for date_str, values in ts.items():
            records.append({
                "date": pd.to_datetime(date_str),
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"]),
            })

        df = pd.DataFrame(records)
        df = df.set_index("date").sort_index()

        # Filter date range
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]

        if df.empty:
            return None

        logger.info(f"Alpha Vantage: fetched {len(df)} days for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"Alpha Vantage price fetch failed for {ticker}: {e}")
        return None


def _av_get_fundamentals(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch company fundamentals from Alpha Vantage OVERVIEW endpoint."""
    try:
        data = _av_request({"function": "OVERVIEW", "symbol": ticker})
        if not data or "Symbol" not in data:
            return None

        fundamentals = {
            "ticker": data.get("Symbol"),
            "name": data.get("Name"),
            "sector": data.get("Sector"),
            "industry": data.get("Industry"),
            "market_cap": _safe_float(data.get("MarketCapitalization")),
            "pe_ratio": _safe_float(data.get("PERatio")),
            "pb_ratio": _safe_float(data.get("PriceToBookRatio")),
            "dividend_yield": _safe_float(data.get("DividendYield")),
            "eps": _safe_float(data.get("EPS")),
            "revenue_per_share": _safe_float(data.get("RevenuePerShareTTM")),
            "profit_margin": _safe_float(data.get("ProfitMargin")),
            "roe": _safe_float(data.get("ReturnOnEquityTTM")),
            "roa": _safe_float(data.get("ReturnOnAssetsTTM")),
            "beta": _safe_float(data.get("Beta")),
            "52_week_high": _safe_float(data.get("52WeekHigh")),
            "52_week_low": _safe_float(data.get("52WeekLow")),
            "50_day_ma": _safe_float(data.get("50DayMovingAverage")),
            "200_day_ma": _safe_float(data.get("200DayMovingAverage")),
        }
        df = pd.DataFrame([fundamentals])
        logger.info(f"Alpha Vantage: fetched fundamentals for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"Alpha Vantage fundamentals failed for {ticker}: {e}")
        return None


def _av_get_technical_indicators(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch RSI + MACD from Alpha Vantage."""
    try:
        # RSI
        rsi_data = _av_request({
            "function": "RSI",
            "symbol": ticker,
            "interval": "daily",
            "time_period": 14,
            "series_type": "close",
        })
        rsi_ts = rsi_data.get("Technical Analysis: RSI", {})

        # MACD (separate request — respect rate limit)
        time.sleep(1)
        macd_data = _av_request({
            "function": "MACD",
            "symbol": ticker,
            "interval": "daily",
            "series_type": "close",
        })
        macd_ts = macd_data.get("Technical Analysis: MACD", {})

        # Merge
        records = {}
        for date_str, vals in rsi_ts.items():
            records.setdefault(date_str, {})["rsi"] = float(vals.get("RSI", 0))
        for date_str, vals in macd_ts.items():
            records.setdefault(date_str, {})["macd"] = float(vals.get("MACD", 0))
            records[date_str]["macd_signal"] = float(vals.get("MACD_Signal", 0))
            records[date_str]["macd_hist"] = float(vals.get("MACD_Hist", 0))

        if not records:
            return None

        df = pd.DataFrame.from_dict(records, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        logger.info(f"Alpha Vantage: fetched indicators for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"Alpha Vantage indicators failed for {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# YFINANCE — BACKUP SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

def _yf_get_price_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch daily price data from yfinance as fallback."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end)
        if df.empty:
            return None
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df.index.name = "date"
        df = df[["open", "high", "low", "close", "volume"]]
        logger.info(f"yfinance: fetched {len(df)} days for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"yfinance price fetch failed for {ticker}: {e}")
        return None


def _yf_get_fundamentals(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch fundamentals from yfinance as fallback."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        if not info.get("symbol"):
            return None

        fundamentals = {
            "ticker": info.get("symbol"),
            "name": info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "eps": info.get("trailingEps"),
            "revenue_per_share": info.get("revenuePerShare"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "beta": info.get("beta"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "50_day_ma": info.get("fiftyDayAverage"),
            "200_day_ma": info.get("twoHundredDayAverage"),
        }
        df = pd.DataFrame([fundamentals])
        logger.info(f"yfinance: fetched fundamentals for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"yfinance fundamentals failed for {ticker}: {e}")
        return None


def _yf_get_technical_indicators(ticker: str) -> Optional[pd.DataFrame]:
    """Compute RSI + MACD from yfinance historical data as fallback."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo")
        if hist.empty or len(hist) < 30:
            return None

        close = hist["Close"]

        # RSI (14-day)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal

        df = pd.DataFrame({
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": macd_hist,
        }, index=hist.index)
        df.index.name = "date"
        df = df.dropna()
        logger.info(f"yfinance: computed indicators for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"yfinance indicators failed for {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — PRIMARY + FALLBACK + CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV price data for a ticker.

    Tries Alpha Vantage first, falls back to yfinance.
    Results are cached for 24 hours.

    Args:
        ticker: Stock symbol, e.g. "AAPL"
        start:  Start date string, e.g. "2023-01-01"
        end:    End date string, e.g. "2024-01-01"

    Returns:
        DataFrame with columns: open, high, low, close, volume
        DatetimeIndex named 'date'

    Example:
        >>> df = get_price_data("AAPL", "2023-01-01", "2024-01-01")
        >>> print(df.head())
    """
    cache_key = _cache_key("price", ticker, start, end)
    cached = _read_cache(cache_key, PRICE_CACHE_TTL)
    if cached is not None:
        return cached

    # Try Alpha Vantage
    df = _av_get_price_data(ticker, start, end)

    # Fallback to yfinance
    if df is None:
        logger.info(f"Falling back to yfinance for {ticker} price data")
        df = _yf_get_price_data(ticker, start, end)

    if df is None:
        logger.error(f"All sources failed for {ticker} price data")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    _write_cache(cache_key, df)
    return df


def get_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Fetch company fundamentals (PE, PB, EPS, margins, etc.).

    Tries Alpha Vantage first, falls back to yfinance.
    Results are cached for 7 days.

    Args:
        ticker: Stock symbol, e.g. "AAPL"

    Returns:
        Single-row DataFrame with fundamental metrics.

    Example:
        >>> df = get_fundamentals("AAPL")
        >>> print(df[["pe_ratio", "market_cap", "sector"]])
    """
    cache_key = _cache_key("fund", ticker)
    cached = _read_cache(cache_key, FUNDAMENTAL_CACHE_TTL)
    if cached is not None:
        return cached

    df = _av_get_fundamentals(ticker)
    if df is None:
        logger.info(f"Falling back to yfinance for {ticker} fundamentals")
        df = _yf_get_fundamentals(ticker)

    if df is None:
        logger.error(f"All sources failed for {ticker} fundamentals")
        return pd.DataFrame()

    _write_cache(cache_key, df)
    return df


def get_technical_indicators(ticker: str) -> pd.DataFrame:
    """
    Fetch technical indicators (RSI, MACD) for a ticker.

    Tries Alpha Vantage first, falls back to yfinance computation.
    Results are cached for 24 hours.

    Args:
        ticker: Stock symbol, e.g. "AAPL"

    Returns:
        DataFrame with columns: rsi, macd, macd_signal, macd_hist
        DatetimeIndex named 'date'

    Example:
        >>> df = get_technical_indicators("AAPL")
        >>> print(df.tail())
    """
    cache_key = _cache_key("tech", ticker)
    cached = _read_cache(cache_key, INDICATOR_CACHE_TTL)
    if cached is not None:
        return cached

    df = _av_get_technical_indicators(ticker)
    if df is None:
        logger.info(f"Falling back to yfinance for {ticker} indicators")
        df = _yf_get_technical_indicators(ticker)

    if df is None:
        logger.error(f"All sources failed for {ticker} indicators")
        return pd.DataFrame(columns=["rsi", "macd", "macd_signal", "macd_hist"])

    _write_cache(cache_key, df)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE API (used by existing document_analyzer pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(val) -> Optional[float]:
    """Safely convert a value to float."""
    if val is None or val == "None" or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _try_yfinance(symbol: str) -> Optional[dict]:
    """Fetch enrichment data from yfinance. Returns dict or None on failure."""
    try:
        import yfinance as yf
        suffixes = [".NS", ".BO", ""]
        for suffix in suffixes:
            ticker_str = symbol.upper() + suffix
            ticker = yf.Ticker(ticker_str)
            info = ticker.info or {}

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price and price > 0:
                volatility = None
                try:
                    hist = ticker.history(period="1mo")
                    if len(hist) >= 5:
                        returns = hist["Close"].pct_change().dropna()
                        volatility = round(float(returns.std() * math.sqrt(252)), 4)
                except Exception:
                    pass

                return {
                    "current_price": round(float(price), 2),
                    "price_change": round(float(info.get("regularMarketChangePercent", 0)), 2),
                    "sector": info.get("sector", None),
                    "industry": info.get("industry", None),
                    "volatility_proxy": volatility,
                    "source": f"yfinance ({ticker_str})",
                    "enriched": "yes",
                }
        return None
    except Exception:
        return None


def _try_alpha_vantage(symbol: str) -> Optional[dict]:
    """Fetch enrichment data from Alpha Vantage. Returns dict or None on failure."""
    api_key = ALPHA_VANTAGE_KEY
    if not api_key or api_key == "your_key_here":
        return None

    try:
        url = BASE_URL
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": f"{symbol}.BSE",
            "apikey": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        quote = data.get("Global Quote", {})

        price = quote.get("05. price")
        if price:
            change_pct = quote.get("10. change percent", "0%")
            change_pct = float(change_pct.replace("%", "")) if change_pct else 0

            return {
                "current_price": round(float(price), 2),
                "price_change": round(change_pct, 2),
                "sector": None,
                "industry": None,
                "volatility_proxy": None,
                "source": "Alpha Vantage",
                "enriched": "partial",
            }
        return None
    except Exception:
        return None


def enrich_stock(symbol: str, stock_name: str = None) -> dict:
    """
    Enrich a single stock with market data.
    Tries yfinance first, then Alpha Vantage, then returns unenriched stub.

    (Backward-compatible with existing pipeline)
    """
    clean_sym = symbol.strip().upper().replace(" ", "")

    result = _try_yfinance(clean_sym)
    if result:
        return result

    if stock_name and stock_name != clean_sym:
        alt_sym = stock_name.strip().upper().split()[0]
        result = _try_yfinance(alt_sym)
        if result:
            return result

    result = _try_alpha_vantage(clean_sym)
    if result:
        return result

    return {
        "current_price": None,
        "price_change": None,
        "sector": None,
        "industry": None,
        "volatility_proxy": None,
        "source": "not_enriched",
        "enriched": "no",
    }


def enrich_holdings(holdings: list) -> list:
    """
    Enrich a list of holdings dicts with real-time market data.
    (Backward-compatible with existing pipeline)
    """
    enriched = []
    for h in holdings:
        symbol = h.get("symbol") or ""
        name = h.get("stock_name") or ""
        lookup_key = symbol if symbol else name.split()[0] if name else ""

        if not lookup_key:
            h["market_data"] = {"enriched": "no", "source": "no_symbol"}
            enriched.append(h)
            continue

        market = enrich_stock(lookup_key, name)
        h["market_data"] = market

        if market.get("current_price") and market["enriched"] != "no":
            h["current_price"] = market["current_price"]
            if h.get("quantity"):
                h["total_value"] = round(h["quantity"] * market["current_price"], 2)

        enriched.append(h)
    return enriched


def compute_sector_exposure(holdings: list) -> dict:
    """Compute sector allocation from enriched holdings."""
    sector_values = {}
    total = 0

    for h in holdings:
        market = h.get("market_data", {})
        sector = market.get("sector", "Unknown") or "Unknown"
        val = h.get("total_value", 0) or 0
        sector_values[sector] = sector_values.get(sector, 0) + val
        total += val

    if total == 0:
        return {}

    return {
        sector: round((val / total) * 100, 2)
        for sector, val in sorted(sector_values.items(), key=lambda x: -x[1])
    }


def compute_portfolio_volatility(holdings: list) -> Optional[float]:
    """Estimate portfolio volatility from weighted individual volatilities."""
    weighted_sum = 0
    total_weight = 0

    for h in holdings:
        market = h.get("market_data", {})
        vol = market.get("volatility_proxy")
        weight = h.get("portfolio_weight", 0) or 0
        if vol and weight:
            weighted_sum += vol * (weight / 100)
            total_weight += weight / 100

    if total_weight > 0:
        return round(weighted_sum / total_weight, 4)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Fetch price data
    prices = get_price_data("AAPL", "2024-01-01", "2024-06-01")
    print("=== Price Data ===")
    print(prices.head())
    print(f"Shape: {prices.shape}\n")

    # Fetch fundamentals
    fund = get_fundamentals("AAPL")
    print("=== Fundamentals ===")
    print(fund.to_string())
    print()

    # Fetch technical indicators
    tech = get_technical_indicators("AAPL")
    print("=== Technical Indicators ===")
    print(tech.tail())
