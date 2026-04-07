"""
News + Sentiment Pipeline — QuantSim

Pipeline:
    1. Fetch company-specific news from Finnhub API
    2. Store news in database (News table)
    3. Run headlines through FinBERT for sentiment analysis
    4. Generate sentiment_score per stock
    5. Store scores in database (SentimentScores table)

Features:
    - Batch processing for multiple tickers
    - Graceful fallback if Finnhub or FinBERT unavailable
    - Database integration via SQLAlchemy
    - Logging throughout
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS FETCHING — FINNHUB
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_company_news(ticker: str, days_back: int = 30) -> list[dict]:
    """
    Fetch company-specific news from Finnhub API.

    Args:
        ticker:    Stock symbol, e.g. "AAPL"
        days_back: Number of days to look back for news (default 30)

    Returns:
        List of news dicts with keys: headline, summary, url, source,
        published_at, ticker

    Example:
        >>> news = fetch_company_news("AAPL", days_back=7)
        >>> print(f"Found {len(news)} articles")
    """
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "your_key_here":
        logger.warning("FINNHUB_API_KEY not configured — skipping news fetch")
        return []

    try:
        import finnhub
        client = finnhub.Client(api_key=FINNHUB_API_KEY)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        raw_news = client.company_news(
            ticker,
            _from=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d"),
        )

        if not raw_news:
            logger.info(f"No news found for {ticker}")
            return []

        articles = []
        for item in raw_news:
            articles.append({
                "ticker": ticker,
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("datetime", 0)
                ).strftime("%Y-%m-%d %H:%M:%S") if item.get("datetime") else None,
                "category": item.get("category", ""),
            })

        logger.info(f"Fetched {len(articles)} news articles for {ticker}")
        return articles

    except ImportError:
        logger.warning("finnhub-python not installed. Run: pip install finnhub-python")
        return []
    except Exception as e:
        logger.error(f"Finnhub news fetch failed for {ticker}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS — FINBERT
# ═══════════════════════════════════════════════════════════════════════════════

_finbert_pipeline = None


def _get_finbert_pipeline():
    """Lazy-load FinBERT sentiment pipeline (downloads ~440 MB on first use)."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading FinBERT model (this may take a moment on first run)...")
            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,  # return all scores
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return None
    return _finbert_pipeline


def analyze_sentiment(texts: list[str]) -> list[dict]:
    """
    Run a list of text strings through FinBERT.

    Args:
        texts: List of headline/summary strings

    Returns:
        List of dicts with keys: label ("positive"/"negative"/"neutral"),
        score (float), raw_scores (dict)

    Example:
        >>> results = analyze_sentiment(["Apple beats Q3 earnings expectations"])
        >>> print(results[0]["label"], results[0]["score"])
    """
    if not texts:
        return []

    pipe = _get_finbert_pipeline()
    if pipe is None:
        logger.warning("FinBERT unavailable — returning neutral scores")
        return [{"label": "neutral", "score": 0.0, "raw_scores": {}} for _ in texts]

    results = []
    # Process in batches to avoid OOM
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Truncate long texts to 512 tokens worth (FinBERT max)
        batch = [t[:512] if t else "" for t in batch]

        try:
            predictions = pipe(batch)
            for pred in predictions:
                # pred is a list of dicts: [{"label": "positive", "score": 0.9}, ...]
                raw_scores = {p["label"]: round(p["score"], 4) for p in pred}
                best = max(pred, key=lambda x: x["score"])

                # Convert to numeric: positive=+1, negative=-1, neutral=0
                label = best["label"].lower()
                if label == "positive":
                    numeric_score = best["score"]
                elif label == "negative":
                    numeric_score = -best["score"]
                else:
                    numeric_score = 0.0

                results.append({
                    "label": label,
                    "score": round(numeric_score, 4),
                    "raw_scores": raw_scores,
                })
        except Exception as e:
            logger.error(f"FinBERT batch prediction failed: {e}")
            results.extend([{"label": "neutral", "score": 0.0, "raw_scores": {}} for _ in batch])

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def store_news_in_db(articles: list[dict]) -> int:
    """
    Store news articles in the database News table.

    Args:
        articles: List of article dicts from fetch_company_news()

    Returns:
        Number of articles stored
    """
    if not articles:
        return 0

    try:
        from utils.db_models import get_session, News
        session = get_session()
        count = 0
        for article in articles:
            try:
                news = News(
                    ticker=article["ticker"],
                    headline=article["headline"][:500],
                    summary=article.get("summary", "")[:1000],
                    url=article.get("url", "")[:500],
                    source=article.get("source", "")[:128],
                    published_at=datetime.strptime(
                        article["published_at"], "%Y-%m-%d %H:%M:%S"
                    ) if article.get("published_at") else None,
                )
                session.add(news)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to store article: {e}")
                continue
        session.commit()
        session.close()
        logger.info(f"Stored {count} news articles in database")
        return count
    except Exception as e:
        logger.error(f"Database storage failed: {e}")
        return 0


def store_sentiment_in_db(ticker: str, score: float, label: str) -> bool:
    """
    Store a sentiment score in the database SentimentScores table.

    Args:
        ticker: Stock symbol
        score:  Numeric sentiment score (-1 to +1)
        label:  Sentiment label (positive/negative/neutral)

    Returns:
        True if stored successfully
    """
    try:
        from utils.db_models import get_session, SentimentScore
        session = get_session()
        record = SentimentScore(
            ticker=ticker,
            date=datetime.now().strftime("%Y-%m-%d"),
            sentiment_score=round(score, 4),
            sentiment_label=label,
            model_version="ProsusAI/finbert",
        )
        session.add(record)
        session.commit()
        session.close()
        return True
    except Exception as e:
        logger.error(f"Failed to store sentiment for {ticker}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sentiment_score(ticker: str, days_back: int = 30,
                            store_db: bool = True) -> dict:
    """
    Full sentiment pipeline for a single ticker:
    1. Fetch news from Finnhub
    2. Analyze sentiment with FinBERT
    3. Aggregate to single score
    4. Optionally store in database

    Args:
        ticker:    Stock symbol, e.g. "AAPL"
        days_back: Days of news to analyze (default 30)
        store_db:  Whether to persist results to database

    Returns:
        Dict with: sentiment_score, sentiment_label, num_articles,
        positive_pct, negative_pct, neutral_pct

    Example:
        >>> result = compute_sentiment_score("AAPL")
        >>> print(f"Sentiment: {result['sentiment_score']:.2f} ({result['sentiment_label']})")
    """
    # Step 1: Fetch news
    articles = fetch_company_news(ticker, days_back=days_back)

    if not articles:
        logger.info(f"No news for {ticker} — returning neutral sentiment")
        return {
            "ticker": ticker,
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "num_articles": 0,
            "positive_pct": 0,
            "negative_pct": 0,
            "neutral_pct": 100,
        }

    # Optionally store news
    if store_db:
        store_news_in_db(articles)

    # Step 2: Analyze sentiment
    headlines = [a["headline"] for a in articles if a.get("headline")]
    sentiments = analyze_sentiment(headlines)

    if not sentiments:
        return {
            "ticker": ticker,
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "num_articles": len(articles),
            "positive_pct": 0,
            "negative_pct": 0,
            "neutral_pct": 100,
        }

    # Step 3: Aggregate scores
    scores = [s["score"] for s in sentiments]
    avg_score = sum(scores) / len(scores)

    labels = [s["label"] for s in sentiments]
    total = len(labels)
    pos_pct = round(labels.count("positive") / total * 100, 1)
    neg_pct = round(labels.count("negative") / total * 100, 1)
    neu_pct = round(labels.count("neutral") / total * 100, 1)

    if avg_score > 0.1:
        agg_label = "positive"
    elif avg_score < -0.1:
        agg_label = "negative"
    else:
        agg_label = "neutral"

    # Step 4: Store sentiment
    if store_db:
        store_sentiment_in_db(ticker, avg_score, agg_label)

    result = {
        "ticker": ticker,
        "sentiment_score": round(avg_score, 4),
        "sentiment_label": agg_label,
        "num_articles": len(articles),
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct": neu_pct,
    }
    logger.info(f"Sentiment for {ticker}: {avg_score:.4f} ({agg_label}) from {len(articles)} articles")
    return result


def compute_sentiment_batch(tickers: list[str], days_back: int = 30,
                            store_db: bool = True) -> pd.DataFrame:
    """
    Run sentiment pipeline for multiple tickers.

    Args:
        tickers:   List of stock symbols
        days_back: Days of news to analyze
        store_db:  Whether to persist results

    Returns:
        DataFrame with columns: ticker, sentiment_score, sentiment_label,
        num_articles, positive_pct, negative_pct, neutral_pct

    Example:
        >>> df = compute_sentiment_batch(["AAPL", "MSFT", "GOOGL"])
        >>> print(df[["ticker", "sentiment_score", "sentiment_label"]])
    """
    results = []
    for ticker in tickers:
        result = compute_sentiment_score(ticker, days_back=days_back, store_db=store_db)
        results.append(result)

    df = pd.DataFrame(results)
    logger.info(f"Sentiment analysis complete for {len(tickers)} tickers")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Single ticker
    result = compute_sentiment_score("AAPL", days_back=7, store_db=False)
    print("=== Single Ticker Sentiment ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Batch
    df = compute_sentiment_batch(["AAPL", "MSFT", "GOOGL"], days_back=7, store_db=False)
    print("\n=== Batch Sentiment ===")
    print(df.to_string(index=False))
