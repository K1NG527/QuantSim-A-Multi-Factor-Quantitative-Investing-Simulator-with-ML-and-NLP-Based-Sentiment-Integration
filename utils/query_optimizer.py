"""
Query Optimizer — QuantSim

Provides tools for analyzing and optimizing database query performance:
    - analyze_query_performance(): profile and log slow queries
    - suggest_index_improvements(): recommend missing indexes
    - log_query_stats(): persistent query statistics

Usage:
    from utils.query_optimizer import analyze_query_performance
    report = analyze_query_performance()
"""

import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

QUERY_LOG_DIR = Path(os.path.dirname(__file__)).parent / "data" / "query_logs"
QUERY_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Slow query threshold in seconds
SLOW_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD", "1.0"))


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY PROFILER
# ═══════════════════════════════════════════════════════════════════════════════

class QueryProfiler:
    """
    Context manager for profiling individual queries.

    Usage:
        with QueryProfiler("fetch holdings") as qp:
            session.query(Holding).filter_by(symbol="AAPL").all()
        print(f"Query took {qp.elapsed:.3f}s")
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.elapsed > SLOW_THRESHOLD:
            logger.warning(f"[SLOW QUERY] '{self.label}' took {self.elapsed:.3f}s")
        else:
            logger.debug(f"[QUERY] '{self.label}' took {self.elapsed:.3f}s")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY PERFORMANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_query_performance() -> dict:
    """
    Profile key database queries and report performance.

    Tests common query patterns:
        - User lookup
        - Snapshot retrieval with joins
        - Holdings by symbol
        - News by ticker
        - Sentiment scores by ticker + date
        - Model predictions by ticker

    Returns:
        Dict with: query_results (list of timing dicts), slow_queries (list),
        recommendations (list), total_time (float)

    Example:
        >>> report = analyze_query_performance()
        >>> for q in report["slow_queries"]:
        ...     print(f"SLOW: {q['label']} — {q['elapsed']:.3f}s")
    """
    try:
        from utils.db_models import (
            get_session, User, PortfolioSnapshot, Holding,
            Transaction, News, SentimentScore, ModelPrediction,
        )
    except ImportError as e:
        logger.error(f"Cannot import db_models: {e}")
        return {"error": str(e)}

    session = get_session()
    results = []

    # Define test queries
    test_queries = [
        ("User lookup by ID", lambda: session.query(User).filter_by(user_id="investor_1").first()),
        ("All snapshots for user", lambda: session.query(PortfolioSnapshot).filter_by(user_id="investor_1").all()),
        ("Latest snapshot with holdings (JOIN)", lambda: (
            session.query(PortfolioSnapshot)
            .filter_by(user_id="investor_1")
            .order_by(PortfolioSnapshot.timestamp.desc())
            .first()
        )),
        ("Holdings by symbol", lambda: session.query(Holding).filter_by(symbol="AAPL").limit(100).all()),
        ("Recent news by ticker", lambda: (
            session.query(News)
            .filter_by(ticker="AAPL")
            .order_by(News.published_at.desc())
            .limit(50)
            .all()
        )),
        ("Sentiment scores by ticker", lambda: (
            session.query(SentimentScore)
            .filter_by(ticker="AAPL")
            .order_by(SentimentScore.date.desc())
            .limit(30)
            .all()
        )),
        ("Model predictions by ticker", lambda: (
            session.query(ModelPrediction)
            .filter_by(ticker="AAPL")
            .order_by(ModelPrediction.date.desc())
            .limit(30)
            .all()
        )),
        ("Transactions by user + date range", lambda: (
            session.query(Transaction)
            .filter(Transaction.user_id == "investor_1")
            .filter(Transaction.date >= "2024-01-01")
            .all()
        )),
    ]

    total_time = 0
    slow_queries = []

    for label, query_fn in test_queries:
        with QueryProfiler(label) as qp:
            try:
                query_fn()
            except Exception as e:
                logger.debug(f"Query '{label}' failed (table may not exist): {e}")

        entry = {
            "label": label,
            "elapsed": round(qp.elapsed, 4),
            "is_slow": qp.elapsed > SLOW_THRESHOLD,
        }
        results.append(entry)
        total_time += qp.elapsed

        if entry["is_slow"]:
            slow_queries.append(entry)

    session.close()

    # Generate recommendations
    recommendations = _generate_recommendations(results, slow_queries)

    report = {
        "timestamp": datetime.now().isoformat(),
        "query_results": results,
        "slow_queries": slow_queries,
        "total_time": round(total_time, 4),
        "num_queries": len(results),
        "num_slow": len(slow_queries),
        "recommendations": recommendations,
    }

    # Save report
    _save_report(report)

    logger.info(
        f"Query analysis complete: {len(results)} queries in {total_time:.3f}s, "
        f"{len(slow_queries)} slow"
    )
    return report


def _generate_recommendations(results: list, slow_queries: list) -> list:
    """Generate optimization recommendations based on query timing results."""
    recs = []

    if not slow_queries:
        recs.append("✅ All queries are within the performance threshold. No immediate action needed.")
        return recs

    for sq in slow_queries:
        label = sq["label"]
        elapsed = sq["elapsed"]

        if "JOIN" in label:
            recs.append(
                f"⚠️ '{label}' is slow ({elapsed:.3f}s). "
                f"Consider using eager loading (joinedload) or adding composite indexes."
            )
        elif "symbol" in label.lower() or "ticker" in label.lower():
            recs.append(
                f"⚠️ '{label}' is slow ({elapsed:.3f}s). "
                f"Verify that 'ticker'/'symbol' columns have B-tree indexes."
            )
        elif "date" in label.lower():
            recs.append(
                f"⚠️ '{label}' is slow ({elapsed:.3f}s). "
                f"Consider adding a composite index on (user_id, date)."
            )
        else:
            recs.append(
                f"⚠️ '{label}' is slow ({elapsed:.3f}s). "
                f"Review query plan and consider indexing frequently filtered columns."
            )

    if len(slow_queries) >= 3:
        recs.append(
            "🔴 Multiple slow queries detected. Consider: "
            "1) Adding connection pooling, "
            "2) Reviewing table indexes, "
            "3) Enabling query plan caching."
        )

    return recs


def suggest_index_improvements() -> list:
    """
    Suggest indexing improvements based on table schema analysis.

    Returns:
        List of suggestion strings

    Example:
        >>> suggestions = suggest_index_improvements()
        >>> for s in suggestions:
        ...     print(s)
    """
    suggestions = []

    try:
        from utils.db_models import get_engine
        from sqlalchemy import inspect

        engine = get_engine()
        inspector = inspect(engine)

        tables_to_check = {
            "news": ["ticker", "published_at"],
            "sentiment_scores": ["ticker", "date"],
            "model_predictions": ["ticker", "date", "model_version"],
            "holdings": ["symbol", "snapshot_id"],
            "transactions": ["user_id", "date"],
            "portfolio_snapshots": ["user_id", "timestamp"],
        }

        for table, expected_indexed_cols in tables_to_check.items():
            try:
                indexes = inspector.get_indexes(table)
                indexed_cols = set()
                for idx in indexes:
                    for col in idx["column_names"]:
                        indexed_cols.add(col)

                for col in expected_indexed_cols:
                    if col not in indexed_cols:
                        suggestions.append(
                            f"📌 Table '{table}': Column '{col}' should be indexed "
                            f"for faster queries."
                        )
            except Exception:
                suggestions.append(f"ℹ️ Table '{table}' does not exist yet. Run init_db() first.")

    except Exception as e:
        suggestions.append(f"⚠️ Could not inspect database: {e}")

    if not suggestions:
        suggestions.append("✅ All recommended indexes are in place.")

    return suggestions


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _save_report(report: dict) -> str:
    """Save query performance report to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = QUERY_LOG_DIR / f"query_report_{timestamp}.json"
    try:
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Query report saved to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to save query report: {e}")
        return ""


def get_query_history(limit: int = 10) -> list:
    """
    Load recent query performance reports.

    Args:
        limit: Max number of reports to load

    Returns:
        List of report dicts, newest first
    """
    reports = []
    try:
        files = sorted(QUERY_LOG_DIR.glob("query_report_*.json"), reverse=True)[:limit]
        for f in files:
            with open(f, "r") as fh:
                reports.append(json.load(fh))
    except Exception as e:
        logger.error(f"Failed to load query history: {e}")
    return reports


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Query Performance Analysis ===")
    report = analyze_query_performance()

    print(f"\nTotal queries: {report['num_queries']}")
    print(f"Total time:   {report['total_time']:.3f}s")
    print(f"Slow queries: {report['num_slow']}")

    print("\n--- Results ---")
    for q in report["query_results"]:
        flag = "🔴" if q["is_slow"] else "✅"
        print(f"  {flag} {q['label']}: {q['elapsed']:.4f}s")

    print("\n--- Recommendations ---")
    for r in report["recommendations"]:
        print(f"  {r}")

    print("\n=== Index Suggestions ===")
    for s in suggest_index_improvements():
        print(f"  {s}")
