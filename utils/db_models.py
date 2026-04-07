"""
Database ORM Models — SQLAlchemy models for EquiSense portfolio intelligence platform.

Tables:
    - users
    - portfolio_snapshots
    - holdings
    - transactions
    - metrics
    - news (NEW)
    - sentiment_scores (NEW)
    - model_predictions (NEW)

Features:
    - PostgreSQL support (via .env) with SQLite fallback
    - Proper indexing on ticker, date, user_id
    - Query performance logging
    - Optimized session management
"""

import os
import logging
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, ForeignKey,
    Index, create_engine, event,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

Base = declarative_base()

# Query performance threshold (seconds)
SLOW_QUERY_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD", "1.0"))


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTING ORM MODELS (preserved)
# ═══════════════════════════════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    snapshots = relationship("PortfolioSnapshot", back_populates="user", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_users_created_at", "created_at"),
    )


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_value = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    risk_trend = Column(String(20), nullable=True)
    investor_profile = Column(String(64), nullable=True)
    raw_json = Column(Text, nullable=True)

    user = relationship("User", back_populates="snapshots")
    holdings = relationship("Holding", back_populates="snapshot", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="snapshot", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_snapshots_user_timestamp", "user_id", "timestamp"),
    )


class Holding(Base):
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("portfolio_snapshots.snapshot_id"), nullable=False, index=True)
    stock_name = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=True, index=True)
    quantity = Column(Integer, nullable=True)
    avg_price = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    total_value = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    sector = Column(String(64), nullable=True)
    volatility = Column(Float, nullable=True)
    enriched = Column(String(10), default="no")

    snapshot = relationship("PortfolioSnapshot", back_populates="holdings")

    __table_args__ = (
        Index("ix_holdings_symbol_snapshot", "symbol", "snapshot_id"),
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False, index=True)
    stock_name = Column(String(128), nullable=False)
    date = Column(String(10), nullable=True, index=True)
    transaction_type = Column(String(10), nullable=True)
    quantity = Column(Integer, nullable=True)
    price = Column(Float, nullable=True)

    user = relationship("User", back_populates="transactions")

    __table_args__ = (
        Index("ix_transactions_user_date", "user_id", "date"),
    )


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("portfolio_snapshots.snapshot_id"), nullable=False, unique=True)
    diversification_score = Column(Float, nullable=True)
    concentration_ratio = Column(Float, nullable=True)
    turnover_rate = Column(Float, nullable=True)
    volatility_estimate = Column(Float, nullable=True)
    trade_frequency = Column(Float, nullable=True)
    buy_sell_ratio = Column(Float, nullable=True)
    avg_holding_period = Column(Float, nullable=True)

    snapshot = relationship("PortfolioSnapshot", back_populates="metrics")


# ═══════════════════════════════════════════════════════════════════════════════
# NEW ORM MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class News(Base):
    """Stores news articles fetched from Finnhub or other sources."""
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False, index=True)
    headline = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    url = Column(String(500), nullable=True)
    source = Column(String(128), nullable=True)
    published_at = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_news_ticker_published", "ticker", "published_at"),
    )


class SentimentScore(Base):
    """Stores FinBERT sentiment scores per ticker per date."""
    __tablename__ = "sentiment_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)
    sentiment_label = Column(String(20), nullable=False)  # positive/negative/neutral
    model_version = Column(String(64), default="ProsusAI/finbert")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_sentiment_ticker_date", "ticker", "date"),
    )


class ModelPrediction(Base):
    """Stores ML model predictions for backtesting and tracking."""
    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), nullable=False, index=True)
    date = Column(String(10), nullable=False, index=True)
    predicted_return = Column(Float, nullable=False)
    model_version = Column(String(64), default="xgboost_v1")
    features_json = Column(Text, nullable=True)  # JSON string of input features
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_predictions_ticker_date", "ticker", "date"),
        Index("ix_predictions_model_version", "model_version"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE & SESSION FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_engine = None  # Singleton engine


def get_engine():
    """
    Build SQLAlchemy engine (singleton).
    Uses PostgreSQL if .env is configured, otherwise falls back to SQLite.
    Includes query performance logging.
    """
    global _engine
    if _engine is not None:
        return _engine

    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if all([user, password, host, port, db_name]):
        encoded_pw = quote_plus(password)
        url = f"postgresql+psycopg2://{user}:{encoded_pw}@{host}:{port}/{db_name}"
        logger.info("[DB] Connecting to PostgreSQL...")
    else:
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "equisense.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        url = f"sqlite:///{os.path.abspath(db_path)}"
        logger.info(f"[DB] PostgreSQL not configured. Using SQLite: {db_path}")

    _engine = create_engine(url, echo=False, pool_pre_ping=True)

    # Query performance logging
    @event.listens_for(_engine, "before_cursor_execute")
    def _before_execute(conn, cursor, statement, parameters, context, executemany):
        import time
        conn.info.setdefault("query_start_time", []).append(time.time())

    @event.listens_for(_engine, "after_cursor_execute")
    def _after_execute(conn, cursor, statement, parameters, context, executemany):
        import time
        start_times = conn.info.get("query_start_time", [])
        if start_times:
            elapsed = time.time() - start_times.pop()
            if elapsed > SLOW_QUERY_THRESHOLD:
                logger.warning(
                    f"[DB] SLOW QUERY ({elapsed:.3f}s): {statement[:200]}..."
                )

    return _engine


def get_session():
    """Create and return a new SQLAlchemy session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Create all tables in the configured database."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("[DB] ✅ All tables created successfully (including News, SentimentScores, ModelPredictions).")
    return engine
