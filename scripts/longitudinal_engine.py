"""
Longitudinal Analysis Engine — Stage 7
Compares current portfolio snapshot with historical data from the database
to detect behavioral shifts and risk trends.
"""

import json
from datetime import datetime


def load_history(user_id: str) -> list:
    """
    Load previous portfolio snapshots from the database.
    Returns list of dicts sorted by timestamp (most recent first).
    """
    try:
        from utils.db_models import get_session, PortfolioSnapshot
        session = get_session()
        snapshots = (
            session.query(PortfolioSnapshot)
            .filter_by(user_id=user_id)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(20)
            .all()
        )

        history = []
        for snap in snapshots:
            entry = {
                "snapshot_id": snap.snapshot_id,
                "timestamp": snap.timestamp.isoformat() if snap.timestamp else None,
                "total_value": snap.total_value,
                "risk_score": snap.risk_score,
                "risk_trend": snap.risk_trend,
                "investor_profile": snap.investor_profile,
            }
            # Parse stored JSON for detailed comparison
            if snap.raw_json:
                try:
                    entry["full_data"] = json.loads(snap.raw_json)
                except json.JSONDecodeError:
                    entry["full_data"] = None
            else:
                entry["full_data"] = None

            history.append(entry)

        session.close()
        return history

    except Exception as e:
        print(f"[Longitudinal] Could not load history: {e}")
        return []


def save_snapshot(user_id: str, analysis_result: dict) -> int | None:
    """
    Save current analysis as a new portfolio snapshot in the database.
    Returns the snapshot_id or None on failure.
    """
    try:
        from utils.db_models import (
            get_session, User, PortfolioSnapshot, Holding, Transaction, Metric
        )
        session = get_session()

        # Ensure user exists
        user = session.query(User).filter_by(user_id=user_id).first()
        if not user:
            user = User(user_id=user_id)
            session.add(user)
            session.flush()

        # Create snapshot
        pm = analysis_result.get("portfolio_metrics", {})
        ra = analysis_result.get("risk_analysis", {})
        snapshot = PortfolioSnapshot(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            total_value=pm.get("total_portfolio_value"),
            risk_score=float(ra.get("risk_score", 0)),
            risk_trend=analysis_result.get("snapshot", {}).get("risk_trend", "stable"),
            investor_profile=analysis_result.get("investor_profile"),
            raw_json=json.dumps(analysis_result, default=str),
        )
        session.add(snapshot)
        session.flush()
        sid = snapshot.snapshot_id

        # Save holdings
        for h in analysis_result.get("extracted_data", {}).get("holdings", []):
            holding = Holding(
                snapshot_id=sid,
                stock_name=h.get("stock_name", ""),
                symbol=h.get("standardized_symbol") or h.get("symbol"),
                quantity=h.get("quantity"),
                avg_price=h.get("average_buy_price"),
                current_price=h.get("current_price"),
                total_value=h.get("total_value"),
                weight=h.get("portfolio_weight"),
                sector=h.get("market_data", {}).get("sector") if isinstance(h.get("market_data"), dict) else None,
                volatility=h.get("market_data", {}).get("volatility_proxy") if isinstance(h.get("market_data"), dict) else None,
                enriched=h.get("market_data", {}).get("enriched", "no") if isinstance(h.get("market_data"), dict) else "no",
            )
            session.add(holding)

        # Save transactions
        for t in analysis_result.get("extracted_data", {}).get("trades", []):
            txn = Transaction(
                user_id=user_id,
                stock_name=t.get("stock_name", ""),
                date=t.get("date"),
                transaction_type=t.get("transaction_type"),
                quantity=t.get("quantity"),
                price=t.get("price"),
            )
            session.add(txn)

        # Save metrics
        bm = analysis_result.get("behavioral_metrics", {})
        metric = Metric(
            snapshot_id=sid,
            diversification_score=pm.get("diversification_score"),
            concentration_ratio=pm.get("top_3_concentration_ratio"),
            turnover_rate=bm.get("churn_rate"),
            volatility_estimate=pm.get("portfolio_volatility"),
            trade_frequency=bm.get("trade_frequency"),
            buy_sell_ratio=bm.get("buy_sell_ratio") if bm.get("buy_sell_ratio") != float("inf") else None,
            avg_holding_period=bm.get("average_holding_period") if isinstance(bm.get("average_holding_period"), (int, float)) else None,
        )
        session.add(metric)

        session.commit()
        session.close()
        return sid

    except Exception as e:
        print(f"[Longitudinal] Could not save snapshot: {e}")
        return None


def compute_longitudinal(current: dict, history: list) -> dict:
    """
    Compare current analysis with historical snapshots.
    Returns longitudinal insights including trends and behavioral shifts.
    """
    result = {
        "has_history": len(history) > 0,
        "snapshots_count": len(history),
        "risk_trend": "stable",
        "value_change": None,
        "risk_change": None,
        "trend_direction": "stable",
        "behavioral_shifts": [],
        "previous_snapshot": None,
    }

    if not history:
        return result

    # Most recent previous snapshot
    prev = history[0]
    result["previous_snapshot"] = {
        "timestamp": prev.get("timestamp"),
        "total_value": prev.get("total_value"),
        "risk_score": prev.get("risk_score"),
        "investor_profile": prev.get("investor_profile"),
    }

    # Value change
    curr_val = current.get("portfolio_metrics", {}).get("total_portfolio_value", 0)
    prev_val = prev.get("total_value", 0)
    if curr_val and prev_val:
        result["value_change"] = {
            "absolute": round(curr_val - prev_val, 2),
            "percentage": round(((curr_val - prev_val) / prev_val) * 100, 2) if prev_val > 0 else None,
        }

    # Risk score change
    curr_risk = float(current.get("risk_analysis", {}).get("risk_score", 0))
    prev_risk = prev.get("risk_score", 0) or 0
    if curr_risk and prev_risk:
        risk_delta = curr_risk - prev_risk
        result["risk_change"] = round(risk_delta, 1)
        if risk_delta > 0.5:
            result["risk_trend"] = "increasing"
            result["trend_direction"] = "increasing"
        elif risk_delta < -0.5:
            result["risk_trend"] = "decreasing"
            result["trend_direction"] = "decreasing"
        else:
            result["risk_trend"] = "stable"

    # Behavioral shifts (compare with previous full data)
    prev_data = prev.get("full_data", {})
    if prev_data:
        prev_bm = prev_data.get("behavioral_metrics", {})
        curr_bm = current.get("behavioral_metrics", {})

        # Trading frequency change
        prev_freq = prev_bm.get("trade_frequency", 0)
        curr_freq = curr_bm.get("trade_frequency", 0)
        if prev_freq and curr_freq:
            if curr_freq > prev_freq * 1.5:
                result["behavioral_shifts"].append(
                    f"Trading frequency increased significantly: {prev_freq:.1f} → {curr_freq:.1f} trades/month"
                )
            elif curr_freq < prev_freq * 0.5:
                result["behavioral_shifts"].append(
                    f"Trading frequency decreased: {prev_freq:.1f} → {curr_freq:.1f} trades/month"
                )

        # Concentration change
        prev_conc = prev_data.get("portfolio_metrics", {}).get("top_3_concentration_ratio", 0)
        curr_conc = current.get("portfolio_metrics", {}).get("top_3_concentration_ratio", 0)
        if prev_conc and curr_conc:
            if curr_conc > prev_conc + 10:
                result["behavioral_shifts"].append(
                    f"Portfolio concentration increased: {prev_conc:.1f}% → {curr_conc:.1f}%"
                )
            elif curr_conc < prev_conc - 10:
                result["behavioral_shifts"].append(
                    f"Portfolio became more diversified: {prev_conc:.1f}% → {curr_conc:.1f}%"
                )

        # Profile change
        prev_profile = prev.get("investor_profile")
        curr_profile = current.get("investor_profile")
        if prev_profile and curr_profile and prev_profile != curr_profile:
            result["behavioral_shifts"].append(
                f"Investor profile shifted: {prev_profile} → {curr_profile}"
            )

    # Multi-snapshot trend analysis (if enough history)
    if len(history) >= 3:
        recent_risks = [h.get("risk_score", 0) for h in history[:5] if h.get("risk_score")]
        if len(recent_risks) >= 3:
            if all(recent_risks[i] <= recent_risks[i + 1] for i in range(len(recent_risks) - 1)):
                result["behavioral_shifts"].append("Consistent risk reduction trend over recent snapshots")
            elif all(recent_risks[i] >= recent_risks[i + 1] for i in range(len(recent_risks) - 1)):
                result["behavioral_shifts"].append("⚠️ Risk has been steadily increasing across snapshots")

    return result
