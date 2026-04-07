"""
Risk Analysis Engine — Stages 4-8 (Enhanced)
Computes portfolio metrics, behavioral metrics, risk scores (with market risk),
investor profiles, insights, and recommendations.
"""

import math
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_portfolio_metrics(holdings: list) -> dict:
    """Compute portfolio-level metrics from holdings data."""
    metrics = {
        "total_portfolio_value": 0,
        "number_of_stocks": 0,
        "top_3_concentration_ratio": 0,
        "diversification_score": 0,
        "sector_exposure": {},
        "portfolio_volatility": None,
    }

    values = [h["total_value"] for h in holdings if h.get("total_value")]
    if not values:
        return metrics

    total = sum(values)
    n = len(values)
    metrics["total_portfolio_value"] = round(total, 2)
    metrics["number_of_stocks"] = n

    # Top-3 concentration
    sorted_vals = sorted(values, reverse=True)
    top3 = sum(sorted_vals[:3])
    metrics["top_3_concentration_ratio"] = round((top3 / total) * 100, 2) if total > 0 else 0

    # Diversification via normalized HHI
    weights = [v / total for v in values]
    hhi = sum(w ** 2 for w in weights)
    if n > 1:
        hhi_norm = (hhi - 1 / n) / (1 - 1 / n)
    else:
        hhi_norm = 1
    metrics["diversification_score"] = round(1 - hhi_norm, 4)

    # Sector exposure (from enriched data)
    sector_values = {}
    for h in holdings:
        market = h.get("market_data", {})
        if isinstance(market, dict):
            sector = market.get("sector", "Unknown") or "Unknown"
        else:
            sector = "Unknown"
        val = h.get("total_value", 0) or 0
        sector_values[sector] = sector_values.get(sector, 0) + val
    if total > 0:
        metrics["sector_exposure"] = {
            s: round((v / total) * 100, 2)
            for s, v in sorted(sector_values.items(), key=lambda x: -x[1])
        }

    # Portfolio volatility (weighted average)
    weighted_vol = 0
    vol_weight = 0
    for h in holdings:
        market = h.get("market_data", {})
        if isinstance(market, dict):
            vol = market.get("volatility_proxy")
            w = h.get("portfolio_weight", 0) or 0
            if vol and w:
                weighted_vol += vol * (w / 100)
                vol_weight += w / 100
    if vol_weight > 0:
        metrics["portfolio_volatility"] = round(weighted_vol / vol_weight, 4)

    return metrics


def compute_behavioral_metrics(trades: list) -> dict:
    """Compute trading behavior metrics from order history."""
    metrics = {
        "total_trades": 0,
        "trade_frequency": 0,
        "buy_sell_ratio": 0,
        "average_holding_period": None,
        "churn_rate": None,
    }

    if not trades:
        return metrics

    metrics["total_trades"] = len(trades)

    buys = [t for t in trades if t.get("transaction_type") == "BUY"]
    sells = [t for t in trades if t.get("transaction_type") == "SELL"]
    n_buys = len(buys)
    n_sells = len(sells)
    metrics["buy_sell_ratio"] = round(n_buys / n_sells, 2) if n_sells > 0 else (float("inf") if n_buys > 0 else 0)

    dates = [t.get("date") for t in trades if t.get("date")]
    if len(dates) >= 2:
        parsed = sorted([datetime.strptime(d, "%Y-%m-%d") for d in dates])
        span_days = (parsed[-1] - parsed[0]).days
        span_months = max(span_days / 30.44, 1)
        metrics["trade_frequency"] = round(len(trades) / span_months, 2)

    stock_buys = {}
    holding_days = []
    for t in sorted(trades, key=lambda x: x.get("date") or ""):
        name = t.get("stock_name")
        if not name or not t.get("date"):
            continue
        if t["transaction_type"] == "BUY":
            stock_buys.setdefault(name, []).append(t["date"])
        elif t["transaction_type"] == "SELL" and stock_buys.get(name):
            buy_date = stock_buys[name].pop(0)
            delta = (datetime.strptime(t["date"], "%Y-%m-%d") -
                     datetime.strptime(buy_date, "%Y-%m-%d")).days
            if delta >= 0:
                holding_days.append(delta)

    if holding_days:
        metrics["average_holding_period"] = round(sum(holding_days) / len(holding_days), 1)
    else:
        metrics["average_holding_period"] = "Unable to compute (insufficient buy-sell pairs)"

    if metrics["total_trades"] > 0:
        metrics["churn_rate"] = round(n_sells / metrics["total_trades"], 2)

    return metrics


def compute_performance_metrics(pnl: dict, trades: list) -> dict:
    """Compute performance metrics from P&L data."""
    metrics = {
        "total_realized_return": None,
        "win_loss_ratio": None,
        "consistency_score": None,
    }

    if pnl.get("realized_profit_loss") is not None:
        metrics["total_realized_return"] = pnl["realized_profit_loss"]

    records = pnl.get("records", [])
    if records:
        wins = sum(1 for r in records if r.get("pnl") and r["pnl"] > 0)
        losses = sum(1 for r in records if r.get("pnl") and r["pnl"] < 0)
        metrics["win_loss_ratio"] = round(wins / losses, 2) if losses > 0 else (float("inf") if wins > 0 else 0)

        pnl_values = [r["pnl"] for r in records if r.get("pnl") is not None]
        if len(pnl_values) >= 2:
            mean_pnl = sum(pnl_values) / len(pnl_values)
            if mean_pnl != 0:
                variance = sum((x - mean_pnl) ** 2 for x in pnl_values) / len(pnl_values)
                std = math.sqrt(variance)
                cv = abs(std / mean_pnl)
                metrics["consistency_score"] = round(max(0, 1 - min(cv, 1)), 4)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: RISK SCORING MODEL (Enhanced with Market Risk)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_concentration(portfolio_metrics: dict) -> float:
    conc = portfolio_metrics.get("top_3_concentration_ratio", 0)
    n_stocks = portfolio_metrics.get("number_of_stocks", 1)

    if conc >= 80:
        c_score = 9
    elif conc >= 65:
        c_score = 7
    elif conc >= 50:
        c_score = 5
    elif conc >= 35:
        c_score = 3
    else:
        c_score = 1

    if n_stocks <= 3:
        c_score = min(10, c_score + 2)
    elif n_stocks <= 5:
        c_score = min(10, c_score + 1)
    elif n_stocks >= 20:
        c_score = max(1, c_score - 1)

    return c_score


def _score_behavior(behavioral_metrics: dict) -> float:
    freq = behavioral_metrics.get("trade_frequency", 0)
    churn = behavioral_metrics.get("churn_rate", 0) or 0

    if freq >= 20:
        f_score = 9
    elif freq >= 10:
        f_score = 7
    elif freq >= 5:
        f_score = 5
    elif freq >= 2:
        f_score = 3
    else:
        f_score = 1

    churn_score = min(10, max(1, int(churn * 10)))
    return round((f_score * 0.6 + churn_score * 0.4), 1)


def _score_performance(performance_metrics: dict) -> float:
    wl = performance_metrics.get("win_loss_ratio")
    consistency = performance_metrics.get("consistency_score")

    score = 5
    if wl is not None and wl != float("inf"):
        if wl >= 2:
            score -= 2
        elif wl >= 1:
            score -= 1
        elif wl < 0.5:
            score += 2
        elif wl < 1:
            score += 1

    if consistency is not None:
        if consistency >= 0.7:
            score -= 1
        elif consistency <= 0.3:
            score += 2

    return max(1, min(10, score))


def _score_market_risk(portfolio_metrics: dict) -> float:
    """Score market risk based on portfolio volatility and sector concentration."""
    vol = portfolio_metrics.get("portfolio_volatility")
    sector_exp = portfolio_metrics.get("sector_exposure", {})

    score = 5  # default

    if vol is not None:
        if vol >= 0.5:
            score = 9
        elif vol >= 0.35:
            score = 7
        elif vol >= 0.25:
            score = 5
        elif vol >= 0.15:
            score = 3
        else:
            score = 1

    # Penalize sector concentration
    if sector_exp:
        top_sector_pct = max(sector_exp.values()) if sector_exp else 0
        if top_sector_pct >= 60:
            score = min(10, score + 2)
        elif top_sector_pct >= 40:
            score = min(10, score + 1)

    return max(1, min(10, score))


def compute_risk_score(portfolio_metrics: dict, behavioral_metrics: dict,
                       performance_metrics: dict) -> dict:
    """Compute composite risk score with 4 sub-scores including market risk."""
    conc_score = _score_concentration(portfolio_metrics)
    behav_score = _score_behavior(behavioral_metrics)
    perf_score = _score_performance(performance_metrics)
    market_score = _score_market_risk(portfolio_metrics)

    # Weighted: concentration 30%, behavior 25%, performance 25%, market 20%
    composite = round(
        conc_score * 0.30 + behav_score * 0.25 + perf_score * 0.25 + market_score * 0.20, 1
    )
    composite = max(1, min(10, composite))

    explanation_parts = [
        f"Concentration {conc_score}/10: Top-3 = {portfolio_metrics.get('top_3_concentration_ratio', 'N/A')}%, {portfolio_metrics.get('number_of_stocks', 'N/A')} stocks.",
        f"Behavior {behav_score}/10: {behavioral_metrics.get('trade_frequency', 'N/A')} trades/mo, churn = {behavioral_metrics.get('churn_rate', 'N/A')}.",
        f"Performance {perf_score}/10: W/L = {performance_metrics.get('win_loss_ratio', 'N/A')}, consistency = {performance_metrics.get('consistency_score', 'N/A')}.",
        f"Market {market_score}/10: volatility = {portfolio_metrics.get('portfolio_volatility', 'N/A')}, top sector = {max(portfolio_metrics.get('sector_exposure', {}).values(), default='N/A')}%.",
    ]

    return {
        "risk_score": composite,
        "sub_scores": {
            "concentration_score": conc_score,
            "behavior_score": behav_score,
            "performance_score": perf_score,
            "market_risk_score": market_score,
        },
        "explanation": " | ".join(explanation_parts),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6: INVESTOR PROFILING
# ═══════════════════════════════════════════════════════════════════════════════

def classify_investor(risk_analysis: dict, behavioral_metrics: dict,
                      portfolio_metrics: dict) -> str:
    risk = risk_analysis.get("risk_score", 5)
    freq = behavioral_metrics.get("trade_frequency", 0)
    holding = behavioral_metrics.get("average_holding_period")
    conc = portfolio_metrics.get("top_3_concentration_ratio", 0)

    if isinstance(holding, (int, float)):
        hold_days = holding
    else:
        hold_days = None

    if freq >= 15 or (hold_days is not None and hold_days < 7 and freq >= 5):
        return "Speculative / High-Risk Trader"
    if risk >= 7 or conc >= 70 or freq >= 8:
        return "Aggressive"
    if risk <= 3 and freq <= 2:
        if hold_days is None or hold_days >= 180:
            return "Conservative"
    return "Moderate"


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7-8: INSIGHT & RECOMMENDATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_insights(portfolio_metrics: dict, behavioral_metrics: dict,
                      performance_metrics: dict, risk_analysis: dict,
                      holdings: list) -> dict:
    strengths = []
    risks = []
    observations = []

    n_stocks = portfolio_metrics.get("number_of_stocks", 0)
    conc = portfolio_metrics.get("top_3_concentration_ratio", 0)
    div_score = portfolio_metrics.get("diversification_score", 0)
    freq = behavioral_metrics.get("trade_frequency", 0)
    churn = behavioral_metrics.get("churn_rate", 0) or 0
    wl = performance_metrics.get("win_loss_ratio")
    consistency = performance_metrics.get("consistency_score")
    holding = behavioral_metrics.get("average_holding_period")
    bsr = behavioral_metrics.get("buy_sell_ratio", 0)
    sector_exp = portfolio_metrics.get("sector_exposure", {})
    vol = portfolio_metrics.get("portfolio_volatility")

    # Strengths
    if n_stocks >= 10:
        strengths.append(f"Well-diversified portfolio with {n_stocks} stocks")
    elif n_stocks >= 5:
        strengths.append(f"Reasonable diversification with {n_stocks} stocks")
    if div_score >= 0.7:
        strengths.append(f"Excellent diversification score ({div_score:.2f})")
    if isinstance(holding, (int, float)) and holding >= 90:
        strengths.append(f"Patient holding behavior (avg {holding:.0f} days)")
    if wl is not None and wl != float("inf") and wl >= 1.5:
        strengths.append(f"Strong win/loss ratio of {wl:.2f}")
    if consistency is not None and consistency >= 0.6:
        strengths.append(f"Consistent return pattern (score: {consistency:.2f})")
    if freq <= 3:
        strengths.append("Low-frequency trading — disciplined approach")
    if vol is not None and vol < 0.2:
        strengths.append(f"Low portfolio volatility ({vol:.2%})")
    if not strengths:
        strengths.append("Portfolio data extracted successfully for analysis")

    # Risks
    if conc >= 60:
        top_names = sorted(holdings, key=lambda h: h.get("total_value") or 0, reverse=True)[:3]
        names = ", ".join(h.get("stock_name") or h.get("symbol", "?") for h in top_names)
        risks.append(f"High concentration: top 3 ({names}) = {conc:.1f}% of portfolio")
    if n_stocks <= 3:
        risks.append(f"Extremely concentrated with only {n_stocks} stock(s)")
    if freq >= 10:
        risks.append(f"Overtrading risk: {freq:.1f} trades/month")
    if wl is not None and wl != float("inf") and wl < 1:
        risks.append(f"More losing trades than winning (W/L = {wl:.2f})")
    if isinstance(holding, (int, float)) and holding < 30:
        risks.append(f"Short avg holding ({holding:.0f} days) — potential panic selling")
    if churn >= 0.5:
        risks.append(f"High churn rate ({churn:.2f})")
    if vol is not None and vol >= 0.35:
        risks.append(f"High portfolio volatility ({vol:.2%})")
    if sector_exp:
        top_sector = max(sector_exp.items(), key=lambda x: x[1], default=("Unknown", 0))
        if top_sector[1] >= 50:
            risks.append(f"Sector concentration: {top_sector[0]} = {top_sector[1]:.1f}%")
    if not risks:
        risks.append("No major structural risks detected from available data")

    # Behavioral observations
    if freq >= 10:
        observations.append("Overtrading: Frequent trading erodes returns through costs")
    if isinstance(holding, (int, float)) and holding < 14:
        observations.append("Panic selling: Very short holding periods suggest reactive exits")
    if conc >= 50:
        observations.append(f"Concentration bias: Top stocks = {conc:.0f}% of portfolio")
    if freq <= 3 and isinstance(holding, (int, float)) and holding >= 60:
        observations.append("Disciplined: Long holds + low frequency show patience")
    if bsr and bsr != float("inf") and bsr < 0.5:
        observations.append("Net seller: More sells than buys — possibly liquidating")
    if not observations:
        observations.append("Insufficient trade data for detailed behavioral analysis")

    return {
        "key_strengths": strengths[:3],
        "key_risks": risks[:3],
        "behavioral_observations": observations,
    }


def generate_recommendations(portfolio_metrics: dict, behavioral_metrics: dict,
                             performance_metrics: dict, risk_analysis: dict,
                             investor_profile: str) -> list:
    recs = []
    conc = portfolio_metrics.get("top_3_concentration_ratio", 0)
    n_stocks = portfolio_metrics.get("number_of_stocks", 0)
    freq = behavioral_metrics.get("trade_frequency", 0)
    churn = behavioral_metrics.get("churn_rate", 0) or 0
    holding = behavioral_metrics.get("average_holding_period")
    wl = performance_metrics.get("win_loss_ratio")
    risk_score = risk_analysis.get("risk_score", 5)
    vol = portfolio_metrics.get("portfolio_volatility")
    sector_exp = portfolio_metrics.get("sector_exposure", {})

    if conc >= 50:
        recs.append(
            f"Reduce concentration: Top 3 = {conc:.1f}%. Cap individual positions at 10-15%."
        )
    if n_stocks < 8:
        recs.append(
            f"Increase diversification: Only {n_stocks} stocks. Expand to 15-20 across sectors."
        )
    if freq >= 8:
        recs.append(
            f"Reduce trading: {freq:.1f} trades/month. Set minimum 30-day holding period."
        )
    if isinstance(holding, (int, float)) and holding < 30:
        recs.append(
            f"Extend holds: Avg {holding:.0f} days is short. 6-12 months improves returns."
        )
    if wl is not None and wl != float("inf") and wl < 1:
        recs.append(
            f"Improve selection: W/L = {wl:.2f}. Tighten entry criteria and use stop-losses."
        )
    if vol is not None and vol >= 0.35:
        recs.append(
            f"Reduce volatility: Portfolio vol = {vol:.2%}. Add large-cap stable stocks."
        )
    if sector_exp:
        top = max(sector_exp.items(), key=lambda x: x[1], default=("Unknown", 0))
        if top[1] >= 50:
            recs.append(
                f"Diversify sectors: {top[0]} = {top[1]:.1f}%. Spread across 4-5 sectors."
            )
    if risk_score >= 7:
        recs.append(
            f"Overall risk elevated ({risk_score}/10). Allocate 20-30% to index funds."
        )
    if investor_profile == "Speculative / High-Risk Trader":
        recs.append(
            "Split capital: 70% core long-term portfolio, 30% active trading account."
        )
    if not recs:
        recs.append("Portfolio well-structured. Continue monitoring and maintain discipline.")

    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(extracted_data: dict) -> dict:
    holdings = extracted_data.get("holdings", [])
    trades = extracted_data.get("trades", [])
    pnl = extracted_data.get("pnl", {})

    portfolio_metrics = compute_portfolio_metrics(holdings)
    behavioral_metrics = compute_behavioral_metrics(trades)
    performance_metrics = compute_performance_metrics(pnl, trades)
    risk_analysis = compute_risk_score(portfolio_metrics, behavioral_metrics, performance_metrics)
    investor_profile = classify_investor(risk_analysis, behavioral_metrics, portfolio_metrics)
    insights = generate_insights(
        portfolio_metrics, behavioral_metrics, performance_metrics, risk_analysis, holdings
    )
    recommendations = generate_recommendations(
        portfolio_metrics, behavioral_metrics, performance_metrics, risk_analysis, investor_profile
    )

    return {
        "portfolio_metrics": portfolio_metrics,
        "behavioral_metrics": behavioral_metrics,
        "performance_metrics": performance_metrics,
        "risk_analysis": risk_analysis,
        "investor_profile": investor_profile,
        "insights": insights,
        "recommendations": recommendations,
    }
